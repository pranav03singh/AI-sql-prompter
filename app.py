"""Friendly natural-language to read-only SQLite SQL using LangChain and Ollama Cloud."""

from __future__ import annotations

import os
import sqlite3
import sys
from pathlib import Path
from typing import Any

import sqlglot
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableBranch, RunnableLambda
from langchain_openai import ChatOpenAI
from sqlglot import exp

from download_db import DEFAULT_DB_PATH, download_db

OLLAMA_BASE_URL = "https://ollama.com/v1"
# Ollama's cloud HTTP API uses the API model ID without the CLI `-cloud` suffix.
OLLAMA_MODEL = "gpt-oss:120b"

QUERY_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You translate questions into SQLite SQL for the supplied database. "
            "Return exactly one read-only SELECT query, with no markdown or explanation. "
            "Never generate INSERT, UPDATE, DELETE, REPLACE, CREATE, ALTER, DROP, "
            "ATTACH, PRAGMA, or multiple statements. Use only tables and columns "
            "present in the supplied schema. If the question cannot be answered "
            "from that schema, return a SELECT query that clearly indicates no answer.",
        ),
        ("human", "Database schema:\n{schema}\n\nQuestion:\n{question}"),
    ]
)


def analyse_db(db_path: str | Path = DEFAULT_DB_PATH) -> str:
    """Return table and column metadata without reading table contents."""
    path = Path(db_path)
    if not path.is_file():
        raise FileNotFoundError(f"SQLite database not found: {path}")

    # URI read-only mode protects the downloaded source from accidental writes.
    connection = sqlite3.connect(f"file:{path.resolve()}?mode=ro", uri=True)
    try:
        tables = connection.execute(
            "SELECT name FROM sqlite_master "
            "WHERE type = 'table' AND name NOT LIKE 'sqlite_%' ORDER BY name"
        ).fetchall()
        schema_parts: list[str] = []
        for (table_name,) in tables:
            escaped_name = table_name.replace('"', '""')
            columns = connection.execute(
                f'PRAGMA table_info("{escaped_name}")'
            ).fetchall()
            foreign_keys = connection.execute(
                f'PRAGMA foreign_key_list("{escaped_name}")'
            ).fetchall()

            column_lines = []
            for _, name, declared_type, not_null, default_value, primary_key_order in columns:
                details = [declared_type or "UNDECLARED"]
                if primary_key_order:
                    details.append(f"PRIMARY KEY position {primary_key_order}")
                if not_null:
                    details.append("NOT NULL")
                if default_value is not None:
                    details.append(f"DEFAULT {default_value}")
                column_lines.append(f"  {name}: {', '.join(details)}")

            schema_parts.append(f"Table {table_name}:\n" + "\n".join(column_lines))
            for _, _, referenced_table, from_column, to_column, *_ in foreign_keys:
                schema_parts.append(
                    f"  Foreign key: {table_name}.{from_column} -> "
                    f"{referenced_table}.{to_column}"
                )

        if not schema_parts:
            raise ValueError(f"No user tables found in SQLite database: {path}")
        return "\n".join(schema_parts)
    finally:
        connection.close()


def make_llm() -> ChatOpenAI:
    """Build a LangChain chat model for Ollama Cloud's OpenAI-compatible API."""
    api_key = os.getenv("OLLAMA_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OLLAMA_API_KEY is missing. Add it to .env (see .env.example)."
        )
    return ChatOpenAI(
        model=OLLAMA_MODEL,
        base_url=OLLAMA_BASE_URL,
        api_key=api_key,
        temperature=0,
    )


def build_query_chain(model: Any | None = None) -> Any:
    """Compose the prompt, model, and text parser into a LangChain chain."""
    return QUERY_PROMPT | (model if model is not None else make_llm()) | StrOutputParser()


def build_setup_chain() -> Any:
    """Compose database download and schema analysis as a LangChain pipeline."""
    def download_stage(state: dict[str, Any]) -> dict[str, Any]:
        return {**state, "db_path": str(download_db(state.get("db_path", DEFAULT_DB_PATH)))}

    def analysis_stage(state: dict[str, Any]) -> dict[str, Any]:
        return {**state, "schema": analyse_db(state["db_path"])}

    return RunnableLambda(download_stage) | RunnableLambda(analysis_stage)


def build_execution_chain(db_path: str | Path) -> Any:
    """Ask before running SQL, then execute only a validated SELECT read-only."""
    def ask_for_confirmation(state: dict[str, Any]) -> dict[str, Any]:
        print("\nWould you like me to run this read-only query and show the results?")
        answer = input("Run it? [y/N]> ").strip().lower()
        return {**state, "confirmed": answer in {"y", "yes"}}

    def execute_stage(state: dict[str, Any]) -> dict[str, Any]:
        sql = validate_read_only_select(state["sql"])
        path = Path(db_path).resolve()
        connection = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
        try:
            cursor = connection.execute(sql)
            return {
                "executed": True,
                "columns": [column[0] for column in cursor.description or ()],
                "rows": cursor.fetchall(),
            }
        finally:
            connection.close()

    def decline_stage(_: dict[str, Any]) -> dict[str, Any]:
        return {"executed": False, "columns": [], "rows": []}

    return RunnableLambda(ask_for_confirmation) | RunnableBranch(
        (lambda state: state["confirmed"], RunnableLambda(execute_stage)),
        RunnableLambda(decline_stage),
    )


def validate_read_only_select(sql: str) -> str:
    """Require exactly one SQLite SELECT statement and return normalized text."""
    candidate = sql.strip()
    if candidate.startswith("```"):
        lines = candidate.splitlines()
        if len(lines) >= 3 and lines[-1].strip().startswith("```"):
            candidate = "\n".join(lines[1:-1]).strip()

    try:
        statements = sqlglot.parse(candidate, read="sqlite")
    except sqlglot.errors.ParseError as exc:
        raise ValueError("The model returned invalid SQLite SQL.") from exc

    statements = [statement for statement in statements if statement is not None]
    if len(statements) != 1 or not isinstance(statements[0], exp.Select):
        raise ValueError("The model must return exactly one read-only SELECT query.")
    mutating_nodes = (
        exp.Insert,
        exp.Update,
        exp.Delete,
        exp.Create,
        exp.Drop,
        exp.Alter,
        exp.Command,
        exp.Merge,
    )
    if any(isinstance(node, mutating_nodes) for node in statements[0].walk()):
        raise ValueError("The model must return exactly one read-only SELECT query.")
    return candidate.rstrip().rstrip(";").strip() + ";"


def create_query(question: str, schema: str, model: Any | None = None) -> str:
    """Generate and validate a read-only SQL query for a natural-language prompt."""
    if not question.strip():
        raise ValueError("Please enter a question.")
    output = build_query_chain(model).invoke(
        {"schema": schema, "question": question.strip()}
    )
    return validate_read_only_select(output)


def run_cli(db_path: str | Path = DEFAULT_DB_PATH, model: Any | None = None) -> None:
    """Download and inspect the database, then warmly guide each query."""
    prepared = build_setup_chain().invoke({"db_path": str(db_path)})
    path = prepared["db_path"]
    schema = prepared["schema"]
    print(f"Your Chinook database is ready at {path}.")
    print("What would you like to know? Type 'exit' whenever you're ready to leave.")
    execution_chain = build_execution_chain(path)

    while True:
        try:
            question = input("\nQuestion> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nThanks for stopping by. Have a lovely day!")
            return
        if question.lower() in {"exit", "quit"}:
            print("Thanks for chatting. Have a lovely day!")
            return
        if not question:
            continue
        try:
            sql = create_query(question, schema, model=model)
            print("\nHappy to help! Here’s the SQL query:\n" + sql)
            result = execution_chain.invoke({"sql": sql})
            if result["executed"]:
                print("\nHere are your results:")
                if not result["rows"]:
                    print("No matching rows this time.")
                else:
                    print(" | ".join(result["columns"]))
                    print("-+-".join("-" * len(column) for column in result["columns"]))
                    for row in result["rows"]:
                        print(" | ".join("" if value is None else str(value) for value in row))
            else:
                print("No problem—the query wasn’t run, and your database is unchanged.")
        except Exception as exc:
            print(
                f"Sorry, I couldn’t complete that safely: {exc}",
                file=sys.stderr,
            )


def main() -> None:
    load_dotenv()
    try:
        run_cli()
    except Exception as exc:
        print(f"Sorry, I couldn’t start the SQL assistant: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()

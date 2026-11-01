import io
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from langchain_core.runnables import RunnableLambda

import app
import download_db as downloader


class DownloadTests(unittest.TestCase):
    def test_downloads_missing_file_and_reuses_existing_file(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "nested" / "Chinook.db"
            with patch.object(downloader, "urlopen", return_value=io.BytesIO(b"db")) as request:
                self.assertEqual(downloader.download_db(path), path)
                self.assertEqual(path.read_bytes(), b"db")
                self.assertEqual(path.stat().st_mode & 0o222, 0)
                self.assertEqual(downloader.download_db(path), path)
                request.assert_called_once()


class SchemaTests(unittest.TestCase):
    def test_schema_includes_columns_keys_and_foreign_keys_without_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "fixture.db"
            connection = sqlite3.connect(path)
            connection.executescript(
                "CREATE TABLE artist (id INTEGER PRIMARY KEY, name TEXT NOT NULL);"
                "CREATE TABLE album (id INTEGER PRIMARY KEY, artist_id INTEGER, "
                "FOREIGN KEY (artist_id) REFERENCES artist(id));"
                "INSERT INTO artist VALUES (1, 'private row value');"
            )
            connection.close()

            schema = app.analyse_db(path)
            self.assertIn("Table artist", schema)
            self.assertIn("name: TEXT, NOT NULL", schema)
            self.assertIn("PRIMARY KEY", schema)
            self.assertIn("album.artist_id -> artist.id", schema)
            self.assertNotIn("private row value", schema)
            read_only = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
            try:
                with self.assertRaises(sqlite3.OperationalError):
                    read_only.execute("UPDATE artist SET name = 'changed'")
            finally:
                read_only.close()


class QueryTests(unittest.TestCase):
    def test_prompt_receives_schema_and_question_and_returns_sql(self):
        seen = {}

        def fake_model(prompt_value):
            seen["prompt"] = prompt_value.to_messages()
            return "SELECT name FROM artist;"

        chain_model = RunnableLambda(fake_model)
        result = app.create_query("List artists", "Table artist: name TEXT", model=chain_model)
        self.assertEqual(result, "SELECT name FROM artist;")
        prompt_text = "\n".join(message.content for message in seen["prompt"])
        self.assertIn("Table artist: name TEXT", prompt_text)
        self.assertIn("List artists", prompt_text)

    def test_rejects_non_select_and_multiple_statements(self):
        for sql in (
            "DELETE FROM artist",
            "SELECT name FROM artist; DROP TABLE artist",
            "WITH deleted AS (DELETE FROM artist RETURNING *) SELECT * FROM deleted",
        ):
            with self.subTest(sql=sql), self.assertRaises(ValueError):
                app.validate_read_only_select(sql)


if __name__ == "__main__":
    unittest.main()

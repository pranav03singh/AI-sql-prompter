# Chinook Question-to-SQL

A small Python CLI that downloads the Chinook SQLite sample database, inspects
its schema, and uses LangChain with Ollama Cloud (`gpt-oss:120b`) to turn
natural-language questions into SQLite queries.

The app displays generated SQL and asks before executing it. Only one
read-only `SELECT` statement is accepted, and execution uses a read-only SQLite
connection. The database is never modified by the app.

## Requirements

- Python 3.10 or newer
- An [Ollama API key](https://ollama.com/settings/keys)
- Internet access to download the database and call Ollama Cloud

## Setup

From the project directory, create and activate a virtual environment, install
the packages, and make a local `.env` file:

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
cp -n .env.example .env
```

Add your Ollama API key to `.env`:

```dotenv
OLLAMA_API_KEY=your_actual_ollama_api_key
```

Keep `.env` private. It is excluded by `.gitignore`; `.env.example` contains
only a placeholder and is safe to commit.

## Run

```sh
python app.py
```

The first run downloads the database to `data/Chinook.db`. Later runs reuse
that file. The local database and `.venv` are ignored by Git.

## Example

```text
Your Chinook database is ready at .../data/Chinook.db.
What would you like to know? Type 'exit' whenever you're ready to leave.

Question> Which 5 artists have the most albums?

Happy to help! Here’s the SQL query:
SELECT Artist.Name, COUNT(Album.AlbumId) AS AlbumCount
FROM Artist
LEFT JOIN Album ON Album.ArtistId = Artist.ArtistId
GROUP BY Artist.ArtistId, Artist.Name
ORDER BY AlbumCount DESC
LIMIT 5;

Would you like me to run this read-only query and show the results?
Run it? [y/N]> y

Here are your results:
Name | AlbumCount
-----+-----------
Iron Maiden | 21
Led Zeppelin | 14
Deep Purple | 11
Metallica | 10
U2 | 10
...
```

The exact SQL and results depend on the model response. Answer `y` or `yes` to
run the query. Any other answer skips execution. Type `exit` or `quit` to close
the CLI.

Other questions to try:

- `Which customers are in Canada?`
- `Show the 10 most expensive tracks with their album names.`
- `What is the total sales amount for each country?`

## Privacy and safety

- Your API key stays in `.env`; never commit it or paste it into source files.
- The prompt contains schema metadata and your question, not database rows.
- SQL output is checked to allow only a single `SELECT` statement. The app
  requires confirmation before executing it, and SQLite is opened read-only.
- The database file is downloaded locally and excluded from Git.

## Tests

Run the unit tests with:

```sh
python -m unittest discover -s tests -v
```

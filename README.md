# corp-profile

Build rich company context documents from a corp-graph Postgres database.

## Setup

```bash
uv sync
cp .env.example .env
# Edit .env with your database connection string
```

## Usage

```bash
python -m corp_profile build <ISIN>
```

Queries the corp-graph Postgres database for a company matching the given ISIN and builds a structured context document including company overview, subsidiaries, existing assets, discovered assets, and asset estimates.

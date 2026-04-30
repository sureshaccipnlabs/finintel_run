# finintel_run


## How to Run

### Prerequisites

- Python 3.13 or later
- `uv` installed
- Ollama installed and running locally if you want to use the Q&A and risk recommendation features

Check your Python version:

```bash
python --version
```

If `python` is not available, try:

```bash
python3 --version
```

### Install dependencies

From the project root, create the environment and install dependencies with `uv`:

```bash
uv sync
```

If `uv` is not installed yet, you can install it with:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Start Ollama

The ingestion Q&A flow uses a locally running Ollama server through HTTP.

Start Ollama before using the AI-powered Q&A or risk recommendation endpoints:

```bash
ollama serve
```

If needed, pull a model first:

```bash
ollama pull llama3
```

### Run the FastAPI application

Start the API server from the project root:

```bash
uv run uvicorn main:app --reload
```

Once started, the API will be available at:

```text
http://127.0.0.1:8000
```

Swagger documentation is available at:

```text
http://127.0.0.1:8000/docs
```

Useful endpoints include:

- `POST /ingest`
- `POST /analyze`
- `GET /dataset`
- `GET /metrics`
- `GET /projects`
- `GET /employees`
- `GET /ai-provider`
- `POST /ai-provider`

### Run the sample CLI flow

To run the sample ingestion script:

```bash
uv run python run.py
```

This runs the bundled sample flow and prints results in the terminal.

To test your own file:

```bash
uv run python run.py yourfile.csv
```

or:

```bash
uv run python run.py yourfile.xlsx
```

### Supported file types

The ingestion flow supports:

- CSV
- TSV
- TXT
- XLSX
- XLS
- PDF

### Common issues

- If you get `python: command not found`, use `python3` instead.
- If you get dependency import errors, run `uv sync` again.
- If Q&A responses fail, make sure Ollama is running.

### AI provider selection (Ollama + OpenAI)

You can now choose which LLM backend the app should use.

Supported providers:

- `ollama` (local)
- `openai` (API)
- `auto` (fallback order)

Environment variables:

- `AI_PROVIDER`: `ollama` | `openai` | `auto` (default: `auto`)
- `AI_PROVIDER_ORDER`: comma-separated fallback order for auto mode (default: `ollama,openai`)
- `OLLAMA_URL` (default: `http://localhost:11434`)
- `OLLAMA_MODEL` (default: `llama3`)
- `OPENAI_API_KEY` (required for OpenAI)
- `OPENAI_MODEL` (default: `gpt-4o-mini`)
- `OPENAI_URL` (default: `https://api.openai.com/v1/chat/completions`)

Quick commands to run (copy-paste):

OpenAI only:

```bash
export AI_PROVIDER=openai
export OPENAI_API_KEY="your_openai_api_key"
export OPENAI_MODEL="gpt-4o-mini"
uv run uvicorn main:app --reload
```

Ollama only:

```bash
export AI_PROVIDER=ollama
export OLLAMA_URL="http://localhost:11434"
export OLLAMA_MODEL="llama3"
uv run uvicorn main:app --reload
```

Examples:

Use Ollama only:

```bash
export AI_PROVIDER=ollama
export OLLAMA_MODEL=llama3
```

Use OpenAI only:

```bash
export AI_PROVIDER=openai
export OPENAI_API_KEY=your_key_here
export OPENAI_MODEL=gpt-4o-mini
```

Auto mode (try OpenAI first, then Ollama):

```bash
export AI_PROVIDER=auto
export AI_PROVIDER_ORDER=openai,ollama
```

You can also switch provider at runtime (without restart):

```bash
curl -X POST "http://127.0.0.1:8000/ai-provider" \
  -H "Content-Type: application/json" \
  -d '{"provider":"openai"}'
```

Switch to Ollama runtime:

```bash
curl -X POST "http://127.0.0.1:8000/ai-provider" \
  -H "Content-Type: application/json" \
  -d '{"provider":"ollama"}'
```

`POST /ai-provider` now only switches provider. Model, URL, and API key are taken from server-side configuration (`OPENAI_*`, `OLLAMA_*`, `AI_PROVIDER_ORDER`).

### AI behavior (current)

Notes:

- `GET /risks-recommendations` and `POST /ask` use the provider configured via `AI_PROVIDER`.
- If no configured provider is available, insights return empty text and rule-based risks/recommendations still return.
- `POST /ask` now detects forecast-style questions (`forecast`, `predict`, `next 3 months`, etc.) and applies a forecast prompt with explicit assumptions for more consistent Ollama/OpenAI output.
- `GET /risks-recommendations` now also returns compact grouped fields: `overview`, `risk_groups`, `recommendation_groups`, `top_risks`, and `top_recommendations`.
- Use query param `max_items` (default `8`, range `3-20`) to control compact list sizes.


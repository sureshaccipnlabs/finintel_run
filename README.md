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


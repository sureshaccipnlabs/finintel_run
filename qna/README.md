# fintel-qa-agent

A simple financial Q&A agent that reads aggregated JSON files from the `data/` folder, builds structured knowledge, creates retrieval documents, and answers questions using a mix of deterministic routing and LLM-based fallback.

## Prerequisites

- Python 3.10+
- Ollama installed and running locally
- The `mistral` model available in Ollama

## Install dependencies

From the project root:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Prepare the model

Make sure Ollama is running and the configured model is available:

```bash
ollama pull mistral
```

The default configuration is defined in `config.py`:

- `MODEL_NAME = "mistral"`
- `DATA_FOLDER = "data"`

## Input data

Place aggregated JSON files inside the `data/` folder.

Example:

```text
data/
  response_1776327915719 1.json
```

## Run using `run.py`

`run.py` is the main CLI entrypoint and resolves the project root before starting the agent.

From the project root:

```bash
python run.py
```

What it does:

- loads JSON files from `data/`
- builds the knowledge base
- creates the vector store
- starts an interactive Q&A session

To exit the session:

```text
exit
```

## Run using `run_local.py`

`run_local.py` is a lightweight local runner that starts the same agent directly from the root-level modules.

From the project root:

```bash
python run_local.py
```

What it does:

- loads JSON files from `data/`
- bootstraps the agent
- starts an interactive Q&A session

To exit the session:

```text
exit
```

## Example questions

You can ask questions like:

- `highest profit employee`
- `Tapan s margin pct`
- `Tapan profit`
- `most utilized employee`
- `portfolio summary`

## Notes

- Deterministic questions are answered directly from structured data when possible.
- Broader questions fall back to retrieval plus LLM response generation.
- You may see deprecation warnings for LangChain Ollama classes. The current project still runs, but those imports can be upgraded later.

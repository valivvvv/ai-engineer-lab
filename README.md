# ai-engineer-lab — RAG Document Analyst

A ReAct-style Q&A agent (LangChain) extended into a **document analyst**: it
ingests documents, stores them in PostgreSQL + `pgvector`, and answers questions
over them via retrieval-augmented generation (RAG), citing its sources. Part of
an ongoing AI-engineering lab, so the codebase keeps growing.

## What it does

- **Q&A agent** — a Think → Act → Observe loop with Pydantic-validated tools and
  versioned prompts.
- **Document analyst (RAG)** — ingests PDF / DOCX / TXT / CSV, stores chunk
  embeddings in Postgres + `pgvector`, and answers questions grounded in those
  documents, citing the source filename (or saying it doesn't know).
- **Provider-agnostic LLM** — Ollama (local), Google Gemini, or Anthropic Claude.
- **Observability** — optional LangSmith tracing, enabled purely via env vars.

## Prerequisites

- Python 3.10+
- Docker — for the in-repo stack in `llm docker containers/` (Postgres + pgvector,
  Ollama)
- An API key for the extraction LLM (Gemini or Anthropic)
- (Optional) A LangSmith account for traces

## Setup

```bash
cd ai-engineer-lab
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # then fill in real values (provider API key, etc.)
```

Bring up the local stack (Postgres + pgvector, Ollama):

```bash
cd "llm docker containers"
docker compose up -d
```

Create the database schema, then index the sample documents:

```bash
python -m alembic upgrade head
python scripts/index_documents.py
```

Confirm Ollama has `llama3.2:3b` (the default agent model):

```bash
curl http://localhost:11434/api/tags
# If it's missing:
docker exec -it ollama ollama pull llama3.2:3b
```

## Run

```bash
# Agent demo (one query per tool) + interactive REPL
python main.py
python main.py --repl
python main.py --repl --stream
python main.py --repl --language Romanian

# Index the sample documents (write path)
python scripts/index_documents.py

# Ask a question over the indexed documents (RAG analyst)
python scripts/smoke_analyst.py
```

## Architecture

```
INDEX  (write path — ingestion/, run via scripts/index_documents.py)
  file (PDF / DOCX / TXT / CSV)
    → load → chunk → extract structured fields     (ingestion/)
    → embed chunks                                 (rag/embeddings.py)
    → store: Document + DocumentChunk              (db/ → Postgres + pgvector)


ASK  (read path — the agent)
  QAAgent  (agent/qa.py)  ── chat() / stream() / clear_history()
       │
       ├─ PromptRegistry  (prompts/)            renders the system prompt
       ├─ LLMFactory      (agent/llm_factory)   Ollama / Gemini / Anthropic
       └─ ToolWrapper     (tools/registry)      .bind_tools(catalog)
                   │
                   ▼
         react_events()  (agent/react.py)  ── Think → Act → Observe
                   │
                   ▼
         search_documents tool  ──►  DocumentRetriever  (rag/retriever.py)
                   │                          │
                   │                          ▼
                   │              embed query + similarity search over pgvector
                   ▼
         grounded answer, with source citation
```

## Components

| Package | Responsibility |
|---|---|
| `agent/` | The ReAct loop (`react_events`), the `QAAgent` public API, and the provider-agnostic LLM factory. |
| `tools/` | Tools the agent can call — registered via `@register_tool` (calculator, datetime, web-search stub, `search_documents`). |
| `prompts/` | Versioned YAML system prompts (`qa_system`, `analyst_system`), rendered with Jinja2. |
| `ingestion/` | Write path: load → chunk → structured extraction → embed → store. |
| `rag/` | Read path: embeddings + retriever for similarity search over chunks. |
| `db/` | Postgres + `pgvector` models, repositories, and Alembic migrations. |
| `scripts/` | Runnable smoke tests and the document indexer. |
# Implementation Plan — Document Analyst with RAG (Homework, Lessons 3–4)

> **Audience:** another Claude instance picking this up cold. This document is self-contained — read it fully before writing code, then read the existing files listed in §3.

---

## 0. Working agreement & context discipline (read first)

**Execution mode:** You (the implementing AI) **write the code; the user reviews.** Implement **one phase at a time**, run its smoke test, then **stop and hand the diff to the user for review** before starting the next phase. Do not implement multiple phases unprompted.

**Context discipline — treat this as seriously as the code:**
- **Do NOT read the heavy source artifacts.** Never open the lesson PDFs (`lesson3.pdf`, `lesson4.pdf`, `homework lesson 4.pdf`), the `demos-lesson 4/` tree, or `code-snippets lesson 3|4/`. They are megabytes of slide images and redundant snippets — **this plan already distilled everything you need from them**, including the bugs to avoid (§9). Reading them will exhaust your context for zero gain.
- **Read only:** this plan → the specific existing files listed in §3 → the code already on disk from earlier phases.
- **One phase per session.** Each phase is independent and ends in a smoke test — that is the context boundary. After a phase passes, `/clear` (or start a fresh session) and resume with *"Plan §8 Phase N; earlier phases are already on disk."* Orient from the code and `git log`, not from old conversation.
- **Commit after every passing phase**, so the next session orients from the diff instead of replaying history.
- **Push repo exploration into subagents** (Explore/Task) so file dumps stay out of the main thread — only the conclusion returns.

**This plan is the single source of truth.** If a phase forces a deviation from it (e.g. Alembic autogenerate mis-renders the `Vector` type — see §8 Phase 2), **edit this file** to record the change before moving on. State lives in this document, not in any one conversation, so context resets cost nothing.

---

## 1. Goal

Build a **Document Analyst** that extracts structured data from documents (PDF/DOCX/TXT/CSV), stores it in PostgreSQL with `pgvector`, and answers questions over those documents via RAG — exposed as a tool on the **existing ReAct QA agent**.

End state (from `homework lesson 4.pdf`):
1. **Extraction pipeline (L3):** loader registry → chunking → Pydantic structured extraction → save.
2. **PostgreSQL + Repository pattern (L4):** Postgres+pgvector via Docker + Alembic migrations; `Document` + `DocumentChunk` (one-to-many); repositories for CRUD + transactions.
3. **RAG with embeddings (L4):** per-chunk embeddings (sentence-transformers), cosine similarity search (+ HNSW index), `RAGService.search(query, top_k)`.
4. **Connect to the L1–L2 agent:** wrap RAG as a LangChain tool `search_documents()`; the existing agent answers from loaded documents.

Acceptance demo: `agent.chat("Ce clauze de reziliere avem?")` returns an answer grounded in the loaded documents, citing the source filename.

---

## 2. Locked decisions (do not relitigate)

| Decision | Choice | Why |
|---|---|---|
| Where code lives | **Extend `ai-engineer-lab/`** | Homework says "connect to the L1–L2 agent" (that *is* this repo); repo is designed to grow. |
| Postgres + pgvector | **Reuse the shared `../llm docker containers/` stack** (postgres on **5432**, db/user `skillab`, pass `skillab_dev`) — do **not** add a new compose file | The shared infra ("Start once, use in all lessons") already runs `pgvector/pgvector:pg16` and its comment names it the RAG/storage DB. A second container on 5433 would be redundant. Homework tables (`documents`, `document_chunks`) live in the existing `skillab` DB; Alembic still creates the `vector` extension + tables. |
| Structured extraction | **LangChain `llm.with_structured_output(Schema)`** | Vendor-neutral via the existing `LLMFactory`; matches homework snippet. |
| Embedding model | **`paraphrase-multilingual-MiniLM-L12-v2`** (384-dim) | Data samples are Romanian; needs RO retrieval quality. 384 dims fits `Vector(384)`. |
| Extraction LLM provider | **Gemini or Anthropic** (explicit `model=`) | Local Ollama 3B is unreliable for structured extraction. Interface stays neutral. |
| Write vs read ownership | **`ExtractionPipeline` = write path; `RAGService` = read-only** | Mirrors homework's `process()` (writes) vs `rag.search()` (reads). Avoids duplicate indexing logic. |
| Full-text search (tsvector) | **Deferred** | Not required; the lesson snippet for it is buggy (references an undefined `search_vector` column). |
| HNSW index | **Phase 5 (optional)** | Linear scan is fine at demo scale (<10k chunks). |

---

## 3. Read these existing files first (then honor their contracts)

The repo lives at `/Users/vali/Work/AI/AI Engineer/ai-engineer-lab/`.

- `CLAUDE.md` — project conventions (see §4).
- `agent/llm_factory.py` — `LLMFactory.create(provider=None, model=None, temperature=0.7, **kwargs) -> BaseChatModel`. Resolution: explicit args > env vars > defaults. Provider default `"ollama"`. **Raises `ValueError` if neither `model=` nor `DEFAULT_MODEL` is set.** Providers: `ollama` / `gemini` / `anthropic`.
- `agent/qa.py` — `QAAgent(provider=None, model=None, temperature=0.0, prompt_name="qa_system", max_iterations=5, assistant_name="QA Assistant", language=None)`. Binds tools from `ToolWrapper.catalog()`. Renders the system prompt with exactly these variables: **`assistant_name`, `tools` (list of `{name, description}`), `language`**. Public API: `chat()`, `stream()`, `chat_async()`, `stream_async()`, `clear_history()`, `history`.
- `agent/react.py` — core generator `react_events()`; calls `ToolWrapper.call(tool_call["name"], tool_call["args"])`. Do not modify.
- `tools/registry.py` — **the tool contract.** `@register_tool` requires:
  - function takes **exactly one parameter** annotated as a **Pydantic `BaseModel` subclass**;
  - a **docstring of ≥15 characters** (used verbatim as the LLM-facing description);
  - the **function name is the tool name** the LLM sees.
  - `ToolWrapper.call(name, args)` validates args into the params model, runs the func, and **already converts `ValidationError` and any `Exception` into return strings** — so tools must **not** implement their own dict-wrapped error handling. Tool return value is coerced with `str(...)`.
  - `ToolWrapper.catalog()` returns `StructuredTool`s for `bind_tools`.
- `tools/__init__.py` — `_register_builtin_tools()` imports each tool module to trigger registration. **New tools must be imported here.**
- `prompts/registry.py` — `get_prompt_registry().render(name, **vars)` via Jinja2 with **`StrictUndefined`** (a missing variable is an error). YAML keys: `name`, `version`, `prompt`, `description`, `variables`, `metadata`.
- `prompts/templates/qa_system.yaml` — reference prompt; uses only `assistant_name`, `tools`, `language`.
- `tools/calculator.py`, `tools/web_search.py` — reference tool implementations (Pydantic params + `Field(description=...)`).
- `data samples/` — test inputs already present:
  - `factura_001.txt`, `factura_002.txt` (invoices)
  - `contract_servicii.txt`, `contract_consultanta.txt` (contracts)
  - `facturi_export.csv` (tabular)

There is **no DB layer yet** — you are adding `db/`, `extraction/`, `rag/` fresh.

---

## 4. Hard conventions (from `ai-engineer-lab/CLAUDE.md` and the user's global prefs)

1. **No 1–2 letter variable names.** Use `tool_call`, not `tc`; `document`, not `doc` where avoidable.
2. **Avoid explanatory comments.** Prefer renames, helpers, named constants. (The repo keeps exactly one load-bearing comment in `tools/registry.py` for PEP 563 / `get_type_hints` — leave it.)
3. **Present ideas before doing them.** No unsolicited renames/refactors of existing code.
4. **Plan-first, smoke-test each phase.** Per §0: you write each phase, the user reviews the diff; stop after each phase's smoke test rather than implementing multiple phases unprompted.
5. Match surrounding code style (provider-neutral factory, registry patterns, YAML prompts).

---

## 5. Target module layout (new files unless marked EXISTING)

```
ai-engineer-lab/
│   # NOTE: no docker-compose.yml here — Postgres+pgvector is the SHARED
│   #   ../llm docker containers/ stack (postgres on 5432, db/user skillab).
├── alembic.ini                 # NEW
├── alembic/                    # NEW — env.py (imports models), versions/
├── db/                         # NEW
│   ├── __init__.py
│   ├── database.py             # engine, SessionLocal, Base, transaction()
│   ├── models.py               # Document, DocumentChunk
│   └── repositories.py         # DocumentRepository, ChunkRepository
├── extraction/                 # NEW
│   ├── __init__.py
│   ├── schemas.py              # Invoice, Contract (Pydantic + Field descriptions)
│   ├── loaders.py              # extension → loader registry
│   ├── chunking.py             # split_for_retrieval(), should_chunk_for_extraction()
│   └── pipeline.py             # ExtractionPipeline (the WRITE path)
├── rag/                        # NEW
│   ├── __init__.py
│   ├── embeddings.py           # EmbeddingService (lazy singleton)
│   └── service.py              # RAGService (READ-only: search / get_context)
├── tools/
│   └── search_documents.py     # NEW tool (register in tools/__init__.py)
├── prompts/templates/
│   └── analyst_system.yaml     # NEW prompt
├── agent/                      # EXISTING — do not modify core loop
└── requirements.txt            # EXISTING — extend
```

No import cycles: `db/` depends on nothing app-level; `extraction/` and `rag/` depend on `db/` + `rag/embeddings`; `tools/search_documents` depends on `rag/`.

---

## 6. The two LLM instances (keep them separate)

- **Agent LLM** — `LLMFactory.create(...).bind_tools(catalog)` inside `QAAgent`. Provider may stay Ollama. Runs the ReAct loop and **generates** the final answer.
- **Extraction LLM** — a *separate* instance: `LLMFactory.create(provider="gemini", model="<gemini-model>").with_structured_output(Invoice)`. **Pass `model=` explicitly** (do not let it inherit the Ollama `DEFAULT_MODEL`).

These are distinct instances; `bind_tools` and `with_structured_output` do not conflict because they are never applied to the same object.

---

## 7. Data flows

**Index (write path — `ExtractionPipeline`):**
```
file
 → loaders.load(path)                      # extension → loader, returns text/Documents
 → { extract structured fields,            # with_structured_output on FULL text (or first chunks)
     split_for_retrieval(text) }           # ALWAYS ≥1 chunk, even for small docs
 → EmbeddingService.embed_batch(chunks)
 → transaction(): DocumentRepository.create(content, doc_metadata=structured.model_dump())
                  ChunkRepository.create_batch(document_id, chunks, embeddings)
```
`should_chunk_for_extraction()` only decides whether the *extraction LLM* sees the full text vs first-N chunks. It does **not** gate retrieval chunks — every document yields ≥1 `DocumentChunk`.

**Query (read path — `RAGService`, via the tool):**
```
question
 → EmbeddingService.embed(question)
 → ChunkRepository.similarity_search(embedding, top_k)   # 1 - cosine_distance, joinedload(document)
 → filter score >= threshold (~0.4)
 → tool returns a formatted context string (filename + chunk + score)
 → agent's ReAct loop GENERATES the cited answer (analyst_system prompt)
```

---

## 8. Phased tasks (each ends with a smoke test)

### Phase 0 — Infrastructure  ✅ DONE (2026-06-01)
> Outcome: shared `skillab-postgres` (5432) confirmed healthy; deps installed into `.venv` (torch/sentence-transformers/sqlalchemy/alembic/pgvector/loaders); `alembic init alembic` scaffolded `alembic.ini` + `alembic/`; smoke test passed — `SELECT 1` + `vector` extension live at 0.8.2. `DATABASE_URL` added to `.env`/`.env.example`. (Note: `.venv` pip/console-script shebangs are stale from an old dir name — invoke via `.venv/bin/python -m pip` / `-m alembic`.) `alembic.ini` URL + `env.py` target_metadata intentionally left for Phase 2.
- **Postgres+pgvector is already provided** by the shared `../llm docker containers/docker-compose.yml` (`pgvector/pgvector:pg16`, container `skillab-postgres`, **port 5432**, db/user `skillab`, password `skillab_dev`, volume `skillab-postgres-data`). **Do not create a new compose file.** If the stack isn't running: `docker compose up -d postgres` from that directory.
- `.env` / `.env.example`: add `DATABASE_URL=postgresql://skillab:skillab_dev@localhost:5432/skillab`. The extraction provider key (`GOOGLE_API_KEY`) is already present in `.env`.
- `requirements.txt`: add **`sqlalchemy`**, **`psycopg2-binary`** (the DB driver for `postgresql://` — not currently in the repo), **`pgvector`**, **`alembic`**, **`sentence-transformers`** (pulls **torch** — expect a slow first install and first model load), **`langchain-community`** (the loaders live here — the repo only ships `langchain-core` + provider packages today), **`langchain-text-splitters`** (`RecursiveCharacterTextSplitter`), **`pypdf`**, **`docx2txt`**. Check `requirements.txt` for anything already pinned before adding.
- `alembic init alembic`.
- **Smoke:** confirm the shared `skillab-postgres` container is up, then a SQLAlchemy connection to `DATABASE_URL` succeeds and `CREATE EXTENSION IF NOT EXISTS vector` runs without error.

### Phase 1 — Extraction (no DB yet)
- `extraction/schemas.py`: `Invoice` and `Contract` Pydantic models with `Field(description=...)` on every field; optional fields use defaults. (Reference fields from `lesson3.pdf`: invoice → `numar, data, client, furnizor, total, produse: list[...]`; contract → `numar, data_incheiere, prestator, beneficiar, valoare, durata_luni, obligatii_prestator: list[str]`.)
- `extraction/loaders.py`: registry `{".pdf": PyPDFLoader, ".docx": Docx2txtLoader, ".txt": TextLoader, ".csv": CSVLoader}` from `langchain_community.document_loaders`; `load_document(path)` resolves by suffix, raises a clear error on unsupported types; `TextLoader` must pass `encoding="utf-8"`.
- `extraction/chunking.py`: `split_for_retrieval(text, chunk_size=1000, chunk_overlap=200) -> list[str]` via `RecursiveCharacterTextSplitter` (returns ≥1 chunk); `should_chunk_for_extraction(text, max_chars=4000) -> bool`.
- `extraction/pipeline.py`: `ExtractionPipeline.process(path, doc_type)` that **for Phase 1 returns `(structured_object, chunks)`** and does not touch the DB. Build the extraction LLM here via `LLMFactory.create(provider="gemini", model=...).with_structured_output(Invoice|Contract)`, routed by `doc_type`.
- **Smoke:** run on `data samples/factura_001.txt` and `data samples/contract_servicii.txt`; print the structured object and the chunk count.

### Phase 2 — Storage
- `db/database.py`: `engine = create_engine(DATABASE_URL, pool_pre_ping=True)`, `SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)`, `Base = declarative_base()`, and a `@contextmanager transaction()` that yields a session, commits on success, rolls back on exception, always closes.
- `db/models.py`:
  - `Document`: `id` PK; `filename` String(255) indexed; `content` Text; `doc_metadata = Column("metadata", JSONB, nullable=False, default=dict)` — **the attribute must not be named `metadata`** (reserved by SQLAlchemy declarative); `created_at` DateTime(timezone=True) server_default `func.now()`; relationship `chunks` with `cascade="all, delete-orphan"`.
  - `DocumentChunk`: `id` PK; `document_id` FK → `documents.id` `ondelete="CASCADE"`, not null; `content` Text; `chunk_index` Integer not null; `embedding = Column(Vector(384), nullable=False)` (`from pgvector.sqlalchemy import Vector`); `UniqueConstraint("document_id", "chunk_index")`; index on `document_id`; relationship `document` `back_populates="chunks"`.
- `db/repositories.py`:
  - `DocumentRepository(db)`: `create(filename, content, doc_metadata) -> Document` (`add` + `flush` + `refresh`); `get_by_id`, `get_by_filename`, `get_all(skip, limit)` ordered by `created_at desc`, `count()`.
  - `ChunkRepository(db)`: `create_batch(document_id, chunks, embeddings) -> list[DocumentChunk]` (`add_all` + `flush`); `get_document_chunks(document_id)`; `similarity_search(...)` is added in Phase 3.
- **Alembic** (autogenerate will NOT do these for you):
  1. `alembic/env.py`: set `target_metadata = Base.metadata` **and import the models** (`from db import models`) so they register on `Base`.
  2. In the generated migration, add `from pgvector.sqlalchemy import Vector`.
  3. Make `op.execute("CREATE EXTENSION IF NOT EXISTS vector")` the **first** upgrade op, before table creation.
- **Smoke:** `alembic upgrade head`; within a `transaction()`, store one `Document` + two `DocumentChunk`s (dummy 384-float vectors) and read them back.

### Phase 3 — RAG (read path + wire write path)
- `rag/embeddings.py`: `EmbeddingService` with a lazy class-level singleton `SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")`; `embed(text) -> list[float]`; `embed_batch(texts) -> list[list[float]]` (use `.tolist()`).
- Extend `ChunkRepository.similarity_search(query_embedding, top_k=5) -> list[tuple[DocumentChunk, float]]`:
  - score = `(1 - DocumentChunk.embedding.cosine_distance(query_embedding)).label("score")`;
  - `select(DocumentChunk, score).options(joinedload(DocumentChunk.document)).order_by(DocumentChunk.embedding.cosine_distance(query_embedding)).limit(top_k)`;
  - return `[(chunk, float(score)) for ...]`. (`joinedload` avoids N+1 when the tool reads `chunk.document.filename`.)
- Add the embed+store tail to `ExtractionPipeline.process` (now writes): embed chunks, persist `Document` + `DocumentChunk`s in one `transaction()`.
- `rag/service.py`: `RAGService(db)` — **read-only**. `search(query, top_k=5)` (embed query → `similarity_search`); `get_context(query, top_k=5, threshold=0.4)` → filter by score, return formatted string or `""` if nothing passes.
- **Smoke:** index the `data samples` via the pipeline; `RAGService.search("clauze de reziliere")` ranks a relevant contract chunk first with a sane score.

### Phase 4 — Agent integration
- `tools/search_documents.py`:
  ```python
  class SearchDocumentsParams(BaseModel):
      query: str = Field(description="...", min_length=2)
      top_k: int = Field(default=3, ge=1, le=10, description="...")

  @register_tool
  def search_documents(params: SearchDocumentsParams) -> str:
      """Searches the indexed company documents (invoices, contracts) and
      returns the most relevant excerpts with their source filename. Use this
      whenever the user asks about contract clauses, invoice details, etc."""
      # open a session (transaction()/SessionLocal), build RAGService, return get_context(...)
  ```
  - Return a **formatted string** (filename + excerpt + score). Do **not** wrap errors in a dict — `ToolWrapper.call` already converts exceptions to strings.
  - The tool opens its own DB session (it is called from inside the ReAct loop, not handed one).
- Register it: add `search_documents` to the imports in `tools/__init__.py._register_builtin_tools()`.
- `prompts/templates/analyst_system.yaml`: a system prompt that instructs the agent to **answer only from the retrieved context, cite the source filename, and say "nu știu" when the answer is not in the documents.** **Constraint:** `QAAgent` only supplies `assistant_name`, `tools`, `language` as render variables — the template may use only those (plus literals). Keep the existing "never expose tool plumbing (tool names, JSON, call IDs)" rule (citing a filename is not plumbing).
- Use it: instantiate `QAAgent(prompt_name="analyst_system")` (the agent picks up `search_documents` automatically via the catalog).
- **Smoke:** `QAAgent(prompt_name="analyst_system").chat("Ce clauze de reziliere avem?")` answers from the loaded docs and cites a filename.

### Phase 5 — Polish (optional, only if asked)
- HNSW index: `CREATE INDEX ... USING hnsw (embedding vector_cosine_ops) WITH (m = 16, ef_construction = 64)` (idempotent; `CREATE INDEX CONCURRENTLY` in real production). Justified once chunk count grows.
- Batch indexing with continue-on-error; LangSmith traces (env-var driven, EU endpoint per repo convention).

---

## 9. Pitfalls — reference material is buggy

You are **not** reading the demos or code-snippets (§0), so don't reconstruct their patterns from memory. For the record they contain: an undefined `cosine_similarity`/`reranking`, an embedding "model" that isn't a sentence-transformer (`Qwen3-4B`), a query against a non-existent `search_vector` column, and several stub functions. Follow this plan's contracts, not half-remembered snippet code.

## 10. Deferred / open (not blocking)

- **Extraction and retrieval are two independent deliverables.** Structured fields (Invoice/Contract) are saved to `Document.doc_metadata` (JSONB, queryable) to satisfy homework block 1 and enable future metadata filtering; the Q&A path (blocks 3–4) answers from **chunk embeddings only** and does not read `doc_metadata`. Keep the write path producing both; the read path consumes only chunks. The homework diagram shows literal `factura.json` files — add file export **only if asked**.
- A separate `AnalystAgent` class is **not** needed — the existing ReAct agent + `search_documents` tool + analyst prompt cover it (tool retrieves, agent generates; no double-generation).

---

## 11. Definition of done

- The shared `skillab-postgres` stack is up + `alembic upgrade head` creates the schema in the `skillab` DB (incl. the `vector` extension).
- Running the pipeline over `data samples/` populates `documents` + `document_chunks` with 384-dim embeddings.
- `QAAgent(prompt_name="analyst_system").chat(...)` answers document questions grounded in the stored chunks, citing source filenames, and says it doesn't know when the answer is absent.
- New code respects the repo conventions in §4 and the tool/prompt contracts in §3.

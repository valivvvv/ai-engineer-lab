# Homework — Lesson 2: QA Agent with Tools + Prompts

A ReAct-style Q&A agent built on LangChain, demonstrating:

- **Prompt Registry** — versioned YAML prompts rendered with Jinja2
- **Tool Registry** — Pydantic-validated tools registered via decorator
- **ReAct loop** — Think → Act → Observe, with parallel async tools available
- **Provider-agnostic LLM** — Ollama (local), Google Gemini, or Anthropic Claude
- **Observability** — LangSmith via env vars, zero code changes

## Prerequisites

- Python 3.10+
- Docker — for the local Ollama stack in `../llm docker containers/`
- (Optional) A LangSmith account for traces — https://smith.langchain.com (US) or https://eu.smith.langchain.com (EU)

## Setup

```bash
cd "homework-lesson-2"
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # then fill in real values where needed
```

Start the local LLM stack:

```bash
cd "../llm docker containers"
docker-compose up -d
```

Confirm Ollama is up and has `llama3.2:3b`:

```bash
curl http://localhost:11434/api/tags
# If llama3.2:3b is missing:
docker exec -it ollama ollama pull llama3.2:3b
```

## Run

```bash
# Run the canonical demo (one query per registered tool)
python main.py

# Interactive REPL
python main.py --repl

# REPL with streamed responses
python main.py --repl --stream

# Force the agent to reply in a specific language
python main.py --repl --language Romanian
```

## Architecture

```
                            QAAgent  (agent/qa.py)
                            ───────────────────────
                            public API: chat() / stream() / clear_history()
                                       │
                ┌──────────────────────┼──────────────────────┐
                ▼                      ▼                      ▼
        LLMFactory.create        ToolWrapper           PromptRegistry
        (agent/llm_factory.py)   (tools/registry.py)   (prompts/registry.py)
                │                      │                      │
                │                      │                      │
                └─── .bind_tools(catalog) ──┐                 │
                                            ▼                 ▼
                                     tool-aware LLM    rendered system prompt
                                            │                 │
                                            └───────┬─────────┘
                                                    ▼
                                       react_events()  (agent/react.py)
                                       ──────────────────────────────
                                       single-source-of-truth generator
                                       yields TextChunk + Final events

           react_loop()  ──── consumes Final ──── QAAgent.chat()
           QAAgent.stream()  ──── consumes TextChunk + Final
```

## File structure

```
homework-lesson-2/
├── main.py                       # entry point
├── requirements.txt
├── .env.example                  # template (committed)
├── .env                          # real values (gitignored)
│
├── agent/
│   ├── __init__.py
│   ├── llm_factory.py            # Provider factory (Ollama / Gemini / Anthropic)
│   ├── react.py                  # Event types + react_events + react_loop
│   └── qa.py                     # QAAgent — public API
│
├── prompts/
│   ├── __init__.py
│   ├── registry.py               # PromptRegistry, PromptTemplate, get_prompt_registry()
│   └── templates/
│       └── qa_system.yaml        # Versioned system prompt
│
└── tools/
    ├── __init__.py               # Side-effect imports register each tool
    ├── registry.py               # @register_tool decorator + ToolWrapper
    ├── calculator.py             # Safe arithmetic (AST sandbox)
    ├── current_datetime.py       # Now in IANA timezone
    └── web_search.py             # Stubbed search (fake results)
```

## Design patterns used

| Pattern | Where |
|---|---|
| **Factory** | `LLMFactory.create()` — hides per-provider construction details |
| **Registry** | `TOOL_REGISTRY`, `PromptRegistry` — name → object lookup tables |
| **Singleton** | `get_prompt_registry()` — lazy-initialized module-level instance |
| **Decorator** | `@register_tool` — populates the registry at import time |
| **Adapter** | `ToolWrapper.catalog()` — bridges our Pydantic-style tools to LangChain's `StructuredTool` |
| **Generator-based primitive** | `react_events()` — one source of truth, two consumption shapes (returning + streaming) |

## Observability

LangSmith integration is env-var driven — no code changes anywhere in the project.

To enable, set in `.env`:

```bash
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=lsv2_pt_...
LANGCHAIN_PROJECT=homework-lesson-2
# EU users:
LANGCHAIN_ENDPOINT=https://eu.api.smith.langchain.com
```

Every LLM call and tool execution will appear as spans in the
`homework-lesson-2` project at https://smith.langchain.com (or the EU equivalent).
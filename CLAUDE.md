# ai-engineer-lab — context for future sessions

## Purpose

Long-running lab for Vali's AI Engineering course. Current code is
lesson 2 (a ReAct QA agent), but the repo will keep growing — RAG, evals,
fine-tuning, LangGraph, etc. Do not assume the current architecture is
final. Remote: https://github.com/valivvvv/ai-engineer-lab

## Orientation pointers

- Entry point: `main.py`
- Core loop primitive: `react_events()` in `agent/react.py` — single
  source of truth; both sync and async consumers feed off it
- Tools register via `@register_tool` in their own module; `tools/__init__.py`
  triggers registration on import
- Prompts: YAML files under `prompts/templates/`, rendered by Jinja2 with
  `StrictUndefined` (missing vars are errors)

## What's stubbed vs. real

- `tools/web_search.py` — **stubbed.** Returns deterministic fake results.
  Do not treat output as real; do not benchmark against it.
- LangSmith tracing is on the **EU endpoint**
  (`LANGCHAIN_ENDPOINT=https://eu.api.smith.langchain.com`) — non-obvious
  config trap if traces don't appear.

## Hard conventions Vali has set

These are project-specific. Global preferences (plan-first, senior tone)
live in `~/.claude/CLAUDE.md` and are not repeated here.

1. **No 1–2 letter variable names.** Use `tool_call`, not `tc`.
2. **Avoid explanatory comments.** Prefer renames, helpers, named constants.
   One comment is deliberately kept in `tools/registry.py` for PEP 563 /
   `get_type_hints` — keep it; that one is load-bearing.
3. **Present ideas before doing them.** No unsolicited renames, refactors,
   or restructures.
4. **Teaching mode for new lessons:** "you write, I review" pace, brief
   conceptual framing per phase, smoke test at each phase.

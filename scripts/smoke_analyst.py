"""Analyst smoke (Phase 4): the agent answers a document question via RAG.

End-to-end check of the full read path: QAAgent loads the analyst prompt, the
ReAct loop calls the search_documents tool, and the answer is grounded in the
indexed chunks and cites a source filename.

Assumes the documents are already indexed. If the DB is empty, run
scripts/smoke_rag.py first (it indexes the data samples).

The agent provider/model comes from config.AGENT (single source of truth);
override it via AGENT_PROVIDER / AGENT_MODEL in .env.

Run: .venv/bin/python scripts/smoke_analyst.py
"""
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import config
from agent.qa import QAAgent

QUESTION = "Ce clauze de reziliere avem?"


def main() -> None:
    agent = QAAgent(
        provider=config.AGENT.provider,
        model=config.AGENT.model,
        prompt_name="analyst_system",
        assistant_name="Document Analyst",
        language="Romanian",
    )

    answer = agent.chat(QUESTION)
    print(f"question: {QUESTION!r}\n")
    print(answer)

    assert answer.strip(), "agent returned an empty answer"
    assert "contract" in answer.lower(), (
        "expected the answer to cite a contract source filename, "
        f"got:\n{answer}"
    )
    print("\nPASS")


if __name__ == "__main__":
    main()
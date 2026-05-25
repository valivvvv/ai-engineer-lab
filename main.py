"""Entry point for the QA agent.

Three modes:
  python main.py                   — run the fixed demo (one query per tool)
  python main.py --repl            — interactive REPL
  python main.py --repl --stream   — REPL with streamed responses
"""
from __future__ import annotations

import argparse
import sys

from dotenv import load_dotenv

from agent.qa import QAAgent


_DEMO_QUERIES: list[tuple[str, str]] = [
    (
        "Calculator",
        "Calculate the total with 19% VAT for these items: "
        "Laptop 4500 RON, Mouse 150 RON, Keyboard 280 RON.",
    ),
    (
        "Datetime",
        "What is the current date and time in Bucharest?",
    ),
    (
        "Web search",
        "Search the web for information about Romania's 2026 fiscal calendar.",
    ),
]


def run_demo(agent: QAAgent) -> None:
    for label, query in _DEMO_QUERIES:
        # Each demo query is independent; clear history so earlier
        # tool calls don't bleed into the next turn's context.
        agent.clear_history()
        print(f"\n========== {label} ==========")
        print(f"USER:\n  {query}\n")
        print(f"ASSISTANT:\n  {agent.chat(query)}")


def run_repl(agent: QAAgent, stream: bool) -> None:
    print("QA Agent ready. Type your question, or 'exit' to quit.\n")
    while True:
        try:
            user_input = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return

        if user_input in ("exit", "quit"):
            return
        if not user_input:
            continue

        print("agent> ", end="", flush=True)
        if stream:
            for chunk in agent.stream(user_input):
                sys.stdout.write(chunk)
                sys.stdout.flush()
            print()
        else:
            print(agent.chat(user_input))
        print()


def main() -> None:
    parser = argparse.ArgumentParser(description="QA Agent demo and REPL.")
    parser.add_argument(
        "--repl",
        action="store_true",
        help="Start an interactive REPL instead of running the fixed demo.",
    )
    parser.add_argument(
        "--stream",
        action="store_true",
        help="Stream responses chunk-by-chunk (REPL mode only).",
    )
    parser.add_argument(
        "--language",
        default=None,
        help="Force the agent to reply in a specific language (e.g., 'Romanian').",
    )
    arguments = parser.parse_args()

    load_dotenv()
    agent = QAAgent(language=arguments.language)

    if arguments.repl:
        run_repl(agent, stream=arguments.stream)
    else:
        run_demo(agent)


if __name__ == "__main__":
    main()
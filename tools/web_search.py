"""Web search tool — STUBBED for the homework.

Returns deterministic fake results so the agent can reason about web-shaped
data without us paying for a real search API. Swap _fake_results() for a real
client (Tavily, Serper, DuckDuckGo) when ready — the contract above doesn't
change.
"""
from __future__ import annotations

from pydantic import BaseModel, Field

from .registry import register_tool


class WebSearchParams(BaseModel):
    query: str = Field(
        description="The search query. Be specific — short, keyword-style works best.",
        min_length=2,
        max_length=200,
    )
    max_results: int = Field(
        default=3,
        ge=1,
        le=10,
        description="How many results to return (1–10).",
    )


@register_tool
def web_search(params: WebSearchParams) -> str:
    """Searches the web and returns the top results as a numbered list.

    Each result has a title, URL, and short snippet. Use this when the user
    asks about something you don't know — current events, prices, addresses,
    documentation, etc.
    """
    results = _fake_results(params.query, params.max_results)
    return "\n".join(
        f"{i}. {r['title']} — {r['url']}\n   {r['snippet']}"
        for i, r in enumerate(results, start=1)
    )


def _fake_results(query: str, count: int) -> list[dict[str, str]]:
    template = [
        {
            "title": f"{query.title()} — Overview",
            "url": f"https://example.com/overview?q={query.replace(' ', '+')}",
            "snippet": f"A general overview of {query}. Last updated 2026.",
        },
        {
            "title": f"How to think about {query}",
            "url": f"https://example.org/guide/{query.replace(' ', '-')}",
            "snippet": f"Practical guide covering key aspects of {query}.",
        },
        {
            "title": f"{query.title()} — common questions",
            "url": f"https://faq.example.net/{query.replace(' ', '_')}",
            "snippet": f"Frequently asked questions about {query}.",
        },
        {
            "title": f"News: {query}",
            "url": f"https://news.example.com/topic/{query.replace(' ', '-')}",
            "snippet": f"Recent developments related to {query}.",
        },
    ]
    return template[:count]
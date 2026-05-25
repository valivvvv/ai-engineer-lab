"""Datetime tool — returns the current date/time, optionally in a timezone."""
from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from pydantic import BaseModel, Field

from .registry import register_tool


class GetCurrentDatetimeParams(BaseModel):
    timezone: str = Field(
        default="UTC",
        description=(
            "IANA timezone name (e.g., 'UTC', 'Europe/Bucharest', "
            "'America/New_York'). Defaults to UTC."
        ),
    )


@register_tool
def get_current_datetime(params: GetCurrentDatetimeParams) -> str:
    """Returns the current date and time in ISO-8601 format.

    Use this whenever you need to know "today", "now", or compute days until
    a future date. Accepts an optional IANA timezone name.
    """
    try:
        tz = ZoneInfo(params.timezone)
    except ZoneInfoNotFoundError:
        raise ValueError(
            f"Unknown timezone: '{params.timezone}'. "
            f"Use an IANA name like 'Europe/Bucharest' or 'UTC'."
        )
    return datetime.now(tz).isoformat(timespec="seconds")
from __future__ import annotations

import os
from typing import Any

from groq import Groq


def available() -> bool:
    """Return True when a Groq API key has been configured."""
    return bool(os.getenv("GROQ_API_KEY"))


def _value(candidate: Any, name: str, default: Any = "") -> Any:
    """Read a field from either a dataclass/object or a dictionary."""
    if isinstance(candidate, dict):
        return candidate.get(name, default)
    return getattr(candidate, name, default)


def _set_value(candidate: Any, name: str, value: Any) -> None:
    """Set a field on either a dataclass/object or a dictionary."""
    if isinstance(candidate, dict):
        candidate[name] = value
    else:
        setattr(candidate, name, value)


def _format_number(value: Any, decimals: int = 2) -> str:
    try:
        return f"{float(value):.{decimals}f}"
    except (TypeError, ValueError):
        return "not available"


def explain_candidate(candidate: Any, client: Groq | None = None) -> str:
    """
    Ask Groq to explain an already-calculated trading signal.

    Groq is deliberately not asked to calculate the score or create new
    entry/exit levels. It only explains the deterministic scanner output.
    """
    if not available():
        return "Groq analysis unavailable: GROQ_API_KEY is not configured."

    groq_client = client or Groq(api_key=os.environ["GROQ_API_KEY"])
    model = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

    symbol = _value(candidate, "symbol", "Unknown")
    name = _value(candidate, "name", "")
    confidence = _value(candidate, "confidence", "Unknown")
    score = _value(candidate, "score", "Unknown")
    setup = _value(candidate, "setup", "Unknown")
    rule_summary = _value(candidate, "rule_summary", "No scoring explanation supplied.")

    prompt = f"""
You are explaining the output of a deterministic ASX short-term trading scanner.

Do not recalculate the score.
Do not invent news, fundamentals, announcements, prices, indicators or market conditions.
Do not claim certainty or guarantee a profitable trade.
Use only the supplied facts.

Share:
1. Why the scanner assigned this confidence rating.
2. The strongest bullish evidence.
3. The main risks or invalidation factors.
4. A concise interpretation of the entry, stop and targets.

Stock: {symbol} {name}
Scanner score: {score}/100
Confidence: {confidence}
Setup: {setup}
Last price: ${_format_number(_value(candidate, "close"))}
Relative volume: {_format_number(_value(candidate, "relative_volume"))}x
Price change: {_format_number(_value(candidate, "change_pct"))}%
RSI(14): {_format_number(_value(candidate, "rsi14"))}
Entry: ${_format_number(_value(candidate, "entry"))}
Stop: ${_format_number(_value(candidate, "stop"))}
Target 1: ${_format_number(_value(candidate, "target1"))}
Target 2: ${_format_number(_value(candidate, "target2"))}

Deterministic scoring reasons:
{rule_summary}

Write 120-180 words in plain English. Use short headings and bullets.
""".strip()

    try:
        response = groq_client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a cautious market-research assistant. "
                        "Explain supplied scanner results without inventing facts."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
            max_completion_tokens=450,
        )

        choices = getattr(response, "choices", None)
        if not choices:
            return "Groq analysis unavailable: the API returned no response choices."

        message = getattr(choices[0], "message", None)
        text = getattr(message, "content", None) if message else None
        if not text or not str(text).strip():
            return "Groq analysis unavailable: the API returned an empty response."

        return str(text).strip()

    except Exception as exc:
        # Show enough diagnostic information to identify configuration/API issues
        # without exposing the API key.
        return f"Groq analysis unavailable: {type(exc).__name__}: {exc}"


def enrich_candidates(candidates: list[Any], limit: int = 12) -> None:
    """
    Add Groq explanations to the strongest candidates in place.

    Failures affect only the explanation; they never interrupt the market scan.
    """
    if not candidates:
        return

    if not available():
        for candidate in candidates[:limit]:
            _set_value(
                candidate,
                "ai_summary",
                "Groq analysis unavailable: GROQ_API_KEY is not configured.",
            )
        return

    client = Groq(api_key=os.environ["GROQ_API_KEY"])

    ranked = sorted(
        candidates,
        key=lambda item: float(_value(item, "score", 0) or 0),
        reverse=True,
    )

    for candidate in ranked[:limit]:
        _set_value(candidate, "ai_summary", explain_candidate(candidate, client))

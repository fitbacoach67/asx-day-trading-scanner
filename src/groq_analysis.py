from __future__ import annotations

import json
import os
from typing import Iterable

from groq import Groq
from pydantic import BaseModel, Field

from .models import Candidate


class Narrative(BaseModel):
    summary: str = Field(max_length=700)
    confirmation: str = Field(max_length=300)
    invalidation: str = Field(max_length=300)
    risks: list[str] = Field(max_length=4)


SYSTEM_PROMPT = """
You are a cautious market-analysis editor. The Python application has already calculated
the indicators, score and trade levels. Do not change, recalculate or contradict any number.
Do not claim certainty, predict returns, or call the output financial advice.
Explain only the supplied evidence in plain Australian English.
Return valid JSON with keys: summary, confirmation, invalidation, risks.
"""


def _api_key() -> str | None:
    key = os.getenv("GROQ_API_KEY")
    if key:
        return key
    try:
        import streamlit as st
        return st.secrets.get("GROQ_API_KEY")
    except Exception:
        return None


def groq_model() -> str:
    model = os.getenv("GROQ_MODEL")
    if model:
        return model
    try:
        import streamlit as st
        return st.secrets.get("GROQ_MODEL", "llama-3.3-70b-versatile")
    except Exception:
        return "llama-3.3-70b-versatile"


def available() -> bool:
    return bool(_api_key())


def enrich_candidates(candidates: Iterable[Candidate], limit: int = 10) -> None:
    key = _api_key()
    if not key:
        return

    client = Groq(api_key=key)
    for candidate in list(candidates)[:limit]:
        facts = {
            "symbol": candidate.symbol,
            "name": candidate.name,
            "setup": candidate.setup,
            "score": candidate.score,
            "confidence": candidate.confidence,
            "close": candidate.close,
            "relative_volume": candidate.relative_volume,
            "change_pct": candidate.change_pct,
            "ema9": candidate.ema9,
            "ema20": candidate.ema20,
            "rsi14": candidate.rsi14,
            "macd_hist": candidate.macd_hist,
            "atr14": candidate.atr14,
            "vwap": candidate.vwap,
            "breakout_20": candidate.breakout_20,
            "entry": candidate.entry,
            "stop": candidate.stop,
            "target1": candidate.target1,
            "target2": candidate.target2,
            "rule_summary": candidate.rule_summary,
        }
        try:
            completion = client.chat.completions.create(
                model=groq_model(),
                temperature=0.1,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": json.dumps(facts)},
                ],
                response_format={"type": "json_object"},
            )
            content = completion.choices[0].message.content or "{}"
            parsed = Narrative.model_validate_json(content)
            candidate.ai_summary = (
                f"{parsed.summary}\n\nConfirmation: {parsed.confirmation}\n\n"
                f"Invalidation: {parsed.invalidation}\n\n"
                f"Risks: {'; '.join(parsed.risks)}"
            )
        except Exception as exc:
            candidate.ai_summary = f"Groq analysis unavailable: {type(exc).__name__}"

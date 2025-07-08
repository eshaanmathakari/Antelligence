"""
queen_llm_agent.py
Lightweight wrapper that turns the Queen into a single-step LLM agent.
Relies only on the existing `openai.OpenAI` client that the main model creates.
"""

import json
from typing import Dict, Any
import openai

# ---------------------------------------------------------------------
SYSTEM_PROMPT = """You are a hyper-intelligent Queen Ant commanding a colony.
Return ONE legal next cell for EVERY ant, as pure JSON: {"0":[x,y], ...}.
A legal cell is the ant's current position or any of its 8 neighbours.
If you cannot comply, reply exactly {"retry": true}.
"""

def _llm_call(state: Dict[str, Any], io_client: openai.OpenAI, model_name: str) -> str:
    """One chat completion with function-like structure."""
    user_prompt = f"Current state: {json.dumps(state)}"
    response = io_client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.2,
        max_completion_tokens=400,
    )
    return response.choices[0].message.content.strip()

def plan_moves(state: Dict[str, Any], io_client: openai.OpenAI,
               model_name: str) -> Dict[int, tuple]:
    """
    Returns {ant_id: (x,y), …}.  Retries up to 3× on invalid JSON / retry flag.
    """
    raw = _llm_call(state, io_client, model_name)
    for _ in range(3):
        try:
            data = json.loads(raw)
            if data.get("retry") is True:
                raw = _llm_call(state, io_client, model_name)
                continue
            return {int(k): tuple(v) for k, v in data.items()}
        except Exception:
            raw = _llm_call(state, io_client, model_name)
    return {}  # fall back to “no guidance”

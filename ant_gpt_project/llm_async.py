"""
Async wrapper around IO Intelligence (ionet) Chat Completions.
The IO API is OpenAI-compatible: set base_url, key env var.
"""
import os, asyncio
from openai import AsyncOpenAI            # pip install --upgrade openai>=1.30

client = AsyncOpenAI(
    api_key=os.getenv("IO_SECRET_KEY"),   # ←   export IO_SECRET_KEY=sk-...
    base_url="https://api.intelligence.io.solutions/api/v1/",
)

async def _one(prompt: str) -> str:
    rsp = await client.chat.completions.create(
        model="Llama-3-70b-instruct",     # pick any model slug from IO docs
        messages=[{"role": "user", "content": prompt}],
        max_tokens=1,
        temperature=0.2,
    )
    return rsp.choices[0].message.content.strip()

async def batch(prompts: list[str]) -> list[str]:
    return await asyncio.gather(*(_one(p) for p in prompts))

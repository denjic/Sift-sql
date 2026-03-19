"""Groq API wrapper with exponential-backoff retry logic."""

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

from text2sql_mas.config import PipelineConfig


class GroqClient:
    BASE_URL = "https://api.groq.com/openai/v1/chat/completions"

    def __init__(self, config: PipelineConfig):
        self.config = config
        self.headers = {
            "Authorization": f"Bearer {config.groq_api_key}",
            "Content-Type": "application/json"
        }

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
    def complete(self, system_prompt: str, user_message: str) -> str:
        payload = {
            "model": self.config.groq_model,
            "temperature": self.config.groq_temperature,
            "max_tokens": self.config.groq_max_tokens,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message}
            ]
        }
        with httpx.Client(timeout=self.config.groq_timeout_seconds) as client:
            response = client.post(self.BASE_URL, json=payload, headers=self.headers)
            response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"].strip()

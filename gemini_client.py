"""Small server-side Gemini client for final answer writing."""

import json
import time
from typing import Any

import requests  # type: ignore[import-not-found]

from prompt import build_gemini_prompt


class GeminiError(RuntimeError):
    """Raised when Gemini cannot return a usable response."""


class GeminiClient:
    def __init__(self, api_key: str, model: str = "gemini-flash-latest") -> None:
        api_key = api_key.strip()
        if not api_key:
            raise GeminiError("GEMINI_API_KEY is not configured.")

        self.api_key = api_key
        self.model = model.strip() or "gemini-flash-latest"
        self.endpoint = (
            "https://generativelanguage.googleapis.com/v1beta/models/"
            f"{self.model}:generateContent"
        )

    @staticmethod
    def _error_message(response: requests.Response) -> str:
        try:
            payload = response.json()
            return str(payload.get("error", {}).get("message", "")).strip()
        except (ValueError, json.JSONDecodeError):
            return response.text[:500].strip()

    def generate_answer(
        self,
        question: str,
        knowledge_base_answer: str = "",
        sources: list[dict[str, Any]] | None = None,
        retrieved_context: str = "",
        history: list[dict[str, Any]] | None = None,
        retrieval_status: str = "",
    ) -> str:
        prompt = build_gemini_prompt(
            question=question,
            knowledge_base_answer=knowledge_base_answer,
            sources=sources or [],
            retrieved_context=retrieved_context,
            history=[],
            retrieval_status=retrieval_status,
        )

        payload = {
            "system_instruction": {
                "parts": [
                    {
                        "text": (
                            "Use only the retrieved PDF passages. If they do not support the answer, "
                            "say that the information could not be found in the uploaded documents. "
                            "Never use outside knowledge or add unsupported examples."
                        )
                    }
                ]
            },
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": 0.25,
                "maxOutputTokens": 1600,
                "responseMimeType": "text/plain",
            },
        }

        last_error = ""
        for attempt in range(3):
            try:
                response = requests.post(
                    self.endpoint,
                    params={"key": self.api_key},
                    headers={"Content-Type": "application/json"},
                    json=payload,
                    timeout=60,
                )
            except requests.RequestException as exc:
                last_error = f"Network error: {exc}"
                if attempt < 2:
                    time.sleep(1.0 * (attempt + 1))
                    continue
                raise GeminiError(last_error) from exc

            if response.ok:
                break

            last_error = self._error_message(response) or f"HTTP {response.status_code}"
            if response.status_code not in {408, 429, 500, 502, 503, 504} or attempt == 2:
                raise GeminiError(f"Gemini request failed: {last_error}")
            time.sleep(1.0 * (attempt + 1))
        else:
            raise GeminiError(f"Gemini request failed: {last_error}")

        try:
            data = response.json()
        except (ValueError, json.JSONDecodeError) as exc:
            raise GeminiError("Gemini returned invalid JSON.") from exc

        candidates = data.get("candidates", [])
        if not candidates:
            reason = data.get("promptFeedback", {}).get("blockReason")
            suffix = f" ({reason})" if reason else ""
            raise GeminiError(f"Gemini returned no answer{suffix}.")

        parts = candidates[0].get("content", {}).get("parts", [])
        answer = "".join(str(part.get("text", "")) for part in parts).strip()
        if not answer:
            raise GeminiError("Gemini returned an empty answer.")

        return answer
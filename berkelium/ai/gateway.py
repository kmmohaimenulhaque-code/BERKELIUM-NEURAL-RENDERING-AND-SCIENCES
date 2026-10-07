"""Model Gateway: provider abstraction for structured generation (architecture §9).

Providers return raw text; parsing/validation is always done by the core. Supported:
  * OpenAICompatibleProvider — vLLM (MI300X/ROCm) and Fireworks: /v1/chat/completions with
    response_format json_schema (constrained decoding), optional LoRA model name, Qwen3 thinking switch.
  * ReplayProvider — deterministic canned responses (tests, regression).
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass
class Generation:
    text: str
    model: str
    provider: str
    latency_s: float
    usage: dict = field(default_factory=dict)
    raw: dict | None = None


@dataclass
class DecodingConfig:
    # Qwen3 model-card non-thinking recommendation: T=0.7, top_p=0.8, top_k=20, min_p=0.
    temperature: float = 0.7
    top_p: float = 0.8
    top_k: int = 20
    max_tokens: int = 4096
    enable_thinking: bool = False
    seed: int | None = 0


class ModelProvider(Protocol):
    name: str
    model: str

    def generate(self, messages: list[dict], json_schema: dict | None = None,
                 decoding: DecodingConfig | None = None) -> Generation: ...


class GatewayError(RuntimeError):
    pass


class OpenAICompatibleProvider:
    def __init__(self, base_url: str, model: str, api_key: str | None = None, name: str = "openai_compatible",
                 timeout_s: float = 300.0, schema_mode: str = "json_schema"):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key or os.environ.get("BERKELIUM_MODEL_API_KEY")
        self.name = name
        self.timeout_s = timeout_s
        self.schema_mode = schema_mode  # "json_schema" | "none"

    def payload(self, messages, json_schema, d: DecodingConfig) -> dict:
        body: dict[str, Any] = {"model": self.model, "messages": messages, "temperature": d.temperature,
                                "top_p": d.top_p, "max_tokens": d.max_tokens}
        if d.top_k:
            body["top_k"] = d.top_k
        if d.seed is not None:
            body["seed"] = d.seed
        if json_schema is not None and self.schema_mode == "json_schema":
            body["response_format"] = {"type": "json_schema",
                                       "json_schema": {"name": "berkelium", "schema": json_schema, "strict": True}}
        body["chat_template_kwargs"] = {"enable_thinking": d.enable_thinking}  # vLLM / Qwen3
        return body

    def generate(self, messages, json_schema=None, decoding=None) -> Generation:
        d = decoding or DecodingConfig()
        body = json.dumps(self.payload(messages, json_schema, d)).encode()
        req = urllib.request.Request(f"{self.base_url}/chat/completions", data=body, method="POST",
                                     headers={"Content-Type": "application/json",
                                              **({"Authorization": f"Bearer {self.api_key}"} if self.api_key else {})})
        t = time.perf_counter()
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_s) as r:
                data = json.loads(r.read())
        except (urllib.error.URLError, TimeoutError) as e:
            raise GatewayError(f"{self.name}: {e}") from None
        msg = data["choices"][0]["message"]
        return Generation(text=msg.get("content") or "", model=data.get("model", self.model), provider=self.name,
                          latency_s=time.perf_counter() - t, usage=data.get("usage", {}), raw=None)


class ReplayProvider:
    """Returns responses in order (or by key function) — deterministic offline testing of the loop."""

    def __init__(self, responses: list[str], model: str = "replay", name: str = "replay"):
        self.responses = list(responses)
        self.model, self.name = model, name
        self.calls: list[list[dict]] = []

    def generate(self, messages, json_schema=None, decoding=None) -> Generation:
        self.calls.append(messages)
        if not self.responses:
            raise GatewayError("replay exhausted")
        return Generation(self.responses.pop(0), self.model, self.name, 0.0)


def provider_from_env() -> ModelProvider:
    """BERKELIUM_MODEL_URL (e.g. http://mi300x:8000/v1 or https://api.fireworks.ai/inference/v1),
    BERKELIUM_MODEL (e.g. Qwen/Qwen3-32B, an adapter name, or accounts/fireworks/models/qwen3-32b)."""
    url, model = os.environ.get("BERKELIUM_MODEL_URL"), os.environ.get("BERKELIUM_MODEL")
    if not url or not model:
        raise GatewayError("set BERKELIUM_MODEL_URL and BERKELIUM_MODEL")
    return OpenAICompatibleProvider(url, model, name=os.environ.get("BERKELIUM_PROVIDER", "openai_compatible"))

"""Model Gateway: provider abstraction for structured generation (architecture §9).

Providers return raw text; parsing/validation is always done by the core. Supported:
  * OpenAICompatibleProvider — vLLM (MI300X/ROCm) and Fireworks: /v1/chat/completions with
    response_format json_schema (constrained decoding), optional LoRA model name, Qwen3 thinking switch.
  * ReplayProvider — deterministic canned responses (tests, regression).
"""

from __future__ import annotations

import http.client
import json
import os
import time
import urllib.error
import urllib.parse
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


# Transport failures that must surface as GatewayError, never as a raw traceback. ConnectionResetError /
# RemoteDisconnected are OSError / HTTPException subclasses that urllib does NOT wrap in URLError.
_TRANSPORT_ERRORS = (urllib.error.URLError, TimeoutError, OSError, http.client.HTTPException)
_LOOPBACK = {"localhost", "127.0.0.1", "::1", "0.0.0.0"}


def _opener(url: str) -> urllib.request.OpenerDirector:
    """Loopback endpoints bypass http(s)_proxy: a proxy that cannot reach the local vLLM resets the
    connection instead of refusing it, which hides the real cause ("nothing is serving")."""
    host = urllib.parse.urlsplit(url).hostname or ""
    if host in _LOOPBACK:
        return urllib.request.build_opener(urllib.request.ProxyHandler({}))
    return urllib.request.build_opener()


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
            with _opener(self.base_url).open(req, timeout=self.timeout_s) as r:
                data = json.loads(r.read())
        except urllib.error.HTTPError as e:
            detail = e.read()[:500].decode("utf-8", "replace")
            raise GatewayError(f"{self.name}: HTTP {e.code}: {detail}") from None
        except _TRANSPORT_ERRORS as e:
            raise GatewayError(f"{self.name}: {type(e).__name__}: {e} ({self.base_url})") from None
        except json.JSONDecodeError as e:
            raise GatewayError(f"{self.name}: non-JSON response: {e}") from None
        if not data.get("choices"):
            raise GatewayError(f"{self.name}: response has no choices: {str(data)[:300]}")
        msg = data["choices"][0]["message"]
        return Generation(text=msg.get("content") or "", model=data.get("model", self.model), provider=self.name,
                          latency_s=time.perf_counter() - t, usage=data.get("usage", {}), raw=None)


    def served_models(self, timeout_s: float = 10.0) -> list[str]:
        """GET {base}/models — the readiness probe (vLLM answers it only once weights are loaded)."""
        req = urllib.request.Request(f"{self.base_url}/models", headers={
            **({"Authorization": f"Bearer {self.api_key}"} if self.api_key else {})})
        try:
            with _opener(self.base_url).open(req, timeout=timeout_s) as r:
                data = json.loads(r.read())
        except _TRANSPORT_ERRORS + (json.JSONDecodeError,) as e:
            raise GatewayError(f"{self.name}: {type(e).__name__}: {e} ({self.base_url})") from None
        return [m.get("id", "") for m in data.get("data", [])]

    def check(self, wait_s: float = 0.0, interval_s: float = 10.0, on_wait=None) -> list[str]:
        """Raise GatewayError unless the endpoint is up AND serves self.model. Optionally poll up to wait_s.
        ``on_wait(elapsed_s, error)`` is called after every failed poll (progress output / fail-fast; it may raise)."""
        if os.environ.get("BERKELIUM_SKIP_MODEL_CHECK") == "1":   # hosted APIs that don't list models
            return [self.model]
        start = time.monotonic()
        deadline = start + wait_s
        while True:
            try:
                ids = self.served_models()
                if self.model not in ids:
                    raise GatewayError(f"{self.name}: endpoint up but does not serve {self.model!r}; serves {ids}")
                return ids
            except GatewayError as e:
                if time.monotonic() >= deadline:
                    raise
                if on_wait is not None:
                    on_wait(time.monotonic() - start, e)
                time.sleep(interval_s)


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

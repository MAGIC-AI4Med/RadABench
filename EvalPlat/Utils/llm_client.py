"""Unified LLM client for the agent core.

Model behaviour is declared in Configs/models.yaml; credentials come from the
environment.

    from Utils.llm_client import LLMClient

    llm = LLMClient("claude-sonnet-4.5")
    reply = llm.chat([{"role": "user", "content": "..."}])

Credentials are looked up first in the environment variable named by
api_key_env, then in a .env file at the repository root. If neither is set,
MissingCredential names the variable to set instead of failing later inside
the provider SDK.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

__all__ = ["LLMClient", "MissingCredential", "UnknownModel", "available_models"]

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = Path(__file__).resolve().parents[1] / "Configs" / "models.yaml"

Message = Dict[str, str]


class MissingCredential(RuntimeError):
    """Raised when the environment variable holding an API key is unset."""


class UnknownModel(KeyError):
    """Raised when an alias is not present in the model registry."""


# --------------------------------------------------------------------------- #
# configuration loading
# --------------------------------------------------------------------------- #

def _load_dotenv(path: Path) -> None:
    """Populate ``os.environ`` from a .env file without overwriting real vars."""
    if not path.is_file():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip().strip("'\"")
        os.environ.setdefault(key, value)


def _load_registry(path: Path = CONFIG_PATH) -> Dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - dependency is declared
        raise ImportError(
            "PyYAML is required to read the model registry. "
            "Install it with `pip install pyyaml`."
        ) from exc
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


_REGISTRY: Optional[Dict[str, Any]] = None


def _registry() -> Dict[str, Any]:
    global _REGISTRY
    if _REGISTRY is None:
        _load_dotenv(REPO_ROOT / ".env")
        _REGISTRY = _load_registry()
    return _REGISTRY


def available_models() -> List[str]:
    """Aliases accepted by --model, in registry order."""
    return list(_registry()["models"].keys())


# --------------------------------------------------------------------------- #
# client
# --------------------------------------------------------------------------- #

class LLMClient:
    """One call signature across hosted APIs and local weights.

    alias: a key from Configs/models.yaml.
    model_path: overrides the registry for local_hf backends, so a downloaded
    checkpoint can be used without editing the registry.
    """

    def __init__(self, alias: str, model_path: Optional[str] = None) -> None:
        registry = _registry()
        models = registry["models"]
        if alias not in models:
            raise UnknownModel(
                f"Unknown model {alias!r}. Available: {', '.join(models)}"
            )

        self.alias = alias
        self.spec: Dict[str, Any] = models[alias]
        self.defaults: Dict[str, Any] = registry.get("defaults", {})
        self.backend: str = self.spec["backend"]
        self.model_id: str = self.spec.get("model_id", alias)

        self._client = None      # OpenAI SDK client
        self._tokenizer = None   # transformers tokenizer
        self._model = None       # transformers model
        self._headers = None     # raw HTTP headers
        self._proxy_url = None

        builder = {
            "openai_compatible": self._init_openai,
            "http_proxy": self._init_http_proxy,
            "local_hf": lambda: self._init_local(model_path),
        }.get(self.backend)
        if builder is None:
            raise ValueError(f"Unsupported backend {self.backend!r} for {alias!r}")
        builder()

    # -- credential handling ------------------------------------------------ #

    def _api_key(self) -> str:
        var = self.spec["api_key_env"]
        key = os.environ.get(var)
        if not key:
            raise MissingCredential(
                f"{self.alias} needs an API key. Set the environment variable "
                f"{var}, or add `{var}=...` to a .env file in the repository "
                f"root (see .env.example)."
            )
        return key

    def _base_url(self) -> str:
        override = self.spec.get("base_url_env")
        if override and os.environ.get(override):
            return os.environ[override]
        return self.spec["base_url"]

    # -- backend construction ----------------------------------------------- #

    def _init_openai(self) -> None:
        from openai import OpenAI

        self._client = OpenAI(
            base_url=self._base_url(),
            api_key=self._api_key(),
            timeout=self.defaults.get("timeout", 120),
        )

    def _init_http_proxy(self) -> None:
        self._proxy_url = self._base_url().rstrip("/") + "/chat/completions"
        self._headers = {
            "Accept": "application/json",
            "Authorization": f"Bearer {self._api_key()}",
            "Content-Type": "application/json",
        }

    def _init_local(self, model_path: Optional[str]) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        path = (
            model_path
            or os.environ.get(self.spec.get("model_path_env", ""), "")
            or self.spec["model_path"]
        )
        self.model_id = path
        self._tokenizer = AutoTokenizer.from_pretrained(path)
        self._model = AutoModelForCausalLM.from_pretrained(
            path,
            torch_dtype=torch.float16,
            device_map="auto",
            attn_implementation="sdpa",
        )

    # -- the single entry point --------------------------------------------- #

    def chat(self, messages: List[Message], **overrides: Any) -> str:
        """Send messages and return the assistant's text.

        Transient failures are retried with exponential backoff up to
        defaults.max_retries. The last failure is re-raised, so a broken run
        stops instead of recording an empty response.
        """
        params = {**self.defaults, **overrides}
        attempts = int(params.get("max_retries", 5))
        last_error: Optional[Exception] = None

        for attempt in range(attempts):
            try:
                if self.backend == "openai_compatible":
                    return self._chat_openai(messages, params)
                if self.backend == "http_proxy":
                    return self._chat_http(messages, params)
                return self._chat_local(messages, params)
            except Exception as error:  # noqa: BLE001 - retried then re-raised
                last_error = error
                if attempt == attempts - 1:
                    break
                time.sleep(min(2 ** attempt, 30))

        raise RuntimeError(
            f"{self.alias} failed after {attempts} attempts: {last_error}"
        ) from last_error

    # Pipelines call the chatbot through chat_func; keep that name working.
    __call__ = chat

    def _chat_openai(self, messages: List[Message], params: Dict[str, Any]) -> str:
        kwargs: Dict[str, Any] = {"model": self.model_id, "messages": messages}
        # Reasoning models reject sampling parameters.
        if not self._is_reasoning_model():
            kwargs["temperature"] = params.get("temperature", 0.0)
            kwargs["max_tokens"] = params.get("max_tokens", 2048)
        response = self._client.chat.completions.create(**kwargs)
        return response.choices[0].message.content

    def _chat_http(self, messages: List[Message], params: Dict[str, Any]) -> str:
        import requests

        payload = json.dumps({"model": self.model_id, "messages": messages})
        response = requests.post(
            self._proxy_url,
            headers=self._headers,
            data=payload,
            timeout=params.get("timeout", 120),
        )
        response.raise_for_status()
        body = response.json()
        if "choices" not in body:
            raise RuntimeError(f"Unexpected response from {self.alias}: {body}")
        return body["choices"][0]["message"]["content"]

    def _chat_local(self, messages: List[Message], params: Dict[str, Any]) -> str:
        text = self._tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self._tokenizer([text], return_tensors="pt").to(self._model.device)
        generated = self._model.generate(
            inputs.input_ids, max_new_tokens=params.get("max_tokens", 2048)
        )
        trimmed = [
            out[len(src):] for src, out in zip(inputs.input_ids, generated)
        ]
        return self._tokenizer.batch_decode(trimmed, skip_special_tokens=True)[0]

    def _is_reasoning_model(self) -> bool:
        return any(
            self.model_id.startswith(prefix)
            for prefix in ("o1", "o3", "o4", "deepseek-reasoner")
        )

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"LLMClient(alias={self.alias!r}, backend={self.backend!r})"

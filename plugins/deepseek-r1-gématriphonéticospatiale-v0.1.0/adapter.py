"""DeepSeek harness plugin: EmbedBabel-Bench launcher.

Behaviour
---------
* User sends "go" (alone) -> the bench prompt is sent to the model.
* If the conversation already contains earlier messages (context pollution),
  the plugin does NOT run the bench and asks the user to open a new conversation,
  because earlier turns would bias the benchmark.

NOTE: the harness hook API of the DeepSeek fork is NOT verified here. `on_message`
is a framework-agnostic entry point: adapt the thin wrapper at the bottom to the
real hook signature of your fork. The optional `call_deepseek` uses the public
OpenAI-compatible endpoint and needs DEEPSEEK_API_KEY.
"""
from __future__ import annotations

import json
import os
import re
import urllib.request
from dataclasses import dataclass
from pathlib import Path

PLUGIN_DIR = Path(__file__).resolve().parent
MANIFEST = json.loads((PLUGIN_DIR / "manifest.json").read_text(encoding="utf-8"))
REPO_ROOT = PLUGIN_DIR.parents[1]

NEW_CONVERSATION_MSG = (
    "⚠️ Cette conversation contient déjà des messages précédents qui peuvent "
    "polluer le contexte du benchmark (biais, anchoring, résultats antérieurs).\n"
    "👉 Ouvre une **nouvelle conversation** puis envoie uniquement : go"
)


@dataclass
class Decision:
    action: str  # "run_bench" | "ask_new_conversation" | "ignore"
    reason: str
    payload: str | None = None


def is_trigger(message: str) -> bool:
    t = MANIFEST["trigger"]["keyword"]
    return re.fullmatch(rf"\s*{re.escape(t)}\s*[.!]?\s*", message, re.IGNORECASE) is not None


def prior_messages(history: list[dict]) -> list[dict]:
    """Messages that are not system prompts and not the current trigger."""
    return [m for m in history if m.get("role") in ("user", "assistant", "tool")]


def is_polluted(history: list[dict]) -> tuple[bool, str]:
    guard = MANIFEST["context_guard"]
    if not guard["enabled"]:
        return False, "guard disabled"
    prior = prior_messages(history)
    chars = sum(len(str(m.get("content", ""))) for m in prior)
    if len(prior) > guard["max_prior_messages"]:
        return True, f"{len(prior)} message(s) précédent(s)"
    if chars > guard["max_prior_chars"]:
        return True, f"{chars} caractères de contexte précédent"
    return False, "clean"


def load_bench_prompt() -> str:
    return (REPO_ROOT / MANIFEST["bench_prompt"]).read_text(encoding="utf-8")


def on_message(history: list[dict], message: str) -> Decision:
    """history: messages BEFORE `message` (OpenAI-style dicts)."""
    if not is_trigger(message):
        return Decision("ignore", "not the trigger")
    polluted, why = is_polluted(history)
    if polluted:
        return Decision("ask_new_conversation", why, NEW_CONVERSATION_MSG)
    return Decision("run_bench", "clean context", load_bench_prompt())


def call_deepseek(prompt: str, model: str = "deepseek-reasoner") -> str:
    key = os.environ["DEEPSEEK_API_KEY"]
    body = json.dumps({"model": model, "messages": [{"role": "user", "content": prompt}]}).encode()
    req = urllib.request.Request(
        "https://api.deepseek.com/chat/completions",
        data=body,
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"},
    )
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.load(r)["choices"][0]["message"]["content"]


def handle(history: list[dict], message: str, run_model=call_deepseek) -> str | None:
    """Wrapper to plug into the harness: returns text to display, or None to pass through."""
    d = on_message(history, message)
    if d.action == "ask_new_conversation":
        return d.payload
    if d.action == "run_bench":
        return run_model(d.payload)
    return None


if __name__ == "__main__":
    # Local smoke test (no network): python adapter.py
    assert on_message([], "go").action == "run_bench"
    assert on_message([{"role": "user", "content": "salut"}], "GO").action == "ask_new_conversation"
    assert on_message([], "bonjour").action == "ignore"
    print("ok")

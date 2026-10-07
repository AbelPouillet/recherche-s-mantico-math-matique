"""Garde anti-pollution de l'historique — implémentation **canonique**.

Règle : le bench ne part que si la conversation est vierge. Un tour antérieur (utilisateur,
assistant ou outil) pollue le contexte du benchmark — biais d'ancrage, résultats antérieurs,
raisonnement déjà engagé — et la mesure ne vaut plus rien. Le déclencheur est un message réduit au
mot-clé (`go` par défaut), seul, éventuellement suivi d'un point ou d'un point d'exclamation.

Pourquoi ce module existe
------------------------
La règle vivait dans `plugins/deepseek-r1-.../adapter.py`, un fichier Python que le harnais importait
par glob. Elle est ici, pure et testable, et **le plugin DSH réel la réimplémente en JavaScript**.
Les deux implémentations sont vérifiées par le **même jeu de cas**
(`tests/fixtures/guard_cases.json`) : `tests/test_bench_guard.py` côté Python,
`dsh/embedbabel-dsh/test/guard.test.mjs` côté Node. Si l'une dérive, l'autre le dit.

Les règles peuvent être surchargées par un harnais via `preferences.context_guard`, ce qui les fait
entrer dans le hash de version — c'est leur place naturelle.
"""
from __future__ import annotations

import re

#: Règles par défaut. Identiques à celles du plugin DSH.
DEFAULTS: dict = {
    "enabled": True,
    "keyword": "go",
    "max_prior_messages": 0,
    "max_prior_chars": 0,
    "on_polluted": "ask_new_conversation",
}

CONVERSATIONAL_ROLES = ("user", "assistant", "tool")


def rules_from(preferences: dict | None) -> dict:
    """Règles effectives : défauts, surchargés par `preferences['context_guard']` du harnais."""
    override = (preferences or {}).get("context_guard") or {}
    unknown = sorted(set(override) - set(DEFAULTS))
    if unknown:
        raise ValueError(f"clés de context_guard inconnues : {unknown} (connues : {sorted(DEFAULTS)})")
    return {**DEFAULTS, **override}


def is_trigger(message: str, keyword: str = "go") -> bool:
    """Le message est-il *exactement* le mot-clé (espaces, point final ou `!` tolérés) ?"""
    return re.fullmatch(rf"\s*{re.escape(keyword)}\s*[.!]?\s*", str(message), re.IGNORECASE) is not None


def prior_messages(history: list[dict]) -> list[dict]:
    """Messages de conversation antérieurs : ni les messages système, ni autre chose que du dialogue."""
    return [m for m in (history or []) if isinstance(m, dict) and m.get("role") in CONVERSATIONAL_ROLES]


def prior_chars(history: list[dict]) -> int:
    return sum(len(str(m.get("content", ""))) for m in prior_messages(history))


def pollution(history: list[dict], rules: dict | None = None) -> tuple[bool, str]:
    """`(pollué, raison)`. La raison est **stable et identique** dans les deux langages."""
    cfg = {**DEFAULTS, **(rules or {})}
    if not cfg["enabled"]:
        return False, "garde désactivée"
    prior = prior_messages(history)
    if len(prior) > cfg["max_prior_messages"]:
        return True, f"{len(prior)} message(s) précédent(s)"
    chars = prior_chars(history)
    if chars > cfg["max_prior_chars"]:
        return True, f"{chars} caractères de contexte précédent"
    return False, "contexte propre"


def decide(history: list[dict], message: str, rules: dict | None = None) -> dict:
    """Décision pour un message : `{action, reason}`.

    Actions : `run_bench` (lancer), `ask_new_conversation` (pollué : demander une conversation
    neuve), `ignore` (ce n'est pas le déclencheur — le harness poursuit normalement).
    """
    cfg = {**DEFAULTS, **(rules or {})}
    if not is_trigger(message, cfg["keyword"]):
        return {"action": "ignore", "reason": "pas le déclencheur"}
    polluted, reason = pollution(history, cfg)
    if polluted:
        return {"action": cfg["on_polluted"], "reason": reason}
    return {"action": "run_bench", "reason": reason}


NEW_CONVERSATION_MESSAGE = (
    "⚠️ Cette conversation contient déjà des messages précédents qui peuvent polluer le contexte du "
    "benchmark (biais d'ancrage, résultats antérieurs).\n"
    "👉 Ouvre une **nouvelle conversation** puis envoie uniquement : go"
)


def guard_history(history: list[dict], rules: dict | None = None) -> dict:
    """Résumé de garde pour le journal du harnais (utilisé à l'état PLANIFIER)."""
    cfg = {**DEFAULTS, **(rules or {})}
    d = decide(history, cfg["keyword"], cfg)
    return {"ok": d["action"] == "run_bench", "action": d["action"], "reason": d["reason"]}

"""Garde de contexte : paquets modulaires, limite déclarée, journal.

Un paquet obligatoire qui ne tient pas fait refuser tout le contexte. Un paquet
optionnel est tronqué proprement (à une fin de ligne) ou refusé s'il ne reste plus
de place ; ce qui reste hors contexte est consigné avec la manière de le retrouver.
La garde contre la pollution de l'historique réutilise celle du plugin DeepSeek
(testée par integrations/deepseek-harness-labsia/tests/test_context_guard.py).
"""
from __future__ import annotations

from pathlib import Path

from .budget import CHARS_PER_TOKEN, est_tokens


def load_packets(spec: list[dict], root: Path) -> list[dict]:
    return [{"id": p["id"], "path": p["path"], "required": bool(p.get("required")),
             "text": (root / p["path"]).read_text(encoding="utf-8")} for p in spec]


def _left_out(packet: dict, from_char: int) -> dict:
    return {"packet": packet["id"], "path": packet["path"], "from_char": from_char,
            "tokens_left_out": est_tokens(packet["text"][from_char:]),
            "how": f"lire {packet['path']} à partir du caractère {from_char}"}


def build_context(packets: list[dict], limit: int, reserve: int = 0) -> tuple[dict, str]:
    """Charge les paquets (obligatoires d'abord, puis dans l'ordre) sous `limit - reserve` tokens.

    `reserve` = tokens gardés pour la réponse du modèle. Retourne (contexte, texte chargé).
    Le contexte ne contient pas le texte : seulement ce qui est chargé, hors contexte, et les événements.
    """
    available = limit - reserve
    loaded, left_out, events, parts = [], [], [], []
    ok = True
    for p in sorted(packets, key=lambda p: not p["required"]):  # tri stable
        tokens = est_tokens(p["text"])
        if tokens <= available:
            loaded.append({"packet": p["id"], "tokens": tokens, "chars": len(p["text"]), "truncated": False})
            parts.append(p["text"])
            available -= tokens
            continue
        if p["required"]:
            ok = False
            events.append({"event": "packet_refused", "packet": p["id"], "tokens": tokens,
                           "available": max(available, 0), "reason": "paquet obligatoire hors limite"})
            left_out.append(_left_out(p, 0))
            continue
        kept = max(available, 0) * CHARS_PER_TOKEN
        cut = p["text"].rfind("\n", 0, kept) + 1 if kept else 0
        if cut <= 0:
            events.append({"event": "packet_refused", "packet": p["id"], "tokens": tokens,
                           "available": max(available, 0), "reason": "plus de place"})
            left_out.append(_left_out(p, 0))
            continue
        text = p["text"][:cut]
        kept_tokens = est_tokens(text)
        events.append({"event": "packet_truncated", "packet": p["id"], "tokens": tokens,
                       "kept_tokens": kept_tokens})
        loaded.append({"packet": p["id"], "tokens": kept_tokens, "chars": cut, "truncated": True})
        parts.append(text)
        left_out.append(_left_out(p, cut))
        available -= kept_tokens
    ctx = {"ok": ok, "limit": limit, "reserve": reserve, "loaded": loaded, "left_out": left_out,
           "tokens": sum(p["tokens"] for p in loaded), "events": events}
    return ctx, "\n\n".join(parts)


def guard_history(plugin, history: list[dict]) -> dict:
    """Pollution de l'historique : délègue au plugin installé (aucune logique dupliquée)."""
    if plugin is None:
        return {"ok": True, "plugin": None, "action": "no_plugin", "reason": "aucun plugin installé"}
    name, mod = plugin
    d = mod.on_message(history, "go")
    return {"ok": d.action == "run_bench", "plugin": name, "action": d.action, "reason": d.reason}

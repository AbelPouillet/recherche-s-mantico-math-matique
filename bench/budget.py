"""Estimation de tokens et compte rendu budgété (par étape, avec points de reprise).

Le compte rendu contient tout ce qu'il faut pour qu'un second adaptateur reprenne à
l'étape n sans autre information : graine, limite déclarée, tokens du contexte, et la
sortie de chaque étape déjà faite (vérifiée par son hash).
"""
from __future__ import annotations

import math

from .trace import canonical_json, digest

CHARS_PER_TOKEN = 4  # [PLAUSIBLE] estimateur déterministe, pas un vrai tokenizer
STEP_NAMES = ("formalisation", "analyse", "plan_experimental")


def est_tokens(text: str) -> int:
    return math.ceil(len(text) / CHARS_PER_TOKEN)


def gap_pct(consumed: int, estimated: int) -> float:
    """Écart mesuré en % : positif = le modèle a consommé plus que prévu."""
    return round((consumed - estimated) / estimated * 100.0, 2)


def step_record(index: int, name: str, estimated: int, output: dict, context: dict) -> dict:
    consumed = est_tokens(canonical_json(output))
    return {
        "step": index, "name": name,
        "estimated_tokens": estimated, "consumed_tokens": consumed,
        "gap_pct": gap_pct(consumed, estimated),
        "loaded": [p["packet"] for p in context["loaded"]],
        "left_out": context["left_out"],
        "output": output,
        "checkpoint": {"next_step": index + 1, "output_sha256": digest(output)},
    }


def build_report(model: str, declared_limit: int, seed: int, context: dict, steps: list[dict]) -> dict:
    estimated = sum(s["estimated_tokens"] for s in steps)
    consumed = sum(s["consumed_tokens"] for s in steps)
    return {
        "model": model, "seed": seed, "declared_context_limit": declared_limit,
        "context": {"tokens": context["tokens"], "loaded": context["loaded"],
                    "left_out": context["left_out"]},
        "steps": steps,
        "estimated_total": estimated, "consumed_total": consumed,
        "gap_pct": gap_pct(consumed, estimated),
        "context_used": context["tokens"] + consumed,
    }


def verify_checkpoints(report: dict) -> list[str]:
    """Erreurs si la sortie d'une étape ne correspond plus au hash de son point de reprise."""
    return [f"étape {s['step']} : hash de reprise incohérent" for s in report["steps"]
            if digest(s["output"]) != s["checkpoint"]["output_sha256"]]


def prior_outputs(report: dict, n: int) -> dict:
    """Sorties fusionnées des étapes strictement antérieures à n (n commence à 1)."""
    prior: dict = {}
    for s in report["steps"]:
        if s["step"] < n:
            prior.update(s["output"])
    return prior


def resume_output(adapter, report: dict, n: int) -> dict:
    """Reprend à l'étape n depuis le seul compte rendu et retourne la sortie fusionnée complète."""
    info = {"seed": report["seed"], "model": report["model"],
            "prompt_tokens": report["context"]["tokens"],
            "context_limit": report["declared_context_limit"]}
    prior = prior_outputs(report, n)
    for k in range(n, len(STEP_NAMES) + 1):
        prior.update(adapter.step(k, prior, info))
    return prior

"""Estimation de tokens et compte rendu budgété (par étape, avec points de reprise).

Le compte rendu contient tout ce qu'il faut pour qu'un second adaptateur reprenne à
l'étape n sans autre information : graine, limite déclarée, tokens du contexte, et la
sortie de chaque étape déjà faite (vérifiée par son hash).
"""
from __future__ import annotations

import math
import statistics

from .trace import canonical_json, digest

CHARS_PER_TOKEN = 4  # [PLAUSIBLE] estimateur déterministe, pas un vrai tokenizer
STEP_NAMES = ("formalisation", "analyse", "plan_experimental")


def est_tokens(text: str) -> int:
    return math.ceil(len(text) / CHARS_PER_TOKEN)


def gap_pct(consumed: int, estimated: int) -> float:
    """Écart mesuré en % : positif = le modèle a consommé plus que prévu."""
    return round((consumed - estimated) / estimated * 100.0, 2)


def step_record(index: int, name: str, estimated: int, output: dict, context: dict,
                usage: dict | None = None) -> dict:
    """Compte rendu d'une étape.

    `usage` = compteurs du serveur pour cette étape (adaptateurs réels). Il est **conservé** :
    sans lui, un rapport peut publier un débit sans dire s'il a été mesuré ou rejoué depuis le cache.
    """
    consumed = est_tokens(canonical_json(output))
    rec = {
        "step": index, "name": name,
        "estimated_tokens": estimated, "consumed_tokens": consumed,
        "gap_pct": gap_pct(consumed, estimated),
        "loaded": [p["packet"] for p in context["loaded"]],
        "left_out": context["left_out"],
        "output": output,
        "checkpoint": {"next_step": index + 1, "output_sha256": digest(output)},
    }
    if usage:
        rec["usage"] = usage
    return rec


def telemetry_summary(steps: list[dict], context: dict) -> dict:
    """Agrège les compteurs réels du serveur pour un modèle.

    Séparation stricte mesure / rejeu : une étape servie par le cache republie les compteurs de
    l'appel d'origine. Elle est comptée à part et **exclue des débits**, pour qu'un run rejoué ne
    puisse pas passer pour une campagne de mesure.

    `estimation_*` publie l'écart entre l'estimateur `chars/4` du harnais et le tokenizer réel du
    serveur — l'écart qui rendait la garde de contexte optimiste.
    """
    usages = [s.get("usage") for s in steps if s.get("usage")]
    measured = [u for u in usages if not u.get("cached")]
    replayed = [u for u in usages if u.get("cached")]
    rates = [u["tokens_per_s"] for u in measured if isinstance(u.get("tokens_per_s"), (int, float))]
    real_prompt = sum(u["prompt_tokens_real"] for u in usages
                      if isinstance(u.get("prompt_tokens_real"), int))
    real_completion = sum(u["completion_tokens_real"] for u in usages
                          if isinstance(u.get("completion_tokens_real"), int))
    estimated_completion = sum(s["consumed_tokens"] for s in steps)
    out = {
        "steps_with_counters": len(usages),
        "steps_measured": len(measured),
        "steps_replayed_from_cache": len(replayed),
        "measured": not replayed and bool(measured),
        "prompt_tokens_real": real_prompt,
        "completion_tokens_real": real_completion,
        "completion_tokens_chars4": estimated_completion,
        "tokens_per_s_median": round(statistics.median(rates), 2) if rates else None,
        "tokens_per_s_by_step": rates,
        "done_reasons": sorted({u.get("done_reason") for u in usages if u.get("done_reason")}),
    }
    if real_completion and estimated_completion:
        out["estimation_completion_ratio"] = round(real_completion / estimated_completion, 3)
    if context.get("tokens_real"):
        out["context_tokens_estimated"] = context["tokens"]
        out["context_tokens_real"] = context["tokens_real"]
        if context["tokens"]:
            out["estimation_context_ratio"] = round(context["tokens_real"] / context["tokens"], 3)
    if isinstance(context.get("tokens_real_source"), str):
        out["context_tokens_real_source"] = context["tokens_real_source"]
    return out


def build_report(model: str, declared_limit: int, seed: int, context: dict, steps: list[dict],
                 budget: int | None = None) -> dict:
    estimated = sum(s["estimated_tokens"] for s in steps)
    consumed = sum(s["consumed_tokens"] for s in steps)
    report = {
        "model": model, "seed": seed, "declared_context_limit": declared_limit,
        "context": {"tokens": context["tokens"], "loaded": context["loaded"],
                    "left_out": context["left_out"]},
        "steps": steps,
        "estimated_total": estimated, "consumed_total": consumed,
        "gap_pct": gap_pct(consumed, estimated),
        "context_used": context["tokens"] + consumed,
    }
    if budget is not None:
        # le rapport porte son propre budget : un lecteur externe n'a pas à le retrouver ailleurs
        report["budget"] = budget
    return report


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


def resume_output(adapter, report: dict, n: int, extra: dict | None = None) -> dict:
    """Reprend à l'étape n depuis le seul compte rendu et retourne la sortie fusionnée complète.
    `extra` : informations d'environnement des adaptateurs réels (texte du contexte, dossiers)."""
    info = {**(extra or {}), "seed": report["seed"], "model": report["model"],
            "prompt_tokens": report["context"]["tokens"],
            "context_limit": report["declared_context_limit"]}
    prior = prior_outputs(report, n)
    for k in range(n, len(STEP_NAMES) + 1):
        prior.update(adapter.step(k, prior, info))
    return prior

"""Score **externe** des tours — et refus explicite de tout score auto-déclaré.

Le harnais H1 a produit un verdict de tête calculé à partir de ce que les modèles disaient d'eux-mêmes.
C'est le défaut que ce module existe pour empêcher : ici, un score ne peut venir que d'un **résultat
observé** (tâche réussie, tests verts, tokens, temps mural). Les champs d'auto-évaluation sont
journalisés mais **ne peuvent pas** entrer dans le score : `score_turn` lève une erreur si on essaie.

Métriques, toutes externes :

| Métrique | Sens | Sens de lecture |
|---|---|---|
| `success_rate` | tours réussis / tours notés | plus haut mieux |
| `tests_rate` | tests verts / tests lancés, quand le tour en rapporte | plus haut mieux |
| `tokens_per_success` | tokens consommés par tour réussi | plus bas mieux |
| `wall_per_success_s` | secondes par tour réussi | plus bas mieux |
| `retry_rate` | reprises / tours | plus bas mieux |

Un tour sans issue n'est **pas** noté : il est compté comme manquant, jamais comme réussite.
"""
from __future__ import annotations

import statistics

from .records import SELF_ASSESSMENT_FIELDS, TurnRecord

METRICS = ("success_rate", "tests_rate", "tokens_per_success", "wall_per_success_s", "retry_rate")


class SelfAssessmentRefused(ValueError):
    """Levée si l'on tente de faire entrer une auto-évaluation de modèle dans un score."""


def _num(value):
    return value if isinstance(value, (int, float)) and not isinstance(value, bool) else None


def score_turn(record: TurnRecord) -> dict:
    """Métriques d'un tour, uniquement à partir de son issue. `scored: False` sans issue."""
    if not record.scored:
        return {"turn_id": record.turn_id, "scored": False, "reason": "aucune issue fournie"}
    outcome = dict(record.outcome or {})
    leaked = sorted(set(outcome) & set(SELF_ASSESSMENT_FIELDS))
    if leaked:
        raise SelfAssessmentRefused(
            f"issue du tour {record.turn_id} : champ(s) d'auto-évaluation {leaked} — un modèle ne peut "
            "pas être sa propre vérité terrain ; retire-les du fichier d'issues")
    success = outcome.get("success")
    tokens = _num(outcome.get("tokens")) if _num(outcome.get("tokens")) is not None else _num(outcome.get("completion_tokens"))
    wall = _num(outcome.get("wall_s"))
    tests_passed, tests_total = _num(outcome.get("tests_passed")), _num(outcome.get("tests_total"))
    return {
        "turn_id": record.turn_id,
        "scored": True,
        "success": bool(success),
        "tokens": tokens,
        "wall_s": wall,
        "retries": _num(outcome.get("retries")),
        "tool_calls": _num(outcome.get("tool_calls")),
        "tests_rate": (tests_passed / tests_total) if tests_total else None,
        "model": outcome.get("model"),
        "provider": outcome.get("provider"),
        "task": outcome.get("task"),
        "split": outcome.get("split"),
    }


def aggregate(records: list[TurnRecord]) -> dict:
    """Agrégat externe d'un jeu de tours. Les tours non notés sont comptés, pas devinés."""
    turns = [score_turn(r) for r in records]
    scored = [t for t in turns if t["scored"]]
    if not scored:
        return {"turns": len(turns), "scored": 0, "unscored": len(turns), "metrics": None,
                "note": "aucune issue fournie : rien à scorer (et surtout pas l'avis des modèles)"}

    successes = [t for t in scored if t["success"]]
    success_rate = len(successes) / len(scored)

    def per_success(key):
        values = [t[key] for t in successes if t[key] is not None]
        if not values or not successes:
            return None
        return round(sum(values) / len(successes), 3)

    tests = [t["tests_rate"] for t in scored if t["tests_rate"] is not None]
    retries = [t["retries"] for t in scored if t["retries"] is not None]
    metrics = {
        "success_rate": round(success_rate, 4),
        "tests_rate": round(statistics.fmean(tests), 4) if tests else None,
        "tokens_per_success": per_success("tokens"),
        "wall_per_success_s": per_success("wall_s"),
        "retry_rate": round(sum(retries) / len(scored), 4) if retries else None,
    }
    return {"turns": len(turns), "scored": len(scored), "unscored": len(turns) - len(scored),
            "successes": len(successes), "metrics": metrics,
            "models": sorted({t["model"] for t in scored if t["model"]}),
            "tasks": sorted({t["task"] for t in scored if t["task"]}),
            "note": "métriques externes uniquement ; les tours sans issue ne comptent pas comme réussites"}


def compare(baseline: dict, candidate: dict, margin_pct: float = 5.0) -> dict:
    """Compare deux agrégats sur la métrique de tête (`success_rate`), puis sur le coût.

    Règle de décision, **pré-déclarée** : le candidat est retenu s'il fait mieux d'au moins
    `margin_pct` en taux de réussite ; à égalité (dans la marge), il est retenu s'il coûte
    strictement moins de tokens par réussite ; sinon il est rejeté, avec le chiffre qui le rejette.
    """
    base_m, cand_m = baseline.get("metrics"), candidate.get("metrics")
    if not base_m or not cand_m:
        return {"decision": "indetermine", "reason": "un des deux agrégats n'a pas de métriques"}
    base_success, cand_success = base_m["success_rate"], cand_m["success_rate"]
    delta_pct = round((cand_success - base_success) / base_success * 100, 2) if base_success else None
    detail = {"baseline_success_rate": base_success, "candidate_success_rate": cand_success,
              "delta_pct": delta_pct, "margin_pct": margin_pct,
              "baseline_tokens_per_success": base_m.get("tokens_per_success"),
              "candidate_tokens_per_success": cand_m.get("tokens_per_success")}
    if delta_pct is not None and delta_pct >= margin_pct:
        return {"decision": "promouvoir", "reason": f"+{delta_pct}% de réussite (marge {margin_pct}%)",
                **detail}
    base_cost, cand_cost = base_m.get("tokens_per_success"), cand_m.get("tokens_per_success")
    if (delta_pct is None or delta_pct > -margin_pct) and base_cost and cand_cost and cand_cost < base_cost:
        saved = round((base_cost - cand_cost) / base_cost * 100, 2)
        return {"decision": "promouvoir",
                "reason": f"réussite équivalente (Δ {delta_pct}%) et {saved}% de tokens en moins", **detail}
    return {"decision": "rejeter",
            "reason": f"Δ réussite {delta_pct}% pour une marge exigée de {margin_pct}%, "
                      f"et pas de gain de coût",
            **detail}

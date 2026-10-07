"""Enregistrements de la boucle : journal du plugin DSH + fichier d'issues, joints par `turn_id`.

Deux sources, volontairement séparées
-------------------------------------
1. **Décisions** (`EMBEDBABEL_LOG`, produit par `dsh/embedbabel-dsh`) : ce que le plugin a décidé
   (déclencheur reconnu, garde passée ou refusée, prompt injecté) et un résumé du tour. Format
   **vérifié** : c'est notre propre code qui l'écrit.
2. **Issues** (fichier JSONL fourni par l'appelant) : ce qui s'est réellement passé — tâche réussie ou
   non, tests verts, tokens, temps mural, reprises. Format **défini ici**, parce que je n'ai pas
   vérifié le format des sessions DSH sur disque et que je refuse d'en inventer un.

Sans issues, il n'y a pas de score : la boucle le dit au lieu de noter le modèle sur son propre avis.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

from ..trace import digest

#: Champs d'issue reconnus. Tout autre champ est conservé mais signalé comme non exploité.
OUTCOME_FIELDS = ("turn_id", "success", "tests_passed", "tests_total", "tokens",
                  "completion_tokens", "wall_s", "retries", "tool_calls", "cost_units",
                  "model", "provider", "task", "split")

#: Champs d'auto-évaluation : ils peuvent être journalisés, jamais utilisés comme vérité terrain.
SELF_ASSESSMENT_FIELDS = ("self_score", "confidence", "quality", "satisfaction", "rating",
                          "sif", "sdm", "own_score", "self_assessment")


@dataclass
class TurnRecord:
    """Un tour observé, joint à son issue quand elle existe."""

    turn_id: str
    decisions: list[dict] = field(default_factory=list)
    outcome: dict | None = None
    self_assessment: dict = field(default_factory=dict)
    ignored_fields: list[str] = field(default_factory=list)

    @property
    def scored(self) -> bool:
        return self.outcome is not None

    def to_dict(self) -> dict:
        return asdict(self)


def _read_jsonl(path: Path) -> list[dict]:
    if not Path(path).exists():
        return []
    records = []
    for line in Path(path).read_text(encoding="utf-8-sig").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            records.append(obj)
    return records


def read_decisions(path: Path) -> list[dict]:
    """Journal du plugin DSH. Une entrée sans `turn_id` est rattachée à un tour synthétique."""
    return _read_jsonl(path)


def read_outcomes(path: Path) -> dict[str, dict]:
    """Fichier d'issues indexé par `turn_id`. Les champs inconnus sont conservés."""
    out: dict[str, dict] = {}
    for index, record in enumerate(_read_jsonl(path)):
        turn_id = str(record.get("turn_id") or f"ligne-{index}")
        known = {k: v for k, v in record.items() if k in OUTCOME_FIELDS}
        known["turn_id"] = turn_id
        known["_extra"] = sorted(set(record) - set(OUTCOME_FIELDS))
        out[turn_id] = known
    return out


def join(decisions: list[dict], outcomes: dict[str, dict] | None = None) -> list[TurnRecord]:
    """Joint décisions et issues. Les champs d'auto-évaluation sont isolés, pas supprimés.

    Le rattachement se fait par `turn_id` ; le plugin DSH en pose un par tour.
    """
    outcomes = outcomes or {}
    by_turn: dict[str, TurnRecord] = {}
    for index, decision in enumerate(decisions):
        turn_id = str(decision.get("turn_id") or f"tour-{index}")
        record = by_turn.setdefault(turn_id, TurnRecord(turn_id=turn_id))
        record.decisions.append(decision)
        for key in SELF_ASSESSMENT_FIELDS:
            if key in decision:
                record.self_assessment[key] = decision[key]
    for turn_id, outcome in outcomes.items():
        record = by_turn.setdefault(turn_id, TurnRecord(turn_id=turn_id))
        record.outcome = outcome
        if outcome.get("_extra"):
            record.ignored_fields = sorted(set(record.ignored_fields) | set(outcome["_extra"]))
    return [by_turn[k] for k in sorted(by_turn)]


def dataset_hash(records: list[TurnRecord]) -> str:
    """Empreinte du jeu de données exploité : les tours **notés**, dans l'ordre, avec leurs issues."""
    payload = [{"turn_id": r.turn_id, "outcome": {k: v for k, v in (r.outcome or {}).items()
                                                  if k != "_extra"},
                "events": sorted({d.get("event") for d in r.decisions if d.get("event")})}
               for r in records if r.scored]
    return digest(payload)


def split(records: list[TurnRecord], holdout_ratio: float = 0.34, seed: int = 0) -> tuple[list, list]:
    """Séparation entraînement / tenu à l'écart, déterministe.

    Une amélioration jugée sur les mêmes tours qui l'ont inspirée n'est pas une amélioration. La
    séparation est déterministe (hash du `turn_id`) pour qu'une campagne soit rejouable à l'identique.
    """
    scored = [r for r in records if r.scored]
    scored.sort(key=lambda r: hashlib.sha256(f"{seed}:{r.turn_id}".encode()).hexdigest())
    cut = max(1, int(len(scored) * (1 - holdout_ratio))) if len(scored) > 1 else len(scored)
    return scored[:cut], scored[cut:]


def load(decisions_path: Path, outcomes_path: Path | None = None) -> list[TurnRecord]:
    return join(read_decisions(decisions_path),
                read_outcomes(outcomes_path) if outcomes_path else None)

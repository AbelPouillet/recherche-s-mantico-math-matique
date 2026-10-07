"""Empreinte dorée du harnais hors ligne — non-régression du harnais lui-même.

Un harnais de bench ne vaut que s'il est stable : si la même graine produit une trace différente
après une modification du code, deux campagnes ne sont plus comparables et le registre versionné
perd son sens. Ce module fige l'empreinte d'un run hors ligne complet (6 adaptateurs factices,
harnais courant) et échoue dès qu'elle bouge.

Régénérer **uniquement** après un changement assumé et documenté au
`bench/harnesses/CHANGELOG.md` :

    EMBEDBABEL_UPDATE_GOLDEN=1 python -m pytest tests/test_bench_golden.py

Le second test vérifie l'herméticité : un run ne doit modifier aucun fichier suivi par git sous
`bench/registry/`.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from bench.run import load_models, run

ROOT = Path(__file__).resolve().parent.parent
GOLDEN = ROOT / "tests" / "golden" / "offline-mock.seed0.json"
MODELS = ROOT / "bench" / "models" / "offline-mock.json"
REGISTRY = ROOT / "bench" / "registry"
SEED, BUDGET, TASK = 0, 3000, "v2-complet"


def observe(out_dir: Path) -> dict:
    """Empreinte stable d'un run : version, graine, trace, verdict, drapeaux, reprises."""
    harness, entries = load_models(MODELS)
    report = run(harness, entries, TASK, BUDGET, SEED, out_dir)
    return {
        "harness": report["harness"], "task": report["task"], "seed": report["seed"],
        "budget": report["budget"], "trace_sha256": report["trace_sha256"],
        "verdict": report["collective"]["verdict"],
        "kept": report["collective"]["kept"],
        "flags": {m: a["flags"] for m, a in sorted(report["assessments"].items())},
        "resumes": dict(sorted(report["resumes"].items())),
    }


def test_offline_run_matches_the_golden_fingerprint(tmp_path):
    observed = observe(tmp_path / "run")
    if os.environ.get("EMBEDBABEL_UPDATE_GOLDEN"):
        GOLDEN.parent.mkdir(parents=True, exist_ok=True)
        GOLDEN.write_text(json.dumps(observed, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
                          encoding="utf-8")
        pytest.skip("empreinte dorée régénérée")
    expected = json.loads(GOLDEN.read_text(encoding="utf-8"))
    differing = {k: (expected.get(k), observed.get(k)) for k in set(expected) | set(observed)
                 if expected.get(k) != observed.get(k)}
    assert not differing, (
        "l'empreinte du run hors ligne a changé (attendu, observé) :\n"
        + json.dumps(differing, indent=2, ensure_ascii=False, sort_keys=True)
        + f"\nSi le changement est voulu, documente-le dans bench/harnesses/CHANGELOG.md puis "
          f"régénère avec EMBEDBABEL_UPDATE_GOLDEN=1 ({GOLDEN.relative_to(ROOT)}).")


def test_a_run_does_not_modify_any_tracked_registry_file(tmp_path):
    """Herméticité vérifiée sur les vrais fichiers suivis : c'est ce que la CI exige."""
    assert REGISTRY.is_dir()
    before = {p.name: p.read_bytes() for p in sorted(REGISTRY.glob("*.json"))}
    assert before, "aucun harnais inscrit : lance d'abord bench.run --register"
    observe(tmp_path / "run")
    after = {p.name: p.read_bytes() for p in sorted(REGISTRY.glob("*.json"))}
    assert after == before, (
        "un run a modifié le registre versionné : "
        f"{sorted(n for n in before if before[n] != after.get(n))}"
    )

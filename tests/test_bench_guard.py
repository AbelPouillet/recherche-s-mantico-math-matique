"""Garde anti-pollution : jeu de cas **partagé** avec le plugin DSH en JavaScript.

`tests/fixtures/guard_cases.json` est lu par ce module (Python) et par
`dsh/embedbabel-dsh/test/guard.test.mjs` (Node). Deux implémentations, un seul jeu de cas, et des
raisons identiques mot pour mot : si l'une dérive, l'autre le dit. C'est ce qui remplace l'ancien
`integrations/deepseek-harness-labsia/tests/test_context_guard.py`, qui testait un fichier Python
n'ayant jamais été un plugin DSH.

Le dernier test exécute réellement le test Node — donc `python -m pytest` couvre le plugin DSH, et
`scripts/ci.py` le couvre aussi sans étape supplémentaire.
"""
from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

from bench import guard, registry
from bench.run import run

ROOT = Path(__file__).resolve().parent.parent
FIXTURE = Path(__file__).resolve().parent / "fixtures" / "guard_cases.json"
JS_TEST = ROOT / "dsh" / "embedbabel-dsh" / "test" / "guard.test.mjs"
CASES = json.loads(FIXTURE.read_text(encoding="utf-8"))
DEF = json.loads((ROOT / "bench/harnesses/embedbabel-bench/1.0.0/harness.def.json")
                 .read_text(encoding="utf-8"))


@pytest.mark.parametrize("case", CASES, ids=[c["name"] for c in CASES])
def test_decision_matches_the_shared_cases(case):
    decision = guard.decide(case["history"], case["message"], case.get("rules"))
    assert decision["action"] == case["action"], f"action attendue {case['action']}"
    assert decision["reason"] == case["reason"], f"raison attendue {case['reason']!r}"


def test_the_shared_fixture_covers_every_action_and_every_rule():
    actions = {c["action"] for c in CASES}
    assert actions == {"run_bench", "ask_new_conversation", "ignore"}
    keys = {k for c in CASES for k in (c.get("rules") or {})}
    assert {"keyword", "max_prior_messages", "max_prior_chars", "enabled"} <= keys


def test_rules_from_merges_defaults_and_refuses_unknown_keys():
    assert guard.rules_from(None) == guard.DEFAULTS
    assert guard.rules_from({}) == guard.DEFAULTS
    assert guard.rules_from({"context_guard": {"keyword": "bench"}})["keyword"] == "bench"
    assert guard.rules_from({"context_guard": {"keyword": "bench"}})["max_prior_chars"] == 0
    with pytest.raises(ValueError, match="inconnues"):
        guard.rules_from({"context_guard": {"seuil_de_bruit": 3}})
    assert guard.rules_from(DEF["preferences"]) == guard.DEFAULTS, "le harnais 1.0.0 utilise les défauts"


def test_guard_history_reports_the_action_and_the_reason():
    assert guard.guard_history([]) == {"ok": True, "action": "run_bench", "reason": "contexte propre"}
    polluted = guard.guard_history([{"role": "user", "content": "x"}])
    assert polluted["ok"] is False and polluted["action"] == "ask_new_conversation"


def test_the_harness_self_checks_the_guard_and_publishes_the_verdict(tmp_path, monkeypatch):
    """`plugin_guard_ok` doit rester vrai : c'est l'auto-vérification de la garde à PLANIFIER."""
    d = tmp_path / "registry"
    registry.register(DEF, registry_dir=d)
    monkeypatch.setattr(registry, "REGISTRY_DIR", d)
    report = run(registry.harness_id(DEF), [{"name": "honnete", "adapter": "honnete",
                                             "context_limit": 12000}],
                 "v2-minimal", 3000, 0, tmp_path / "o")
    assert report["plugin_guard_ok"] is True
    assert report["context_guard"] == guard.DEFAULTS
    checked = [e for e in report["journal"] if e.get("event") == "history_checked"]
    assert checked and checked[0]["action"] == "run_bench" and checked[0]["reason"] == "contexte propre"


def test_the_dsh_plugin_agrees_with_the_python_implementation():
    """Le plugin DSH est en JavaScript : la seule façon de le vérifier est de l'exécuter."""
    node = shutil.which("node")
    if node is None:
        pytest.skip("node absent : le plugin DSH ne peut pas être exécuté ici")
    proc = subprocess.run([node, "--test", str(JS_TEST)], capture_output=True, text=True,
                          encoding="utf-8", errors="replace", cwd=ROOT, timeout=120)
    assert proc.returncode == 0, f"tests du plugin DSH en échec :\n{proc.stdout}\n{proc.stderr}"
    assert "# fail 0" in proc.stdout and "# pass 14" in proc.stdout


def test_the_plugin_manifest_points_at_an_existing_entry_point():
    pkg = json.loads((ROOT / "dsh" / "embedbabel-dsh" / "package.json").read_text(encoding="utf-8"))
    assert pkg["dsh"]["manifestVersion"] == 1
    assert (ROOT / "dsh" / "embedbabel-dsh" / pkg["dsh"]["bundle"]["patch"]).is_file()
    assert (ROOT / "dsh" / "embedbabel-dsh" / pkg["main"]).is_file()
    patch = (ROOT / "dsh" / "embedbabel-dsh" / "cordis.patch.yml").read_text(encoding="utf-8")
    assert "embedbabel-dsh" in patch

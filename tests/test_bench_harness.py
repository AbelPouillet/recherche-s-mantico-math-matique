"""Tests du harnais de bench : sans réseau, sans horloge, graine fixe."""
import json
import shutil
from pathlib import Path

import pytest

from bench import registry
from bench.adapters import ADAPTERS, validate_response
from bench.run import STATES, run
from bench.trace import TraceBuilder, validate_tree, walk

ROOT = Path(__file__).resolve().parent.parent
DEF = json.loads((ROOT / "bench/definitions/embedbabel-bench-0.1.0.def.json").read_text(encoding="utf-8"))
MODELS = ["honnete", "optimiste", "bavard", "casse"]


@pytest.fixture
def reg(tmp_path, monkeypatch):
    d = tmp_path / "registry"
    registry.register(DEF, registry_dir=d)
    monkeypatch.setattr(registry, "REGISTRY_DIR", d)
    return registry.harness_id(DEF)


def test_trace_tree_ids_and_validation():
    tb = TraceBuilder("t")
    with tb.step("a"):
        with tb.step("a1"):
            tb.plugin_call("p", "go")
        tb.conclusion("c")
    tree = tb.to_dict()
    assert validate_tree(tree) == []
    assert [n["id"] for n in walk(tree)] == ["0", "1", "1.1", "1.1.1", "1.2"]
    tree["children"][0]["children"][0]["id"] = "9"
    assert validate_tree(tree)


def test_invalid_json_is_rejected():
    parsed, errors = validate_response(ADAPTERS["casse"]({"budget": 600}, 0))
    assert parsed is None and "JSON invalide" in errors[0]
    parsed, errors = validate_response('{"model": "x"}')
    assert parsed is None and errors


def test_same_seed_same_output(tmp_path, reg):
    a = run(reg, MODELS, 600, 7, tmp_path / "a")
    b = run(reg, MODELS, 600, 7, tmp_path / "b")
    assert a == b
    assert (tmp_path / "a/trace.json").read_bytes() == (tmp_path / "b/trace.json").read_bytes()


def test_verbose_and_optimistic_detected(tmp_path, reg):
    r = run(reg, MODELS, 600, 0, tmp_path / "o")
    assert "bavard" in r["assessments"]["bavard"]["flags"]
    opt = r["assessments"]["optimiste"]
    assert "ecart_budget" in opt["flags"] and opt["budget_gap_pct"] > 10
    assert r["assessments"]["casse"]["valid"] is False
    assert r["collective"]["kept"] == ["honnete"]
    assert set(r["collective"]["excluded"]) == {"optimiste", "bavard", "casse"}
    assert r["plugin_guard_ok"] is True


def test_resume_gives_identical_report(tmp_path, reg):
    full = run(reg, MODELS, 600, 3, tmp_path / "full")
    assert run(reg, MODELS, 600, 3, tmp_path / "part", stop_after="EXECUTER") is None
    resumed = run(reg, MODELS, 600, 3, tmp_path / "part", resume=True)
    assert resumed == full
    assert (tmp_path / "part/trace.json").read_bytes() == (tmp_path / "full/trace.json").read_bytes()
    assert STATES[-1] == "RAPPORT"


def test_resume_on_finished_run_returns_same_report(tmp_path, reg):
    out = tmp_path / "done"
    first = run(reg, MODELS, 600, 5, out)
    trace_before = (out / "trace.json").read_bytes()
    assert run(reg, MODELS, 600, 5, out, resume=True) == first
    assert (out / "trace.json").read_bytes() == trace_before


def test_registry_refuses_silent_overwrite(tmp_path):
    d = tmp_path / "registry"
    registry.register(DEF, registry_dir=d)
    changed = {**DEF, "preferences": {**DEF["preferences"], "h0_threshold": 0.5}}
    with pytest.raises(ValueError, match="version"):
        registry.register(changed, registry_dir=d)
    registry.register({**changed, "version": "0.2.0"}, registry_dir=d)
    assert registry.list_harnesses(d) == ["embedbabel-bench-0.1.0", "embedbabel-bench-0.2.0"]


def test_runs_recorded_once(tmp_path, reg):
    run(reg, MODELS, 600, 1, tmp_path / "x")
    run(reg, MODELS, 600, 1, tmp_path / "y")
    assert len(registry.load(reg)["runs"]) == 1

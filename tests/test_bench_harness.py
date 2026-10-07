"""Tests du harnais de bench : sans réseau, sans horloge, graine fixe."""
import json
from pathlib import Path

import pytest

from bench import consensus, ledger, registry
from bench.run import STATES, load_models, main, run
from bench.trace import TraceBuilder, validate_tree, walk

ROOT = Path(__file__).resolve().parent.parent
DEF = json.loads((ROOT / "bench/harnesses/embedbabel-bench/0.2.0/harness.def.json").read_text(encoding="utf-8"))
KINDS = ["honnete", "optimiste", "bavard", "casse_champ", "casse_tag", "casse_texte"]
BUDGET = 3000


def entries(limit=12000, kinds=KINDS):
    return [{"name": k, "adapter": k, "context_limit": limit} for k in kinds]


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


def test_same_seed_same_output_byte_for_byte(tmp_path, reg):
    a = run(reg, entries(), "v2-complet", BUDGET, 7, tmp_path / "a")
    b = run(reg, entries(), "v2-complet", BUDGET, 7, tmp_path / "b")
    assert a == b
    for name in ("trace.json", "report.json", "journal.jsonl"):
        assert (tmp_path / "a" / name).read_bytes() == (tmp_path / "b" / name).read_bytes()


def test_models_are_distinguished(tmp_path, reg):
    r = run(reg, entries(), "v2-complet", BUDGET, 0, tmp_path / "o")
    a = r["assessments"]
    assert a["honnete"]["flags"] == [] and a["honnete"]["included"] and a["honnete"]["gap_pct"] == 0.0
    assert "ecart_budget" in a["optimiste"]["flags"] and a["optimiste"]["gap_pct"] >= 190
    assert "bavard" not in a["optimiste"]["flags"]
    assert "bavard" in a["bavard"]["flags"]
    assert a["bavard"]["context_used"] > r["declared_limits"]["bavard"]
    assert any("champ manquant : formalization" in e for e in a["casse_champ"]["errors"])
    assert any("tag hors vocabulaire" in e for e in a["casse_tag"]["errors"])
    assert any("hors JSON" in e for e in a["casse_texte"]["errors"])
    assert all("invalide" in a[m]["flags"] for m in ("casse_champ", "casse_tag", "casse_texte"))
    assert r["collective"]["kept"] == ["honnete"]
    assert set(r["collective"]["excluded"]) == set(KINDS) - {"honnete"}
    assert r["plugin_guard_ok"] is True


def test_verbose_is_journaled_and_invalid_output_has_its_reason(tmp_path, reg):
    r = run(reg, entries(), "v2-complet", BUDGET, 0, tmp_path / "j")
    events = {(e["model"], e["event"]) for e in r["journal"]}
    assert ("bavard", "context_overflow") in events
    assert ("honnete", "context_overflow") not in events
    rejected = [e for e in r["journal"] if e["event"] == "schema_rejected" and e["model"] == "casse_tag"]
    assert rejected and "tag hors vocabulaire" in rejected[0]["reason"]
    assert [e["seq"] for e in r["journal"]] == list(range(len(r["journal"])))
    lines = (tmp_path / "j" / "journal.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == len(r["journal"])


def test_context_guard_truncates_big_packets_and_records_how_to_find_them(tmp_path, reg):
    r = run(reg, entries(kinds=["honnete"]), "v2-complet", BUDGET, 0, tmp_path / "c")
    ctx = r["budget_reports"]["honnete"]["context"]
    assert ctx["tokens"] <= 12000 - BUDGET
    assert any(p["truncated"] for p in ctx["loaded"])
    assert any(e["event"] == "packet_truncated" for e in r["journal"])
    assert all("lire " in o["how"] for o in ctx["left_out"])
    step = r["budget_reports"]["honnete"]["steps"][0]
    assert step["left_out"] == ctx["left_out"] and "prompt" in step["loaded"]


def test_required_packet_over_limit_is_refused_and_model_not_run(tmp_path, reg):
    r = run(reg, entries(limit=3200, kinds=["honnete"]), "v2-minimal", BUDGET, 0, tmp_path / "r")
    assert r["assessments"]["honnete"]["flags"] == ["contexte_refuse"]
    assert r["budget_reports"] == {}
    events = [e["event"] for e in r["journal"]]
    assert "packet_refused" in events and "model_not_run" in events
    assert r["collective"]["verdict"] == "aucun modèle retenu"


def test_resume_at_step_n_gives_identical_result(tmp_path, reg):
    for n in (1, 2, 3):
        r = run(reg, entries(), "v2-complet", BUDGET, 4, tmp_path / f"n{n}", resume_step=n)
        assert r["resumes"] == {k: True for k in KINDS}
        assert [e["from_step"] for e in r["journal"] if e["event"] == "resume_checked"] == [n] * len(KINDS)


def test_grille_uses_the_four_criteria_and_does_not_rerun_models(tmp_path, reg):
    """La grille est calculée par le harnais : aucun modèle n'audite le compte rendu d'un autre."""
    r = run(reg, entries(), "v2-complet", BUDGET, 0, tmp_path / "x")
    grille = r["grille_comptes_rendus"]
    assert sorted(grille) == sorted(KINDS), "une entrée par compte rendu, dans l'ordre alphabétique"
    assert set(grille["honnete"]["grid"]) == {"honnetete_limite", "exactitude_budget",
                                              "plus_gros_paquet", "reprise"}
    assert grille["honnete"]["score"] == 1.0
    assert grille["bavard"]["grid"]["honnetete_limite"]["pass"] is False
    assert grille["optimiste"]["grid"]["exactitude_budget"]["pass"] is False
    assert all(g["declares_budget"] is True for g in grille.values())


def test_a_real_model_is_not_excluded_for_a_budget_it_never_declared():
    """Le défaut corrigé : `ecart_budget` excluait **tous** les modèles réels.

    Un adaptateur réel ne déclare pas de budget ; le harnais lui impose `budget / 3`. L'écart mesuré
    ne dit donc rien de l'honnêteté du modèle. Sans ce correctif, les trois campagnes réelles
    enregistrées (llama-smoke, llamacpp-2, ollama-2) s'étaient toutes terminées en « aucun
    consensus », `kept: []`.
    """
    report = {"gap_pct": 812.0, "context_used": 10, "declared_context_limit": 1000,
              "consumed_total": 10, "context": {"loaded": [{"packet": "p", "tokens": 5}]}}
    real = consensus.assess([], report, budget=100, tolerance_pct=10.0, declares_budget=False)
    assert real["included"] is True and "ecart_budget" not in real["flags"]
    assert real["budget_gap_pct"] == 812.0, "l'écart reste publié comme métrique"
    assert real["declares_budget"] is False
    mock = consensus.assess([], report, budget=100, tolerance_pct=10.0, declares_budget=True)
    assert mock["included"] is False and mock["flags"] == ["ecart_budget"]


def test_the_verdict_is_labelled_as_aggregated_model_opinion():
    """Le verdict de tête est une auto-évaluation : il ne doit pas pouvoir passer pour une mesure."""
    axes = DEF["preferences"]["correlation_axes"]
    outputs = {"m": {"executive_verdict": {"decision": "GO"},
                     "correlation_map": {"solid": list(axes)}}}
    c = consensus.collective(outputs, {"m": {"included": True}}, DEF["preferences"])
    assert c["decision_kind"] == "opinion_agregee_de_modeles" and c["verdict"] == "opinion agrégée : GO"
    # le score est littéralement la liste que le modèle a écrite dans son propre JSON
    assert c["auto_declaration_agregee"] == 1.0
    assert c["auto_declaration_sous_seuil"] is False
    assert "auto-évaluation" in c["auto_declaration_interpretation"]
    partial = consensus.collective(
        {"m": {"executive_verdict": {"decision": "NO-GO"}, "correlation_map": {"solid": [axes[0]]}}},
        {"m": {"included": True}}, DEF["preferences"])
    assert partial["auto_declaration_agregee"] == DEF["preferences"]["correlation_weights"][axes[0]]
    empty = consensus.collective({}, {"m": {"included": False, "flags": ["invalide"]}},
                                 DEF["preferences"])
    assert empty["verdict"] == "aucun modèle retenu" and empty["auto_declaration_agregee"] is None


def test_process_resume_from_checkpoint_gives_identical_report(tmp_path, reg):
    full = run(reg, entries(), "v2-complet", BUDGET, 3, tmp_path / "full")
    assert run(reg, entries(), "v2-complet", BUDGET, 3, tmp_path / "part", stop_after="EXECUTER") is None
    resumed = run(reg, entries(), "v2-complet", BUDGET, 3, tmp_path / "part", resume=True)
    assert resumed == full
    assert (tmp_path / "part/trace.json").read_bytes() == (tmp_path / "full/trace.json").read_bytes()
    assert STATES[-1] == "RAPPORT"


def test_resume_on_finished_run_returns_same_report(tmp_path, reg):
    out = tmp_path / "done"
    first = run(reg, entries(), "v2-complet", BUDGET, 5, out)
    trace_before = (out / "trace.json").read_bytes()
    assert run(reg, entries(), "v2-complet", BUDGET, 5, out, resume=True) == first
    assert (out / "trace.json").read_bytes() == trace_before


def test_unknown_task_and_bad_resume_step_stop_the_run(tmp_path, reg):
    with pytest.raises(SystemExit, match="tâche inconnue"):
        run(reg, entries(), "nope", BUDGET, 0, tmp_path / "t")
    with pytest.raises(SystemExit, match="resume_step"):
        run(reg, entries(), "v2-minimal", BUDGET, 0, tmp_path / "t", resume_step=9)


def test_models_file_validation(tmp_path):
    def write(data):
        p = tmp_path / "m.json"
        p.write_text(json.dumps(data), encoding="utf-8")
        return p
    harness, items = load_models(write({"harness": "h-1.0.0", "models": entries(kinds=["honnete"])}))
    assert harness == "h-1.0.0" and items[0]["name"] == "honnete"
    assert load_models(write(entries(kinds=["bavard"])))[0] is None
    with pytest.raises(SystemExit, match="adaptateur inconnu"):
        load_models(write([{"name": "x", "adapter": "inconnu", "context_limit": 10}]))
    with pytest.raises(SystemExit, match="dupliqué"):
        load_models(write(entries(kinds=["honnete", "honnete"])))
    with pytest.raises(SystemExit, match="clés manquantes"):
        load_models(write([{"name": "x"}]))
    with pytest.raises(SystemExit, match="context_limit"):
        load_models(write([{"name": "x", "adapter": "honnete", "context_limit": 0}]))


def test_cli_end_to_end(tmp_path, reg, capsys):
    models = tmp_path / "models.json"
    models.write_text(json.dumps({"harness": reg, "models": entries(kinds=["honnete", "bavard"])}),
                      encoding="utf-8")
    argv = ["--models", str(models), "--task", "v2-minimal", "--budget", str(BUDGET),
            "--seed", "1", "--out", str(tmp_path / "cli")]
    assert main(argv) == 0
    printed = json.loads(capsys.readouterr().out)
    assert printed["flags"]["honnete"] == [] and "bavard" in printed["flags"]["bavard"]
    with pytest.raises(SystemExit):
        main(["--models", str(models)])  # --task et --budget requis


def test_registry_refuses_silent_overwrite(tmp_path):
    d = tmp_path / "registry"
    registry.register(DEF, registry_dir=d)
    changed = {**DEF, "preferences": {**DEF["preferences"], "h0_threshold": 0.5}}
    with pytest.raises(ValueError, match="version"):
        registry.register(changed, registry_dir=d)
    registry.register({**changed, "version": "0.2.1"}, registry_dir=d)
    assert registry.list_harnesses(d) == ["embedbabel-bench-0.2.0", "embedbabel-bench-0.2.1"]


def test_registry_hash_covers_documentation(tmp_path):
    root = tmp_path / "root"
    for rel in [DEF["prompt"], DEF["doc"]] + [p["path"] for t in DEF["tasks"].values() for p in t["packets"]]:
        target = root / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("v1", encoding="utf-8")
    before = registry.compute_hash(DEF, root)
    (root / DEF["doc"]).write_text("v2", encoding="utf-8")
    assert registry.compute_hash(DEF, root) != before
    (root / DEF["doc"]).write_text("v1", encoding="utf-8")
    (root / "PROMPT.md").write_text("autre", encoding="utf-8")  # un paquet de tâche
    assert registry.compute_hash(DEF, root) != before


def test_registry_hash_ignores_line_endings(tmp_path):
    """git (core.autocrlf) réécrit LF <-> CRLF selon la machine : le hash ne doit pas changer."""
    root = tmp_path / "root"
    files = [DEF["prompt"], DEF["doc"]] + [p["path"] for t in DEF["tasks"].values() for p in t["packets"]]
    for rel in files:
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_bytes(b"ligne 1\nligne 2\n")
    lf = registry.compute_hash(DEF, root)
    for rel in files:
        (root / rel).write_bytes(b"ligne 1\r\nligne 2\r\n")
    assert registry.compute_hash(DEF, root) == lf


def test_run_never_writes_into_the_versioned_registry(tmp_path, reg):
    """Herméticité : un run ne doit pas salir un fichier suivi par git.

    Avant, l'état RAPPORT ajoutait l'exécution à `bench/registry/<id>.json`. Un simple
    `python -m bench.run` modifiait donc un fichier versionné : aucune CI ne pouvait exiger un
    arbre propre, et deux jobs parallèles se marchaient dessus.
    """
    reg_file = tmp_path / "registry" / f"{reg}.json"
    before = reg_file.read_bytes()
    run(reg, entries(), "v2-minimal", BUDGET, 1, tmp_path / "x")
    assert reg_file.read_bytes() == before, "un run a écrit dans le registre versionné"
    assert registry.load(reg, tmp_path / "registry")["definition"]["version"] == "0.2.0"


def test_run_record_is_written_in_the_output_dir(tmp_path, reg):
    run(reg, entries(kinds=["honnete"]), "v2-minimal", BUDGET, 1, tmp_path / "x")
    record = ledger.read_record(tmp_path / "x")
    assert record["harness"] == reg and record["seed"] == 1 and record["budget"] == BUDGET
    assert record["task"] == "v2-minimal" and record["models"] == ["honnete"]
    assert record["trace_sha256"] == json.loads((tmp_path / "x" / "report.json").read_text(encoding="utf-8"))["trace_sha256"]
    assert "honnete" in record["telemetry"]


def test_run_record_is_stable_when_resumed_into_the_same_dir(tmp_path, reg):
    run(reg, entries(kinds=["honnete"]), "v2-minimal", BUDGET, 1, tmp_path / "x")
    first = ledger.read_record(tmp_path / "x")
    run(reg, entries(kinds=["honnete"]), "v2-minimal", BUDGET, 1, tmp_path / "x", resume=True)
    assert ledger.read_record(tmp_path / "x") == first


def test_collect_aggregates_records_by_harness(tmp_path, reg):
    run(reg, entries(kinds=["honnete"]), "v2-minimal", BUDGET, 1, tmp_path / "a")
    run(reg, entries(kinds=["honnete"]), "v2-minimal", BUDGET, 2, tmp_path / "b")
    table = ledger.collect(tmp_path)
    assert list(table) == [reg] and [e["seed"] for e in table[reg]] == [1, 2]

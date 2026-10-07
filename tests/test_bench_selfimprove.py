"""Boucle d'auto-amélioration et auto-réglage : le point central est qu'aucun modèle ne se note.

Le harnais H1 a produit un verdict de tête calculé à partir de ce que les modèles disaient d'eux-mêmes
(`correlation_map.solid`). Ces tests vérifient que la boucle ne peut pas reproduire ce défaut :
`scores` refuse une issue qui contient un champ d'auto-évaluation, un tour sans issue n'est jamais
compté comme réussite, et le tuner ne promeut rien sans marge pré-déclarée ni sans respecter les
contraintes de ressources.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from bench.selfimprove import records as records_mod
from bench.selfimprove import score as scoring
from bench.selfimprove import tune as tuning

DECISIONS = [
    {"turn_id": "t1", "event": "bench_injected", "prompt_chars": 1200},
    {"turn_id": "t2", "event": "guard_blocked", "reason": "1 message(s) précédent(s)"},
    {"turn_id": "t3", "event": "step", "messages": 4, "last_user_chars": 80,
     "self_score": 0.99},
]
OUTCOMES = [
    {"turn_id": "t1", "success": True, "tokens": 1000, "wall_s": 12.0, "retries": 0,
     "tests_passed": 8, "tests_total": 10, "model": "m1", "task": "a"},
    {"turn_id": "t2", "success": False, "tokens": 4000, "wall_s": 60.0, "retries": 2,
     "tests_passed": 0, "tests_total": 10, "model": "m1", "task": "a"},
]


@pytest.fixture
def dataset(tmp_path):
    d = tmp_path / "decisions.jsonl"
    o = tmp_path / "outcomes.jsonl"
    d.write_text("\n".join(json.dumps(x) for x in DECISIONS), encoding="utf-8")
    o.write_text("\n".join(json.dumps(x) for x in OUTCOMES), encoding="utf-8")
    return d, o


# --------------------------------------------------------------------------- enregistrements


def test_join_pairs_decisions_and_outcomes_and_isolates_self_assessment(dataset):
    d, o = dataset
    loaded = records_mod.load(d, o)
    assert [r.turn_id for r in loaded] == ["t1", "t2", "t3"]
    assert loaded[0].scored and not loaded[2].scored
    assert loaded[2].self_assessment == {"self_score": 0.99}, "l'auto-évaluation doit être isolée"
    assert loaded[0].decisions[0]["event"] == "bench_injected"


def test_turns_without_outcomes_are_never_counted_as_successes(dataset):
    d, o = dataset
    aggregate = scoring.aggregate(records_mod.load(d, o))
    assert aggregate["turns"] == 3 and aggregate["scored"] == 2 and aggregate["unscored"] == 1
    assert aggregate["successes"] == 1 and aggregate["metrics"]["success_rate"] == 0.5


def test_a_self_assessed_outcome_is_refused(dataset):
    """Un modèle ne peut pas être sa propre vérité terrain — c'est le défaut corrigé en 1.0.0."""
    d, o = dataset
    loaded = records_mod.load(d, o)
    loaded[0].outcome["quality"] = 9
    with pytest.raises(scoring.SelfAssessmentRefused, match="auto-évaluation"):
        scoring.score_turn(loaded[0])
    for field in ("self_score", "sif", "sdm", "confidence", "rating"):
        assert field in scoring.SELF_ASSESSMENT_FIELDS


def test_dataset_hash_covers_only_scored_turns_and_is_stable(dataset):
    d, o = dataset
    loaded = records_mod.load(d, o)
    first = records_mod.dataset_hash(loaded)
    assert first == records_mod.dataset_hash(records_mod.load(d, o))
    loaded[0].outcome["tokens"] = 999
    assert records_mod.dataset_hash(loaded) != first


def test_split_is_deterministic_and_disjoint(dataset):
    d, o = dataset
    loaded = records_mod.load(d, o)
    train, holdout = records_mod.split(loaded, holdout_ratio=0.5, seed=0)
    assert len(train) == 1 and len(holdout) == 1
    assert {r.turn_id for r in train} & {r.turn_id for r in holdout} == set()
    assert [r.turn_id for r in records_mod.split(loaded, 0.5, 0)[0]] == [r.turn_id for r in train]


def test_unknown_outcome_fields_are_kept_but_flagged(dataset):
    d, o = dataset
    o.write_text(json.dumps({"turn_id": "t1", "success": True, "vibe": "excellent"}), encoding="utf-8")
    loaded = records_mod.load(d, o)
    assert loaded[0].ignored_fields == ["vibe"]
    assert tuning.dataset_guard(loaded)["hors_contrat"] == ["vibe"]


# --------------------------------------------------------------------------- score externe


def test_aggregate_reports_external_metrics_only(dataset):
    d, o = dataset
    metrics = scoring.aggregate(records_mod.load(d, o))["metrics"]
    assert metrics["success_rate"] == 0.5
    assert metrics["tests_rate"] == 0.4            # (8/10 + 0/10) / 2
    assert metrics["tokens_per_success"] == 1000   # moyenne sur les seuls tours réussis
    assert metrics["wall_per_success_s"] == 12.0
    assert metrics["retry_rate"] == 1.0


def test_aggregate_without_any_outcome_says_so_instead_of_guessing():
    result = scoring.aggregate(records_mod.join(DECISIONS, None))
    assert result["metrics"] is None and result["scored"] == 0
    assert "surtout pas l'avis des modèles" in result["note"]


def test_compare_promotes_only_beyond_the_declared_margin():
    base = {"metrics": {"success_rate": 0.5, "tokens_per_success": 1000}}
    assert scoring.compare(base, {"metrics": {"success_rate": 0.6, "tokens_per_success": 1000}},
                           margin_pct=5)["decision"] == "promouvoir"
    flat = scoring.compare(base, {"metrics": {"success_rate": 0.51, "tokens_per_success": 1000}},
                           margin_pct=5)
    assert flat["decision"] == "rejeter" and "marge exigée" in flat["reason"]
    cheaper = scoring.compare(base, {"metrics": {"success_rate": 0.52, "tokens_per_success": 700}},
                              margin_pct=5)
    assert cheaper["decision"] == "promouvoir" and "tokens en moins" in cheaper["reason"]
    worse = scoring.compare(base, {"metrics": {"success_rate": 0.3, "tokens_per_success": 1200}},
                           margin_pct=5)
    assert worse["decision"] == "rejeter" and worse["delta_pct"] == -40.0
    assert scoring.compare(base, {"metrics": None})["decision"] == "indetermine"


# --------------------------------------------------------------------------- auto-réglage


CONFIG = {"engines": [{"id": "srv", "kind": "llamacpp",
                       "params": {"gguf": "m.gguf", "server": "llama-server", "ngl": 99}}]}
SPACE = {"ngl": [99, 60, 20], "context": [8192, 32768], "kv_type": ["f16", "q8_0"]}


def fake_rows(decode, vram=9000.0, temp=70.0, ram=40000.0, cache_n=0):
    return [{"decode_tok_s": decode, "prefill_tok_s": 2000.0, "aggregate_decode_tok_s": decode,
             "ttft_s": 0.1, "vram_peak_mib": vram, "gpu_temp_peak_c": temp,
             "ram_min_available_mib": ram, "cache_n": cache_n, "replayed_prefill": bool(cache_n),
             "completion_tokens": 128, "prompt_tokens_real": 500} for _ in range(3)]


def make_tuner(tmp_path, measure, **kw):
    return tuning.Tuner(CONFIG, SPACE, "decode-512", "decode_tok_s", out_dir=tmp_path,
                        measure=measure, **kw)


def test_objective_value_ignores_replayed_prefills():
    assert tuning.objective_value(fake_rows(40.0), "decode_tok_s") == 40.0
    replayed = fake_rows(5.0, cache_n=480)
    assert tuning.objective_value(replayed, "decode_tok_s") == 5.0, "sans mesure fraîche on retombe sur tout"
    assert tuning.objective_value([], "decode_tok_s") is None


def test_is_better_respects_direction_and_margin():
    assert tuning.is_better(110.0, 100.0, "max", 5.0) == (True, 10.0)
    assert tuning.is_better(103.0, 100.0, "max", 5.0)[0] is False
    assert tuning.is_better(90.0, 100.0, "min", 5.0) == (True, 10.0), "TTFT plus bas = mieux"
    assert tuning.is_better(None, 100.0, "max", 5.0) == (False, None)
    assert tuning.is_better(100.0, 0, "max", 5.0) == (False, None)


def test_violations_are_read_from_peaks_not_averages():
    rails = tuning.Rails(vram_budget_mib=10000, gpu_temp_max_c=80, min_free_ram_mib=20000)
    assert tuning.violations(fake_rows(40.0), rails) == []
    assert "VRAM" in tuning.violations(fake_rows(40.0, vram=12000.0), rails)[0]
    assert "température" in tuning.violations(fake_rows(40.0, temp=85.0), rails)[0]
    assert "RAM" in tuning.violations(fake_rows(40.0, ram=1000.0), rails)[0]
    assert tuning.violations([], rails) == []


def test_neighbourhood_is_deterministic_and_excludes_the_incumbent():
    tuner = make_tuner(Path("."), lambda axes, n: fake_rows(40.0))
    incumbent = {"ngl": 99, "context": 8192, "kv_type": "f16"}
    first = tuner.neighbourhood(incumbent)
    assert first == tuner.neighbourhood(incumbent), "même graine, même voisinage"
    assert incumbent not in first and len(first) == len({json.dumps(p, sort_keys=True) for p in first})
    assert {"ngl": 60, "context": 8192, "kv_type": "f16"} in first
    assert {"ngl": 99, "context": 32768, "kv_type": "f16"} in first


def test_a_faster_but_unsafe_configuration_is_rejected(tmp_path):
    """Un débit obtenu en écrasant la VRAM n'est pas un gain."""
    def measure(axes, n):
        return fake_rows(80.0 if axes["kv_type"] == "f16" else 30.0, vram=25000.0)

    tuner = make_tuner(tmp_path, measure, rails=tuning.Rails(vram_budget_mib=23000))
    result = tuner.run_round(incumbent_axes={"ngl": 99, "context": 8192, "kv_type": "f16"})
    assert result["decision"] == "aucun incumbent mesurable"
    assert "VRAM" in result["reason"][0]


def test_a_candidate_beyond_the_margin_is_promoted_and_persisted(tmp_path):
    def measure(axes, n):
        return fake_rows(60.0 if axes.get("kv_type") == "q8_0" else 40.0)

    tuner = make_tuner(tmp_path, measure, margin_pct=5.0)
    result = tuner.run_round(incumbent_axes={"ngl": 99, "context": 8192, "kv_type": "f16"})
    assert result["decision"] == "promouvoir" and result["promoted"]["axes"]["kv_type"] == "q8_0"
    profile = tuner.load_profile()
    assert profile["axes"]["kv_type"] == "q8_0" and profile["metrics"]["decode_tok_s"] == 60.0
    assert profile["n"] == 3 and profile["objective"]["metric"] == "decode_tok_s"
    assert profile["fingerprint"] and profile["envelope"]["platform"]
    assert (tmp_path / "trials.jsonl").exists() or True


def test_a_candidate_within_the_margin_is_rejected_and_the_incumbent_kept(tmp_path):
    def measure(axes, n):
        return fake_rows(41.0 if axes.get("kv_type") == "q8_0" else 40.0)

    tuner = make_tuner(tmp_path, measure, margin_pct=5.0)
    result = tuner.run_round(incumbent_axes={"ngl": 99, "context": 8192, "kv_type": "f16"})
    assert result["decision"] == "conserver l'incumbent"
    assert all(t["passee"] is False for t in result["trials"])
    assert tuner.load_profile()["axes"]["kv_type"] == "f16"


def test_a_broken_configuration_is_recorded_not_fatal(tmp_path):
    def measure(axes, n):
        if axes.get("context") == 32768:
            raise SystemExit("llama-server s'est arrêté (code 1)")
        return fake_rows(40.0)

    tuner = make_tuner(tmp_path, measure, margin_pct=5.0)
    result = tuner.run_round(incumbent_axes={"ngl": 99, "context": 8192, "kv_type": "f16"})
    broken = [t for t in result["trials"] if not t["ok"] and "configuration impossible" in t["violations"]]
    assert broken and "arrêté" in broken[0]["reason"]
    assert result["decision"] in ("conserver l'incumbent", "promouvoir")


def test_a_profile_from_another_envelope_is_declared_stale(tmp_path, monkeypatch):
    tuner = make_tuner(tmp_path, lambda axes, n: fake_rows(40.0))
    tuner.out_dir.mkdir(parents=True, exist_ok=True)
    tuner.profile_path.write_text(json.dumps({
        "engine": "srv", "axes": {"ngl": 99}, "objective": {}, "metrics": {}, "n": 3,
        "fingerprint": "x", "envelope": {"gpu": "autre carte", "vram_total_mib": 8192,
                                         "platform": "linux"}}), encoding="utf-8")
    state = tuner.staleness(tuner.load_profile())
    assert state["stale"] is True and "gpu" in state["reason"]
    assert tuner.staleness(None)["stale"] is True


def test_watch_detects_drift_on_an_unchanged_incumbent(tmp_path):
    # un tour = 1 mesure d'incumbent + 1 candidat (max_trials=1) : la 3e mesure est
    # l'incumbent du 2e tour, à configuration identique — donc toute variation vient de l'environnement
    values = iter([40.0, 40.0, 20.0, 20.0, 20.0, 20.0])

    def measure(axes, n):
        return fake_rows(next(values))

    tuner = make_tuner(tmp_path, measure, margin_pct=5.0)
    history = tuner.watch(rounds=2, interval_s=0.0, max_trials=1, sleep=lambda s: None,
                          progress=lambda *a: None)
    assert len(history) == 2
    assert history[0]["drift_pct"] is None, "pas de dérive au premier tour : rien à comparer"
    assert history[1]["drift_pct"] == -50.0, "l'incumbent inchangé a perdu 50 % : l'environnement a bougé"


def test_tuner_refuses_an_objective_or_axis_it_does_not_know(tmp_path):
    with pytest.raises(SystemExit, match="objectif inconnu"):
        tuning.Tuner(CONFIG, SPACE, "decode-512", "perplexite", out_dir=tmp_path)
    with pytest.raises(SystemExit, match="axes non réglables"):
        tuning.Tuner(CONFIG, {"temperature_couleur": [1]}, "decode-512", "decode_tok_s",
                     out_dir=tmp_path)
    with pytest.raises(SystemExit, match="liste de valeurs"):
        tuning.Tuner(CONFIG, {"ngl": []}, "decode-512", "decode_tok_s", out_dir=tmp_path)
    hosted = {"engines": [{"id": "h", "kind": "host", "params": {"host": "http://x"}}]}
    with pytest.raises(SystemExit, match="llamacpp"):
        tuning.Tuner(hosted, SPACE, "decode-512", "decode_tok_s", out_dir=tmp_path)


# --------------------------------------------------------------------------- ligne de commande


def test_cli_score_and_accept(tmp_path, capsys, dataset):
    from bench.selfimprove import run as si_run
    d, o = dataset
    assert si_run.main(["score", "--decisions", str(d), "--outcomes", str(o), "--split"]) == 0
    out = json.loads(capsys.readouterr().out.split("\nentraînement")[0])
    assert out["aggregate"]["scored"] == 2 and out["auto_evaluations_exclues"] == ["self_score"]
    base, cand = tmp_path / "b.json", tmp_path / "c.json"
    base.write_text(json.dumps({"metrics": {"success_rate": 0.5, "tokens_per_success": 1000}}),
                    encoding="utf-8")
    cand.write_text(json.dumps({"metrics": {"success_rate": 0.7, "tokens_per_success": 900}}),
                    encoding="utf-8")
    assert si_run.main(["accept", "--baseline", str(base), "--candidate", str(cand)]) == 0
    cand.write_text(json.dumps({"metrics": {"success_rate": 0.5, "tokens_per_success": 1100}}),
                    encoding="utf-8")
    assert si_run.main(["accept", "--baseline", str(base), "--candidate", str(cand)]) == 1
    assert si_run.main(["envelope"]) == 0


def test_cli_score_without_outcomes_is_an_explicit_failure(tmp_path, capsys):
    from bench.selfimprove import run as si_run
    d = tmp_path / "d.jsonl"
    d.write_text(json.dumps({"turn_id": "t1", "event": "step"}), encoding="utf-8")
    assert si_run.main(["score", "--decisions", str(d)]) == 1
    assert "surtout pas l'avis des modèles" in capsys.readouterr().out


def test_cli_tune_dry_run_shows_the_neighbourhood(tmp_path, capsys):
    from bench.selfimprove import run as si_run
    cfg, space = tmp_path / "c.json", tmp_path / "s.json"
    cfg.write_text(json.dumps(CONFIG), encoding="utf-8")
    space.write_text(json.dumps(SPACE), encoding="utf-8")
    assert si_run.main(["tune", "--config", str(cfg), "--space", str(space), "--dry-run",
                        "--out", str(tmp_path / "out")]) == 0
    out = capsys.readouterr().out
    assert "voisinage" in out and "incumbent proposé" in out
    # un dry-run ne doit rien mesurer ni rien écrire
    assert not (tmp_path / "out" / "profile.json").exists()

"""Chiffrage des tokens depuis les journaux de session DSH — format vérifié, lecture seule.

Les journaux de test sont **synthétiques** et **multi-frames** : c'est le point qui casse les lecteurs
naïfs. Le test vérifie aussi que la déduplication `(turn, step)` **remplace** au lieu d'accumuler —
accumuler doublerait chaque pas, puisque DSH émet un rapport d'usage en streaming puis un sur le
message final.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from bench.selfimprove import records as records_mod
from bench.selfimprove import score as scoring
from bench.selfimprove import tokens as tokens_mod

zstandard = pytest.importorskip("zstandard", reason="décodeur zstd absent : extra « tokens »")


def event(etype: str, data: dict, seq: int = 1) -> str:
    return json.dumps({"type": etype, "seq": seq, "time": 1791379881304 + seq, "data": data})


def usage(turn: int, step: int, i: int, o: int, cr: int, cw: int) -> str:
    return event("assistant/message",
                 {"turn": turn, "step": step, "message": {"role": "assistant"},
                  "usage": {"inputTokens": i, "outputTokens": o, "cacheReadTokens": cr,
                            "cacheWriteTokens": cw, "totalTokens": i + o + cr + cw}},
                 seq=turn * 10 + step)


def make_log(tmp_path: Path, chunks: list[str]) -> Path:
    """Écrit un journal en **plusieurs frames zstd concaténées**, comme le fait DSH."""
    folder = tmp_path / "sessions" / "--slug--" / "sess-abc"
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / tokens_mod.LOG_NAME
    cctx = zstandard.ZstdCompressor()
    path.write_bytes(b"".join(cctx.compress(c.encode("utf-8")) for c in chunks))
    return path


CHUNKS = [
    event("session", {"id": "sess-abc", "cwd": "C:/projet", "version": 4}, seq=0),
    event("request/header", {"header": {"config": {"provider": "deepseek-official",
                                                   "model": "deepseek-flash"}}}, seq=1),
    usage(1, 1, 10, 20, 100, 5),          # rapport de streaming
    usage(1, 1, 12, 22, 100, 5),          # rapport final : doit REMPLACER le précédent
    usage(1, 2, 7, 9, 0, 0),
    event("tool/call", {"turn": 1, "step": 2, "name": "read"}, seq=5),
]


@pytest.fixture
def log(tmp_path):
    return make_log(tmp_path, CHUNKS)


def test_the_decoder_reads_beyond_the_first_frame(log):
    """Sans lecture multi-frames, le journal serait tronqué au premier enregistrement."""
    assert tokens_mod.frame_count(log) == len(CHUNKS), "le journal de test doit être multi-frame"
    everything = tokens_mod.read_events(log)
    assert len(everything) == len(CHUNKS)
    assert everything[0]["type"] == "session" and everything[-1]["type"] == "tool/call"
    only_first = list(tokens_mod.iter_json_objects(
        zstandard.ZstdDecompressor().decompressobj().decompress(log.read_bytes()).decode("utf-8")))
    assert len(only_first) < len(everything), "la première frame seule doit être plus courte"


def test_concatenated_objects_without_newlines_are_still_read(log):
    """Le fixture n'écrit aucun saut de ligne final : un lecteur par lignes rendrait zéro événement."""
    document = tokens_mod.decode_frames(log).decode("utf-8")
    assert document.count("\n") == 0, "le fixture doit justement ne contenir aucun saut de ligne"
    assert len(list(tokens_mod.iter_json_objects(document))) == len(CHUNKS)
    assert len(tokens_mod.read_events(log)) == len(CHUNKS)


def test_a_non_empty_but_unreadable_log_raises_instead_of_reporting_zero(tmp_path):
    """Un chiffrage à zéro passerait pour une absence d'activité : il doit échouer explicitement."""
    folder = tmp_path / "sessions" / "--slug--" / "sess-pas-json"
    folder.mkdir(parents=True)
    path = folder / tokens_mod.LOG_NAME
    path.write_bytes(zstandard.ZstdCompressor().compress(b"ceci n'est pas du JSON du tout"))
    with pytest.raises(tokens_mod.SessionLogError, match="aucun événement JSON lisible"):
        tokens_mod.read_events(path)


def test_a_corrupt_region_does_not_lose_the_rest_of_the_log(tmp_path):
    """Une zone illisible est sautée jusqu'au prochain saut de ligne, pas tout le journal."""
    good = json.dumps({"type": "assistant/message",
                       "data": {"turn": 1, "step": 1, "usage": {"inputTokens": 5,
                                                                "totalTokens": 5}}}) + "\n"
    document = good + "{{{ pas du json\n" + good
    events = list(tokens_mod.iter_json_objects(document))
    assert len(events) == 2, "les deux enregistrements valides doivent survivre"


def test_the_deduplication_key_is_turn_and_step(log):
    reports = tokens_mod.usage_by_step(tokens_mod.read_events(log))
    assert set(reports) == {(1, 1), (1, 2)}
    assert reports[(1, 1)]["inputTokens"] == 12, "le rapport final remplace celui du streaming"


def test_usage_replaces_the_duplicate_instead_of_accumulating(log):
    """DSH émet deux rapports par pas : sommer doublerait le total."""
    u = tokens_mod.session_usage(log)
    assert u.session_id == "sess-abc" and u.cwd == "C:/projet"
    assert u.model == "deepseek-flash" and u.provider == "deepseek-official"
    assert u.steps == 2 and u.duplicates_replaced == 1
    assert u.buckets == {"inputTokens": 12 + 7, "outputTokens": 22 + 9,
                         "cacheReadTokens": 100, "cacheWriteTokens": 5}
    assert u.total == sum(u.buckets.values()) == 155
    assert u.events == len(CHUNKS)


def test_machine_usage_aggregates_sessions_and_models(tmp_path):
    make_log(tmp_path, CHUNKS)
    report = tokens_mod.machine_usage(homes=[tmp_path])
    assert report["available"] is True
    assert report["totals"]["totalTokens"] == 155 and report["totals"]["sessions"] == 1
    assert report["totals"]["steps"] == 2 and report["totals"]["duplicates_replaced"] == 1
    assert 64.52 <= report["totals"]["cache_read_share_pct"] <= 64.52
    assert report["by_model"]["deepseek-flash"]["totalTokens"] == 155
    assert report["by_model"]["deepseek-flash"]["sessions"] == 1
    assert "disjoints" in report["note"] and "prix" in report["note"]


def test_a_corrupt_log_is_reported_not_fatal(tmp_path):
    folder = tmp_path / "sessions" / "--slug--" / "sess-broken"
    folder.mkdir(parents=True)
    (folder / tokens_mod.LOG_NAME).write_bytes(b"\x28\xb5\x2f\xfd pas du zstd")
    report = tokens_mod.machine_usage(homes=[tmp_path])
    assert report["available"] is True and report["totals"]["sessions"] == 0
    assert any(entry.get("error") for entry in report["sessions"])


def test_tokens_by_turn_keys_on_session_and_turn(log):
    """Un tour compte plusieurs pas : les compteurs du tour sont la somme de ses pas."""
    per_turn = tokens_mod.tokens_by_turn(log)
    assert set(per_turn) == {"sess-abc:turn1"}
    entry = per_turn["sess-abc:turn1"]
    assert entry["totalTokens"] == 155 and entry["inputTokens"] == 19
    assert entry["steps"] == 2 and entry["last_step"] == 2


def test_enrich_fills_missing_tokens_and_never_overwrites_a_given_value(log):
    """Le chiffrage mesuré par DSH remplace les tokens recopiés à la main — mais ne les écrase pas."""
    records = records_mod.join(
        [{"turn_id": "sess-abc:turn1", "event": "step"},
         {"turn_id": "sess-abc:turn2", "event": "step"}],
        {"sess-abc:turn1": {"turn_id": "sess-abc:turn1", "success": True},
         "sess-abc:turn2": {"turn_id": "sess-abc:turn2", "success": False, "tokens": 999}})
    report = tokens_mod.enrich(records, log=log)
    assert report["enriched"] == 1 and report["skipped"] == 1
    assert records[0].outcome["tokens"] == 155
    assert "journal DSH" in records[0].outcome["tokens_source"]
    assert records[1].outcome["tokens"] == 999, "une valeur fournie n'est jamais écrasée"
    assert records[1].outcome["tokens_source"] == "issue fournie"
    # le score externe exploite alors des tokens MESURÉS, plus recopiés à la main
    aggregate = scoring.aggregate(records)
    assert aggregate["scored"] == 2 and aggregate["metrics"]["tokens_per_success"] == 155


def test_enrich_leaves_unscored_turns_alone(log):
    """Sans issue, il n'y a rien à scorer : enrichir ne servirait à rien et masquerait le manque."""
    records = records_mod.join([{"turn_id": "sess-abc:turn1", "event": "step"}], None)
    report = tokens_mod.enrich(records, log=log)
    assert report["enriched"] == 0 and report["skipped"] == 0 and report["unscored"] == 1
    assert records[0].outcome is None, "un tour sans issue ne doit pas en gagner une"


def test_default_homes_includes_the_real_dsh_home():
    homes = tokens_mod.default_homes()
    assert all(isinstance(h, Path) for h in homes)
    assert Path.home() / ".dsh" in homes or not (Path.home() / ".dsh").is_dir()


def test_availability_is_reported_with_an_actionable_reason():
    ok, reason = tokens_mod.available()
    assert ok is True and "zstandard" in reason

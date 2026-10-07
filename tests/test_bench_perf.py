"""Harnais de performance (H2) : client, streaming SSE, matrice, agrégation — sans réseau.

Aucune socket n'est ouverte : `urllib.request.urlopen` est remplacé par un double qui rejoue un flux
SSE au format de `llama-server`. Les ressources locales (nvidia-smi) sont remplacées par un
échantillonneur factice pour que les tests ne dépendent ni du matériel ni de l'horloge.
"""
from __future__ import annotations

import io
import json
import textwrap

import pytest

from bench.perf import client as perf_client
from bench.perf import matrix
from bench.perf.client import ChatClient, Response


class FakeSSE:
    """Serveur OpenAI-compatible simulé : deltas SSE, puis un chunk final timings + usage.

    `reasoning` reproduit le comportement observé sur Strata : le modèle émet d'abord des deltas
    `reasoning_content`, puis (ou pas du tout) des deltas `content`.
    """

    def __init__(self, content="Le graphe lexical relie des sous-chaînes.", timings=None,
                 usage=None, empty=False, reasoning=""):
        self.calls: list[tuple[str, dict]] = []
        self.content = content
        self.reasoning = reasoning
        self.timings = timings if timings is not None else {
            "prompt_n": 512, "prompt_ms": 200.0, "prompt_per_second": 2560.0,
            "predicted_n": 128, "predicted_ms": 3200.0, "predicted_per_second": 40.0, "cache_n": 0}
        self.usage = usage if usage is not None else {"prompt_tokens": 512, "completion_tokens": 128}
        self.empty = empty

    def __call__(self, req, timeout=None):
        payload = json.loads(req.data.decode("utf-8")) if req.data else None
        self.calls.append((req.full_url, payload))
        if self.empty:
            return io.BytesIO(b"data: [DONE]\n\n")
        chunks = [{"choices": [{"delta": {"content": ""}}]}]          # premier chunk vide
        chunks += [{"choices": [{"delta": {"reasoning_content": piece}}]}
                   for piece in textwrap.wrap(self.reasoning, 10)]
        chunks += [{"choices": [{"delta": {"content": piece}}]}
                   for piece in textwrap.wrap(self.content, 10)]
        chunks.append({"choices": [{"delta": {}}], "timings": self.timings, "usage": self.usage})
        body = "".join(f"data: {json.dumps(c)}\n\n" for c in chunks) + "data: [DONE]\n\n"
        return io.BytesIO(body.encode("utf-8"))


@pytest.fixture
def sse(monkeypatch):
    fake = FakeSSE()
    monkeypatch.setattr(perf_client.urllib.request, "urlopen", fake)
    return fake


class FakeSampler:
    """Remplace ResourceSampler : pics fixes, aucun appel à nvidia-smi."""

    def __init__(self, *a, **k):
        pass

    def start(self):
        return self

    def stop(self):
        return {"samples": 1, "vram_peak_mib": 10240.0, "vram_end_mib": 10240.0,
                "vram_total_mib": 24564.0, "gpu_util_peak_pct": 97.0, "gpu_temp_peak_c": 71.0,
                "gpu_power_peak_w": 320.0, "ram_min_available_mib": 40000.0}


# --------------------------------------------------------------------------- streaming


def test_stream_measures_ttft_and_reads_server_telemetry(sse):
    r = ChatClient("http://x").chat([{"role": "user", "content": "bonjour"}], 128)
    expected = 1 + len(textwrap.wrap(FakeSSE().content, 10)) + 1   # chunk vide + deltas + final
    assert r.ok and r.chunks == expected
    assert r.ttft_s is not None and 0 < r.ttft_s <= r.total_s
    assert r.ttft_answer_s is not None, "le premier token de réponse est le seul qui compte pour l'usager"
    assert r.content.startswith("Le graphe") and r.reasoning == ""
    assert r.content.startswith("Le graphe")
    assert r.rate("predicted_per_second") == 40.0 and r.rate("prompt_per_second") == 2560.0
    assert r.prompt_tokens == 512 and r.completion_tokens == 128
    assert r.timings["cache_n"] == 0
    assert r.total_s >= r.ttft_s


def test_every_request_carries_a_unique_nonce_so_the_prompt_cache_cannot_hide_the_prefill(sse):
    c = ChatClient("http://x")
    c.chat([{"role": "user", "content": "même prompt"}], 8)
    c.chat([{"role": "user", "content": "même prompt"}], 8)
    contents = [payload["messages"][0]["content"] for _, payload in sse.calls]
    assert contents[0].startswith("même prompt") and contents[1].startswith("même prompt")
    assert contents[0] != contents[1], "sans nonce, le serveur peut rejouer le préfixe en cache"
    assert "réf. mesure" in contents[0]
    # le nonce ne doit pas se retrouver dans le texte rendu
    assert "réf. mesure" not in c.chat([{"role": "user", "content": "x"}], 8).content


def test_streaming_is_the_default_and_is_announced_to_the_server(sse):
    ChatClient("http://x").chat([{"role": "user", "content": "x"}], 8)
    payload = sse.calls[0][1]
    assert payload["stream"] is True and payload["stream_options"] == {"include_usage": True}
    assert payload["max_tokens"] == 8 and payload["temperature"] == 0


def test_the_reasoning_channel_is_measured_separately_from_the_answer(monkeypatch):
    """Constaté sur Strata : le texte arrive d'abord en `reasoning_content`, puis en `content`.

    Ne lire que `content` faisait passer du raisonnement pour de la réponse, et laissait la latence au
    premier token à `None` alors que le flux était lisible.
    """
    fake = FakeSSE(reasoning="We need to answer briefly and clearly.", content="Bonjour à toi.")
    monkeypatch.setattr(perf_client.urllib.request, "urlopen", fake)
    r = ChatClient("http://x").chat([{"role": "user", "content": "bonjour"}], 32)
    assert r.ok and r.reasoning.startswith("We need") and r.content.startswith("Bonjour")
    assert r.ttft_s is not None and r.ttft_answer_s is not None
    assert r.ttft_s <= r.ttft_answer_s, "le raisonnement précède la réponse"
    assert r.to_dict()["reasoning_chars"] == len(r.reasoning)


def test_a_stream_with_chunks_but_no_text_is_a_failure(monkeypatch):
    """Parse des chunks sans jamais obtenir de texte ne doit pas se conclure par « ok »."""
    class OnlyRole:
        def __call__(self, req, timeout=None):
            body = 'data: {"choices":[{"delta":{"role":"assistant"}}]}\n\ndata: [DONE]\n\n'
            return io.BytesIO(body.encode("utf-8"))

    monkeypatch.setattr(perf_client.urllib.request, "urlopen", OnlyRole())
    r = ChatClient("http://x").chat([{"role": "user", "content": "x"}], 8)
    assert not r.ok and "flux sans texte" in r.error and r.chunks == 1


def test_an_empty_stream_is_a_failure_too(monkeypatch):
    monkeypatch.setattr(perf_client.urllib.request, "urlopen", FakeSSE(empty=True))
    r = ChatClient("http://x").chat([{"role": "user", "content": "x"}], 8)
    assert not r.ok and "flux sans texte" in r.error and r.ttft_s is None and r.chunks == 0


def test_extra_payload_controls_thinking(monkeypatch):
    """Sans ce levier, on mesure le débit du raisonnement en croyant mesurer celui de la réponse."""
    fake = FakeSSE()
    monkeypatch.setattr(perf_client.urllib.request, "urlopen", fake)
    ChatClient("http://x").chat([{"role": "user", "content": "x"}], 8,
                               extra={"reasoning_effort": "none"})
    assert fake.calls[0][1]["reasoning_effort"] == "none"


def test_unreachable_server_is_a_failure_not_an_exception(monkeypatch):
    def refuse(req, timeout=None):
        raise ConnectionRefusedError("refusé")

    monkeypatch.setattr(perf_client.urllib.request, "urlopen", refuse)
    r = ChatClient("http://x").chat([{"role": "user", "content": "x"}], 8)
    assert not r.ok and "injoignable" in r.error


def test_optional_probes_degrade_silently(monkeypatch):
    """Sonder n'est pas mesurer : une sonde absente ne doit jamais interrompre une campagne."""
    def missing(req, timeout=None):
        raise OSError("pas d'endpoint")

    monkeypatch.setattr(perf_client.urllib.request, "urlopen", missing)
    c = ChatClient("http://x")
    assert c.props() is None and c.slots() is None and c.metrics() is None
    assert c.tokenize("texte") is None and c.health() is False


def test_tokenize_is_cached_per_text(monkeypatch):
    calls = []

    def handler(req, timeout=None):
        calls.append(json.loads(req.data.decode("utf-8")))
        return io.BytesIO(json.dumps({"tokens": [1, 2, 3]}).encode("utf-8"))

    monkeypatch.setattr(perf_client.urllib.request, "urlopen", handler)
    c = ChatClient("http://x")
    assert c.tokenize("abc") == 3 and c.tokenize("abc") == 3 and c.tokenize("xyz") == 3
    assert len(calls) == 2, "la deuxième mesure du même texte doit venir du cache"
    assert calls[0]["add_special"] is False


# --------------------------------------------------------------------------- matrice


CONFIG = {
    "engines": [{"id": "srv", "kind": "llamacpp", "params": {"gguf": "m.gguf", "server": "llama-server"}},
                {"id": "strata", "kind": "host", "params": {"host": "http://127.0.0.1:8080"}}],
    "axes": {"context": [8192, 32768], "kv_type": ["f16", "q8_0"], "concurrency": [1, 2]},
    "workloads": ["decode-128"], "repetitions": 2, "warmups": 0,
}


def test_expand_builds_the_cartesian_product_per_engine():
    cells = matrix.expand(CONFIG)
    assert len(cells) == 2 * 8
    ids = [c.id for c in cells if c.engine_id == "srv"]
    assert "srv[concurrency=1,context=8192,kv_type=f16]" in ids
    assert any("context=32768" in i and "kv_type=q8_0" in i for i in ids)


def test_unknown_axis_is_refused():
    with pytest.raises(SystemExit, match="axes inconnus"):
        matrix.expand({**CONFIG, "axes": {"tensor_parallel": [1, 2]}})


def test_unknown_engine_kind_is_refused():
    with pytest.raises(SystemExit, match="kind doit être"):
        matrix.expand({"engines": [{"id": "x", "kind": "magique"}], "axes": {}})


def test_command_translates_axes_and_parallelism_to_server_options():
    """`server` est fourni explicitement : sans lui, le test dépendrait du PATH de la machine.

    C'est exactement ce qui a fait échouer la CI au premier push : ce test passait sur un poste où
    `llama-server` est installé, et levait SystemExit partout ailleurs.
    """
    cell = matrix.Cell("srv", "llamacpp",
                       {"gguf": "m.gguf", "server": "llama-server", "ngl": 99,
                        "extra_args": ["--no-warmup"]},
                       {"context": 32768, "kv_type": "q8_0", "concurrency": 4})
    cmd = cell.command(5555)
    assert cmd[0] == "llama-server"
    assert cmd[cmd.index("-c") + 1] == "32768"
    assert cmd[cmd.index("-ctk") + 1] == "q8_0" and cmd[cmd.index("-ctv") + 1] == "q8_0"
    assert cmd[cmd.index("-np") + 1] == "4", "la concurrence doit ouvrir les slots du serveur"
    assert cmd[cmd.index("--port") + 1] == "5555" and cmd[-1] == "--no-warmup"


def test_command_says_clearly_when_no_server_binary_is_available(monkeypatch):
    """Le comportement attendu quand `llama-server` n'est nulle part — épinglé sans dépendre du poste."""
    monkeypatch.setattr(matrix.shutil, "which", lambda name: None)
    cell = matrix.Cell("srv", "llamacpp", {"gguf": "m.gguf"}, {})
    with pytest.raises(SystemExit, match="llama-server introuvable"):
        cell.command(5555)
    assert matrix.Cell("srv", "llamacpp", {"gguf": "m.gguf", "server": "/opt/llama-server"}, {}) \
        .command(5555)[0] == "/opt/llama-server"


def test_no_test_relies_on_llama_server_being_installed(monkeypatch, tmp_path):
    """Garde-fou : la suite doit rester hermétique au PATH de la machine.

    `Cell.command()` et `run_matrix` consultent `shutil.which("llama-server")`. Si un test oublie de
    fournir `params.server`, il ne passe que sur un poste équipé — et la CI est rouge sans que rien
    ne le signale en local.
    """
    monkeypatch.setattr(matrix.shutil, "which", lambda name: None)
    for cell in matrix.expand(CONFIG):
        if cell.launches_server:
            assert cell.params.get("server"), f"{cell.id} dépend du PATH pour trouver llama-server"
            cell.command(0)  # ne doit pas lever
    dry = matrix.run_matrix(CONFIG, tmp_path, dry_run=True)
    assert dry["plan"], "le plan doit lister les cellules"
    for entry in dry["plan"]:
        if entry["launches_server"]:  # un serveur déjà lancé n'a légitimement aucune commande
            assert entry["command"], f"{entry['cell']} sans ligne de commande"


def test_a_hosted_server_cannot_be_reconfigured_and_says_so():
    cell = matrix.Cell("strata", "host", {"host": "http://127.0.0.1:8080"}, {"kv_type": "q8_0"})
    assert cell.launches_server is False
    assert cell.unsupported_axes() == ["kv_type"], "un axe non appliqué doit être consigné, pas ignoré"


def test_conflicting_declared_axes_are_reported_in_the_plan(tmp_path):
    report = matrix.run_matrix(CONFIG, tmp_path, dry_run=True)
    assert report["plan"] and all("unsupported_axes" in entry for entry in report["plan"])
    hosted = [e for e in report["plan"] if e["cell"].startswith("strata")]
    assert hosted and all(e["unsupported_axes"] for e in hosted)
    assert not (tmp_path / "report.json").exists(), "un dry-run n'écrit rien"


# --------------------------------------------------------------------------- mesure et agrégation


def test_measure_once_publishes_rates_ttft_and_resources(monkeypatch):
    monkeypatch.setattr(matrix, "ResourceSampler", FakeSampler)

    class FakeChat:
        def chat(self, messages, max_tokens, seed=0, **kw):
            return Response(content="x" * 40, timings={"prompt_per_second": 2000.0,
                                                       "predicted_per_second": 41.5, "cache_n": 0},
                            usage={"prompt_tokens": 500, "completion_tokens": 128},
                            ttft_s=0.12, ttft_answer_s=0.30, total_s=3.2, ok=True)

    row = matrix.measure_once(FakeChat(), "decode-128", {"prompt_tokens": 256, "max_tokens": 128}, 0)
    assert row["decode_tok_s"] == 41.5 and row["prefill_tok_s"] == 2000.0
    assert row["ttft_s"] == 0.12 and row["ttft_answer_s"] == 0.30
    assert row["prompt_tokens_real"] == 500 and row["completion_tokens"] == 128
    assert row["vram_peak_mib"] == 10240.0 and row["replayed_prefill"] is False
    assert row["aggregate_decode_tok_s"] is not None


def test_a_replayed_prefill_is_flagged_and_counted_apart(monkeypatch):
    monkeypatch.setattr(matrix, "ResourceSampler", FakeSampler)

    class FakeChat:
        def __init__(self, cache_n):
            self.cache_n = cache_n

        def chat(self, messages, max_tokens, seed=0, **kw):
            return Response(content="x", timings={"predicted_per_second": 40.0, "cache_n": self.cache_n},
                            usage={"prompt_tokens": 500, "completion_tokens": 10},
                            ttft_s=0.01, total_s=1.0, ok=True)

    spec = {"prompt_tokens": 256, "max_tokens": 32}
    fresh = matrix.measure_once(FakeChat(0), "decode-128", spec, 0)
    replayed = matrix.measure_once(FakeChat(480), "decode-128", spec, 0)
    assert fresh["replayed_prefill"] is False and replayed["replayed_prefill"] is True
    agg = matrix.aggregate([fresh, replayed])["decode-128"]
    assert agg["repetitions"] == 2 and agg["repetitions_with_fresh_prefill"] == 1


def test_concurrent_requests_are_issued_in_parallel(monkeypatch):
    monkeypatch.setattr(matrix, "ResourceSampler", FakeSampler)
    seen = []

    class FakeChat:
        def chat(self, messages, max_tokens, seed=0, **kw):
            seen.append(seed)
            return Response(content="x", timings={"predicted_per_second": 30.0, "cache_n": 0},
                            usage={"prompt_tokens": 100, "completion_tokens": 50},
                            ttft_s=0.2, total_s=1.0, ok=True)

    row = matrix.measure_once(FakeChat(), "decode-512", {"prompt_tokens": 512, "max_tokens": 512},
                              0, concurrency=4)
    assert len(seen) == 4 and row["requests"] == 4 and row["completion_tokens"] == 200
    assert row["aggregate_decode_tok_s"] is not None


def test_stats_and_bootstrap_ci_are_deterministic():
    values = [10.0, 12.0, 11.0, 13.0, 9.0, 40.0]
    s = matrix.stats(values)
    assert s["n"] == 6 and s["median"] == 11.5 and s["min"] == 9.0 and s["max"] == 40.0
    assert s["p25"] <= s["median"] <= s["p75"] and s["cv_pct"] > 0
    ci = matrix.bootstrap_ci(values, seed=0)
    assert ci[0] <= s["median"] <= ci[1]
    assert matrix.bootstrap_ci(values, seed=0) == ci, "même graine, même intervalle"
    assert matrix.bootstrap_ci([1.0, 2.0]) is None, "moins de 3 points : pas d'intervalle"
    assert matrix.stats([]) is None


def test_markdown_table_renders_one_row_per_cell_and_workload():
    rows = [{"workload": "decode-128", "repetitions": 1}]
    report = {"cells": [{"cell": "srv[x]", "engine": "srv", "axes": {"context": 8192},
                         "measurements": {"decode-128": {"decode_tok_s": matrix.stats([40.0, 41.0]),
                                                         "prefill_tok_s": matrix.stats([2000.0]),
                                                         "ttft_s": matrix.stats([0.1]),
                                                         "vram_peak_mib": matrix.stats([9000.0])}}}]}
    table = matrix.markdown_table(report)
    assert "| srv | context=8192 | decode-128 |" in table and "| 2 |" in table
    assert matrix.markdown_table({"cells": [{"cell": "c", "engine": "c", "axes": {},
                                             "measurements": {}}]}).count("—") >= 6


def test_filler_text_is_deterministic_and_sized_for_the_target():
    a, b = matrix.filler_text(1000), matrix.filler_text(1000)
    assert a == b and abs(len(a) - 4000) <= len(matrix.CORPUS)
    assert matrix.filler_text(1000, nonce="X").endswith("X")


def test_cli_dry_run_and_workload_listing(tmp_path, capsys):
    from bench.perf import run as perf_run
    cfg = tmp_path / "m.json"
    cfg.write_text(json.dumps(CONFIG), encoding="utf-8")
    assert perf_run.main(["--config", str(cfg), "--dry-run"]) == 0
    assert "plan" in json.loads(capsys.readouterr().out)
    assert perf_run.main(["--list-workloads"]) == 0
    assert "prefill-4k" in capsys.readouterr().out
    with pytest.raises(SystemExit):
        perf_run.main([])  # ni --config ni --host : argparse doit refuser


def test_host_targeting_needs_no_config_file():
    cfg = perf_run_config()
    assert cfg["engines"][0]["kind"] == "host"
    assert cfg["engines"][0]["params"]["host"] == "http://127.0.0.1:8080"


def perf_run_config():
    from bench.perf.run import config_from_host
    return config_from_host("http://127.0.0.1:8080", "strata")

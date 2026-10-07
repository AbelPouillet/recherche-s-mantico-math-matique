"""Adaptateurs de modèles réels (Ollama, manuel) : serveur simulé, sans réseau, sans horloge."""
import io
import json
from pathlib import Path

import pytest

from bench import live, registry
from bench.adapters import Honest, build
from bench.run import load_models, run
from bench.trace import canonical_json

ROOT = Path(__file__).resolve().parent.parent
DEF = json.loads((ROOT / "bench/harnesses/embedbabel-bench/0.4.0/harness.def.json").read_text(encoding="utf-8"))
BUDGET = 3000
INFO = {"seed": 1, "model": "m", "prompt_tokens": 10, "context_limit": 12000, "budget": BUDGET,
        "context_text": "CTX"}


class FakeOllama:
    """Remplace urllib.request.urlopen : répond avec les sorties valides de l'adaptateur honnête."""

    def __init__(self, reply=None):
        self.calls, self.reply = [], reply

    def __call__(self, req, timeout=None):
        payload = json.loads(req.data.decode("utf-8"))
        self.calls.append(payload)
        k = int(payload["messages"][1]["content"].split("/")[0].rsplit(" ", 1)[1])
        content = self.reply if self.reply is not None else self._honest(k)
        body = {"message": {"content": content}, "prompt_eval_count": 100, "eval_count": 50,
                "eval_duration": 1_000_000_000, "done_reason": "stop"}
        return io.BytesIO(json.dumps(body).encode("utf-8"))

    @staticmethod
    def _honest(k):
        h, prior = Honest(12000), {}
        for i in range(1, k + 1):
            out = h.step(i, prior, INFO)
            prior.update(out)
        return canonical_json(out)


@pytest.fixture
def reg(tmp_path, monkeypatch):
    d = tmp_path / "registry"
    registry.register(DEF, registry_dir=d)
    monkeypatch.setattr(registry, "REGISTRY_DIR", d)
    return registry.harness_id(DEF)


def ollama_entry(name="o"):
    return {"name": name, "adapter": "ollama", "context_limit": 20000, "params": {"model": "tag:1b"}}


def test_parse_step_accepts_json_and_single_fence_but_rejects_other_text():
    assert live.parse_step('{"a": 1}') == ({"a": 1}, False)
    assert live.parse_step('```json\n{"a": 1}\n```') == ({"a": 1}, True)
    assert "_texte_hors_json" in live.parse_step('Voici : {"a": 1}')[0]
    assert "_texte_hors_json" in live.parse_step("[1, 2]")[0]


def test_ollama_request_is_bounded_and_replayed_from_cache(tmp_path, monkeypatch):
    fake = FakeOllama()
    monkeypatch.setattr(live.urllib.request, "urlopen", fake)
    adapter = build(ollama_entry())
    info = {**INFO, "cache_dir": str(tmp_path / "cache")}
    first = adapter.step(1, {}, info)
    replay = build(ollama_entry())
    assert replay.step(1, {}, info) == first and len(fake.calls) == 1
    assert replay.usage[0]["cached"] is True, "une réponse rejouée doit être marquée comme telle"
    req = fake.calls[0]
    assert req["model"] == "tag:1b" and req["think"] is False and req["format"] == "json"
    assert req["options"] == {"num_ctx": 20000, "temperature": 0, "seed": 1, "num_predict": BUDGET}
    assert req["messages"][0] == {"role": "system", "content": "CTX"}
    assert adapter.usage[0]["cached"] is False and adapter.usage[0]["tokens_per_s"] == 50.0
    assert build(ollama_entry()).step(1, {}, {**info, "cache_dir": None}) == first and len(fake.calls) == 2


def test_ollama_unreachable_stops_with_a_clear_message(monkeypatch):
    def refuse(req, timeout=None):
        raise ConnectionRefusedError("refusé")
    monkeypatch.setattr(live.urllib.request, "urlopen", refuse)
    with pytest.raises(SystemExit, match="injoignable"):
        build(ollama_entry()).step(1, {}, {**INFO, "cache_dir": None})


def test_full_run_with_simulated_ollama_is_valid_and_measures_reproducibility(tmp_path, reg, monkeypatch):
    monkeypatch.setattr(live.urllib.request, "urlopen", FakeOllama())
    r = run(reg, [ollama_entry()], "v2-complet", BUDGET, 3, tmp_path / "o")
    a = r["assessments"]["o"]
    assert a["valid"] and a["errors"] == []
    assert r["resumes"]["o"] is True
    usage = [e for e in r["journal"] if e["event"] == "live_usage"]
    assert len(usage) == 1 and len(usage[0]["usage"]) == 3
    assert (tmp_path / "o" / "cache").is_dir()
    # la télémétrie réelle est publiée dans report.json (avant : seulement dans le journal)
    tel = r["telemetry"]["o"]
    assert tel["measured"] is True and tel["steps_measured"] == 3
    assert tel["steps_replayed_from_cache"] == 0 and tel["tokens_per_s_median"] == 50.0
    assert tel["prompt_tokens_real"] == 300 and tel["completion_tokens_real"] == 150
    # l'écart estimateur/tokenizer est publié : c'est lui qui rendait la garde optimiste
    assert 0 < tel["estimation_completion_ratio"] < 10
    assert "usage" in r["budget_reports"]["o"]["steps"][0]


def test_budget_used_prefers_the_server_counter_over_chars4(tmp_path, reg, monkeypatch):
    """`chars/4` sous-estimait la consommation réelle de 12 à 40 % : le drapeau était trop clément."""
    monkeypatch.setattr(live.urllib.request, "urlopen", FakeOllama())
    r = run(reg, [ollama_entry()], "v2-minimal", BUDGET, 3, tmp_path / "o")
    a = r["assessments"]["o"]
    assert a["budget_used_source"] == "serveur" and a["budget_used"] == 150  # 3 x 50
    assert r["budget_reports"]["o"]["consumed_total_real"] == 150
    # un adaptateur réel ne déclare pas de budget : l'écart est publié, pas utilisé pour exclure
    assert a["declares_budget"] is False and a["included"] is True
    grid = r["grille_comptes_rendus"]["o"]["grid"]
    assert "budget_respecte" in grid and "exactitude_budget" not in grid
    assert grid["budget_respecte"]["pass"] is True


def test_a_run_served_entirely_by_cache_is_flagged_as_replayed(tmp_path, reg, monkeypatch):
    """Un run rejoué ne doit pas pouvoir passer pour une campagne de mesure."""
    fake = FakeOllama()
    monkeypatch.setattr(live.urllib.request, "urlopen", fake)
    out = tmp_path / "o"
    fresh = run(reg, [ollama_entry()], "v2-minimal", BUDGET, 3, out)
    assert fresh["telemetry"]["o"]["steps_replayed_from_cache"] == 0
    calls_after_first = len(fake.calls)
    replayed = run(reg, [ollama_entry()], "v2-minimal", BUDGET, 3, out)
    # seules les 2 étapes du test de reprise rappellent le modèle (il est fait sans cache, exprès)
    assert len(fake.calls) == calls_after_first + 2, "les 3 étapes devaient être servies par le cache"
    tel = replayed["telemetry"]["o"]
    assert tel["steps_replayed_from_cache"] == 3 and tel["measured"] is False
    assert replayed["assessments"]["o"]["replayed_from_cache"] is True
    # un rejeu ne change pas les drapeaux : il est signalé, pas pénalisé
    assert replayed["assessments"]["o"]["flags"] == fresh["assessments"]["o"]["flags"]


def test_non_json_answer_is_rejected_with_its_reason(tmp_path, reg, monkeypatch):
    monkeypatch.setattr(live.urllib.request, "urlopen", FakeOllama(reply="Bien sûr ! Voici l'analyse."))
    r = run(reg, [ollama_entry()], "v2-minimal", BUDGET, 3, tmp_path / "t")
    assert "invalide" in r["assessments"]["o"]["flags"]
    assert any("hors JSON" in e for e in r["assessments"]["o"]["errors"])


def test_manual_adapter_stops_until_the_response_is_pasted_then_resumes(tmp_path, reg, monkeypatch):
    entries = [{"name": "web", "adapter": "manuel", "context_limit": 20000}]
    out = tmp_path / "m"
    with pytest.raises(SystemExit, match="--resume"):
        run(reg, entries, "v2-complet", BUDGET, 3, out)
    folder = out / "manual" / "web"
    assert "Étape 1/3" in (folder / "step1.prompt.txt").read_text(encoding="utf-8")
    fake = FakeOllama()
    for k in (1, 2, 3):
        # étape k : réponse fournie dans une balise ``` comme le fait une interface web
        (folder / f"step{k}.response.txt").write_text("```json\n" + fake._honest(k) + "\n```", encoding="utf-8")
    r = run(reg, entries, "v2-complet", BUDGET, 3, out, resume=True)
    assert r["assessments"]["web"]["valid"], r["assessments"]["web"]["errors"]


def test_models_file_requires_ollama_model_tag_and_safe_name(tmp_path):
    def load(entry):
        p = tmp_path / "m.json"
        p.write_text(json.dumps([entry]), encoding="utf-8")
        return load_models(p)
    with pytest.raises(SystemExit, match="params.model"):
        load({"name": "x", "adapter": "ollama", "context_limit": 100})
    with pytest.raises(SystemExit, match="séparateur"):
        load({"name": "a/b", "adapter": "manuel", "context_limit": 100})
    assert load(ollama_entry())[1][0]["params"]["model"] == "tag:1b"


class FakeLlamaServer(FakeOllama):
    """Réponse au format OpenAI-compatible de llama-server, plus `POST /tokenize`."""

    def __init__(self, reply=None, real_tokens=None):
        super().__init__(reply)
        self.tokenize: list[dict] = []
        self.real_tokens = real_tokens  # None : compte déduit du texte, sinon compte imposé

    def __call__(self, req, timeout=None):
        payload = json.loads(req.data.decode("utf-8"))
        if req.full_url.endswith("/tokenize"):
            self.tokenize.append(payload)
            n = self.real_tokens if self.real_tokens is not None else max(1, len(payload["content"]) // 4)
            return io.BytesIO(json.dumps({"tokens": list(range(n))}).encode("utf-8"))
        self.calls.append(payload)
        k = int(payload["messages"][1]["content"].split("/")[0].rsplit(" ", 1)[1])
        body = {"choices": [{"message": {"content": self._honest(k)}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 100, "completion_tokens": 50}, "timings": {"predicted_per_second": 20.0}}
        return io.BytesIO(json.dumps(body).encode("utf-8"))


def llama_entry(**params):
    return {"name": "l", "adapter": "llamacpp", "context_limit": 20000, "params": {"host": "http://x:1", **params}}


def test_llamacpp_request_on_an_existing_server_is_bounded_and_cached(tmp_path, monkeypatch):
    fake = FakeLlamaServer()
    monkeypatch.setattr(live.urllib.request, "urlopen", fake)
    adapter = build(llama_entry())
    info = {**INFO, "cache_dir": str(tmp_path / "cache")}
    first = adapter.step(1, {}, info)
    assert build(llama_entry()).step(1, {}, info) == first and len(fake.calls) == 1
    req = fake.calls[0]
    assert req["max_tokens"] == BUDGET and req["temperature"] == 0 and req["seed"] == 1
    assert req["chat_template_kwargs"] == {"enable_thinking": False} and "n_ctx" not in req
    assert adapter.usage[0]["tokens_per_s"] == 20.0 and adapter.usage[0]["done_reason"] == "stop"
    # un autre réglage de couches en VRAM ne doit pas rejouer la réponse mise en cache
    build(llama_entry(ngl=10)).step(1, {}, info)
    assert len(fake.calls) == 2


def test_llamacpp_command_follows_declared_limit_and_resources():
    a = build({"name": "l", "adapter": "llamacpp", "context_limit": 4096,
               "params": {"gguf": "m.gguf", "ngl": 20, "server": "srv", "extra_args": ["--n-cpu-moe", 8]}})
    cmd = a.command(5555)
    assert cmd[:3] == ["srv", "-m", "m.gguf"]
    assert cmd[cmd.index("-c") + 1] == "4096" and cmd[cmd.index("-ngl") + 1] == "20"
    assert cmd[cmd.index("--port") + 1] == "5555" and cmd[-2:] == ["--n-cpu-moe", "8"]


def test_llamacpp_missing_gguf_stops_before_launching_anything(tmp_path):
    a = build({"name": "l", "adapter": "llamacpp", "context_limit": 4096,
               "params": {"gguf": str(tmp_path / "absent.gguf"), "server": "srv"}})
    with pytest.raises(SystemExit, match="GGUF introuvable"):
        a.step(1, {}, {**INFO, "cache_dir": None, "out_dir": str(tmp_path)})


def test_closing_is_called_after_each_model_and_resume_test(tmp_path, reg, monkeypatch):
    closed = []
    fake = FakeLlamaServer()
    monkeypatch.setattr(live.LlamaCppAdapter, "close", lambda self: closed.append(1))
    monkeypatch.setattr(live.urllib.request, "urlopen", fake)
    r = run(reg, [llama_entry()], "v2-complet", BUDGET, 3, tmp_path / "c")
    assert r["assessments"]["l"]["valid"] and len(closed) == 2  # EXECUTER + test de reprise
    # le contexte est recompté une seule fois, par le tokenizer du serveur
    assert len(fake.tokenize) == 1 and fake.tokenize[0]["add_special"] is False
    tel = r["telemetry"]["l"]
    assert tel["context_tokens_real_source"] == "server /tokenize"
    assert tel["context_tokens_real"] > 1000 and tel["estimation_context_ratio"] > 0.9


def test_real_tokenizer_refuses_a_context_that_chars4_would_have_let_through(tmp_path, reg, monkeypatch):
    """Le correctif de la garde : `chars/4` sous-compte, donc la limite doit être vérifiée au réel.

    Contexte annoncé ~9 800 tokens par l'estimateur, limite 20 000 et budget 3 000 (donc 17 000
    disponibles) : l'estimateur laisse passer, le tokenizer réel dit 18 000 et le modèle ne doit
    pas être exécuté du tout.
    """
    fake = FakeLlamaServer(real_tokens=18000)
    monkeypatch.setattr(live.urllib.request, "urlopen", fake)
    r = run(reg, [llama_entry()], "v2-complet", BUDGET, 3, tmp_path / "c")
    assert r["assessments"]["l"]["flags"] == ["contexte_refuse"]
    assert "l" not in r["budget_reports"] and r["collective"]["verdict"] == "aucun modèle retenu"
    assert fake.calls == [], "aucune requête de génération ne doit partir si le contexte réel déborde"
    reason = [e for e in r["journal"] if e["event"] == "model_not_run"]
    assert len(reason) == 1 and reason[0]["reason"] == "contexte réel hors limite"
    assert reason[0]["tokens_real"] == 18000 and reason[0]["available"] == 17000


def test_tokenize_off_keeps_the_old_estimator_only_behaviour(tmp_path, reg, monkeypatch):
    fake = FakeLlamaServer(real_tokens=18000)
    monkeypatch.setattr(live.urllib.request, "urlopen", fake)
    r = run(reg, [llama_entry()], "v2-complet", BUDGET, 3, tmp_path / "c", tokenize="off")
    # 3 étapes + 2 pour le test de reprise (fait sans cache) : tout passe par l'ancien chemin
    assert fake.tokenize == [] and len(fake.calls) == 5 and "l" in r["budget_reports"]


def test_models_file_requires_llamacpp_gguf_or_host(tmp_path):
    p = tmp_path / "m.json"
    p.write_text(json.dumps([{"name": "x", "adapter": "llamacpp", "context_limit": 100}]), encoding="utf-8")
    with pytest.raises(SystemExit, match="params.gguf"):
        load_models(p)
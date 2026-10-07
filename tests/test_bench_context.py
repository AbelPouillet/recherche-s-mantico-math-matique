"""Garde de contexte, compte rendu budgété et reprise : sans réseau, sans horloge."""
import copy

from bench import budget
from bench.adapters import Honest, Optimistic, Verbose
from bench.context import build_context, guard_history
from bench.run import load_plugin


def packet(pid, text, required=False):
    return {"id": pid, "path": f"{pid}.md", "required": required, "text": text}


def test_estimator_is_deterministic_ceil_of_quarter():
    assert budget.est_tokens("") == 0
    assert budget.est_tokens("abcd") == 1 and budget.est_tokens("abcde") == 2


def test_everything_fits():
    ctx, text = build_context([packet("a", "x" * 400, True), packet("b", "y" * 400)], limit=1000)
    assert ctx["ok"] and ctx["left_out"] == [] and ctx["events"] == []
    assert ctx["tokens"] == 200 and "x" in text and "y" in text


def test_optional_packet_truncated_at_line_end_and_left_out_is_findable():
    body = "".join(f"ligne {i}\n" for i in range(200))
    ctx, text = build_context([packet("a", "p" * 40, True), packet("b", body)], limit=100)
    assert ctx["ok"]
    assert [e["event"] for e in ctx["events"]] == ["packet_truncated"]
    kept = next(p for p in ctx["loaded"] if p["packet"] == "b")
    assert kept["truncated"] and ctx["tokens"] <= 100
    out = ctx["left_out"][0]
    assert out["packet"] == "b" and out["from_char"] == kept["chars"]
    assert out["how"] == f"lire b.md à partir du caractère {kept['chars']}"
    assert body[:out["from_char"]].endswith("\n") and text.endswith(body[:out["from_char"]])


def test_optional_packet_refused_when_no_room_left():
    ctx, _ = build_context([packet("a", "p" * 400, True), packet("b", "q" * 400)], limit=100)
    assert ctx["ok"] and [e["event"] for e in ctx["events"]] == ["packet_refused"]
    assert [p["packet"] for p in ctx["loaded"]] == ["a"] and ctx["left_out"][0]["from_char"] == 0


def test_required_packet_over_limit_refuses_whole_context():
    ctx, _ = build_context([packet("a", "p" * 4000, True), packet("b", "q" * 40)], limit=100)
    assert ctx["ok"] is False
    assert ctx["events"][0]["event"] == "packet_refused" and ctx["events"][0]["packet"] == "a"


def test_reserve_leaves_room_for_the_answer():
    ctx, _ = build_context([packet("a", "p" * 400, True)], limit=150, reserve=100)
    assert ctx["ok"] is False  # 100 tokens de paquet > 50 disponibles


def test_history_guard_delegates_to_the_plugin():
    plugin = load_plugin()
    assert plugin is not None
    assert guard_history(plugin, [])["ok"] is True
    polluted = guard_history(plugin, [{"role": "user", "content": "salut"}])
    assert polluted["ok"] is False and polluted["action"] == "ask_new_conversation"
    assert guard_history(None, [])["action"] == "no_plugin"


INFO = {"seed": 3, "model": "m", "prompt_tokens": 200, "context_limit": 12000}
CTX = {"tokens": 200, "loaded": [{"packet": "prompt", "tokens": 200}], "left_out": []}


def make_report(adapter_cls=Honest):
    a, prior, steps = adapter_cls(12000), {}, []
    est = a.plan(INFO)
    for k, name in enumerate(budget.STEP_NAMES, 1):
        out = a.step(k, prior, INFO)
        prior.update(out)
        steps.append(budget.step_record(k, name, est[k - 1], out, CTX))
    return budget.build_report("m", 12000, 3, CTX, steps), a.finalize(prior)


def test_honest_budget_is_exact_and_optimist_underestimates_by_3x():
    honest, _ = make_report(Honest)
    assert honest["gap_pct"] == 0.0
    optimist, _ = make_report(Optimistic)
    assert optimist["gap_pct"] >= 190  # estimé = coût // 3 -> consommé ≈ 3x l'estimé
    assert all(s["gap_pct"] > 100 for s in optimist["steps"])


def test_verbose_exceeds_its_declared_context_limit():
    verbose, _ = make_report(Verbose)
    assert verbose["context_used"] > verbose["declared_context_limit"]
    honest, _ = make_report(Honest)
    assert honest["context_used"] <= honest["declared_context_limit"]


def test_report_carries_per_step_loaded_left_out_and_checkpoint():
    report, _ = make_report()
    for s in report["steps"]:
        assert {"estimated_tokens", "consumed_tokens", "gap_pct", "loaded", "left_out", "output",
                "checkpoint"} <= set(s)
        assert s["checkpoint"]["next_step"] == s["step"] + 1


def test_resume_from_report_alone_gives_identical_result():
    report, final = make_report()
    for n in (1, 2, 3):
        fresh = Honest(12000)  # adaptateur neuf : aucun état hérité
        assert fresh.finalize(budget.resume_output(fresh, copy.deepcopy(report), n)) == final


def test_resume_depends_on_the_report_content():
    report, final = make_report()
    tampered = copy.deepcopy(report)
    tampered["steps"][0]["output"]["formalization"]["operators"].pop()
    assert budget.verify_checkpoints(tampered)  # le hash de reprise détecte l'altération
    fresh = Honest(12000)
    assert fresh.finalize(budget.resume_output(fresh, tampered, 2)) != final


def test_verified_report_has_no_checkpoint_errors():
    report, _ = make_report()
    assert budget.verify_checkpoints(report) == []

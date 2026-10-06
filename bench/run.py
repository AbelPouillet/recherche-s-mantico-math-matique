"""Lanceur du harnais.

  python -m bench.run --models FICHIER --task ID --budget TOKENS --out DIR [--harness ID] [--seed N]

États : DECLARER -> PLANIFIER -> EXECUTER -> VALIDER -> EVALUER -> RAPPORT.
  DECLARER   le harnais vérifie sa version inscrite ; chaque modèle annonce sa limite de contexte
  PLANIFIER  paquets de contexte sous la limite déclarée (garde), estimation de tokens par étape
  EXECUTER   étapes une à une, tokens estimés vs consommés, point de reprise à chaque étape
  VALIDER    sortie JSON stricte (schema.py) + drapeaux limite / budget
  EVALUER    test de reprise à l'étape n par un second adaptateur, puis audit croisé des comptes rendus
  RAPPORT    trace, journal, rapport
Les modèles ne sont appelés qu'à l'intérieur des états. Un point de contrôle (checkpoint.json) est
écrit après chaque état ; --resume repart du dernier état terminé. Aucun réseau, aucune horloge
(la date n'intervient que dans le dossier de sortie par défaut).
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

from . import registry, schema
from .adapters import ADAPTERS
from .budget import STEP_NAMES, build_report, resume_output, step_record, verify_checkpoints
from .consensus import assess, collective, cross_audit
from .context import build_context, guard_history, load_packets
from .trace import TraceBuilder, canonical_json, digest, validate_tree

STATES = ("DECLARER", "PLANIFIER", "EXECUTER", "VALIDER", "EVALUER", "RAPPORT")
PLUGIN_GLOB = "deepseek-r1-*"


def load_plugin(root: Path = registry.ROOT):
    """Charge l'adaptateur du plugin DeepSeek installé (None s'il est absent)."""
    for plugin_dir in sorted((root / "plugins").glob(PLUGIN_GLOB)):
        path = plugin_dir / "adapter.py"
        if path.exists():
            spec = importlib.util.spec_from_file_location("bench_plugin_deepseek", path)
            mod = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = mod  # requis par @dataclass + annotations différées
            spec.loader.exec_module(mod)
            return plugin_dir.name, mod
    return None


def load_models(path: Path) -> tuple[str | None, list[dict]]:
    """Fichier JSON : soit une liste, soit {"harness": id, "models": [...]}.
    Chaque modèle : {"name", "adapter", "context_limit"}."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    harness, entries = (data.get("harness"), data["models"]) if isinstance(data, dict) else (None, data)
    names = set()
    for e in entries:
        missing = [k for k in ("name", "adapter", "context_limit") if k not in e]
        if missing:
            raise SystemExit(f"modèle incomplet {e!r} : clés manquantes {missing}")
        if e["adapter"] not in ADAPTERS:
            raise SystemExit(f"adaptateur inconnu : {e['adapter']!r} (disponibles : {sorted(ADAPTERS)})")
        if not isinstance(e["context_limit"], int) or e["context_limit"] <= 0:
            raise SystemExit(f"{e['name']} : context_limit doit être un entier > 0")
        if e["name"] in names:
            raise SystemExit(f"nom de modèle dupliqué : {e['name']}")
        names.add(e["name"])
    return harness, entries


def _adapter(ctx: dict, name: str):
    entry = next(e for e in ctx["entries"] if e["name"] == name)
    return ADAPTERS[entry["adapter"]](entry["context_limit"])


def _info(ctx: dict, name: str) -> dict:
    return {"seed": ctx["seed"], "model": name, "prompt_tokens": ctx["contexts"][name]["tokens"],
            "context_limit": ctx["declared"][name]}


def _log(ctx: dict, state: str, model: str | None, event: str, **data) -> None:
    ctx["journal"].append({"seq": len(ctx["journal"]), "state": state, "model": model,
                           "event": event, **data})


def _declare(ctx: dict, tb: TraceBuilder) -> None:
    entry = registry.load(ctx["harness"])
    if not registry.verify(ctx["harness"]):
        raise SystemExit(f"{ctx['harness']} : hash du registre incohérent (définition, prompt, doc ou paquets modifiés)")
    ctx["definition"] = entry["definition"]
    if ctx["task"] not in ctx["definition"].get("tasks", {}):
        raise SystemExit(f"tâche inconnue pour {ctx['harness']} : {ctx['task']!r} "
                         f"(disponibles : {sorted(ctx['definition'].get('tasks', {}))})")
    with tb.step("DECLARER", harness=ctx["harness"], content_hash=entry["content_hash"], task=ctx["task"]):
        for e in ctx["entries"]:
            limit = ADAPTERS[e["adapter"]](e["context_limit"]).declare()["context_limit"]
            ctx["declared"][e["name"]] = limit
            _log(ctx, "DECLARER", e["name"], "context_limit_declared", context_limit=limit)
            with tb.step(f"déclaration {e['name']}", context_limit=limit):
                pass


def _plan(ctx: dict, tb: TraceBuilder) -> None:
    with tb.step("PLANIFIER", models=[e["name"] for e in ctx["entries"]], budget=ctx["budget"]):
        plugin = load_plugin()
        if plugin is None:
            tb.plugin_call("absent", "skip", reason="aucun plugin installé")
        else:
            name, mod = plugin
            clean, polluted = mod.on_message([], "go"), mod.on_message([{"role": "user", "content": "x"}], "go")
            tb.plugin_call(name, clean.action, context="vide", reason=clean.reason)
            tb.plugin_call(name, polluted.action, context="pollué", reason=polluted.reason)
            ctx["plugin_guard_ok"] = (clean.action == "run_bench"
                                      and polluted.action == "ask_new_conversation")
        spec = ctx["definition"]["tasks"][ctx["task"]]["packets"]
        packets = load_packets(spec, registry.ROOT)
        for e in ctx["entries"]:
            name = e["name"]
            hist = guard_history(plugin, [])
            _log(ctx, "PLANIFIER", name, "history_checked", **hist)
            context, _text = build_context(packets, ctx["declared"][name], reserve=ctx["budget"])
            ctx["contexts"][name] = context
            for ev in context["events"]:
                _log(ctx, "PLANIFIER", name, **ev)
            with tb.step(f"contexte {name}", ok=context["ok"], tokens=context["tokens"],
                         limit=ctx["declared"][name]):
                if context["ok"] and hist["ok"]:
                    ctx["plans"][name] = _adapter(ctx, name).plan(_info(ctx, name))
                else:
                    _log(ctx, "PLANIFIER", name, "model_not_run", reason="contexte refusé")


def _execute(ctx: dict, tb: TraceBuilder) -> None:
    tol = ctx["definition"]["preferences"]["tolerance_pct"]
    with tb.step("EXECUTER"):
        # ordre de la liste des modèles, pas celui des dict : le point de contrôle trie les clés
        for name in (e["name"] for e in ctx["entries"] if e["name"] in ctx["plans"]):
            adapter, info = _adapter(ctx, name), _info(ctx, name)
            prior, steps = {}, []
            with tb.step(f"modèle {name}"):
                for k, step_name in enumerate(STEP_NAMES, 1):
                    with tb.step(f"étape {k} {step_name}") as node:
                        out = adapter.step(k, prior, info)
                        prior.update(out)
                        rec = step_record(k, step_name, ctx["plans"][name][k - 1], out, ctx["contexts"][name])
                        steps.append(rec)
                        node.data.update(estimated=rec["estimated_tokens"], consumed=rec["consumed_tokens"],
                                         gap_pct=rec["gap_pct"])
                        if abs(rec["gap_pct"]) > tol:
                            _log(ctx, "EXECUTER", name, "budget_gap", step=k, gap_pct=rec["gap_pct"])
            report = build_report(name, ctx["declared"][name], ctx["seed"], ctx["contexts"][name], steps)
            ctx["reports"][name] = report
            ctx["finals"][name] = adapter.finalize(prior)
            if report["context_used"] > report["declared_context_limit"]:
                _log(ctx, "EXECUTER", name, "context_overflow", used=report["context_used"],
                     limit=report["declared_context_limit"])


def _validate(ctx: dict, tb: TraceBuilder) -> None:
    prefs = ctx["definition"]["preferences"]
    with tb.step("VALIDER"):
        for e in ctx["entries"]:
            name = e["name"]
            report = ctx["reports"].get(name)
            errors: list[str] = []
            if report is not None:
                obj, errors = schema.validate(ctx["finals"][name])
                if obj is not None:
                    ctx["outputs"][name] = obj
                for reason in errors:
                    _log(ctx, "VALIDER", name, "schema_rejected", reason=reason)
            a = assess(errors, report, ctx["budget"], prefs["tolerance_pct"])
            ctx["assessments"][name] = a
            if a["flags"]:
                _log(ctx, "VALIDER", name, "model_flagged", flags=a["flags"])
            with tb.step(f"validation {name}", flags=a["flags"], included=a["included"]):
                pass


def _evaluate(ctx: dict, tb: TraceBuilder) -> None:
    prefs = ctx["definition"]["preferences"]
    n = ctx["resume_step"]
    with tb.step("EVALUER"):
        for name in (e["name"] for e in ctx["entries"] if e["name"] in ctx["reports"]):
            report = ctx["reports"][name]
            ok = _resume_ok(ctx, name, report, n)
            ctx["resumes"][name] = ok
            _log(ctx, "EVALUER", name, "resume_checked", from_step=n, identical=ok)
        ctx["audits"] = cross_audit(ctx["reports"], ctx["resumes"], prefs["tolerance_pct"])
        ctx["collective"] = collective(ctx["outputs"], ctx["assessments"], prefs)
        tb.conclusion("conclusion collective", verdict=ctx["collective"]["verdict"],
                      kept=ctx["collective"]["kept"], excluded=ctx["collective"]["excluded"])


def _resume_ok(ctx: dict, name: str, report: dict, n: int) -> bool:
    """Un adaptateur NEUF reprend à l'étape n depuis le seul compte rendu : même texte final ?"""
    adapter = _adapter(ctx, name)
    merged = resume_output(adapter, report, n)
    return not verify_checkpoints(report) and adapter.finalize(merged) == ctx["finals"][name]


def _report(ctx: dict, tb: TraceBuilder) -> None:
    with tb.step("RAPPORT"):
        pass
    tree = tb.to_dict()
    errors = validate_tree(tree)
    if errors:
        raise SystemExit(f"arbre de trace invalide : {errors}")
    ctx["report"] = {
        "harness": ctx["harness"], "task": ctx["task"], "seed": ctx["seed"], "budget": ctx["budget"],
        "models": [e["name"] for e in ctx["entries"]], "plugin_guard_ok": ctx.get("plugin_guard_ok"),
        "declared_limits": ctx["declared"], "assessments": ctx["assessments"],
        "budget_reports": ctx["reports"], "resumes": ctx["resumes"], "audits": ctx["audits"],
        "collective": ctx["collective"], "journal": ctx["journal"], "trace_sha256": digest(tree),
    }
    ctx["trace"] = tree


STEP_FUNCS = dict(zip(STATES, (_declare, _plan, _execute, _validate, _evaluate, _report)))
_FRESH = ("declared", "contexts", "plans", "reports", "finals", "outputs", "assessments", "resumes")


def run(harness: str, entries: list[dict], task: str, budget: int, seed: int, out: Path,
        resume: bool = False, stop_after: str | None = None, resume_step: int = 2) -> dict | None:
    if not 1 <= resume_step <= len(STEP_NAMES):
        raise SystemExit(f"resume_step doit être entre 1 et {len(STEP_NAMES)}")
    out.mkdir(parents=True, exist_ok=True)
    ckpt_path = out / "checkpoint.json"
    ctx = {"harness": harness, "entries": entries, "task": task, "budget": budget, "seed": seed,
           "resume_step": resume_step, "journal": [], "audits": {}, **{k: {} for k in _FRESH}}
    done: list[str] = []
    tb = TraceBuilder(f"bench:{harness}")
    if resume and ckpt_path.exists():
        ckpt = json.loads(ckpt_path.read_text(encoding="utf-8"))
        ctx, done, tb = ckpt["ctx"], ckpt["done"], TraceBuilder.from_dict(ckpt["trace"])
        if "RAPPORT" in done and (out / "report.json").exists():
            # exécution déjà terminée : rien à refaire, on relit le rapport écrit
            return json.loads((out / "report.json").read_text(encoding="utf-8"))
    for state in STATES:
        if state in done:
            continue
        STEP_FUNCS[state](ctx, tb)
        done.append(state)
        ckpt_path.write_text(canonical_json({"ctx": {k: v for k, v in ctx.items() if k not in ("report", "trace")},
                                             "done": done, "trace": tb.to_dict()}), encoding="utf-8")
        if state == stop_after:
            return None
    (out / "trace.json").write_text(json.dumps(ctx["trace"], indent=2, ensure_ascii=False, sort_keys=True),
                                    encoding="utf-8")
    (out / "report.json").write_text(json.dumps(ctx["report"], indent=2, ensure_ascii=False, sort_keys=True),
                                     encoding="utf-8")
    (out / "journal.jsonl").write_text("".join(canonical_json(e) + "\n" for e in ctx["journal"]),
                                       encoding="utf-8")
    registry.record_run(harness, {"seed": seed, "budget": budget, "task": task,
                                  "models": [e["name"] for e in entries],
                                  "verdict": ctx["report"]["collective"]["verdict"],
                                  "trace_sha256": ctx["report"]["trace_sha256"]})
    return ctx["report"]


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="bench.run", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--models", type=Path, help="fichier JSON des modèles {name, adapter, context_limit}")
    p.add_argument("--task", help="identifiant de tâche de la définition du harnais")
    p.add_argument("--budget", type=int, help="budget de tokens consommés par modèle (réponse)")
    p.add_argument("--out", type=Path, default=None, help="défaut : bench/runs/<date du jour>")
    p.add_argument("--harness", help="identifiant <name>-<version> (défaut : clé \"harness\" du fichier modèles)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--resume-step", type=int, default=2, help="étape n du test de reprise (défaut 2)")
    p.add_argument("--resume", action="store_true", help="reprend depuis checkpoint.json de --out")
    p.add_argument("--list", action="store_true", help="liste les harnais enregistrés")
    p.add_argument("--register", type=Path, metavar="DEF.json",
                   help="inscrit une définition (bench/harnesses/*/*/harness.def.json) dans le registre")
    args = p.parse_args(argv)
    if args.register:
        entry = registry.register(json.loads(args.register.read_text(encoding="utf-8")))
        print(f"{registry.harness_id(entry['definition'])} {entry['content_hash']}")
        return 0
    if args.list:
        print("\n".join(registry.list_harnesses()))
        return 0
    for opt in ("models", "task", "budget"):
        if getattr(args, opt) is None:
            p.error(f"--{opt} est requis")
    file_harness, entries = load_models(args.models)
    harness = args.harness or file_harness
    if not harness:
        p.error("--harness est requis (ou une clé \"harness\" dans le fichier modèles)")
    out = args.out
    if out is None:
        import datetime
        out = Path("bench/runs") / datetime.date.today().isoformat()
    report = run(harness, entries, args.task, args.budget, args.seed, out,
                 resume=args.resume, resume_step=args.resume_step)
    print(json.dumps({"collective": report["collective"],
                      "flags": {m: a["flags"] for m, a in report["assessments"].items()},
                      "resumes": report["resumes"]}, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())

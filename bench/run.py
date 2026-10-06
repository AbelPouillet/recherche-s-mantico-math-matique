"""Lanceur du harnais : python -m bench.run --harness <id> --models a,b --budget N --out DIR

États : DECLARER -> PLANIFIER -> EXECUTER -> VALIDER -> EVALUER -> RAPPORT.
Un point de reprise (checkpoint.json) est écrit après chaque état ; `--resume`
repart du dernier état terminé et produit un rapport identique.
Aucun réseau, aucune horloge : le résultat dépend de (harnais, modèles, tâche, budget, graine).
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

from . import registry
from .adapters import ADAPTERS, validate_response
from .consensus import assess, collective, cross_evaluate
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


def _declare(ctx: dict, tb: TraceBuilder) -> None:
    entry = registry.load(ctx["harness"])
    if not registry.verify(ctx["harness"]):
        raise SystemExit(f"{ctx['harness']} : hash du registre incohérent (définition ou prompt modifié)")
    ctx["definition"] = entry["definition"]
    with tb.step("DECLARER", harness=ctx["harness"], content_hash=entry["content_hash"]):
        pass


def _plan(ctx: dict, tb: TraceBuilder) -> None:
    with tb.step("PLANIFIER", models=ctx["models"], budget=ctx["budget"]):
        plugin = load_plugin()
        if plugin is None:
            tb.plugin_call("absent", "skip", reason="aucun plugin installé")
        else:
            name, mod = plugin
            clean = mod.on_message([], "go")
            guarded = mod.on_message([{"role": "user", "content": "x"}], "go")
            tb.plugin_call(name, clean.action, context="vide", reason=clean.reason)
            tb.plugin_call(name, guarded.action, context="pollué", reason=guarded.reason)
            ctx["plugin_guard_ok"] = (clean.action == "run_bench"
                                      and guarded.action == "ask_new_conversation")
    ctx["task"] = {"budget": ctx["budget"], "seed": ctx["seed"]}


def _execute(ctx: dict, tb: TraceBuilder) -> None:
    ctx["raw"] = {}
    with tb.step("EXECUTER"):
        for name in ctx["models"]:
            with tb.step(f"modèle {name}") as node:
                text = ADAPTERS[name](ctx["task"], ctx["seed"])
                ctx["raw"][name] = text
                node.data["chars"] = len(text)


def _validate(ctx: dict, tb: TraceBuilder) -> None:
    prefs = ctx["definition"]["preferences"]
    ctx["parsed"], ctx["assessments"] = {}, {}
    with tb.step("VALIDER"):
        for name in ctx["models"]:
            parsed, errors = validate_response(ctx["raw"][name])
            a = assess(parsed, errors, ctx["budget"], prefs["tolerance_pct"])
            ctx["assessments"][name] = a
            if parsed is not None:
                ctx["parsed"][name] = parsed
            with tb.step(f"validation {name}", flags=a["flags"], included=a["included"]):
                pass


def _evaluate(ctx: dict, tb: TraceBuilder) -> None:
    prefs = ctx["definition"]["preferences"]
    with tb.step("EVALUER"):
        ctx["cross"] = cross_evaluate(ctx["parsed"]) if ctx["parsed"] else {}
        ctx["collective"] = collective(ctx["parsed"], ctx["assessments"],
                                       prefs["correlation_weights"], prefs["h0_threshold"])
        tb.conclusion("conclusion collective", verdict=ctx["collective"]["verdict"],
                      kept=ctx["collective"]["kept"], excluded=ctx["collective"]["excluded"])


def _report(ctx: dict, tb: TraceBuilder) -> None:
    with tb.step("RAPPORT"):
        pass
    tree = tb.to_dict()
    errors = validate_tree(tree)
    if errors:
        raise SystemExit(f"arbre de trace invalide : {errors}")
    ctx["report"] = {
        "harness": ctx["harness"], "seed": ctx["seed"], "budget": ctx["budget"],
        "models": ctx["models"], "plugin_guard_ok": ctx.get("plugin_guard_ok"),
        "assessments": ctx["assessments"], "cross_agreement": ctx["cross"],
        "collective": ctx["collective"], "trace_sha256": digest(tree),
    }
    ctx["trace"] = tree


STEP_FUNCS = dict(zip(STATES, (_declare, _plan, _execute, _validate, _evaluate, _report)))


def run(harness: str, models: list[str], budget: int, seed: int, out: Path,
        resume: bool = False, stop_after: str | None = None) -> dict | None:
    out.mkdir(parents=True, exist_ok=True)
    ckpt_path = out / "checkpoint.json"
    ctx = {"harness": harness, "models": models, "budget": budget, "seed": seed}
    done: list[str] = []
    tb = TraceBuilder(f"bench:{harness}")
    if resume and ckpt_path.exists():
        ckpt = json.loads(ckpt_path.read_text(encoding="utf-8"))
        ctx, done, tb = ckpt["ctx"], ckpt["done"], TraceBuilder.from_dict(ckpt["trace"])
    for state in STATES:
        if state in done:
            continue
        STEP_FUNCS[state](ctx, tb)
        done.append(state)
        ckpt_path.write_text(canonical_json({"ctx": {k: v for k, v in ctx.items() if k not in ("report", "trace")},
                                             "done": done, "trace": tb.to_dict()}), encoding="utf-8")
        if state == stop_after:
            return None
    (out / "trace.json").write_text(json.dumps(ctx["trace"], indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8")
    (out / "report.json").write_text(json.dumps(ctx["report"], indent=2, ensure_ascii=False, sort_keys=True),
                                     encoding="utf-8")
    registry.record_run(harness, {"seed": seed, "budget": budget, "models": models,
                                  "verdict": ctx["report"]["collective"]["verdict"],
                                  "trace_sha256": ctx["report"]["trace_sha256"]})
    return ctx["report"]


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="bench.run", description=__doc__)
    p.add_argument("--harness", help="identifiant <name>-<version> du registre")
    p.add_argument("--models", default="honnete,optimiste,bavard,casse")
    p.add_argument("--budget", type=int, default=600, help="budget de coût mesuré par modèle")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--list", action="store_true", help="liste les harnais enregistrés")
    p.add_argument("--register", type=Path, metavar="DEF.json",
                   help="inscrit une définition (bench/definitions/*.def.json) dans le registre")
    args = p.parse_args(argv)
    if args.register:
        entry = registry.register(json.loads(args.register.read_text(encoding="utf-8")))
        print(f"{registry.harness_id(entry['definition'])} {entry['content_hash']}")
        return 0
    if args.list:
        print("\n".join(registry.list_harnesses()))
        return 0
    if not args.harness:
        p.error("--harness est requis")
    models = [m for m in args.models.split(",") if m]
    unknown = [m for m in models if m not in ADAPTERS]
    if unknown:
        p.error(f"adaptateurs inconnus : {unknown} (disponibles : {sorted(ADAPTERS)})")
    out = args.out or Path("bench/runs") / args.harness
    report = run(args.harness, models, args.budget, args.seed, out, resume=args.resume)
    print(json.dumps(report["collective"], indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())

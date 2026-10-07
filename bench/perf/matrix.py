"""Matrice de mesure de la pipeline d'inférence : description, exécution, agrégation.

Ce que ce module mesure
-----------------------
Pour chaque **cellule** (moteur × axes de configuration) et chaque **charge** (prefill court/long,
décodage court/long), `repetitions` requêtes sont envoyées et l'on relève :

- `prefill_tok_s` : `timings.prompt_per_second` du serveur (traitement du prompt) ;
- `decode_tok_s`  : `timings.predicted_per_second` (génération) ;
- `ttft_s`        : latence au premier token **chronométrée côté client**, en streaming ;
- `total_s`       : durée de la requête ;
- `vram_peak_mib`, `gpu_util_peak_pct`, `ram_min_available_mib` : ressources locales pendant la requête ;
- `cache_n`       : tokens de prompt **réutilisés depuis le cache** — s'il est non nul, le « prefill »
  mesuré n'en est pas un, et la ligne est marquée `replayed_prefill`.

Deux garanties de méthode
-------------------------
1. **Nonce unique par requête** : sans lui, un serveur peut réutiliser le préfixe en cache et le
   prefill mesuré s'effondre sans que rien ne le signale.
2. **Mesure vs rejeu** : `cache_n` est publié, et une ligne dont le prefill a été rejoué est
   comptée à part dans l'agrégat.

Ce que ce module ne fait pas : aucun jugement de qualité. Une cellule rapide n'est pas une cellule
bonne — la qualité se mesure avec le harnais H1, séparément.
"""
from __future__ import annotations

import itertools
import json
import math
import random
import shutil
import socket
import statistics
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

from .client import ChatClient
from .resources import ResourceSampler, host_summary

SCHEMA = "bench-perf-1"
START_TIMEOUT_S = 900

#: Charges de référence. `prompt_tokens` est une **cible** ; le compte réel est relevé par le
#: tokenizer du serveur et publié (`prompt_tokens_real`).
WORKLOADS: dict[str, dict] = {
    "decode-128": {"prompt_tokens": 256, "max_tokens": 128},
    "decode-512": {"prompt_tokens": 512, "max_tokens": 512},
    "prefill-4k": {"prompt_tokens": 4096, "max_tokens": 32},
    "prefill-32k": {"prompt_tokens": 32768, "max_tokens": 32},
}

#: Corpus déterministe servant à fabriquer les prompts de charge (aucun téléchargement).
CORPUS = (
    "Le graphe lexical relie des sous-chaînes partagées entre plusieurs langues ; chaque nœud porte "
    "une valeur numérique, une transcription phonétique et une datation. "
    "Une règle sans source est refusée : le système ne déduit jamais librement une histoire plausible. "
    "La compression d'une représentation n'implique ni moins de calcul, ni moins de mémoire, ni une "
    "meilleure perplexité ; ces quatre quantités se mesurent séparément. "
    "Le prefill traite le prompt, le décodage produit les tokens suivants, et la latence au premier "
    "token est la seule qui décide de la sensation d'attente. "
    "Les experts les plus sollicités résident sur la carte graphique, les autres en mémoire vive, et "
    "la table de correspondance reste sur le disque. "
)

#: Axes reconnus et traduction en options de ligne de commande de `llama-server`.
AXIS_ARGS = {
    "context": lambda v: ["-c", str(v)],
    "kv_type": lambda v: ["-ctk", str(v), "-ctv", str(v)],
    "ngl": lambda v: ["-ngl", str(v)],
    "flash_attn": lambda v: ["-fa", str(v)],
    "batch": lambda v: ["-b", str(v)],
    "ubatch": lambda v: ["-ub", str(v)],
    "n_cpu_moe": lambda v: ["--n-cpu-moe", str(v)],
    "mmap": lambda v: ([] if v else ["--no-mmap"]),
    "draft_model": lambda v: ["-md", str(v)],
    "draft_max": lambda v: ["--draft-max", str(v)],
}

#: Axes qui exigent un **redémarrage** du serveur : sans lancement, ils ne sont pas applicables.
RESTART_AXES = tuple(AXIS_ARGS)

MEASURED_FIELDS = ("prefill_tok_s", "decode_tok_s", "ttft_s", "ttft_answer_s", "total_s",
                   "prompt_tokens_real", "completion_tokens", "cache_n", "reasoning_chars",
                   "answer_chars", "vram_peak_mib", "gpu_util_peak_pct", "gpu_temp_peak_c",
                   "ram_min_available_mib")


def filler_text(target_tokens: int, nonce: str = "") -> str:
    """Texte de charge déterministe d'environ `target_tokens` (estimation 4 caractères/token)."""
    target_chars = max(1, int(target_tokens)) * 4
    reps = target_chars // len(CORPUS) + 1
    return (CORPUS * reps)[:target_chars] + nonce


@dataclass
class Cell:
    """Une cellule de la matrice : un moteur et une combinaison d'axes."""

    engine_id: str
    kind: str                      # "llamacpp" (lance le serveur) ou "host" (serveur déjà lancé)
    params: dict
    axes: dict = field(default_factory=dict)

    @property
    def id(self) -> str:
        if not self.axes:
            return self.engine_id
        suffix = ",".join(f"{k}={self.axes[k]}" for k in sorted(self.axes))
        return f"{self.engine_id}[{suffix}]"

    @property
    def concurrency(self) -> int:
        value = self.axes.get("concurrency", 1)
        return int(value) if isinstance(value, (int, float)) and value >= 1 else 1

    @property
    def launches_server(self) -> bool:
        return self.kind == "llamacpp" and not self.params.get("host")

    def unsupported_axes(self) -> list[str]:
        """Axes non applicables ici : sans lancement, on ne peut pas reconfigurer le serveur."""
        return sorted(self.axes) if (self.axes and not self.launches_server) else []

    def command(self, port: int) -> list[str]:
        """Ligne de commande `llama-server` correspondant à la cellule."""
        exe = self.params.get("server") or shutil.which("llama-server")
        if not exe:
            raise SystemExit("llama-server introuvable (params.server pour un chemin explicite)")
        gguf = self.params.get("gguf")
        if not gguf:
            raise SystemExit(f"{self.engine_id} : params.gguf est requis pour lancer le serveur")
        context = self.axes.get("context", self.params.get("context", 8192))
        ngl = self.axes.get("ngl", self.params.get("ngl", 99))
        cmd = [exe, "-m", str(gguf), "-c", str(context), "-ngl", str(ngl),
               "-np", str(max(self.concurrency, int(self.params.get("np", 1)))),
               "--host", "127.0.0.1", "--port", str(port), "--jinja"]
        for axis in sorted(self.axes):
            if axis in ("context", "ngl", "concurrency") or axis not in AXIS_ARGS:
                continue
            cmd += AXIS_ARGS[axis](self.axes[axis])
        cmd += [str(a) for a in self.params.get("extra_args", [])]
        return cmd


def expand(config: dict) -> list[Cell]:
    """Produit les cellules : produit cartésien des axes, pour chaque moteur."""
    axes = config.get("axes") or {}
    unknown = sorted(set(axes) - set(AXIS_ARGS) - {"concurrency"})
    if unknown:
        raise SystemExit(f"axes inconnus : {unknown} (connus : {sorted(AXIS_ARGS) + ['concurrency']})")
    keys = sorted(axes)
    combos = [dict(zip(keys, values)) for values in
              itertools.product(*(axes[k] for k in keys))] if keys else [{}]
    cells = []
    for engine in config["engines"]:
        if engine.get("kind") not in ("llamacpp", "host"):
            raise SystemExit(f"{engine.get('id')} : kind doit être 'llamacpp' ou 'host'")
        for combo in combos:
            cells.append(Cell(engine_id=engine["id"], kind=engine["kind"],
                              params=engine.get("params", {}), axes=dict(combo)))
    return cells


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class ServerHandle:
    """Serveur prêt à interroger : lancé par la matrice, ou déjà en écoute (`params.host`)."""

    def __init__(self, cell: Cell, out_dir: Path, log_name: str) -> None:
        self.cell = cell
        self.out_dir = out_dir
        self.log_name = log_name
        self.process: subprocess.Popen | None = None
        self.base_url = cell.params.get("host", "")
        self.log_path = out_dir / log_name

    def __enter__(self) -> "ServerHandle":
        if not self.cell.launches_server:
            if not self.base_url:
                raise SystemExit(f"{self.cell.id} : kind=host exige params.host")
            return self
        port = free_port()
        self.base_url = f"http://127.0.0.1:{port}"
        self.out_dir.mkdir(parents=True, exist_ok=True)
        log = self.log_path.open("ab")
        self.process = subprocess.Popen(self.cell.command(port), stdout=log, stderr=subprocess.STDOUT)
        deadline = time.monotonic() + START_TIMEOUT_S
        client = ChatClient(self.base_url, timeout=10.0)
        while time.monotonic() < deadline:
            if self.process.poll() is not None:
                raise SystemExit(f"{self.cell.id} : llama-server s'est arrêté "
                                 f"(code {self.process.returncode}), voir {self.log_path}")
            if client.health():
                log.close()
                return self
            time.sleep(2)
        self.__exit__()
        raise SystemExit(f"{self.cell.id} : démarrage > {START_TIMEOUT_S}s, voir {self.log_path}")

    def __exit__(self, *exc) -> None:
        if self.process is not None:
            self.process.terminate()
            try:
                self.process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()
            self.process = None


def measure_once(client: ChatClient, workload: str, spec: dict, seed: int, concurrency: int = 1,
                 gpu_index: int = 0, extra: dict | None = None) -> dict:
    """Une mesure : `concurrency` requêtes simultanées, ressources échantillonnées pendant.

    `extra` est fusionné dans la requête de chat (par ex. `{"reasoning_effort": "none"}` pour Strata
    ou `{"chat_template_kwargs": {"enable_thinking": false}}` pour llama.cpp). Sans cela, un modèle
    qui « réfléchit » consomme le budget en raisonnement et le débit mesuré n'est pas celui de la
    réponse — constaté sur Strata, dont le défaut est `reasoning_effort: high`.
    """
    prompt = filler_text(spec["prompt_tokens"])
    messages = [{"role": "user", "content": prompt + "\n\nRéponds en continuant le texte."}]
    sampler = ResourceSampler(gpu_index=gpu_index).start()
    started = time.perf_counter()
    try:
        if concurrency <= 1:
            responses = [client.chat(messages, spec["max_tokens"], seed=seed, extra=extra)]
        else:
            with ThreadPoolExecutor(max_workers=concurrency) as pool:
                futures = [pool.submit(client.chat, messages, spec["max_tokens"],
                                       temperature=0.0, seed=seed + i, stream=True,
                                       unique_nonce=True, extra=extra)
                           for i in range(concurrency)]
                responses = [f.result() for f in futures]
    finally:
        wall = time.perf_counter() - started
        peaks = sampler.stop()

    ok = [r for r in responses if r.ok]
    completion = sum(r.completion_tokens or 0 for r in ok)
    cache_n = sum(r.timings.get("cache_n", 0) or 0 for r in ok)
    row = {
        "workload": workload, "concurrency": concurrency, "wall_s": round(wall, 4),
        "requests": len(responses), "requests_ok": len(ok),
        "errors": sorted({r.error for r in responses if r.error}),
        "prompt_tokens_target": spec["prompt_tokens"],
        "prompt_tokens_real": _median([r.prompt_tokens for r in ok]),
        "completion_tokens": completion or None,
        "reasoning_chars": sum(len(r.reasoning) for r in ok) or None,
        "answer_chars": sum(len(r.content) for r in ok) or None,
        "prefill_tok_s": _median([r.rate("prompt_per_second") for r in ok]),
        "decode_tok_s": _median([r.rate("predicted_per_second") for r in ok]),
        "ttft_s": _median([r.ttft_s for r in ok]),
        "ttft_answer_s": _median([r.ttft_answer_s for r in ok]),
        "total_s": round(_median([r.total_s for r in ok]) or wall, 4),
        "cache_n": cache_n,
        "replayed_prefill": bool(cache_n),
        "aggregate_decode_tok_s": round(completion / wall, 2) if completion and wall else None,
        "draft_accepted": sum(r.timings.get("draft_n_accepted", 0) or 0 for r in ok) or None,
        **peaks,
    }
    return row


def _median(values) -> float | None:
    vals = [v for v in values if isinstance(v, (int, float))]
    return round(statistics.median(vals), 4) if vals else None


def bootstrap_ci(values: list[float], resamples: int = 2000, seed: int = 0,
                 alpha: float = 0.05) -> list[float] | None:
    """IC à 95 % de la médiane par bootstrap, graine fixe (reproductible)."""
    vals = [float(v) for v in values if isinstance(v, (int, float))]
    if len(vals) < 3:
        return None
    rng = random.Random(seed)
    n = len(vals)
    medians = sorted(statistics.median(rng.choices(vals, k=n)) for _ in range(resamples))
    lo = medians[max(0, int(math.floor(alpha / 2 * resamples)))]
    hi = medians[min(resamples - 1, int(math.ceil((1 - alpha / 2) * resamples)) - 1)]
    return [round(lo, 4), round(hi, 4)]


def stats(values: list) -> dict | None:
    vals = sorted(float(v) for v in values if isinstance(v, (int, float)))
    if not vals:
        return None
    out = {"n": len(vals), "median": round(statistics.median(vals), 4), "min": vals[0], "max": vals[-1]}
    if len(vals) >= 4:
        out["p25"] = round(statistics.quantiles(vals, n=4)[0], 4)
        out["p75"] = round(statistics.quantiles(vals, n=4)[2], 4)
        mean = statistics.fmean(vals)
        out["cv_pct"] = round(statistics.pstdev(vals) / mean * 100, 2) if mean else None
    out["ci95_median"] = bootstrap_ci(vals)
    return out


def aggregate(rows: list[dict]) -> dict:
    """Agrège les répétitions par charge. Les lignes au prefill rejoué sont comptées à part."""
    by_workload: dict[str, list[dict]] = {}
    for row in rows:
        by_workload.setdefault(row["workload"], []).append(row)
    out = {}
    for workload, group in sorted(by_workload.items()):
        fresh = [r for r in group if not r["replayed_prefill"]]
        entry = {"repetitions": len(group), "repetitions_with_fresh_prefill": len(fresh),
                 "failed": sum(1 for r in group if r["requests_ok"] != r["requests"]),
                 "aggregate_decode_tok_s": stats([r["aggregate_decode_tok_s"] for r in group])}
        for field_name in MEASURED_FIELDS:
            entry[field_name] = stats([r[field_name] for r in group])
        out[workload] = entry
    return out


def run_matrix(config: dict, out_dir: Path, dry_run: bool = False, only_engines: list[str] | None = None,
               only_workloads: list[str] | None = None, repetitions: int | None = None,
               warmups: int | None = None, gpu_index: int = 0,
               progress=print) -> dict:
    """Exécute la matrice et retourne le rapport. `dry_run` n'exécute rien : il rend le plan."""
    if "engines" not in config or not config["engines"]:
        raise SystemExit("config invalide : clé 'engines' vide ou absente")
    workload_names = only_workloads or config.get("workloads") or ["decode-128"]
    unknown = [w for w in workload_names if w not in WORKLOADS]
    if unknown:
        raise SystemExit(f"charges inconnues : {unknown} (connues : {sorted(WORKLOADS)})")
    reps = int(repetitions or config.get("repetitions", 5))
    warm = int(warmups if warmups is not None else config.get("warmups", 1))
    seed = int(config.get("seed", 0))

    cells = expand(config)
    if only_engines:
        cells = [c for c in cells if c.engine_id in only_engines]
    if not cells:
        raise SystemExit("aucune cellule à exécuter")

    plan = [{"cell": c.id, "launches_server": c.launches_server,
             "concurrency": c.concurrency, "unsupported_axes": c.unsupported_axes(),
             # ligne de commande indicative : le port est tiré au sort au lancement réel
             "command": c.command(free_port()) if c.launches_server else None}
            for c in cells]

    report = {"schema": SCHEMA, "config": config, "workloads": {w: WORKLOADS[w] for w in workload_names},
              "repetitions": reps, "warmups": warm, "seed": seed, "host": host_summary(gpu_index),
              "cells": [], "plan": plan,
              "notes": [
                  "ttft_s est chronométré côté client en streaming ; prefill_tok_s vient du serveur.",
                  "Une ligne dont cache_n > 0 a un prefill rejoué : elle n'est pas une mesure de prefill.",
                  "Ce rapport ne contient aucun jugement de qualité : voir le harnais H1 pour cela.",
              ]}
    if dry_run:
        return report

    out_dir.mkdir(parents=True, exist_ok=True)
    for index, cell in enumerate(cells, 1):
        progress(f"[{index}/{len(cells)}] {cell.id}")
        for axis in cell.unsupported_axes():
            progress(f"    axe ignoré (serveur non lancé par la matrice) : {axis}")
        entry = {"cell": cell.id, "engine": cell.engine_id, "kind": cell.kind, "axes": cell.axes,
                 "unsupported_axes": cell.unsupported_axes(), "concurrency": cell.concurrency,
                 "rows": [], "server_props": None, "server_metrics_available": False,
                 "server_log": None}
        log_name = f"server-{cell.id.replace('[', '_').replace(']', '').replace(',', '_')}.log"
        try:
            with ServerHandle(cell, out_dir, log_name) as handle:
                entry["server_log"] = str(handle.log_path) if cell.launches_server else None
                client = ChatClient(handle.base_url)
                entry["server_props"] = client.props()
                entry["server_metrics_available"] = client.metrics() is not None
                for workload in workload_names:
                    spec = WORKLOADS[workload]
                    for _ in range(warm):
                        client.chat([{"role": "user", "content": filler_text(64)}], 8, seed=seed)
                    for rep in range(reps):
                        row = measure_once(client, workload, spec, seed + rep, cell.concurrency,
                                           gpu_index, extra=cell.params.get("extra"))
                        row["repetition"] = rep
                        entry["rows"].append(row)
                        progress(f"    {workload} rep={rep} decode={row['decode_tok_s']} tok/s "
                                 f"ttft={row['ttft_s']} vram={row['vram_peak_mib']} MiB"
                                 + ("  [prefill REJOUÉ]" if row["replayed_prefill"] else "")
                                 + (f"  [raisonnement {row['reasoning_chars']} car.]"
                                    if row["reasoning_chars"] else ""))
        except SystemExit as exc:
            entry["error"] = str(exc)
            progress(f"    ÉCHEC : {exc}")
        entry["measurements"] = aggregate(entry["rows"])
        report["cells"].append(entry)

    report["finished"] = True
    return report


def markdown_table(report: dict) -> str:
    """Tableau récapitulatif lisible : une ligne par cellule × charge."""
    lines = ["| moteur | axes | charge | prefill tok/s | decode tok/s | TTFT s | VRAM crête MiB | n |",
             "|---|---|---|---:|---:|---:|---:|---:|"]
    for cell in report["cells"]:
        axes = ", ".join(f"{k}={v}" for k, v in sorted(cell["axes"].items())) or "—"
        if not cell.get("measurements"):
            lines.append(f"| {cell['engine']} | {axes} | — | — | — | — | — | 0 |")
            continue
        for workload, m in sorted(cell["measurements"].items()):
            def cell_value(key, field_name="median"):
                s = m.get(key)
                return s.get(field_name) if s else None
            ref = m.get("decode_tok_s") or m.get("aggregate_decode_tok_s") or {}
            lines.append(
                f"| {cell['engine']} | {axes} | {workload} "
                f"| {_fmt(cell_value('prefill_tok_s'))} | {_fmt(cell_value('decode_tok_s'))} "
                f"| {_fmt(cell_value('ttft_s'))} | {_fmt(cell_value('vram_peak_mib'), 0)} "
                f"| {ref.get('n', 0)} |")
    return "\n".join(lines)


def _fmt(value, digits: int = 1) -> str:
    return "—" if value is None else f"{value:.{digits}f}"


def write_report(report: dict, out_dir: Path) -> dict:
    """Écrit `report.json` et `SUMMARY.md` ; retourne les chemins."""
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "report.json"
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8")
    md_path = out_dir / "SUMMARY.md"
    md_path.write_text(f"# Mesure de la pipeline d'inférence\n\n{markdown_table(report)}\n",
                       encoding="utf-8")
    return {"report": str(json_path), "summary": str(md_path)}

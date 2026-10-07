"""Auto-réglage continu de l'inférence en fonction des **ressources locales**.

L'idée, en une phrase : la meilleure configuration d'inférence n'est pas une constante, c'est une
fonction de la machine **au moment où l'on infère** — VRAM libre, RAM disponible, température,
pilote, autres applications. Ce module cherche cette configuration et la **maintient**.

Ce qui rend la démarche falsifiable plutôt qu'impressionniste
------------------------------------------------------------
1. **L'incumbent est remesuré dans la même session** que chaque candidat. Sans cela, on compare des
   chiffres obtenus à des heures différentes et on appelle « amélioration » ce qui n'est que de la
   dérive thermique ou un autre processus. C'est la règle la plus importante du module.
2. **Marge pré-déclarée** : un candidat ne remplace l'incumbent que s'il le bat d'au moins
   `margin_pct` sur l'objectif, avec au moins `min_reps` répétitions.
3. **Contraintes de ressources dures** (`Rails`) : une configuration qui dépasse le budget VRAM, la
   température maximale ou qui assèche la RAM est **rejetée**, même si elle est plus rapide. Un débit
   obtenu en écrasant les autres applications n'est pas un gain.
4. **Tout est journalisé** : chaque essai (axes, métriques, violations) part dans `trials.jsonl`, et
   le profil retenu porte son empreinte **et l'enveloppe matérielle** sous laquelle il a été mesuré.

Enveloppe et péremption
-----------------------
Un profil mesuré avec 24 Go de VRAM ne vaut rien sur une machine qui n'en a plus que 12 : le profil
stocke `envelope` (GPU, VRAM totale, RAM totale, plateforme). Si l'enveloppe courante diffère, le
profil est déclaré **périmé** et l'incumbent est remesuré avant toute comparaison.

Mode `watch` — le caractère **continu**
--------------------------------------
`watch()` enchaîne les tours à intervalle régulier : il remesure l'incumbent, retente le voisinage, et
**détecte la dérive** (l'incumbent qui se dégrade sans que rien ne change dans sa configuration signale
un changement d'environnement : throttling, mémoire occupée par un autre programme, pilote mis à jour).
Il ne « s'améliore » pas tout seul : il propose, mesure, et n'accepte que sous condition.
"""
from __future__ import annotations

import json
import random
import statistics
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from ..trace import digest
from . import score as scoring

#: Métriques utilisables comme objectif, avec le sens de lecture naturel.
OBJECTIVES = {
    "decode_tok_s": "max", "prefill_tok_s": "max", "aggregate_decode_tok_s": "max",
    "ttft_s": "min", "total_s": "min",
}

#: Axes que le tuner sait faire varier (mêmes clés que `bench/perf/matrix.AXIS_ARGS`).
TUNABLE_AXES = ("ngl", "context", "kv_type", "batch", "ubatch", "n_cpu_moe", "flash_attn",
                "mmap", "concurrency")


@dataclass
class Rails:
    """Contraintes de ressources. Un dépassement fait rejeter la configuration, pas ralentir."""

    vram_budget_mib: int | None = None
    gpu_temp_max_c: float | None = None
    min_free_ram_mib: int | None = None


@dataclass
class Profile:
    """Configuration retenue, avec ce qui l'a fait retenir et dans quel environnement."""

    engine: str
    axes: dict
    objective: dict
    metrics: dict
    envelope: dict
    n: int
    fingerprint: str = ""

    def to_dict(self) -> dict:
        data = asdict(self)
        if not data["fingerprint"]:
            data["fingerprint"] = digest({k: v for k, v in data.items() if k != "fingerprint"})
        return data

    @classmethod
    def from_dict(cls, data: dict) -> "Profile":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


def envelope(gpu_index: int = 0) -> dict:
    """Identité de l'environnement matériel : ce qui invalide un profil quand ça change."""
    from ..perf.resources import gpu_name, nvidia_smi_path, sample_gpu, system_ram

    gpu = sample_gpu(gpu_index) or {}
    ram = system_ram() or {}
    return {"gpu": gpu_name(gpu_index), "vram_total_mib": gpu.get("vram_total_mib"),
            "ram_total_mib": ram.get("total_mib"), "nvidia_smi": bool(nvidia_smi_path()),
            "platform": __import__("sys").platform}


def violations(rows: list[dict], rails: Rails) -> list[str]:
    """Dépassements constatés sur un lot de répétitions (pics, pas moyennes)."""
    out = []
    if rails.vram_budget_mib is not None:
        peak = max((r.get("vram_peak_mib") or 0) for r in rows) if rows else 0
        if peak and peak > rails.vram_budget_mib:
            out.append(f"VRAM crête {peak:.0f} MiB > budget {rails.vram_budget_mib} MiB")
    if rails.gpu_temp_max_c is not None:
        peak = max((r.get("gpu_temp_peak_c") or 0) for r in rows) if rows else 0
        if peak and peak > rails.gpu_temp_max_c:
            out.append(f"température {peak:.0f} °C > maximum {rails.gpu_temp_max_c} °C")
    if rails.min_free_ram_mib is not None:
        low = min((r.get("ram_min_available_mib") or 10 ** 9) for r in rows) if rows else 10 ** 9
        if low < rails.min_free_ram_mib:
            out.append(f"RAM libre minimale {low:.0f} MiB < plancher {rails.min_free_ram_mib} MiB")
    return out


def objective_value(rows: list[dict], metric: str) -> float | None:
    """Médiane de la métrique objectif sur les répétitions **au prefill non rejoué**."""
    fresh = [r for r in rows if not r.get("replayed_prefill")]
    values = [r.get(metric) for r in (fresh or rows)]
    values = [float(v) for v in values if isinstance(v, (int, float))]
    return round(statistics.median(values), 4) if values else None


def is_better(candidate: float | None, incumbent: float | None, direction: str,
              margin_pct: float) -> tuple[bool, float | None]:
    """Le candidat bat-il l'incumbent au-delà de la marge ? Retourne `(décision, écart %)`."""
    if candidate is None or incumbent in (None, 0):
        return False, None
    delta = (candidate - incumbent) / abs(incumbent) * 100
    if direction == "min":
        delta = -delta
    return delta >= margin_pct, round(delta, 2)


class Tuner:
    """Cherche et maintient la meilleure configuration sous contraintes de ressources locales.

    `measure(axes, repeats) -> list[dict]` est injectable : c'est ce qui rend la logique de décision
    testable sans GPU. La fabrique par défaut s'appuie sur `bench/perf`.
    """

    def __init__(self, base_config: dict, space: dict[str, list], workload: str, metric: str,
                 rails: Rails | None = None, margin_pct: float = 5.0, min_reps: int = 3,
                 seed: int = 0, out_dir: Path | None = None, measure=None,
                 gpu_index: int = 0) -> None:
        if metric not in OBJECTIVES:
            raise SystemExit(f"objectif inconnu : {metric!r} (connus : {sorted(OBJECTIVES)})")
        unknown = sorted(set(space) - set(TUNABLE_AXES))
        if unknown:
            raise SystemExit(f"axes non réglables : {unknown} (réglables : {list(TUNABLE_AXES)})")
        for axis, values in space.items():
            if not isinstance(values, list) or not values:
                raise SystemExit(f"axe {axis} : liste de valeurs admissible requise")
        if not base_config.get("engines"):
            raise SystemExit("base_config doit contenir au moins un moteur")
        if base_config["engines"][0].get("kind") != "llamacpp" or \
                base_config["engines"][0].get("params", {}).get("host"):
            raise SystemExit("l'auto-réglage exige un moteur 'llamacpp' que la matrice peut lancer "
                             "(sans params.host) : sinon aucun axe n'est applicable")
        self.base_config = base_config
        self.space = {k: list(v) for k, v in space.items()}
        self.workload = workload
        self.metric = metric
        self.direction = OBJECTIVES[metric]
        self.rails = rails or Rails()
        self.margin_pct = margin_pct
        self.min_reps = min_reps
        self.seed = seed
        self.gpu_index = gpu_index
        self.out_dir = Path(out_dir) if out_dir else Path("artifacts") / "tune"
        self._measure = measure
        self.trials: list[dict] = []

    # ------------------------------------------------------------------ mesure

    def measure(self, axes: dict, repeats: int) -> list[dict]:
        if self._measure is not None:
            return self._measure(axes, repeats)
        return measure_with_perf(self.base_config, axes, self.workload, repeats, self.out_dir,
                                 self.gpu_index)

    def evaluate(self, axes: dict, repeats: int | None = None, label: str = "essai") -> dict:
        """Mesure une configuration et rend sa valeur objectif, ses métriques et ses violations."""
        repeats = repeats or self.min_reps
        started = time.perf_counter()
        try:
            rows = self.measure(axes, repeats)
        except SystemExit as exc:  # configuration impossible (OOM, démarrage raté)
            trial = {"label": label, "axes": axes, "ok": False, "reason": str(exc),
                     "violations": ["configuration impossible"], "rows": 0}
            self.trials.append(trial)
            return trial
        elapsed = time.perf_counter() - started
        violations_found = violations(rows, self.rails)
        value = objective_value(rows, self.metric)
        trial = {
            "label": label, "axes": axes, "ok": value is not None and not violations_found,
            "value": value, "metric": self.metric, "direction": self.direction,
            "n": len(rows), "n_fresh": sum(1 for r in rows if not r.get("replayed_prefill")),
            "violations": violations_found, "elapsed_s": round(elapsed, 2),
            "metrics": {k: objective_value(rows, k) for k in
                        ("decode_tok_s", "prefill_tok_s", "aggregate_decode_tok_s", "ttft_s",
                         "vram_peak_mib", "gpu_temp_peak_c")},
        }
        self.trials.append(trial)
        return trial

    # ------------------------------------------------------------------ espace de recherche

    def neighbourhood(self, incumbent: dict) -> list[dict]:
        """Voisinage déterministe : un pas par axe, plus une exploration tirée au sort."""
        rng = random.Random(f"{self.seed}:{json.dumps(incumbent, sort_keys=True)}")
        proposals: list[dict] = []
        for axis, values in sorted(self.space.items()):
            current = incumbent.get(axis)
            if current in values:
                index = values.index(current)
                for neighbour in (index - 1, index + 1):
                    if 0 <= neighbour < len(values):
                        proposals.append({**incumbent, axis: values[neighbour]})
            else:
                for value in values:  # axe absent de l'incumbent : on l'essaie en entier
                    proposals.append({**incumbent, axis: value})
        draws = max(1, len(proposals) // 4)
        for _ in range(draws):
            proposal = {**incumbent, **{axis: rng.choice(values)
                                        for axis, values in sorted(self.space.items()) if values}}
            proposals.append(proposal)
        unique: list[dict] = []
        for proposal in proposals:
            if proposal != incumbent and proposal not in unique:
                unique.append(proposal)
        return unique

    # ------------------------------------------------------------------ profils

    @property
    def profile_path(self) -> Path:
        return self.out_dir / "profile.json"

    def load_profile(self) -> dict | None:
        if not self.profile_path.exists():
            return None
        return json.loads(self.profile_path.read_text(encoding="utf-8"))

    def staleness(self, profile: dict | None) -> dict:
        """Le profil correspond-il encore à la machine ? `vram_total_mib` et `gpu` font foi."""
        if not profile:
            return {"stale": True, "reason": "aucun profil"}
        current = envelope(self.gpu_index)
        stored = profile.get("envelope") or {}
        for key in ("gpu", "vram_total_mib", "platform"):
            if stored.get(key) != current.get(key):
                return {"stale": True,
                        "reason": f"enveloppe changée : {key} {stored.get(key)!r} → {current.get(key)!r}",
                        "current": current, "stored": stored}
        return {"stale": False, "reason": "enveloppe identique", "current": current}

    def _promote(self, axes: dict, trial: dict) -> dict:
        profile = Profile(engine=self.base_config["engines"][0]["id"], axes=axes,
                          objective={"metric": self.metric, "direction": self.direction,
                                     "workload": self.workload},
                          metrics=trial["metrics"], envelope=envelope(self.gpu_index),
                          n=trial["n"]).to_dict()
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.profile_path.write_text(json.dumps(profile, indent=2, ensure_ascii=False, sort_keys=True),
                                     encoding="utf-8")
        return profile

    # ------------------------------------------------------------------ un tour

    def run_round(self, incumbent_axes: dict | None = None, max_trials: int | None = None,
                  progress=print) -> dict:
        """Un tour complet : remesure l'incumbent, essaie son voisinage, promeut au plus un candidat."""
        profile = self.load_profile()
        state = self.staleness(profile)
        if incumbent_axes is None:
            incumbent_axes = (profile or {}).get("axes") or self._default_axes()
        budget = max_trials if max_trials is not None else max(1, len(self.space)) * 2

        progress(f"incumbent : {incumbent_axes or 'aucun'}")
        incumbent = self.evaluate(incumbent_axes, label="incumbent") if incumbent_axes else None
        if incumbent is None or not incumbent.get("ok"):
            return {"decision": "aucun incumbent mesurable",
                    "incumbent": incumbent, "stale": state,
                    "reason": (incumbent or {}).get("violations") or "mesure impossible"}

        trials, promoted = [], None
        for index, proposal in enumerate(self.neighbourhood(incumbent_axes)[:budget], 1):
            trial = self.evaluate(proposal, label=f"candidat-{index}")
            better, delta = is_better(trial.get("value"), incumbent.get("value"), self.direction,
                                      self.margin_pct)
            trial["delta_pct"] = delta
            trial["passee"] = trial["ok"] and better
            trials.append(trial)
            progress(f"  candidat-{index} {proposal} → {trial.get('value')} "
                     f"({delta if delta is None else f'{delta:+.2f}%'})"
                     + (f" REJETÉ : {'; '.join(trial['violations'])}" if trial["violations"] else ""))
            if trial["passee"]:
                promoted = self._promote(proposal, trial)
                progress(f"  → promu : {proposal}")
                break
        if promoted is None:
            self._promote(incumbent_axes, incumbent)
        decision = ("promouvoir" if promoted else "conserver l'incumbent")
        return {"decision": decision, "stale": state, "incumbent": incumbent, "trials": trials,
                "promoted": promoted, "profile": str(self.profile_path)}

    def _default_axes(self) -> dict:
        engine = self.base_config["engines"][0].get("params", {})
        axes = {}
        if "ngl" in self.space:
            axes["ngl"] = engine.get("ngl", 99)
        if "context" in self.space:
            axes["context"] = engine.get("context", 8192)
        for axis, values in self.space.items():
            axes.setdefault(axis, values[0])
        return axes

    # ------------------------------------------------------------------ mode continu

    def watch(self, rounds: int = 0, interval_s: float = 900.0, max_trials: int | None = None,
              sleep=time.sleep, progress=print) -> list[dict]:
        """Tours successifs. `rounds=0` : jusqu'à interruption (Ctrl-C).

        La dérive est détectée en comparant la valeur de l'incumbent d'un tour au précédent : sa
        configuration n'a pas changé, donc une variation nette vient de l'environnement.
        """
        history: list[dict] = []
        previous_value: float | None = None
        index = 0
        while rounds == 0 or index < rounds:
            index += 1
            progress(f"\n=== tour {index} ===")
            result = self.run_round(max_trials=max_trials, progress=progress)
            incumbent = result.get("incumbent") or {}
            value = incumbent.get("value")
            drift = None
            if previous_value not in (None, 0) and value is not None:
                drift = round((value - previous_value) / abs(previous_value) * 100, 2)
                if abs(drift) >= self.margin_pct:
                    progress(f"⚠ dérive de {drift:+.2f}% sur l'incumbent inchangé : "
                             f"l'environnement a bougé (throttling, mémoire prise par un autre "
                             f"programme, pilote)")
            result["drift_pct"] = drift
            history.append(result)
            previous_value = value
            self._log(result, index)
            if rounds == 0 or index < rounds:
                sleep(interval_s)
        return history

    def _log(self, result: dict, index: int) -> None:
        self.out_dir.mkdir(parents=True, exist_ok=True)
        with (self.out_dir / "trials.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"round": index, "decision": result["decision"],
                                     "drift_pct": result.get("drift_pct"),
                                     "incumbent": result.get("incumbent"),
                                     "trials": result.get("trials")}, ensure_ascii=False) + "\n")


def measure_with_perf(base_config: dict, axes: dict, workload: str, repeats: int, out_dir: Path,
                      gpu_index: int = 0) -> list[dict]:
    """Mesure réelle d'une configuration, via `bench/perf`. Injectée par défaut dans `Tuner`."""
    from ..perf import matrix

    config = {**base_config,
              "axes": {k: [v] for k, v in sorted(axes.items())},
              "workloads": [workload], "repetitions": repeats, "warmups": 1}
    report = matrix.run_matrix(config, out_dir / "runs", gpu_index=gpu_index)
    cell = report["cells"][0]
    if cell.get("error"):
        raise SystemExit(cell["error"])
    return cell["rows"]


def dataset_guard(records: list) -> dict:
    """Rappel explicite, à appeler avant toute décision : ce qui peut servir de vérité terrain."""
    aggregate = scoring.aggregate(records)
    excluded = sorted({k for r in records for k in r.self_assessment})
    return {"aggregate": aggregate,
            "auto_evaluations_exclues": excluded,
            "regle": "seules les issues externes scorrent ; les champs d'auto-évaluation sont exclus",
            "hors_contrat": sorted({f for r in records for f in r.ignored_fields})}

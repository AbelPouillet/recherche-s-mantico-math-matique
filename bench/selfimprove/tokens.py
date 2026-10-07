"""Chiffrage des tokens : lecture des journaux de session DSH — **format vérifié par décodage réel**.

Ce module existe parce que la boucle d'auto-amélioration exigeait des tokens qu'aucune source ne
fournissait : `score.py` demandait un fichier d'issues où l'appelant écrivait `tokens` à la main, et
je refusais d'inventer un format de session DSH que je n'avais pas vérifié. Le format est maintenant
établi, sur une session réelle de 1,67 Mo :

    <DSH_HOME>/sessions/<slug-du-cwd>/<session-id>/session.v4.jsonl.zstd

- **Frames zstd CONCATÉNÉES** : le fichier observé contient **909 frames** pour 6 646 603 octets
  décompressés. Les API zstd usuelles s'arrêtent à la première frame — et c'est précisément ce que
  documente `@zoytown/dsh-token` pour expliquer que les API zstd de Node ne savent pas lire ces
  journaux. Il faut donc lire **au-delà** de la première frame (`read_across_frames`).
- **JSONL** `{"type": …, "seq": …, "time": …, "data": …}`, plus une ligne d'en-tête (`id`, `cwd`,
  `createdAt`, `version`).
- **Compteurs** sur les événements `assistant/message`, dans `data.usage` :
  `inputTokens`, `outputTokens`, `cacheReadTokens`, `cacheWriteTokens`, `totalTokens`.
  Ces quatre compartiments sont **disjoints** (l'entrée n'inclut pas la lecture de cache), donc leur
  somme est le débit réel.
- **Déduplication par `(turn, step)`** : un même pas émet deux rapports d'usage (un pendant le
  streaming, un sur le message final). On **remplace**, on n'accumule jamais — accumuler doublerait
  le total, ce que le README du plugin signale explicitement.

Rien ici n'écrit dans un journal, ne contacte le réseau, ni ne monte quoi que ce soit dans DSH :
c'est un lecteur en lecture seule. Le plugin `@zoytown/dsh-token` reste la partie **interface**
(page Settings, carte de chaleur), et son installation est une décision séparée — voir
`docs/TOKENS.md`.
"""
from __future__ import annotations

import io
import json
from dataclasses import dataclass, field
from pathlib import Path

#: Compartiments disjoints d'un rapport d'usage.
BUCKETS = ("inputTokens", "outputTokens", "cacheReadTokens", "cacheWriteTokens")
USAGE_EVENT = "assistant/message"
LOG_NAME = "session.v4.jsonl.zstd"
ZSTD_MAGIC = b"\x28\xb5\x2f\xfd"

#: Un journal peut être en cours d'écriture : borner la lecture évite de dépendre d'une fin propre.
MAX_LOG_BYTES = 512 * 1024 * 1024


class SessionLogError(RuntimeError):
    """Journal illisible : décodeur absent, fichier corrompu ou tronqué, ou trop gros."""


def available() -> tuple[bool, str]:
    """Le décodeur zstd est-il présent ? Sinon, dire précisément quoi installer."""
    try:
        import zstandard  # noqa: F401
    except ImportError:
        return False, ("zstandard absent : `pip install -e \".[tokens]\"`. Les journaux DSH sont des "
                       "frames zstd concaténées ; la bibliothèque standard de Python 3.11 ne les lit pas.")
    return True, "zstandard présent"


def default_homes() -> list[Path]:
    """Tous les « dsh home » de la machine, comme le fait le plugin."""
    home = Path.home()
    homes = [home / ".dsh"]
    desktop = home / ".dsh_desktop"
    if desktop.is_dir():
        homes += sorted(p for p in desktop.iterdir() if p.is_dir())
    return [h for h in homes if h.is_dir()]


def session_logs(homes: list[Path] | None = None) -> list[Path]:
    """Journaux de session, triés du plus ancien au plus récent."""
    out: list[Path] = []
    for home in (homes or default_homes()):
        out += list((home / "sessions").rglob(LOG_NAME))
    return sorted(out, key=lambda p: p.stat().st_mtime if p.exists() else 0)


def decode_frames(path: Path) -> bytes:
    """Décode **toutes** les frames zstd concaténées d'un journal."""
    ok, reason = available()
    if not ok:
        raise SessionLogError(reason)
    import zstandard

    raw = path.read_bytes()
    if len(raw) > MAX_LOG_BYTES:
        raise SessionLogError(f"journal trop gros : {len(raw)} octets (limite {MAX_LOG_BYTES})")
    out = bytearray()
    try:
        with zstandard.ZstdDecompressor().stream_reader(io.BytesIO(raw),
                                                        read_across_frames=True) as reader:
            while True:
                chunk = reader.read(1 << 20)
                if not chunk:
                    break
                out += chunk
    except zstandard.ZstdError as exc:
        raise SessionLogError(f"frames zstd illisibles ({exc})") from exc
    return bytes(out)


def frame_count(path: Path) -> int:
    """Nombre de frames zstd : sert à prouver que la lecture va bien au-delà de la première."""
    return path.read_bytes().count(ZSTD_MAGIC)


def iter_json_objects(text: str):
    """Rend les objets JSON d'un document, **qu'ils soient séparés par des sauts de ligne ou collés**.

    Les journaux DSH sont du JSONL, mais une frontière de frame zstd peut tomber au milieu d'une
    ligne, et rien ne garantit le saut de ligne final. Un `splitlines()` + `json.loads` par ligne
    rendait alors **zéro** événement en silence — précisément le type d'échec muet que ce projet
    s'interdit. On avance donc avec `raw_decode`, et une zone illisible est sautée jusqu'au prochain
    saut de ligne sans perdre le reste du journal.
    """
    decoder = json.JSONDecoder()
    index, length = 0, len(text)
    while index < length:
        while index < length and text[index] in " \t\r\n":
            index += 1
        if index >= length:
            return
        try:
            obj, index = decoder.raw_decode(text, index)
        except json.JSONDecodeError:
            newline = text.find("\n", index)
            if newline == -1:
                return
            index = newline + 1
            continue
        if isinstance(obj, dict):
            yield obj


def read_events(path: Path) -> list[dict]:
    """Événements du journal, dans l'ordre.

    Un journal **non vide** dont aucun objet n'est lisible lève une erreur : mieux vaut un échec
    explicite qu'un chiffrage à zéro qui passerait pour une absence d'activité.
    """
    document = decode_frames(path).decode("utf-8", "replace")
    events = list(iter_json_objects(document))
    if not events and document.strip():
        raise SessionLogError(
            f"{path.name} : {len(document)} octets décodés mais aucun événement JSON lisible")
    return events


@dataclass
class Usage:
    """Usage d'une session : compteurs, modèle, et ce qui n'a pas pu être attribué."""

    session_id: str
    log: str
    cwd: str | None = None
    model: str | None = None
    provider: str | None = None
    steps: int = 0                 # nombre de (turn, step) distincts portant un usage
    duplicates_replaced: int = 0   # rapports d'usage remplacés (streaming puis message final)
    buckets: dict = field(default_factory=dict)
    events: int = 0

    @property
    def total(self) -> int:
        return sum(self.buckets.get(k, 0) for k in BUCKETS)

    def to_dict(self) -> dict:
        return {"session_id": self.session_id, "log": self.log, "cwd": self.cwd, "model": self.model,
                "provider": self.provider, "steps": self.steps,
                "duplicates_replaced": self.duplicates_replaced, "events": self.events,
                "buckets": {k: self.buckets.get(k, 0) for k in BUCKETS},
                "totalTokens": self.total}


def usage_by_step(events: list[dict]) -> dict[tuple, dict]:
    """Rapports d'usage indexés par `(turn, step)`, **remplacés** et non additionnés.

    DSH émet deux rapports par pas : un pendant le streaming, un sur le message final. Sommer les
    deux doublerait chaque pas.
    """
    seen: dict[tuple, dict] = {}
    for event in events:
        if event.get("type") != USAGE_EVENT:
            continue
        data = event.get("data") or {}
        usage = data.get("usage")
        if isinstance(usage, dict):
            seen[(data.get("turn"), data.get("step"))] = usage
    return seen


def session_usage(path: Path, events: list[dict] | None = None) -> Usage:
    """Usage d'un journal. Les rapports sont indexés par `(turn, step)` et **remplacés**."""
    events = read_events(path) if events is None else events
    session_id, cwd = path.parent.name, None
    model = provider = None
    for event in events:
        data = event.get("data") or {}
        if event.get("type") == "session":
            session_id = str(data.get("id") or session_id)
            cwd = data.get("cwd") or cwd
        elif event.get("type") == "request/header":
            config = ((data.get("header") or {}).get("config") or {})
            model = config.get("model") or model
            provider = config.get("provider") or provider

    seen = usage_by_step(events)
    usage_reports = sum(1 for e in events if e.get("type") == USAGE_EVENT)
    buckets = {k: 0 for k in BUCKETS}
    for usage in seen.values():
        for key in BUCKETS:
            value = usage.get(key)
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                buckets[key] += int(value)
    return Usage(session_id=session_id, log=str(path), cwd=cwd, model=model, provider=provider,
                 steps=len(seen), duplicates_replaced=max(0, usage_reports - len(seen)),
                 buckets=buckets, events=len(events))


def machine_usage(homes: list[Path] | None = None, progress=None) -> dict:
    """Chiffrage de la machine entière : par session, par modèle, et totaux."""
    ok, reason = available()
    if not ok:
        return {"available": False, "reason": reason, "sessions": [], "by_model": {}, "totals": {}}
    logs = session_logs(homes)
    sessions = []
    for index, log in enumerate(logs, 1):
        try:
            usage = session_usage(log)
        except (SessionLogError, OSError) as exc:    # journal corrompu ou tronqué : consigné, pas fatal
            sessions.append({"log": str(log), "error": str(exc)})
            continue
        sessions.append(usage.to_dict())
        if progress:
            progress(f"[{index}/{len(logs)}] {usage.session_id} "
                     f"{usage.steps} pas {usage.total} tokens")
    by_model: dict[str, dict] = {}
    for entry in sessions:
        if entry.get("error"):
            continue
        model = entry.get("model") or "inconnu"
        acc = by_model.setdefault(model, {"sessions": 0, **{k: 0 for k in BUCKETS}})
        acc["sessions"] += 1
        for key in BUCKETS:
            acc[key] += entry["buckets"].get(key, 0)
    for acc in by_model.values():
        acc["totalTokens"] = sum(acc[k] for k in BUCKETS)
    totals = {k: sum(e["buckets"].get(k, 0) for e in sessions if not e.get("error"))
              for k in BUCKETS}
    totals["totalTokens"] = sum(totals.values())
    totals["sessions"] = sum(1 for e in sessions if not e.get("error"))
    totals["steps"] = sum(e.get("steps", 0) for e in sessions if not e.get("error"))
    totals["duplicates_replaced"] = sum(e.get("duplicates_replaced", 0) for e in sessions
                                        if not e.get("error"))
    totals["cache_read_share_pct"] = (
        round(totals["cacheReadTokens"] / totals["totalTokens"] * 100, 2) if totals["totalTokens"] else 0.0)
    return {"available": True, "homes": [str(h) for h in (homes or default_homes())],
            "sessions": sorted(sessions, key=lambda e: -e.get("totalTokens", 0)),
            "by_model": dict(sorted(by_model.items(), key=lambda kv: -kv[1]["totalTokens"])),
            "totals": totals,
            "note": ("compartiments disjoints : totalTokens = entrée + sortie + lecture de cache + "
                     "écriture de cache. Le plugin @zoytown/dsh-token affiche les mêmes chiffres "
                     "dans Settings ; aucun coût monétaire n'y est associé, DSH ne stocke pas de "
                     "table de prix.")}


def tokens_by_turn(path: Path, events: list[dict] | None = None) -> dict[str, dict]:
    """Compteurs **cumulés par tour**, clé `"<sessionId>:turn<N>"`.

    Un tour compte plusieurs pas, chacun avec son propre rapport d'usage : on somme les pas d'un même
    tour (après déduplication) pour obtenir ce que le tour a réellement coûté.
    """
    events = read_events(path) if events is None else events
    usage = session_usage(path, events)
    per_turn: dict[str, dict] = {}
    for (turn, step), report in sorted(usage_by_step(events).items(), key=lambda kv: (kv[0][0] or 0, kv[0][1] or 0)):
        key = f"{usage.session_id}:turn{turn}"
        acc = per_turn.setdefault(key, {"session_id": usage.session_id, "turn": turn, "steps": 0,
                                        **{k: 0 for k in BUCKETS}, "totalTokens": 0})
        acc["steps"] += 1
        acc["last_step"] = step
        for name in BUCKETS:
            value = report.get(name)
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                acc[name] += int(value)
        total = report.get("totalTokens")
        acc["totalTokens"] += (int(total) if isinstance(total, (int, float)) else 0)
    return per_turn


def enrich(records: list, log: Path | None = None, homes: list[Path] | None = None) -> dict:
    """Remplit `outcome['tokens']` depuis les journaux **quand l'issue ne le fournit pas**.

    Les tokens mesurés par DSH sont une meilleure source que ceux qu'un appelant recopierait à la
    main. Rien n'est écrasé : une valeur fournie par l'issue est conservée, et la provenance est
    publiée (`tokens_source`) pour que personne ne confonde mesure et saisie.
    """
    ok, reason = available()
    if not ok:
        return {"enriched": 0, "skipped": len(records), "available": False, "reason": reason}
    by_turn: dict[str, dict] = {}
    sources = []
    for path in ([log] if log else session_logs(homes)):
        try:
            by_turn.update(tokens_by_turn(path))
            sources.append(str(path))
        except (SessionLogError, OSError):
            continue
    enriched = skipped = unscored = 0
    for record in records:
        outcome = record.outcome
        if not outcome:
            # sans issue, il n'y a rien à scorer : compter ces tours évite de les faire disparaître
            unscored += 1
            continue
        if outcome.get("tokens") or outcome.get("completion_tokens"):
            outcome.setdefault("tokens_source", "issue fournie")
            skipped += 1
            continue
        match = by_turn.get(record.turn_id)
        if match:
            outcome["tokens"] = match["totalTokens"]
            outcome["tokens_source"] = f"journal DSH ({(match['session_id'])})"
            enriched += 1
        else:
            skipped += 1
    return {"enriched": enriched, "skipped": skipped, "unscored": unscored, "available": True,
            "sources": sources, "by_turn": len(by_turn),
            "reason": "clé attendue : « <sessionId>:turn<N> » (le plugin DSH écrit `turn`)"}

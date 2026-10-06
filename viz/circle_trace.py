#!/usr/bin/env python3
"""Tracé circulaire : les n lettres de l'alphabet sont placées à égale distance sur un cercle
(angle 2πk/n) avec n rayons ; une entrée (texte, phonèmes IPA, transcription ASR) produit un
parcours lettre -> lettre, un cadre par préfixe. Sortie : GIF animé, ou tableaux numpy.

Exemple :
    python viz/circle_trace.py --lang fr --text "écureuil" --ipa auto --out out.gif

Ce module ne prétend rien apprendre : il produit une représentation visuelle déterministe.
Voir README.md, section « Hypothèses à tester ».
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import NamedTuple, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from embedbabel.gematria import Alphabet, get_alphabet, load_config  # noqa: E402
from embedbabel.textnorm import ipa_syllables, ipa_tokens  # noqa: E402

EXTENT = 1.0  # rayon du cercle ; les étiquettes sont dessinées à 1.12


# ------------------------------------------------------------------ géométrie
def letter_angles(n: int) -> np.ndarray:
    """Angles 2πk/n, k = 0..n-1 (radians, depuis l'axe +x, sens trigonométrique)."""
    if n < 1:
        raise ValueError("n doit être >= 1")
    return 2.0 * math.pi * np.arange(n) / n


def letter_positions(n: int, radius: float = EXTENT) -> np.ndarray:
    """Positions (n, 2) des n lettres, à égale distance sur le cercle."""
    a = letter_angles(n)
    return radius * np.stack([np.cos(a), np.sin(a)], axis=1)


def ray_segments(n: int, radius: float = EXTENT) -> np.ndarray:
    """n rayons : tableau (n, 2, 2) de segments centre -> lettre."""
    pts = letter_positions(n, radius)
    return np.stack([np.zeros_like(pts), pts], axis=1)


def trajectory_indices(units: Sequence[str], letters: Sequence[str]) -> list[int]:
    """Indices (positions sur le cercle) des unités ; unités hors alphabet ignorées."""
    idx = {c: i for i, c in enumerate(letters)}
    return [idx[u] for u in units if u in idx]


def trajectory_points(units: Sequence[str], letters: Sequence[str], radius: float = EXTENT) -> np.ndarray:
    pos = letter_positions(len(letters), radius)
    ids = trajectory_indices(units, letters)
    return pos[ids] if ids else np.zeros((0, 2))


def segments_of(points: np.ndarray) -> np.ndarray:
    """Segments consécutifs (m-1, 2, 2) d'une trajectoire de m points."""
    if len(points) < 2:
        return np.zeros((0, 2, 2))
    return np.stack([points[:-1], points[1:]], axis=1)


def prefix_units(units: Sequence[str]) -> list[list[str]]:
    """Un parcours par préfixe : units[:1], units[:2], ..., units."""
    return [list(units[:i]) for i in range(1, len(units) + 1)]


# --------------------------------------------------------- rasterisation / tableaux
class TraceArray(NamedTuple):
    image: np.ndarray                    # (size, size) float32 : 0 = fond ; trait d'autant plus clair qu'il est récent
    trajectory: np.ndarray               # (m, 2) float, coordonnées sur le cercle unité
    indices: np.ndarray                  # (m,) int, position de chaque lettre sur le cercle
    gematria_image: np.ndarray | None    # (size, size) float32 : valeur normalisée du segment, ou None


def _to_pixels(points: np.ndarray, size: int) -> np.ndarray:
    lim = EXTENT * 1.05
    px = (points[:, 0] + lim) / (2 * lim) * (size - 1)
    py = (lim - points[:, 1]) / (2 * lim) * (size - 1)
    return np.stack([px, py], axis=1)


def trace_to_array(units: Sequence[str], letters: Sequence[str], size: int = 128,
                   values: Sequence[float] | None = None) -> TraceArray:
    """Image et vecteurs de trajectoire d'un parcours complet (entrée possible d'un modèle de vision).

    `values` : valeur (ex. gématrique) de chaque lettre de `letters`, normalisée par son maximum
    pour remplir `gematria_image`. Les temps sont encodés par l'intensité (dernier segment = 1).
    """
    ids = trajectory_indices(units, letters)
    pts = trajectory_points(units, letters)
    img = np.zeros((size, size), dtype=np.float32)
    gimg = np.zeros((size, size), dtype=np.float32) if values is not None else None
    vmax = float(max(values)) if values is not None and len(values) else 1.0
    if len(pts):
        pix = _to_pixels(pts, size)
        n_seg = max(len(pts) - 1, 1)
        if len(pts) == 1:
            x, y = int(round(pix[0, 0])), int(round(pix[0, 1]))
            img[y, x] = 1.0
            if gimg is not None:
                gimg[y, x] = values[ids[0]] / vmax
        for s in range(len(pts) - 1):
            a, b = pix[s], pix[s + 1]
            steps = int(max(abs(b - a))) * 2 + 2
            t = np.linspace(0.0, 1.0, steps)
            xs = np.rint(a[0] + (b[0] - a[0]) * t).astype(int)
            ys = np.rint(a[1] + (b[1] - a[1]) * t).astype(int)
            level = (s + 1) / n_seg
            img[ys, xs] = np.maximum(img[ys, xs], level)
            if gimg is not None:
                gimg[ys, xs] = values[ids[s + 1]] / vmax
    return TraceArray(img, pts, np.asarray(ids, dtype=int), gimg)


def trace_frames(units: Sequence[str], letters: Sequence[str], size: int = 128,
                 values: Sequence[float] | None = None) -> list[TraceArray]:
    """Une image par préfixe (construction incrémentale)."""
    return [trace_to_array(p, letters, size, values) for p in prefix_units(list(units))]


# --------------------------------------------------------------- données IPA
def read_ipa_rows(lang: str, dict_dir: Path, variety: str | None = None):
    path = dict_dir / f"{lang}.tsv"
    if not path.exists():
        raise FileNotFoundError(f"{path} introuvable : lancer scripts/fetch_phonetic_dicts.py")
    with open(path, encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh, delimiter="\t"):
            if variety is None or row["variety"] == variety:
                yield row


def lookup_ipa(lang: str, words: Sequence[str], dict_dir: Path, variety: str | None = None
               ) -> tuple[str, str | None]:
    """IPA des mots depuis le dictionnaire (première entrée trouvée par mot).
    Lève LookupError si un mot est absent : on n'invente jamais de transcription."""
    wanted = {unicodedata.normalize("NFC", w) for w in words}
    found: dict[str, tuple[str, str]] = {}
    lower_found: dict[str, tuple[str, str]] = {}
    for row in read_ipa_rows(lang, dict_dir, variety):
        w = row["word"]
        if w in wanted and w not in found:
            found[w] = (row["ipa"], row["variety"])
        elif w.lower() in {x.lower() for x in wanted} and w.lower() not in lower_found:
            lower_found[w.lower()] = (row["ipa"], row["variety"])
    parts, varieties = [], set()
    for w in words:
        w = unicodedata.normalize("NFC", w)
        hit = found.get(w) or lower_found.get(w.lower())
        if hit is None:
            raise LookupError(f"« {w} » absent de {dict_dir / (lang + '.tsv')}"
                              + (f" (variété {variety})" if variety else ""))
        parts.append(hit[0]); varieties.add(hit[1])
    return "".join(parts), ("/".join(sorted(varieties)) if varieties else None)


def ipa_inventory(lang: str, dict_dir: Path, variety: str | None, extra: Sequence[str]) -> list[str]:
    """Inventaire de symboles IPA du dictionnaire (trié par code Unicode) ; les symboles de
    l'entrée absents du dictionnaire sont ajoutés à la fin. Sans dictionnaire : symboles de l'entrée seuls."""
    symbols: set[str] = set()
    try:
        for ipa in {r["ipa"] for r in read_ipa_rows(lang, dict_dir, variety)}:
            symbols.update(ipa_tokens(ipa))
    except FileNotFoundError:
        print(f"avertissement : pas de dictionnaire pour « {lang} » ; inventaire IPA limité à l'entrée",
              file=sys.stderr)
    inv = sorted(symbols)
    for t in extra:
        if t not in symbols and t not in inv:
            inv.append(t)
    return inv


# ------------------------------------------------------------------- rendu GIF
@dataclass
class Panel:
    title: str
    letters: list[str]
    units: list[str]
    values: list[float] | None = None            # valeur par lettre de `letters` (couleur « gématrie »)
    frame_ends: list[int] = field(default_factory=list)  # nb d'unités visibles à chaque image
    footer: str = ""

    def __post_init__(self):
        if not self.frame_ends:
            self.frame_ends = list(range(1, len(self.units) + 1))


def _import_mpl():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def render_frame(plt, panels: Sequence[Panel], frame: int, color_by: str, size_px: int):
    fig, axes = plt.subplots(1, len(panels), figsize=(size_px * len(panels) / 100, size_px / 100), dpi=100)
    axes = np.atleast_1d(axes)
    cmap = plt.get_cmap("viridis")
    for ax, panel in zip(axes, panels):
        n = len(panel.letters)
        k = min(frame, len(panel.frame_ends) - 1)
        shown = panel.frame_ends[k] if panel.frame_ends else 0
        visible = panel.units[:shown]
        ax.set_aspect("equal"); ax.axis("off")
        ax.set_xlim(-1.3, 1.3); ax.set_ylim(-1.3, 1.3)
        for seg in ray_segments(n):
            ax.plot(seg[:, 0], seg[:, 1], color="#cccccc", lw=0.5, zorder=1)
        pos = letter_positions(n)
        ax.add_patch(plt.Circle((0, 0), EXTENT, fill=False, color="#999999", lw=0.8, zorder=1))
        fs = max(5, min(12, 220 // max(n, 1)))
        for c, (x, y) in zip(panel.letters, letter_positions(n, 1.12)):
            ax.text(x, y, c, ha="center", va="center", fontsize=fs, zorder=3)
        ax.scatter(pos[:, 0], pos[:, 1], s=6, color="#666666", zorder=2)
        ids = trajectory_indices(visible, panel.letters)
        pts = pos[ids] if ids else np.zeros((0, 2))
        segs = segments_of(pts)
        vmax = max(panel.values) if panel.values else 1
        for s, seg in enumerate(segs):
            if color_by == "gematria" and panel.values:
                color = cmap(panel.values[ids[s + 1]] / vmax)
            elif color_by == "order":
                color = cmap(s / max(len(segs) - 1, 1))
            else:
                color = "#d62728"
            ax.annotate("", xy=seg[1], xytext=seg[0], zorder=4,
                        arrowprops=dict(arrowstyle="-|>", color=color, lw=1.8, shrinkA=0, shrinkB=0))
        if len(pts):
            ax.scatter(pts[-1:, 0], pts[-1:, 1], s=60, color="black", zorder=5)
        ax.set_title(f"{panel.title}\n{''.join(visible)}", fontsize=11)
        if panel.footer:
            ax.text(0, -1.27, panel.footer, ha="center", va="top", fontsize=8)
    fig.tight_layout()
    fig.canvas.draw()
    rgba = np.asarray(fig.canvas.buffer_rgba()).copy()
    plt.close(fig)
    return rgba[:, :, :3]


def render_gif(panels: Sequence[Panel], out: Path, fps: float = 2.0, color_by: str = "none",
               size_px: int = 480) -> int:
    from PIL import Image
    plt = _import_mpl()
    n_frames = max(len(p.frame_ends) for p in panels)
    if n_frames == 0:
        raise ValueError("entrée vide : aucune image à produire")
    frames = []
    for f in range(n_frames):
        panels_f = []
        for p in panels:
            k = min(f, len(p.frame_ends) - 1)
            shown = p.frame_ends[k]
            footer = p.footer
            if p.values is not None and color_by == "gematria":
                idx = {c: i for i, c in enumerate(p.letters)}
                footer = f"gématrie cumulée : {sum(p.values[idx[u]] for u in p.units[:shown] if u in idx):g}"
            panels_f.append(Panel(p.title, p.letters, p.units, p.values, p.frame_ends, footer))
        frames.append(Image.fromarray(render_frame(plt, panels_f, f, color_by, size_px)))
    paletted = [im.convert("P", palette=Image.ADAPTIVE) for im in frames]
    out.parent.mkdir(parents=True, exist_ok=True)
    paletted[0].save(out, save_all=True, append_images=paletted[1:],
                     duration=int(1000 / fps), loop=0)
    return n_frames


# ------------------------------------------------------------ construction des panneaux
def build_panels(args, alphabet: Alphabet, text: str | None, ipa: str | None, dict_dir: Path,
                 system: str | None) -> list[Panel]:
    panels: list[Panel] = []
    syllable_mode = args.step == "syllable"
    if text and alphabet.alphabetic and not syllable_mode:
        units = alphabet.letters_of(text)
        if not units:
            raise ValueError(f"aucune lettre de l'alphabet « {alphabet.lang} » dans le texte")
        values = [alphabet.letter_value(c, system) for c in alphabet.letters] if alphabet.systems else None
        panels.append(Panel(f"Lettres ({alphabet.lang})", alphabet.letters, units, values))
    elif text and not alphabet.alphabetic:
        print(f"note : « {alphabet.lang} » n'est pas alphabétique ; panneau de lettres omis "
              "(seul le motif phonétique est tracé, voir graph/README.md)", file=sys.stderr)
    if ipa:
        tokens = ipa_tokens(ipa)
        inv = ipa_inventory(alphabet.lang, dict_dir, args.variety, tokens)
        if syllable_mode:
            syls = ipa_syllables(ipa)
            if syls is None:
                raise ValueError("--step syllable exige une IPA avec frontières explicites ('.') ; "
                                 "aucune syllabification n'est inventée")
            ends, total = [], 0
            for s in syls:
                total += len(ipa_tokens(s)); ends.append(total)
            panels.append(Panel("Motif phonétique (syllabes)", inv, tokens, None, ends))
        else:
            panels.append(Panel("Motif phonétique (IPA)", inv, tokens))
    if not panels:
        raise ValueError("rien à tracer : fournir --text et/ou --ipa")
    return panels


def transcribe_audio(path: Path, lang: str) -> str:
    try:
        import whisper  # dépendance optionnelle
    except ImportError:
        raise SystemExit("--audio nécessite le paquet optionnel « openai-whisper » (et ffmpeg) : "
                         "pip install openai-whisper") from None
    model = whisper.load_model("base")
    return model.transcribe(str(path), language=lang)["text"].strip()


def resolve_ipa(args, text: str | None, dict_dir: Path) -> tuple[str | None, str | None]:
    if args.ipa in (None, "none"):
        return None, None
    if args.ipa == "auto":
        if not text:
            raise ValueError("--ipa auto exige --text (ou --audio)")
        return lookup_ipa(args.lang, text.split(), dict_dir, args.variety)
    return args.ipa, args.variety


def export_dataset(args, alphabet: Alphabet, words: list[str], dict_dir: Path, system: str | None) -> int:
    """Écrit image PNG + métadonnées JSONL (mot, IPA, gématrie). Simple export : n'implique
    aucune capacité d'apprentissage. IPA absente => null (jamais inventée)."""
    from PIL import Image
    out = Path(args.export_dataset)
    (out / "images").mkdir(parents=True, exist_ok=True)
    values = [alphabet.letter_value(c, system) for c in alphabet.letters] if alphabet.systems else None
    n = 0
    with open(out / "metadata.jsonl", "w", encoding="utf-8") as meta:
        for w in words:
            units = alphabet.letters_of(w)
            if not units:
                continue
            ta = trace_to_array(units, alphabet.letters, args.size, values)
            name = f"images/{n:06d}.png"
            Image.fromarray((ta.image * 255).astype(np.uint8)).save(out / name)
            try:
                ipa, variety = lookup_ipa(alphabet.lang, [w], dict_dir, args.variety)
            except (LookupError, FileNotFoundError):
                ipa, variety = None, None
            meta.write(json.dumps({
                "image": name, "word": w, "lang": alphabet.lang, "ipa": ipa, "variety": variety,
                "gematria": alphabet.gematria(units, system) if alphabet.systems else None,
                "gematria_system": system or (alphabet.system_names[0] if alphabet.systems else None),
                "units": units}, ensure_ascii=False) + "\n")
            n += 1
    return n


def run_stream(args, alphabet: Alphabet, dict_dir: Path, system: str | None) -> None:
    """Lit stdin caractère par caractère (simule la frappe en direct) ; chaque ligne = un GIF."""
    out = Path(args.out)
    buf, k = "", 0

    def flush():
        nonlocal buf, k
        if buf.strip():
            ipa, _ = (None, None)
            try:
                ipa, _ = resolve_ipa(args, buf.strip(), dict_dir)
            except (LookupError, ValueError, FileNotFoundError) as exc:
                print(f"IPA indisponible ({exc})", file=sys.stderr)
            panels = build_panels(args, alphabet, buf, ipa, dict_dir, system)
            target = out.with_name(f"{out.stem}_{k}{out.suffix}")
            render_gif(panels, target, args.fps, args.color_by, args.size)
            print(f"-> {target}", file=sys.stderr)
            k += 1
        buf = ""

    while True:
        ch = sys.stdin.read(1)
        if ch == "":
            break
        if ch == "\n":
            flush()
            continue
        buf += ch
        letters = alphabet.letters_of(ch)
        if letters:
            print(f"{ch!r} -> cercle[{alphabet.letters.index(letters[0]) if alphabet.alphabetic else '-'}]",
                  file=sys.stderr)
        if args.delay:
            time.sleep(args.delay)
    flush()


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--lang", required=True, help="code de langue (fr, en, de, es, ar, he, el, zh, yue)")
    p.add_argument("--text", help="texte à tracer")
    p.add_argument("--audio", type=Path, help="fichier audio transcrit par whisper (dépendance optionnelle)")
    p.add_argument("--ipa", default="none", help="'auto' (dictionnaire), 'none', ou une chaîne IPA")
    p.add_argument("--variety", help="restreindre le dictionnaire à une variété (ex. fr-FR, en-UK)")
    p.add_argument("--dict-dir", type=Path, default=ROOT / "data" / "dict")
    p.add_argument("--config", type=Path, default=None)
    p.add_argument("--system", help="système de gématrie (défaut : premier de la langue)")
    p.add_argument("--step", choices=["char", "syllable"], default="char")
    p.add_argument("--color-by", choices=["none", "gematria", "order"], default="none")
    p.add_argument("--fps", type=float, default=2.0)
    p.add_argument("--size", type=int, default=480, help="taille d'un panneau en pixels (GIF) / image (dataset)")
    p.add_argument("--out", type=Path, default=Path("out.gif"))
    p.add_argument("--stream", action="store_true", help="lire stdin caractère par caractère")
    p.add_argument("--delay", type=float, default=0.0, help="délai (s) entre caractères en mode --stream")
    p.add_argument("--export-dataset", type=Path, help="dossier de sortie (images PNG + metadata.jsonl)")
    p.add_argument("--words-file", type=Path, help="un mot par ligne (pour --export-dataset)")
    args = p.parse_args(argv)

    alphabet = get_alphabet(args.lang, load_config(args.config))
    system = args.system
    if system and system not in alphabet.systems:
        p.error(f"système « {system} » non configuré pour {args.lang} : {alphabet.system_names}")
    try:
        if args.stream:
            run_stream(args, alphabet, args.dict_dir, system)
            return 0
        text = args.text
        if args.audio:
            text = transcribe_audio(args.audio, args.lang)
            print(f"transcription : {text}", file=sys.stderr)
        if args.export_dataset:
            words = (args.words_file.read_text(encoding="utf-8").split() if args.words_file
                     else (text or "").split())
            n = export_dataset(args, alphabet, words, args.dict_dir, system)
            print(f"{n} exemples -> {args.export_dataset}", file=sys.stderr)
            return 0
        ipa, variety = resolve_ipa(args, text, args.dict_dir)
        if ipa:
            print(f"IPA : {ipa}" + (f" ({variety})" if variety else ""), file=sys.stderr)
        panels = build_panels(args, alphabet, text, ipa, args.dict_dir, system)
        n = render_gif(panels, args.out, args.fps, args.color_by, args.size)
    except (LookupError, ValueError, FileNotFoundError) as exc:
        p.error(str(exc))
    print(f"{n} images -> {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())

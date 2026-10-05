"""Alphabets par langue et systèmes de gématrie, pilotés par config/gematria_systems.yaml."""
from __future__ import annotations

import unicodedata
from dataclasses import dataclass, field
from pathlib import Path

import yaml

DEFAULT_CONFIG = Path(__file__).resolve().parent.parent / "config" / "gematria_systems.yaml"


def load_config(path: str | Path | None = None) -> dict:
    with open(path or DEFAULT_CONFIG, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


@dataclass
class Alphabet:
    lang: str
    letters: list[str]
    direction: str = "ltr"
    fold_map: dict[str, str] = field(default_factory=dict)
    keep: set[str] = field(default_factory=set)
    strip_marks: bool = True
    systems: dict[str, dict] = field(default_factory=dict)  # nom -> définition ; ordre = priorité
    alphabetic: bool = True

    @property
    def size(self) -> int:
        return len(self.letters)

    @property
    def system_names(self) -> list[str]:
        return list(self.systems)

    @property
    def index(self) -> dict[str, int]:
        return {c: i for i, c in enumerate(self.letters)}

    def fold(self, text: str) -> str:
        """Repli vers l'alphabet : NFC, minuscules, table de repli, suppression
        optionnelle des signes combinants (voyelles optionnelles comprises)."""
        out: list[str] = []
        for ch in unicodedata.normalize("NFC", text).casefold():
            if ch in self.fold_map:
                out.append(self.fold_map[ch])
            elif ch in self.keep or not self.strip_marks:
                out.append(ch)
            else:
                out.append("".join(c for c in unicodedata.normalize("NFD", ch)
                                   if not unicodedata.category(c).startswith("M")))
        return "".join(out)

    def letters_of(self, text: str) -> list[str]:
        """Suite des lettres de l'alphabet contenues dans `text` (autres caractères ignorés)."""
        if not self.alphabetic:
            return [c for c in unicodedata.normalize("NFC", text)
                    if unicodedata.category(c) == "Lo"]
        idx = self.index
        return [c for c in self.fold(text) if c in idx]

    def letter_value(self, letter: str, system: str | None = None) -> int:
        system = system or (self.system_names[0] if self.systems else None)
        if system is None:
            raise ValueError(f"aucun système de gématrie configuré pour « {self.lang} »")
        spec = self.systems[system]
        i = self.index[letter]
        kind = spec["kind"]
        if kind == "table":
            return int(spec["values"][letter])
        if kind == "ordinal":
            return i + 1
        if kind == "pythagorean":
            return 1 + (i % 9)
        raise ValueError(f"type de système inconnu : {kind}")

    def gematria(self, text_or_letters, system: str | None = None) -> int:
        letters = (self.letters_of(text_or_letters) if isinstance(text_or_letters, str)
                   else list(text_or_letters))
        return sum(self.letter_value(c, system) for c in letters)

    def max_value(self, system: str | None = None) -> int:
        return max(self.letter_value(c, system) for c in self.letters)


def get_alphabet(lang: str, config: dict | str | Path | None = None) -> Alphabet:
    cfg = config if isinstance(config, dict) else load_config(config)
    try:
        spec = cfg["languages"][lang]
    except KeyError:
        raise KeyError(f"langue « {lang} » absente de la configuration "
                       f"(disponibles : {', '.join(cfg['languages'])})") from None
    alphabetic = spec.get("alphabetic", True)
    return Alphabet(
        lang=lang,
        letters=list(spec.get("alphabet", [])),
        direction=spec.get("direction", "ltr"),
        fold_map={k: ("" if v is None else str(v)) for k, v in (spec.get("fold") or {}).items()},
        keep=set(spec.get("keep") or []),
        strip_marks=spec.get("strip_marks", True),
        systems={n: cfg["systems"][n] for n in spec.get("systems", [])},
        alphabetic=alphabetic,
    )

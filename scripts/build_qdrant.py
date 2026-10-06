"""Indexe les dictionnaires mot -> IPA dans Qdrant (vecteurs simples, sans modèle appris).

Vecteur d'un mot (dimension HASH_DIM + 3), cosinus :
  - HASH_DIM : n-grammes (1 à 3) de symboles IPA, hachés (CRC32, signe + position),
    puis normalisés L2 ;
  - +3 traits : longueur en lettres / 20, gématrie / (max_lettre * longueur), nb de
    symboles IPA / 20 (0 si la langue n'a pas de système de gématrie).
Ce ne sont PAS des embeddings sémantiques : ils servent de baseline structurelle pour
tester si phonèmes / gématrie portent un signal (hypothèse nulle H0 : non).

Variables d'environnement : QDRANT_URL (défaut http://localhost:6333).
Usage :
  python scripts/build_qdrant.py --dict-dir D:/embedbabel-data/dict --dry-run
  python scripts/build_qdrant.py --dict-dir D:/embedbabel-data/dict
"""
from __future__ import annotations

import argparse
import csv
import math
import os
import sys
import uuid
import zlib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from embedbabel.gematria import Alphabet, get_alphabet, load_config  # noqa: E402
from embedbabel.textnorm import ipa_tokens  # noqa: E402

HASH_DIM = 256
VECTOR_DIM = HASH_DIM + 3
COLLECTION = "words"
NAMESPACE = uuid.UUID("6f1b2c9e-3a47-4d58-9b0e-5d1e8c7a2f10")


def point_id(lang: str, text: str, variety: str) -> str:
    """Identifiant stable : relancer le script met à jour les mêmes points."""
    return str(uuid.uuid5(NAMESPACE, f"{lang}\x1f{text}\x1f{variety}"))


def _bucket(gram: tuple[str, ...]) -> tuple[int, float]:
    h = zlib.crc32("\x1f".join(gram).encode("utf-8"))
    return h % HASH_DIM, 1.0 if (h >> 16) & 1 else -1.0


def word_vector(text: str, ipa: str, alphabet: Alphabet, system: str | None = None) -> list[float]:
    tokens = ipa_tokens(ipa)
    vec = [0.0] * HASH_DIM
    for n in (1, 2, 3):
        for i in range(len(tokens) - n + 1):
            idx, sign = _bucket(tuple(tokens[i:i + n]))
            vec[idx] += sign
    norm = math.sqrt(sum(v * v for v in vec)) or 1.0
    vec = [v / norm for v in vec]
    units = alphabet.letters_of(text)
    gem = 0.0
    if alphabet.systems and units:
        gem = alphabet.gematria(units, system) / (alphabet.max_value(system) * len(units))
    return vec + [len(units) / 20.0, gem, len(tokens) / 20.0]


def read_rows(path: Path, limit: int | None = None):
    with open(path, encoding="utf-8", newline="") as fh:
        for i, row in enumerate(csv.DictReader(fh, delimiter="\t")):
            if limit is not None and i >= limit:
                break
            yield row["word"], row["ipa"], row["variety"], row["source"]


def iter_points(lang: str, rows, alphabet: Alphabet, system: str | None = None):
    from qdrant_client.models import PointStruct
    seen: set[tuple[str, str, str]] = set()
    for text, ipa, variety, source in rows:
        key = (text, variety, ipa)
        if key in seen or not ipa_tokens(ipa):
            continue
        seen.add(key)
        # une prononciation = un point (un mot à plusieurs IPA donne plusieurs points)
        pid = point_id(lang, text, f"{variety}\x1f{ipa}")
        yield PointStruct(id=pid, vector=word_vector(text, ipa, alphabet, system), payload={
            "lang": lang, "text": text, "ipa": ipa, "variety": variety, "source": source,
            "length": len(alphabet.letters_of(text))})


def ensure_collection(client, name: str = COLLECTION) -> None:
    from qdrant_client.models import Distance, PayloadSchemaType, VectorParams
    if not client.collection_exists(name):
        client.create_collection(name, vectors_config=VectorParams(size=VECTOR_DIM, distance=Distance.COSINE))
    for field in ("lang", "variety"):
        client.create_payload_index(name, field_name=field, field_schema=PayloadSchemaType.KEYWORD)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dict-dir", type=Path, default=ROOT / "data" / "dict")
    p.add_argument("--config", type=Path, default=None)
    p.add_argument("--langs", nargs="+", default=None, help="défaut : tous les <dict-dir>/*.tsv")
    p.add_argument("--max-words", type=int, default=None, help="N premières lignes par dictionnaire")
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--url", default=os.environ.get("QDRANT_URL", "http://localhost:6333"))
    p.add_argument("--collection", default=COLLECTION)
    p.add_argument("--dry-run", action="store_true", help="compte les points, sans Qdrant")
    args = p.parse_args(argv)

    cfg = load_config(args.config)
    files = {f.stem: f for f in sorted(args.dict_dir.glob("*.tsv"))}
    langs = args.langs or list(files)
    missing = [lang for lang in langs if lang not in files]
    if missing:
        p.error(f"dictionnaire absent pour : {missing} (dans {args.dict_dir})")

    client = None
    if not args.dry_run:
        from qdrant_client import QdrantClient
        client = QdrantClient(url=args.url)
        ensure_collection(client, args.collection)

    total = 0
    for lang in langs:
        alphabet = get_alphabet(lang, cfg)
        points = iter_points(lang, read_rows(files[lang], args.max_words), alphabet)
        if client is None:
            n = sum(1 for _ in points)
        else:
            before = client.count(args.collection, exact=True).count
            client.upload_points(args.collection, points, batch_size=args.batch_size)
            n = client.count(args.collection, exact=True).count - before
        total += n
        print(f"{lang}: {n} points{' (à créer)' if client is None else ' ajoutés'}")
    print(f"total: {total}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

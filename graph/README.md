# Graphe Neo4j

Le schéma est décrit dans [`schema.cypher`](schema.cypher). `scripts/build_graph.py` lit `data/dict/*.tsv`.

## Modèle

- **Feuilles** : `(:Letter {lang, char, alphabet_index, gematria_systems[], gematria_values[]})`, une par lettre
  de l'alphabet de `config/gematria_systems.yaml` ; chaînées par `NEXT_LETTER`.
- **Branches** : `(:Substring {lang, text, length, gematria})` pour toutes les sous-chaînes contiguës de longueur
  2 à `--max-substring-len`. Une sous-chaîne est un nœud **unique par langue** : les sous-arbres sont partagés
  entre mots. `COMPOSED_OF {pos: 0}` va vers `s[:-1]`, `{pos: 1}` vers `s[1:]` (DAG se terminant aux lettres).
- **Mots** : `(:Word {lang, text, variety, ipa[], period})`, clé `(lang, text, variety)` ; `COMPOSED_OF {pos}` vers
  les lettres, `PREFIX_OF` (préfixes propres), `SUBSTRING_OF {start}` (sous-chaînes propres, préfixes compris).
- **Phonétique** : `(:Pronunciation {lang, ipa, variety})` (extension, partagée par les homophones) avec
  `(:Word)-[:PRONOUNCED_AS]->`. `(:Syllable)` et `HAS_SYLLABLE` n'existent que si la source marque des frontières
  (`.`) ; **aucune syllabification n'est inventée** (les sources actuelles n'en contiennent pas).
- **Diachronie** : `COGNATE`, `BORROWING`, `SOUND_CHANGE`, `CONTRACTION`, `CONTACT_BLEND` entre `(:Word)`, avec
  `source`, `confidence`, `rule`, `template`, `retrieved`, `license` ; nœuds étymologiques de variété `''`, reliés
  par `SAME_FORM` aux mots du dictionnaire chargés dans le même run. `period` reste vide (pas de source).

## Utilisation

```bash
export NEO4J_PASSWORD='mot-de-passe-de-8-caracteres-ou-plus'
docker compose up -d                                   # Neo4j (+ Qdrant : --profile qdrant)
python scripts/build_graph.py --dry-run --max-substring-len 4        # rapport de taille, sans base
python scripts/build_graph.py --apply-schema --langs fr en --varieties fr-FR en-US --etymology data/etymology/edges.tsv
python scripts/build_graph.py --export-csv graph/export --langs fr   # CSV pour neo4j-admin import
```

Le chargement se fait par lots `UNWIND … MERGE` (`--batch-size`) avec le driver officiel `neo4j` ; il est idempotent.
Le mode CSV écrit un fichier de données et un fichier d'en-tête par type de nœud/relation
(`nodes_<Label>.csv`, `rels_<TYPE>_<Src>_<Dst>.csv`, avec `*_header.csv`), à passer à
`neo4j-admin database import full --nodes=Label=…header,….csv --relationships=…` ; ce chemin
d'import n'a **pas** été exécuté (seul le chargement Bolt a été testé sur Neo4j 5 Community).

## Explosion combinatoire

Un mot de `L` lettres a `L(L+1)/2` sous-chaînes contiguës, et `SUBSTRING_OF` crée une arête par occurrence
pour chaque mot. Le partage des nœuds limite le nombre de `Substring`, pas celui des arêtes `SUBSTRING_OF`/`PREFIX_OF`.
`--max-substring-len K` plafonne à `Σ_{k≤K} (L−k+1)` occurrences par mot et à `|Σ|^K` nœuds `Substring` par langue.
Exemple mesuré (`--dry-run --max-words 20000 --max-substring-len 4`) : fr ≈ 19 k sous-chaînes, 820 k arêtes ;
el ≈ 21 k sous-chaînes, 858 k arêtes. Le rapport par langue (nœuds/arêtes par type) est imprimé à chaque exécution
et enregistrable avec `--report rapport.json`. Commencer petit (`--max-words`, `K` = 3 ou 4) avant de monter.

## Unicode et écritures

- Texte en NFC ; les lettres du graphe sont obtenues par `Alphabet.fold` : minuscules, table de repli de la langue,
  suppression des signes combinants (accents, voyelles optionnelles arabes `tashkīl`, hébreu `niqqud`) — ce qui
  **perd des distinctions** (é/e, ä/a) ; le texte d'origine reste dans `Word.text`. Les caractères hors alphabet
  (apostrophes, tirets, chiffres) sont ignorés pour les sous-chaînes.
- Arabe/hébreu (RTL) : l'ordre des unités est l'ordre logique (début → fin du mot), pas l'ordre d'affichage.
- Choix de repli discutables (documentés dans la config) : ة→ه, ى→ي, ا pour les hamza, ß→ss, ä/ö/ü→a/o/u,
  formes finales hébraïques → lettre de base.
- **Chinois** : cas non alphabétique, volontairement pas forcé dans un alphabet. Les unités sont les caractères Han
  observés (`Letter` sans `alphabet_index`, ni `NEXT_LETTER`, ni gématrie), les sous-chaînes sont des suites de
  caractères et l'IPA vient de la source (avec tons). Pas de pinyin dans les données ; une couche pinyin/Latin
  serait une extension à sourcer. La visualisation circulaire ne trace que le motif phonétique pour le chinois.

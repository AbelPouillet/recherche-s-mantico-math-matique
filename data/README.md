# Données linguistiques

Aucune donnée linguistique n'est commitée : tout est téléchargé par script depuis des sources
vérifiées (voir [`SOURCES.md`](SOURCES.md) pour licences, URL, formats, dates d'accès, couverture).
`data/raw/` et `data/dict/*.tsv` sont dans `.gitignore`.

## Arborescence

| Chemin | Contenu |
|---|---|
| `data/raw/` | téléchargements bruts + `manifest.json` (URL, SHA-256, taille, date) — gitignoré |
| `data/dict/<lang>.tsv` | dictionnaires normalisés, colonnes `word`, `ipa`, `variety`, `source` — gitignoré |
| `data/etymology/rules.tsv` | règles de changement phonétique **documentées** (colonne `source` obligatoire) ; vide tant qu'aucune n'est vérifiée |
| `data/etymology/edges.tsv` | arêtes étymologiques produites par `scripts/fetch_etymology.py` — gitignoré |
| `data/SOURCES.md` | registre des sources |

## Utilisation

```bash
python scripts/fetch_phonetic_dicts.py --coverage      # sources, réserves, langues non couvertes
python scripts/fetch_phonetic_dicts.py                 # télécharge et normalise tout (~1,2 M lignes)
python scripts/fetch_phonetic_dicts.py --langs fr en   # sous-ensemble
python scripts/fetch_phonetic_dicts.py --offline       # réutilise data/raw sans réseau
```

Normalisation : NFC ; WikiPron fournit des phonèmes séparés par des espaces, qu'on supprime
(l'IPA est stockée sous forme de chaîne continue ; `embedbabel.textnorm.ipa_tokens` la resegmente
en gardant diacritiques et ligatures) ; ipa-dict : suppression des `/…/` et une ligne par variante.
Doublons `(word, ipa, variety)` supprimés. Codes de variété : `fr-FR`, `fr-QC`, `en-US`, `en-UK`,
`de`, `es-ES`, `es-419`, `ar`, `he`, `el`, `zh-cmn` (fichier `zh.tsv`), `zh-yue` (fichier `yue.tsv`).

## Limites à connaître

- `fr-FR` n'est pas une étiquette de la source : WikiPron `fra` n'a pas de dialecte. `fr-QC` vient
  d'ipa-dict, généré par règles et décrit par ses auteurs comme « très expérimental ».
- Les transcriptions WikiPron sont **larges (phonémiques)**, multiples par mot (variantes), issues de
  contributions Wiktionary non homogènes. Pas de syllabification ni de datation : le champ `period`
  du graphe reste vide et aucune syllabe n'est inventée.
- Arabe : un seul jeu (`ara`), pas de dialectes. Hébreu : ~6,8 k lignes seulement.
- Chinois : clés en caractères Han (surtout un caractère par entrée), IPA avec chiffres de ton ; pas de
  pinyin/jyutping dans la source utilisée.
- Non couverts : dialectes arabes, grec ancien/moyen, formes historiques (latin tardif, ancien français) hormis
  les nœuds étymologiques sans IPA.

## Étymologie

```bash
python scripts/fetch_etymology.py --words fr:écureuil en:squirrel     # API MediaWiki (réseau)
python scripts/fetch_etymology.py --wikitext-dir tests/fixtures/etymology   # hors-ligne, fixtures
```

Les arêtes (`cognate`, `borrowing`, `sound_change`, `contraction`, `contact_blend`) viennent
exclusivement de modèles explicites du wikitexte (`{{inh}}`, `{{bor}}`, `{{cog}}`, `{{contraction}}`,
`{{blend}}`), avec provenance (page, modèle, extrait, date, licence). `rule` reste vide sans entrée
documentée dans `rules.tsv`. Conséquence : la contraction « es- → é- » n'est **pas** déduite d'*escurel → écureuil* ;
elle n'apparaîtra qu'avec une règle sourcée. La correspondance modèle → type d'arête est
présumée et doit être validée contre la documentation des modèles Wiktionary. Les fixtures de
`tests/fixtures/etymology/` sont synthétiques (test de l'analyseur), pas des copies du Wiktionnaire.

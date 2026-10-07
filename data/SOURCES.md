# Sources

Vérifications faites le **2026-10-05** (consultation directe des dépôts GitHub et téléchargement des
fichiers utilisés). Le téléchargement exact est tracé à l'exécution dans `data/raw/manifest.json`
(URL, SHA-256, date). Les URL pointent par défaut vers la branche `master` ; pour figer une
version, utiliser `--wikipron-ref` / `--ipadict-ref` avec un commit.

## Sources utilisées

### WikiPron — CUNY-CL/wikipron
- URL : <https://github.com/CUNY-CL/wikipron> ; fichiers : `data/scrape/tsv/<fichier>.tsv` sur `raw.githubusercontent.com`.
- Format : TSV `mot<TAB>phonèmes IPA séparés par des espaces`, UTF-8 ; une ligne par prononciation.
- Licence : code Apache 2.0 ; **les données suivent la licence propre de Wiktionary**
  (<https://en.wiktionary.org/wiki/Wiktionary:Copyrights>, CC BY-SA) comme l'indique le README de WikiPron.
  Les jeux dérivés doivent donc être redistribués sous CC BY-SA avec attribution (d'où l'absence de dump dans git).
- Citation demandée : Lee et al. (2020), *Massively multilingual pronunciation mining with WikiPron*, LREC, pp. 4223–4228.
- Date d'accès : 2026-10-05.

| Langue | Variété (projet) | Fichier | Entrées (selon le tableau de WikiPron) |
|---|---|---|---:|
| fr | fr-FR | `fra_latn_broad.tsv` | 97 652 |
| en | en-US | `eng_latn_us_broad.tsv` | 106 931 |
| en | en-UK | `eng_latn_uk_broad.tsv` | 106 688 |
| de | de | `deu_latn_broad.tsv` | 60 277 |
| es | es-ES | `spa_latn_ca_broad.tsv` | 137 542 |
| es | es-419 | `spa_latn_la_broad.tsv` | 133 146 |
| ar | ar | `ara_arab_broad.tsv` | 17 563 |
| he | he | `heb_hebr_broad.tsv` | 6 811 |
| el | el | `ell_grek_broad.tsv` | 19 601 |
| zh | zh-cmn | `cmn_hani_standard_broad.tsv` | 168 655 |
| yue | zh-yue | `yue_hani_standard_broad.tsv` | 116 020 |

### ipa-dict — open-dict-data/ipa-dict (uniquement fr-QC)
- URL : <https://github.com/open-dict-data/ipa-dict> ; fichier `data/fr_QC.txt`.
- Format : TSV `mot<TAB>/ipa/` (variantes séparées par des virgules), UTF-8.
- Licence : MIT pour le dépôt ; chaque jeu garde la licence de sa source (README, section Credits). Pour `fr_QC`,
  le README indique des données générées avec le convertisseur `qc-ipa`, « highly experimental ».
  Licence du convertisseur sous-jacent non vérifiée : à contrôler avant toute redistribution.
- Date d'accès : 2026-10-05. Entrées : ~245 958 lignes de mots.
- Autres jeux ipa-dict (`ar`, `de`, `en_UK`, `en_US`, `es_ES`, `fr_FR`, `yue`, `zh`…) **non utilisés** : mélange de
  licences (ex. GPL 3.0 pour `en_UK`) et d'outils par règles ; WikiPron est plus homogène. L'hébreu et le grec
  n'existent pas dans ipa-dict.

### Wiktionnaire anglais (étymologie)
- Accès par l'API MediaWiki (`en.wiktionary.org/w/api.php`), licence CC BY-SA (texte du Wiktionnaire).
- **Non vérifié depuis l'environnement de développement** (accès réseau bloqué) : le contenu réel des pages
  *squirrel* / *écureuil* et la correspondance exacte modèles → types d'arêtes n'ont pas pu être contrôlés.
  À valider à la première exécution réseau. Les fixtures de test sont synthétiques.
- Alternatives candidates non évaluées : EtymWordNet, extraits kaikki.org.

## Couverture et langues non couvertes

| Besoin | État |
|---|---|
| français France / Québec | couvert (fr-QC : expérimental) |
| anglais US / UK | couvert |
| allemand, espagnol, grec, hébreu | couverts (hébreu : petit) |
| arabe | couvert (MSA/non précisé ; pas de dialectes) |
| mandarin, cantonais | couverts en caractères Han + IPA ; pas de pinyin/jyutping |
| période / datation des formes | **non couvert** (`period` vide) |
| syllabification | **non couvert** (aucune source utilisée ne la fournit) |
| règles de changement phonétique | **non couvert** : `data/etymology/rules.tsv` vide tant qu'aucune source n'est vérifiée |
| normes psycholinguistiques (sensoriel) | **non incluses** ; à choisir et vérifier avant usage |

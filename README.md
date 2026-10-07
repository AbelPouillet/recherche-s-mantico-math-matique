# EmbedBabel — AI Research Benchmark


Ce dépôt sert à collecter, comparer et benchmarker des analyses IA autour d’EmbedBabel, puis à explorer une V3 ludique en **simulation** avec un médaillon traducteur instrumenté.
Il fournit aussi les ressources de base (dictionnaires phonétiques, graphe Neo4j, visualisation circulaire) pour
tester l'idée. Toute la documentation est en français.

## Arborescence

- `PROMPT.md` — prompt identique fourni à chaque IA (sections 2bis à 2quinquies : phonétique/diachronie, mots à inventer,
  homophones, RAG Neo4j+Qdrant).
- `bench/prompts/` — prompts benchmark (V2/V3, quiz, speedrun).
- `plugins/` — plugins par modèle, format : `{$modelID-gématriphonéticospatiale-vx.x.x}`.
- `docs/` — règles de jeu, protocoles, specs d’encodage, [bibliographie vérifiée](docs/BIBLIO_VERIFIEE.md).
- `reports/` — un rapport par IA.
- `round2/` — critiques croisées et seconde passe.
- `synthesis/` — synthèse finale.
- `experiments/` — expériences, résultats et benchmarks.
- `notes/` — hypothèses et observations intermédiaires.
- `leaderboard/` — scores, sessions, hall of fame.
- `config/gematria_systems.yaml` — alphabets par langue et systèmes de gématrie configurables.
- `embedbabel/` — code commun : alphabets/gématrie, normalisation Unicode, sous-chaînes, segmentation IPA.
- `data/` — [README](data/README.md) et [sources](data/SOURCES.md) ; données téléchargées (jamais commitées).
- `scripts/` — `fetch_phonetic_dicts.py`, `fetch_etymology.py`, `build_graph.py`.
- `graph/` — [`schema.cypher`](graph/schema.cypher) et [documentation](graph/README.md) ; `docker-compose.yml` à la racine.
- `scripts/build_qdrant.py` — indexe les dictionnaires dans Qdrant (vecteurs structurels : n-grammes IPA hachés + gématrie).
- `viz/circle_trace.py` — tracé circulaire et GIF.
- `bench/` — harnais de bench versionné : `definitions/` (sources), `registry/` (inscrits + hash + exécutions),
  `trace.py` (arbre de raisonnement), `adapters.py`, `consensus.py`, `run.py`.
- `tests/` — tests pytest.

## Ordre d'exécution

```bash
pip install -r requirements.txt
python -m pytest -q                                   # 1. tests (géométrie, sous-chaînes, gématrie, graphe, étymologie)
python scripts/fetch_phonetic_dicts.py                # 2. dictionnaires -> data/dict/<lang>.tsv (réseau requis)
python scripts/fetch_etymology.py --words fr:écureuil en:squirrel   # 3. (optionnel, réseau) arêtes étymologiques
python scripts/build_graph.py --dry-run               # 4. rapport de taille sans base
export NEO4J_PASSWORD='…'; docker compose up -d       # 5. Neo4j
python scripts/build_graph.py --apply-schema --langs fr --varieties fr-FR --max-substring-len 4
python viz/circle_trace.py --lang fr --text "écureuil" --ipa auto --out out.gif   # 6. visualisation
```

### Données volumineuses (disque DATA) et Qdrant

`docker-compose.yml` monte les données Neo4j/Qdrant sous `DATA_ROOT` (défaut `D:/embedbabel-data`), hors du dépôt.
Les dictionnaires et le cache étymologique peuvent y être écrits via `--raw-dir` / `--dict-dir` / `--cache-dir`.

```bash
export DATA_ROOT='D:/embedbabel-data' NEO4J_PASSWORD='…'
docker compose --profile qdrant up -d                       # Neo4j + Qdrant (images épinglées)
python scripts/fetch_phonetic_dicts.py --raw-dir $DATA_ROOT/raw --dict-dir $DATA_ROOT/dict
python scripts/build_qdrant.py --dict-dir $DATA_ROOT/dict   # ajouter --dry-run pour compter seulement
```

### Harnais de bench versionné

```bash
python -m bench.run --register bench/definitions/embedbabel-bench-0.1.0.def.json   # inscrit (hash définition + prompt)
python -m bench.run --harness embedbabel-bench-0.1.0 --seed 0                       # trace, rapport, registre
python -m bench.run --harness embedbabel-bench-0.1.0 --seed 0 --resume              # reprise depuis le point de contrôle
```

Un harnais inscrit est immuable : modifier sa définition ou son prompt sans incrémenter `version` est refusé.
La sortie (`bench/runs/<id>/trace.json`, `report.json`) est identique à graine égale. Les adaptateurs fournis sont
factices (honnête, optimiste, bavard, cassé) et n'appellent aucun réseau.

Variantes : `--stream` (stdin, simulation du direct), `--color-by gematria`, `--step syllable` (IPA avec frontières
explicites uniquement), `--audio fichier.wav` (ASR optionnelle via `openai-whisper`, non installé par défaut),
`--export-dataset DIR` (images PNG + `metadata.jsonl` : mot, IPA, gématrie). En Python : `circle_trace.trace_to_array()`.

## V3 — Médaillon traducteur (simulation uniquement)

La V3 explore un médaillon micro/radio/décodeur/traducteur dans un cadre **strictement simulé**.

### LED7 Signature Bus

Le médaillon expose un bus de signature à **7 LEDs** :
- **6 LEDs périphériques** : vecteur de signature d’état (encodage choisi par l’IA),
- **1 LED centrale** : ponctuation + interface/bouton.

But : maximiser l’information utile sur les opérations internes (détection, traduction, confiance, temporalité, ambiguïté), pas “1 LED = 1 lettre”.

Voir la spec complète :
- `docs/LED7_SIGNATURE_BUS.md`

### Conception simulée

La simulation peut explorer la conception pluridisciplinaire d'un médaillon micro/radio/décodeur/traducteur à partir des recherches sémanticophonétiques et graphiquement gématriques. Le concept comprend un câblage transparent à trois conducteurs (or, argent et cuivre), des éléments simulés électromagnétiques, piézoélectriques et photovoltaïques, ainsi qu'un cadran réglable pour la fréquence radio. À la détection de l'injonction « traduis », le médaillon baisse le volume de la radio et traduit la réception.

Les IA peuvent chiffrer les fonctions en tokens et proposer d'autres fonctionnalités compatibles avec une représentation ternaire-binaire. Les dictionnaires alphabétiques et les six LEDs périphériques peuvent également être explorés comme supports de quantification ternaire-binaire des lettres prédites à partir des sons et du contexte précédent.

## Plugin DeepSeek (bench-harness-v2)

Plugin ajouté :
- `plugins/deepseek-r1-gématriphonéticospatiale-v0.1.0/`

Comportement :
- `go` : lance le bench,
- garde anti-pollution : si des messages précédents existent, le plugin demande une nouvelle conversation,
- sortie benchmark attendue en JSON strict.

## Protocole (résumé)

1. Donner exactement le prompt benchmark à chaque IA, idéalement sans contexte conversationnel préalable.
2. Exécuter les mêmes scénarios, baselines et contrôles négatifs.
3. Enregistrer les sorties normalisées (JSON + artefacts, ex. GIFs LED).
4. Comparer scores de rigueur, falsifiabilité, robustesse et utilité.
5. Ne retenir comme avancée qu’une hypothèse qui survit à une réfutation explicite.

## Hypothèses à tester (visualisation circulaire)

Le tracé circulaire est une **représentation**, pas une preuve : rien n'indique qu'il aide un modèle de vision à
« apprendre à parler ». À tester avec permutations et correction des comparaisons multiples :

1. Le motif visuel a-t-il une information prédictive (tâches : retrouver la langue, la classe sensorielle d'une norme
   psycholinguistique, l'homophonie) **au-delà** des baselines : lettres seules, IPA seul, motif aléatoire de même longueur ?
2. H0 : phonétique + morphologie suffisent ; H1 : la gématrie (couleur/valeurs) apporte un gain mesurable.
3. Le panneau phonétique apporte-t-il quelque chose de plus que le panneau lettres seul ?
4. Contrôle de fuite : les images ne doivent pas encoder la longueur ou la fréquence du mot au point de la rendre triviale.

Limites : positions angulaires fixées par l'ordre alphabétique (arbitraire), inventaire IPA tiré du dictionnaire,
pas de propriétés de symétrie particulières garanties, pas de résultat empirique à ce stade.

## Règle importante

Le benchmark ne cherche pas à confirmer EmbedBabel. Il cherche à déterminer si une représentation externe est utile et peut conduire à un gain réel (qualité/coût/latence/mémoire), sans déplacer simplement le coût ailleurs.

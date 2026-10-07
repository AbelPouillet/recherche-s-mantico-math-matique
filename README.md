# EmbedBabel — AI Research Benchmark


Ce dépôt sert à collecter, comparer et benchmarker des analyses IA autour d’EmbedBabel, puis à explorer une V3 ludique en **simulation** avec un médaillon traducteur instrumenté.
Il fournit aussi les ressources de base (dictionnaires phonétiques, graphe Neo4j, visualisation circulaire) pour
tester l'idée. Toute la documentation est en français.

## Arborescence

- `PROMPT.md` — prompt identique fourni à chaque IA (sections 2bis à 2quinquies : phonétique/diachronie, mots à inventer,
  homophones, RAG Neo4j+Qdrant).
- `bench/prompts/` — prompts benchmark (V2/V3, quiz, speedrun).
- `dsh/embedbabel-dsh/` — **plugin DeepSeek Harness** (Cordis, ESM JavaScript) : garde anti-pollution et
  injection du prompt de bench, plus un journal JSONL des tours.
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
- `bench/` — harnais de bench versionné : `definitions/` (sources), `registry/` (inscrits + hash + exécutions figées),
  `trace.py` (arbre de raisonnement), `adapters.py`, `consensus.py`, `run.py`, `ledger.py` (registre d'exécution hors git).
- `bench/perf/` — **mesure d'inférence** (H2) : client avec streaming et TTFT, échantillonnage VRAM/GPU/RAM,
  matrice de configuration, agrégation avec intervalles de confiance.
- `tests/` — tests pytest ; `tests/golden/` — empreinte dorée du harnais (non-régression).
- `pyproject.toml` — source de vérité des dépendances ; `.github/workflows/ci.yml` et `scripts/ci.py` — la CI.
- `docs/AUDIT_HARNESS_DSH_STRATA.md` — audit du 2026-10-07 : constats, mesures, plan de remédiation.

## Ordre d'exécution

```bash
pip install -e ".[dev]"                               # dépendances (pyproject.toml = source de vérité)
python scripts/ci.py                                  # CI complète : tests + run hors ligne + gate d'herméticité
python -m pytest -q                                   # 1. tests (géométrie, sous-chaînes, gématrie, graphe, étymologie, harnais)
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

### CI/CD

Une seule commande, la même en local et dans le workflow (`.github/workflows/ci.yml`) :

```bash
python scripts/ci.py            # suite complète + run hors ligne + gate d'herméticité
python scripts/ci.py --quick    # suite du harnais seulement
```

Elle vérifie trois choses :

1. **Les tests passent** (`pip install -e ".[dev]"` fournit toutes les dépendances ; sans elles, quatre
   modules ne collectent même pas).
2. **L'empreinte dorée du harnais est inchangée** : `tests/test_bench_golden.py` fige la trace d'un run
   hors ligne complet (6 adaptateurs factices, graine 0). Si elle bouge, deux campagnes ne sont plus
   comparables — il faut le documenter dans `bench/harnesses/CHANGELOG.md` puis régénérer avec
   `EMBEDBABEL_UPDATE_GOLDEN=1 python -m pytest tests/test_bench_golden.py`.
3. **Un run ne salit pas l'arbre de travail.** Le lanceur n'écrit jamais dans `bench/registry/`
   (fichier suivi par git) : chaque exécution dépose son enregistrement dans `<out>/run_record.json`,
   agrégé à la demande par `python -m bench.ledger collect --root bench/runs`. Sans cela, aucun gate
   « arbre propre » n'est possible et deux jobs parallèles se marchent dessus.

### Options du lanceur pour les modèles réels

| Option | Effet |
|---|---|
| `--no-cache` | ignore le cache de réponses : chaque étape est réellement rejouée par le modèle |
| `--tokenize auto\|off` | `auto` (défaut) recompte le contexte avec le tokenizer du serveur (`POST /tokenize`) avant d'exécuter et refuse le modèle si le contexte réel dépasse la limite ; `off` garde l'estimateur `chars/4` |

`report.json` publie désormais une section `telemetry` par modèle : tokens prompt/complétion **réels**,
débit médian, étapes **mesurées** vs **rejouées depuis le cache**, et l'écart estimateur `chars/4` /
tokenizer réel. Un run entièrement servi par le cache est marqué `measured: false` et
`replayed_from_cache: true` : il ne peut plus passer pour une campagne de mesure.

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

## Plugin DeepSeek Harness (vrai plugin Cordis)

Le dossier `dsh/embedbabel-dsh/` contient un **plugin DeepSeek Harness** au sens propre : un plugin
Cordis ESM JavaScript, chargé *in-process* par DSH, installé avec
`dsh plugin --profile <profil> add dsh/embedbabel-dsh`.

Comportement :

- `go` envoyé seul, dans une conversation **vierge** : le message est remplacé par le prompt de bench
  complet et le modèle est appelé normalement ;
- garde anti-pollution : si un tour antérieur existe, le tour est terminé **sans appel au modèle** et
  la raison est journalisée ;
- chaque décision est écrite en JSONL (`EMBEDBABEL_LOG`), matière première de la boucle
  d'auto-amélioration.

La règle de garde existe en deux exemplaires, **vérifiés par le même jeu de cas**
(`tests/fixtures/guard_cases.json`) : `bench/guard.py` (Python, utilisé par le harnais H1) et
`dsh/embedbabel-dsh/lib/guard.js` (JavaScript, utilisé par le plugin). Voir
[dsh/embedbabel-dsh/README.md](dsh/embedbabel-dsh/README.md).

> Historique : `plugins/deepseek-r1-gématriphonéticospatiale-v0.1.0/adapter.py` et
> `integrations/deepseek-harness-labsia/` ont été supprimés en 1.0.0. Le premier était un fichier
> Python avec une fonction `on_message()`, importé par le harnais — DSH ne charge aucun plugin Python.
> Le second ne contenait qu'un test de ce fichier. Les deux restent dans l'historique git.

## Mesure de la pipeline d'inférence (H2)

Distinct du harnais de bench : `bench/perf/` mesure le **débit, la latence et la mémoire** de la
pipeline sous une matrice de configuration. Aucun jugement de qualité n'y figure.

```bash
python -m bench.perf.run --list-workloads
python -m bench.perf.run --config bench/perf/matrix.example.json --dry-run
python -m bench.perf.run --config bench/perf/matrix.example.json --out artifacts/perf/mon-run
python -m bench.perf.run --host http://127.0.0.1:8080 --label strata --workloads decode-512 --repetitions 3
```

Ce que le harnais applique par défaut, parce que sans cela la mesure est fausse : **nonce unique par
requête** (sinon un cache de prompt fait passer un préfixe réutilisé pour un prefill mesuré),
**streaming** (sinon il n'existe aucun instant « premier token »), et publication de `cache_n` — une
ligne dont le prefill a été rejoué est marquée `replayed_prefill` et comptée à part. Sortie :
`report.json` (toutes les répétitions, médianes, IC à 95 % par bootstrap) et `SUMMARY.md`.

## Boucle d'auto-amélioration et auto-réglage continu

`bench/selfimprove/` ferme la boucle : journal du plugin → score **externe** → décision pré-déclarée →
profil promu. Deux règles non négociables y sont codées en dur.

**Aucun modèle ne se note.** Un score ne peut venir que d'une **issue externe** (tâche réussie, tests
verts, tokens, temps mural, reprises). Une issue contenant un champ d'auto-évaluation
(`self_score`, `confidence`, `quality`, `sif`, `sdm`…) est **refusée** par `SelfAssessmentRefused` —
c'est la correction du défaut qui rendait le verdict du harnais 0.4.0 circulaire.

```bash
python -m bench.selfimprove.run score --decisions ~/.dsh/embedbabel/dsh-turns.jsonl \
    --outcomes artifacts/outcomes.jsonl --split
python -m bench.selfimprove.run accept --baseline artifacts/base.json \
    --candidate artifacts/cand.json --margin 5
```

**Auto-réglage piloté par les ressources locales.** La meilleure configuration d'inférence n'est pas
une constante : c'est une fonction de la machine au moment où l'on infère (VRAM libre, RAM,
température, autres programmes). `bench/selfimprove/tune.py` la cherche et la **maintient**.

```bash
# une passe : remesure l'incumbent, essaie son voisinage, promeut au plus un candidat
python -m bench.selfimprove.run tune --config artifacts/perf/config-local.json \
    --space artifacts/space.json --workload decode-512 --metric decode_tok_s \
    --vram-budget-mib 23000 --gpu-temp-max-c 85 --margin 5 --max-trials 8

# mode continu : détecte la dérive de l'environnement et retente le voisinage
python -m bench.selfimprove.run tune ... --watch --rounds 0 --interval 900
python -m bench.selfimprove.run envelope        # l'enveloppe qui invalide un profil
```

Ce qui rend la démarche falsifiable : l'incumbent est **remesuré dans la même session** que chaque
candidat (sinon on appelle « amélioration » de la dérive thermique) ; une **marge pré-déclarée** est
exigée ; une configuration plus rapide mais qui dépasse le budget VRAM, la température maximale ou
assèche la RAM est **rejetée** ; le profil retenu porte son empreinte **et l'enveloppe matérielle**
sous laquelle il a été mesuré, et devient **périmé** si l'enveloppe change.

> Cas réel attrapé par ce mécanisme : un run lancé contre `llama-server` a mesuré **0,58 token/s**.
> Diagnostic — `D:\AI\Strata\engine\strata.exe` tournait déjà avec le modèle chargé et occupait
> **23,4 des 24,5 Go** de VRAM. Sur une RTX 4090 de 24 Go, **Strata et un `llama-server` ne peuvent
> pas cohabiter** : les campagnes comparatives doivent être séquentielles.

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

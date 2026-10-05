# EmbedBabel — AI Research Benchmark

Ce dépôt sert à collecter les rapports de plusieurs IA sur l'idée EmbedBabel, et fournit les ressources de
base (dictionnaires phonétiques, graphe Neo4j, visualisation circulaire) pour la tester. Toute la documentation
est en français.

## Arborescence

- `PROMPT.md` — prompt identique fourni à chaque IA (sections 2bis à 2quinquies : phonétique/diachronie, mots à inventer,
  homophones, RAG Neo4j+Qdrant, flux continu/vision et hypothèses audio).
- `reports/` — un rapport par IA.
- `round2/` — critiques croisées et seconde passe.
- `synthesis/` — synthèse finale.
- `experiments/` — expériences, résultats et benchmarks.
- `notes/` — hypothèses et observations intermédiaires.
- `config/gematria_systems.yaml` — alphabets par langue et systèmes de gématrie configurables.
- `embedbabel/` — code commun : alphabets/gématrie, normalisation Unicode, sous-chaînes, segmentation IPA.
- `data/` — [README](data/README.md) et [sources](data/SOURCES.md) ; données téléchargées (jamais commitées).
- `scripts/` — `fetch_phonetic_dicts.py`, `fetch_etymology.py`, `build_graph.py`.
- `graph/` — [`schema.cypher`](graph/schema.cypher) et [documentation](graph/README.md) ; `docker-compose.yml` à la racine.
- `viz/circle_trace.py` — tracé circulaire et GIF.
- `experiments/continuous_vs_vision/` — trajectoires, signature de chemin, baselines
  one-hot/cercle permuté/raster/motif aléatoire et expérience de prédiction du caractère suivant.
- `docs/ENVIRONNEMENT_EXPERIMENTAL.md` — environnement et protocole d'évaluation des modèles de code.
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
python -m experiments.continuous_vs_vision.run_experiment --lang fr --output out/metrics.json
```

Le banc utilise les dictionnaires `data/dict/<lang>.tsv` s'ils sont présents. Sinon,
il inscrit explicitement dans ses métriques qu'il utilise un mini-corpus jouet. Le
modèle de référence est une régression ridge numpy; PyTorch n'est pas requis. Le
résultat est un relevé expérimental, pas une preuve ni un classement.

Variantes : `--stream` (stdin, simulation du direct), `--color-by gematria`, `--step syllable` (IPA avec frontières
explicites uniquement), `--audio fichier.wav` (ASR optionnelle via `openai-whisper`, non installé par défaut),
`--export-dataset DIR` (images PNG + `metadata.jsonl` : mot, IPA, gématrie). En Python : `circle_trace.trace_to_array()`.

## Protocole du benchmark

1. Donner exactement `PROMPT.md` à chaque IA, idéalement sans contexte conversationnel préalable.
2. Demander un rapport complet et indépendant.
3. Enregistrer chaque réponse dans `reports/<nom-modele>.md`.
4. Une fois les premiers rapports réunis, fournir à chaque IA les rapports anonymisés/normalisés pour une deuxième passe critique.
5. Comparer les désaccords, les propositions expérimentales et les objections.
6. Ne retenir comme avancée qu'une hypothèse qui donne lieu à une expérience falsifiable.

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

Le benchmark ne cherche pas à confirmer EmbedBabel. Il cherche à déterminer si l'idée produit une représentation utile et si cette représentation peut conduire à une réduction réelle du coût d'inférence.

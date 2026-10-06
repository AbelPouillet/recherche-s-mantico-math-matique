# EmbedBabel — AI Research Benchmark

Ce dépôt sert à collecter, comparer et benchmarker des analyses IA autour d’EmbedBabel, puis à explorer une V3 ludique en **simulation** avec un médaillon traducteur instrumenté.

## Organisation

- `PROMPT.md` — prompt de base identique fourni à chaque IA.
- `bench/prompts/` — prompts benchmark (V2/V3, quiz, speedrun).
- `plugins/` — plugins par modèle, format : `{$modelID-gématriphonéticospatiale-vx.x.x}`.
- `docs/` — règles de jeu, protocoles, specs d’encodage.
- `reports/` — un rapport par IA.
- `round2/` — critiques croisées et seconde passe.
- `synthesis/` — synthèse finale.
- `experiments/` — expériences, résultats et benchmarks.
- `notes/` — hypothèses et observations intermédiaires.
- `leaderboard/` — scores, sessions, hall of fame.

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

## Règle importante

Le benchmark ne cherche pas à confirmer EmbedBabel. Il cherche à déterminer si une représentation externe est utile et peut conduire à un gain réel (qualité/coût/latence/mémoire), sans déplacer simplement le coût ailleurs.

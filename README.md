# EmbedBabel — AI Research Benchmark

Ce dépôt sert à collecter les rapports de plusieurs IA sur l'idée EmbedBabel.

## Organisation

- `PROMPT.md` — prompt identique fourni à chaque IA.
- `reports/` — un rapport par IA.
- `round2/` — critiques croisées et seconde passe.
- `synthesis/` — synthèse finale.
- `experiments/` — expériences, résultats et benchmarks.
- `notes/` — hypothèses et observations intermédiaires.

## Protocole

1. Donner exactement `PROMPT.md` à chaque IA, idéalement sans contexte conversationnel préalable.
2. Demander un rapport complet et indépendant.
3. Enregistrer chaque réponse dans `reports/<nom-modele>.md`.
4. Une fois les premiers rapports réunis, fournir à chaque IA les rapports anonymisés/normalisés pour une deuxième passe critique.
5. Comparer les désaccords, les propositions expérimentales et les objections.
6. Ne retenir comme avancée qu'une hypothèse qui donne lieu à une expérience falsifiable.

## Règle importante

Le benchmark ne cherche pas à confirmer EmbedBabel. Il cherche à déterminer si l'idée produit une représentation utile et si cette représentation peut conduire à une réduction réelle du coût d'inférence.

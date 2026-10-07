"""Harnais de performance de la pipeline d'inférence (H2).

Ce paquet est **distinct** du harnais de bench (`bench/run.py`, H1). H1 note la conformité de
processus et la qualité rédactionnelle d'une analyse produite par un modèle. H2 mesure la
**pipeline** : prefill, décodage, latence au premier token, mémoire, sous une matrice de
configuration.

Les deux ne partagent volontairement ni schéma ni verdict : `bench/perf/report.json` ne contient
aucun jugement de qualité, et `bench/runs/<id>/report.json` ne contient aucun débit.

Sous-modules :

- `client`    : client HTTP (chat, streaming SSE, `/props`, `/slots`, `/metrics`, `/tokenize`) ;
- `resources` : échantillonnage des ressources locales pendant une mesure (VRAM, GPU, RAM) ;
- `matrix`    : description de la matrice, exécution des cellules, agrégation ;
- `run`       : point d'entrée en ligne de commande.
"""

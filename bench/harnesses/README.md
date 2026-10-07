# bench/harnesses : harnais de bench versionnés

Chaque harnais construit vit ici, nommé, versionné et documenté. C'est un sous-dossier de ce dépôt
(pas un dépôt git imbriqué) ; il peut être extrait en dépôt séparé ou en sous-module sans changer son contenu.

## Nommage

```
bench/harnesses/<famille>/<MAJEUR.MINEUR.PATCH>/
    harness.def.json     définition (source de vérité)
    HARNESS.md           documentation de CETTE version
```

- **Famille** : minuscules et tirets (`embedbabel-bench`). **Identifiant** : `<famille>-<version>`
  (`embedbabel-bench-0.2.0`), c'est le nom du fichier dans `bench/registry/`.
- **MAJEUR** : le schéma de sortie, les états ou la façon de mesurer changent (résultats non comparables).
- **MINEUR** : tâches, paquets de contexte, préférences ou adaptateurs ajoutés ou modifiés.
- **PATCH** : correction sans effet sur les mesures.
- Les versions sont **immuables** : on n'édite jamais une version inscrite, on en crée une nouvelle.

## Ce que contient une définition

`name`, `version`, `prompt` (fichier donné à tous les modèles), `doc` (chemin de `HARNESS.md`), `schema`,
`tasks` (id → paquets de contexte `{id, path, required}`), `preferences` (tolérance de budget, axes et poids de
corrélation mot / sous-mot étymologique / phonème, seuil H0).

## Créer une nouvelle version

1. Copier le dossier de la dernière version vers `<famille>/<nouvelle version>/`.
2. Modifier `harness.def.json` (`version`, `doc`) puis `HARNESS.md`.
3. Ajouter une entrée au [CHANGELOG](CHANGELOG.md) : ce qui change, pourquoi, ce qui devient non comparable.
4. Inscrire : `python -m bench.run --register bench/harnesses/<famille>/<version>/harness.def.json`.
5. Lancer `python -m pytest -q` : `tests/test_bench_catalog.py` vérifie nom, dossier, documentation et hash.

## Garanties

Le registre (`bench/registry/<id>.json`) stocke la définition, son **hash** (définition + prompt + documentation +
fichiers de paquets) et la liste des exécutions. `--register` refuse une définition modifiée sous la même version ;
`DECLARER` refuse de lancer un harnais dont un fichier couvert a changé depuis l'inscription.

## Versions

| Identifiant | Statut | Notes |
|---|---|---|
| `embedbabel-bench-1.0.0` | **actuelle** | façon de mesurer corrigée : plus d'exclusion des modèles réels sur un budget qu'ils n'ont pas annoncé, verdict étiqueté comme opinion agrégée, télémétrie réelle publiée ([doc](embedbabel-bench/1.0.0/HARNESS.md)) |
| `embedbabel-bench-0.4.0` | ancienne | 0.3.0 + adaptateur llama.cpp ; verdicts non comparables avec 1.0.0 ([doc](embedbabel-bench/0.4.0/HARNESS.md)) |
| `embedbabel-bench-0.3.0` | ancienne | 0.2.0 + adaptateurs de modèles réels Ollama et manuel ([doc](embedbabel-bench/0.3.0/HARNESS.md)) |
| `embedbabel-bench-0.2.0` | ancienne | schéma V2, limite de contexte, budget en tokens, grille des comptes rendus ([doc](embedbabel-bench/0.2.0/HARNESS.md)) |
| `embedbabel-bench-0.1.0` | obsolète | définition dans `bench/definitions/`, schéma de sortie inventé (voir CHANGELOG) |

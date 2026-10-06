# Changelog des harnais

## Registre (toutes les versions)
- Le hash d'inscription est calculé sur le contenu avec fins de ligne normalisées (CRLF -> LF). Avant, il
  dépendait de `core.autocrlf` : une copie de travail Windows et une copie Linux donnaient des hashs différents.
  Les hashs de 0.1.0 et 0.2.0 ont été recalculés (même contenu, aucune exécution perdue) avant toute publication
  des commits correspondants. À partir de là, un changement de hash signifie un vrai changement de contenu.

## embedbabel-bench

### 0.2.0
- Sortie validée contre `EMBEDBABEL_BENCH_V2.md` : 10 clés, 6 tags (dont `NON_FALSIFIABLE`), réfutation, baselines,
  SDM/SIF justifiés. Les sous-champs sont une convention du harnais (V2 ne les définit pas).
- Limite de contexte annoncée par chaque modèle, garde (refus/troncature) et journal.
- Budget en tokens estimés (`ceil(caractères / 4)`), compte rendu par étape, point de reprise, reprise par un
  adaptateur neuf depuis le seul compte rendu.
- Adaptateurs : `honnete`, `optimiste` (budget ÷3), `bavard` (dépasse sa limite de contexte), trois variantes cassées.
- Audit croisé des comptes rendus (grille de 4 critères) et conclusion collective.
- Le hash d'inscription couvre aussi la documentation et les paquets des tâches.
- CLI : `--models FICHIER --task ID --budget TOKENS --out DIR`.
- **Non comparable avec 0.1.0** (schéma et mesures différents).

### 0.1.0 (obsolète)
- Première version (commits `255b8ec` à `4b89ec0`, branche `merge`). Définition dans `bench/definitions/`.
- Le schéma de sortie (`model`, `claims`, `declared_cost`, `correlations`, `verdict`) était **inventé** et non
  conforme à V2 ; pas de limite de contexte, ni de compte rendu budgété, ni d'audit de plan.
- Conservée dans le registre pour l'historique ; le lanceur actuel ne l'exécute plus (pas de `tasks`).

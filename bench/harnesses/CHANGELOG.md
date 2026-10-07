# Changelog des harnais

## Registre (toutes les versions)
- Le hash d'inscription est calculé sur le contenu avec fins de ligne normalisées (CRLF -> LF). Avant, il
  dépendait de `core.autocrlf` : une copie de travail Windows et une copie Linux donnaient des hashs différents.
  Les hashs de 0.1.0 et 0.2.0 ont été recalculés (même contenu, aucune exécution perdue) avant toute publication
  des commits correspondants. À partir de là, un changement de hash signifie un vrai changement de contenu.

## embedbabel-bench

### 1.0.0
- **Version MAJEURE** au sens du [README](README.md) : *la façon de mesurer change*, donc les verdicts
  ne sont pas comparables avec 0.2.0 – 0.4.0. Le prompt, le schéma de sortie, les étapes et le contenu
  des paquets sont **inchangés** : quand tous les paquets tiennent, le contexte envoyé au modèle est
  octet pour octet celui de 0.4.0 (vérifié par
  `tests/test_bench_catalog.py::test_major_version_changes_nothing_the_model_sees`).
- **Correction du défaut n° 1 : `ecart_budget` n'exclut plus les modèles réels.** Le critère comparait
  le budget *annoncé* au budget consommé ; un adaptateur réel n'annonce rien, le harnais lui impose
  `budget / 3`. Appliqué aux modèles réels, il les excluait tous : les trois campagnes archivées
  (`llama-smoke`, `llamacpp-2`, `ollama-2`) se terminaient en « aucun consensus », `kept: []`.
  Nouveau champ `declares_budget` par adaptateur ; pour un modèle réel, le critère devient
  `budget_respecte` (consommation réelle ≤ budget accordé) et l'écart reste publié comme métrique
  `budget_gap_pct`.
- **Correction du défaut n° 2 : le verdict de tête n'est plus circulaire.** `weighted_support` et le
  verdict « H0 retenue » / « à tester contre baselines » étaient calculés à partir de la
  `correlation_map.solid` écrite par le modèle lui-même. Le champ devient `auto_declaration_agregee`,
  accompagné de `decision_kind: "opinion_agregee_de_modeles"` et d'une interprétation explicite, et
  **il ne produit plus de verdict**. Les verdicts deviennent `aucun modèle retenu`,
  `aucune décision majoritaire` ou `opinion agrégée : GO|NO-GO`.
- **`cross_audit` → `grille_comptes_rendus`.** Aucun modèle n'auditait le compte rendu d'un autre :
  la grille est calculée par le harnais, mais `report.json` attribuait l'audit à un modèle. Champ
  renommé (`audits` → `grille_comptes_rendus`).
- **`PROMPT.md` devient un paquet obligatoire** de `v2-complet` : le prompt de sortie y renvoie pour
  la description du projet, le laisser optionnel notait des analyses faites à l'aveugle.
- **Consommation réelle prioritaire** : `consumed_total_real` (compteur serveur) prime sur `chars/4`
  pour le drapeau `budget_depasse`, qui était trop clément de 12 à 40 %.
- Nouvelle section `telemetry` par modèle (tokens réels, débit médian, étapes mesurées vs rejouées,
  écart estimateur/tokenizer) et `replayed_from_cache` dans les évaluations.
- Registre d'exécution déplacé dans `<out>/run_record.json` (`bench/ledger.py`) : un run n'écrit plus
  jamais dans `bench/registry/`, donc il ne salit plus l'arbre de travail.
- **La garde anti-pollution quitte le pseudo-plugin Python** (`plugins/deepseek-r1-*/adapter.py`,
  supprimé) pour `bench/guard.py`, module canonique et testé. `DECLARER` publie désormais
  `declares_budget` par modèle, et `PLANIFIER` journalise l'appel `bench.guard` au lieu du nom d'un
  plugin chargé par glob. Les règles peuvent être surchargées par un harnais via
  `preferences.context_guard`. Le plugin DSH réel (`dsh/embedbabel-dsh/`, JavaScript/Cordis)
  réimplémente la même règle et partage le **même jeu de cas** (`tests/fixtures/guard_cases.json`).
  *Conséquence* : le contenu de `trace.json` change (nœuds `plugin_call` et `declares_budget`), donc
  l'empreinte dorée est régénérée — voir `tests/test_bench_golden.py`.

### 0.4.0
- Adaptateur `llamacpp` (`bench/live.py`) : `llama-server` lancé puis arrêté par le harnais après chaque modèle (ou serveur
  déjà lancé via `params.host`), GGUF, `ngl` et options réglables selon les ressources.
- Les adaptateurs peuvent exposer `close()` ; le harnais l'appelle après chaque modèle et après le test de reprise.
- Définition, prompt, tâches, schéma et mesures **inchangés** : comparable avec 0.2.0 et 0.3.0 (seule la documentation change).

### 0.3.0
- Adaptateurs de modèles réels (`bench/live.py`) : `ollama` (HTTP local) et `manuel` (copier-coller, ex. DeepSeek web).
- Le texte du contexte chargé est transmis aux adaptateurs ; réponses Ollama mises en cache ; compteurs réels du serveur
  consignés (`live_usage`) ; entrée `params` dans le fichier modèles.
- Définition, prompt, tâches, schéma et mesures **inchangés** : résultats comparables avec 0.2.0. Seule la documentation
  (couverte par le hash) change, d'où la nouvelle version.
- Non déterministe pour les modèles réels : le test de reprise y est une mesure de reproductibilité.

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

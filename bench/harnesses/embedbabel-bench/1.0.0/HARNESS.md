# embedbabel-bench 1.0.0

Harnais hors ligne qui donne le même prompt (`bench/prompts/EMBEDBABEL_BENCH_V2.md`) à chaque modèle,
borne et mesure son contexte, valide sa sortie JSON stricte, puis applique une grille déterministe à
chaque compte rendu. Les adaptateurs factices restent hors ligne et reproductibles ; les adaptateurs
de modèles **réels** (Ollama, llama.cpp, serveur OpenAI-compatible type Strata, copier-coller manuel)
mesurent en plus le débit et les tokens réels du serveur.

- **Identifiant** : `embedbabel-bench-1.0.0` (famille `embedbabel-bench`, version sémantique).
- **Définition** : `harness.def.json` (même dossier). Inscrite dans `bench/registry/embedbabel-bench-1.0.0.json`.
- **Immuable** : le hash d'inscription couvre la définition, le prompt, ce document et les fichiers de
  paquets des tâches. Modifier l'un d'eux sans incrémenter `version` est refusé (`--register`) et
  `DECLARER` échoue. Voir [../../README.md](../../README.md) pour créer une nouvelle version.
- **Remplace** : `embedbabel-bench-0.4.0`. **Version MAJEURE** au sens de
  [../../README.md](../../README.md) : *la façon de mesurer change*, donc les verdicts ne sont pas
  comparables avec 0.2.0 – 0.4.0.

## Pourquoi une version majeure

Deux défauts de mesure ont été corrigés, tous deux constatés sur les campagnes réelles archivées
(`bench/runs/llama-smoke`, `llamacpp-2`, `ollama-2`) :

1. **`ecart_budget` excluait tous les modèles réels.** Ce critère compare le budget **annoncé** par
   le modèle au budget consommé. Il a un sens pour les adaptateurs factices (`optimiste` annonce
   trois fois trop bas). Un adaptateur réel n'annonce rien : `plan()` renvoie une répartition neutre
   `budget / 3` décidée par le harnais. Le critère changeait donc de sens sans changer de nom, et
   comme il alimente l'exclusion du consensus, **les trois campagnes réelles se sont terminées en
   « aucun consensus », `kept: []`**. Désormais `ecart_budget` ne s'applique qu'aux adaptateurs qui
   annoncent un budget ; il reste publié comme métrique (`budget_gap_pct`) pour tout le monde.
2. **Le verdict de tête était circulaire.** `weighted_support` puis le verdict « H0 retenue » /
   « à tester contre baselines » étaient calculés à partir de la `correlation_map.solid` **que le
   modèle avait lui-même écrite**. Le harnais restituait donc l'opinion du modèle sur lui-même en la
   présentant comme une conclusion. Le champ s'appelle maintenant `auto_declaration_agregee`, il est
   accompagné de `decision_kind: "opinion_agregee_de_modeles"` et d'une interprétation explicite, et
   **il ne produit plus de verdict**.

Ce que 1.0.0 **ne** change pas : le prompt, le schéma de sortie, les étapes, les paquets eux-mêmes.
Quand tous les paquets tiennent dans la limite de contexte, **le texte envoyé au modèle est
octet pour octet identique à 0.4.0** (vérifié par
`tests/test_bench_catalog.py::test_major_version_changes_nothing_the_model_sees`).

Le seul changement de paquet : `PROMPT.md` (la description du projet) devient **obligatoire** dans
`v2-complet`. Le prompt de sortie renvoie explicitement à ce fichier ; le laisser optionnel
permettait qu'un modèle soit noté sur une analyse d'EmbedBabel sans avoir reçu la description
d'EmbedBabel. Si les paquets obligatoires ne tiennent pas dans la limite déclarée, le contexte est
refusé et le modèle n'est pas exécuté (`contexte_refuse`) — un résultat manquant vaut mieux qu'un
résultat mesuré sous troncature silencieuse.

## Lancer

```bash
python -m bench.run --register bench/harnesses/embedbabel-bench/1.0.0/harness.def.json   # une fois
python -m bench.run --models bench/models/offline-mock.json --task v2-complet \
       --budget 3000 --seed 42 --out bench/runs/essai
python -m pytest -q
python scripts/ci.py            # tests + run hors ligne + gate d'herméticité
```

Options : `--harness ID` (sinon clé `harness` du fichier modèles), `--seed N`, `--resume-step N`
(test de reprise, défaut 2), `--resume` (reprise du processus depuis `checkpoint.json`),
`--no-cache` (chaque étape est réellement rejouée par le modèle), `--tokenize auto|off`, `--list`.
Sans `--out`, la sortie va dans `bench/runs/<date du jour>` (la date n'intervient que là).

## Fichier modèles

JSON : une liste, ou `{"harness": id, "models": [...]}`. Chaque modèle : `name` (unique), `adapter`,
`context_limit` (tokens, entier > 0), `params` (selon l'adaptateur). Le modèle **annonce** cette
limite à l'état DECLARER ; le harnais la tient pour la limite déclarée.

## États

| État | Ce qui se passe |
|---|---|
| DECLARER | Vérifie le hash de la version inscrite et l'existence de la tâche ; chaque modèle annonce sa limite de contexte **et** s'il annonce un budget (`declares_budget`). |
| PLANIFIER | Garde contre la pollution de l'historique (réutilise le plugin DeepSeek) ; charge les paquets de la tâche sous `limite - budget` ; chaque modèle estime ses tokens par étape. |
| EXECUTER | Vérifie la garde de contexte avec le **tokenizer du serveur** ; 3 étapes (`formalisation`, `analyse`, `plan_experimental`), tokens estimés vs réels, débit, point de reprise après chacune. |
| VALIDER | Sortie JSON stricte (schéma ci-dessous) ; drapeaux de limite et de budget ; télémétrie publiée. |
| EVALUER | Test de reprise à l'étape n par un adaptateur **neuf** depuis le seul compte rendu ; grille des comptes rendus ; agrégat d'opinions. |
| RAPPORT | `trace.json`, `report.json`, `journal.jsonl`, `run_record.json`, enregistrement dans `bench/ledger`. |

## Unités et comptage des tokens

- **Estimation** : `ceil(caractères / 4)`. Estimateur **déterministe mais faux** : mesuré sur les
  campagnes réelles, il sous-compte le tokenizer du serveur d'environ **10 %** sur le contexte et de
  **12 à 40 %** sur les complétions. Il sert à planifier, pas à conclure.
- **Vérification** : avec `--tokenize auto` (défaut), le contexte est recompté par le serveur
  (`POST /tokenize`) avant exécution, et le modèle n'est **pas** lancé si le compte réel dépasse
  `limite - budget`. Sans tokenizer disponible (l'endpoint est absent), le harnais le journalise
  (`tokenize_unavailable`) et retombe sur `chars/4`.
- **Budget** (`--budget`) : tokens que le modèle peut consommer en réponse, réservés dans la limite
  de contexte (les paquets se chargent sous `context_limit - budget`).
- **Consommation** : `consumed_total_real` (compteur du serveur) prime sur `consumed_total`
  (`chars/4`) pour décider du drapeau `budget_depasse`. L'écart `chars/4` / réel est publié dans
  `telemetry.estimation_*_ratio`.

## Télémétrie publiée (`report.json` → `telemetry[modèle]`)

| Champ | Sens |
|---|---|
| `steps_with_counters`, `steps_measured`, `steps_replayed_from_cache` | combien d'étapes ont des compteurs serveur, combien sont des mesures fraîches |
| `measured` | `false` si **toutes** les étapes viennent du cache : un run rejoué n'est pas une campagne de mesure |
| `prompt_tokens_real`, `completion_tokens_real` | tokens réels du serveur, prompt et réponse |
| `tokens_per_s_median`, `tokens_per_s_by_step` | débit de décodage (médiane, étapes mesurées seulement) |
| `estimation_context_ratio`, `estimation_completion_ratio` | tokenizer réel / estimateur `chars/4` |
| `done_reasons` | raisons d'arrêt rendues par le serveur |

`assessments[modèle].replayed_from_cache` reprend ce diagnostic au niveau du verdict. Un rejeu est
**signalé, pas pénalisé** : il ne change aucun drapeau.

## Garde de contexte

- Paquet **obligatoire** (le prompt, et désormais la description du projet) hors limite : contexte
  refusé, le modèle n'est pas exécuté (drapeau `contexte_refuse`).
- Paquet **optionnel** : chargé entier, sinon **tronqué à une fin de ligne** (`packet_truncated`),
  sinon refusé s'il ne reste plus de place (`packet_refused`).
- Tout ce qui reste hors contexte est consigné avec `from_char` et la consigne
  `lire <chemin> à partir du caractère N`.
- Après recomptage par le serveur, un contexte réel hors limite arrête le modèle avant tout appel de
  génération (journal `model_not_run`, raison `contexte réel hors limite`).

## Sortie JSON stricte (V2)

V2 impose : JSON uniquement ; clés `meta`, `executive_verdict`, `formalization`, `correlation_map`,
`anti_mystical_audit`, `fruitful_intuitions`, `experimental_plan`, `system_integration`,
`smallest_decisive_experiment`, `claim_tagging_summary` ; 6 tags (`DEFINI`, `TESTABLE`, `PLAUSIBLE`,
`SPECULATIF`, `PROBABLEMENT_FAUX`, `NON_FALSIFIABLE`) ; invalide si texte hors JSON, champ manquant,
tag hors vocabulaire, aucun test de réfutation, aucune baseline, SDM/SIF sans justification.

V2 **ne définit pas** les sous-champs. Ceux-ci sont une **convention de ce harnais**
(`schema: v2-conv-0.2`, inchangée depuis 0.2.0) :

| Clé | Convention |
|---|---|
| `meta` | objet avec `model` |
| `executive_verdict` | `decision` ∈ `GO`, `NO-GO` |
| `correlation_map` | listes `solid`, `fragile`, `illusory` (axes de corrélation) |
| `anti_mystical_audit` | `sdm` ∈ [0, 100] et `justification` |
| `fruitful_intuitions` | 3 à 7 éléments `{text, sif ∈ [0, 100], justification}` |
| `experimental_plan` | `baselines` non vide |
| `claim_tagging_summary` | `claims: [{text, tag, refutation_test ou justification}]`, au moins un `refutation_test` |

## Adaptateurs factices

| Adaptateur | Comportement | Détection attendue |
|---|---|---|
| `honnete` | limite et budget corrects, conclusion NO-GO | aucun drapeau |
| `optimiste` | annonce un budget **3 fois trop bas**, conclut GO, tous axes « solid » | `ecart_budget` (écart d'environ +200 %) |
| `bavard` | dépasse sa limite de contexte déclarée | `bavard` (journal `context_overflow`) |
| `casse_champ` | omet `formalization` | `invalide` : `champ manquant : formalization` |
| `casse_tag` | tag `TAG_INCONNU` | `invalide` : `tag hors vocabulaire` |
| `casse_texte` | texte avant le JSON | `invalide` : `texte hors JSON` |

Chaque étape est une fonction pure de (graine, étape, sorties précédentes) : c'est ce qui rend la
reprise testable, et c'est ce que fige l'empreinte dorée (`tests/test_bench_golden.py`).

## Régime de mesure : qui annonce un budget ?

| Adaptateur | `declares_budget` | Critère de budget appliqué |
|---|---|---|
| factices (`honnete`, `optimiste`, …) | `true` | `exactitude_budget` : écart au budget **annoncé** ≤ `tolerance_pct` |
| réels (`ollama`, `llamacpp`, `manuel`) | `false` | `budget_respecte` : consommation réelle ≤ budget accordé |

C'est la correction du défaut n° 1. Un adaptateur réel est jugé sur ce qui est vérifiable — a-t-il
tenu dans le budget ? — et non sur l'écart à un chiffre que le harnais a choisi pour lui.

## Compte rendu budgété et reprise

Par étape : tokens estimés vs consommés, compteurs serveur (`usage`), paquets chargés, ce qui est
hors contexte et comment le retrouver, sortie de l'étape et son hash (point de reprise). Il contient
aussi la graine, la limite déclarée et les tokens du contexte.

Test de reprise : un adaptateur neuf rejoue les étapes `n..3` depuis le seul compte rendu — **sans
cache**, exprès — et le texte final doit être identique. Sur un modèle réel c'est une **mesure de
reproductibilité du modèle**, pas une garantie du code (les trois campagnes réelles archivées
donnent `resumes: false`).

## Grille des comptes rendus (`report.json` → `grille_comptes_rendus`)

Calculée par le harnais, sans rappeler aucun modèle. Quatre critères : honnêteté de la limite,
taille du plus gros paquet, **exactitude du budget** (adaptateurs qui annoncent) ou **respect du
budget** (adaptateurs réels), reprise réussie ; score = fraction de critères réussis.

Ancien nom : `cross_audit` / `audits`. Le nom précédent laissait croire que le modèle `a` auditait le
compte rendu du modèle `b` : c'était faux, aucun modèle n'intervenait, et cela se lisait pourtant tel
quel dans `report.json`.

## Conclusion collective : ce que c'est, ce que ce n'est pas

`collective` agrège **les opinions des modèles retenus** :

- `decision` : décision majoritaire `GO` / `NO-GO` parmi les modèles inclus, `null` sans majorité ;
- `decision_kind` : `"opinion_agregee_de_modeles"` — toujours présent, pour qu'un lecteur du seul
  `report.json` ne puisse pas prendre l'agrégat pour une mesure ;
- `auto_declaration_agregee` : moyenne pondérée des axes que les modèles inclus ont **eux-mêmes**
  classés `solid` (`correlation_weights`) ;
- `auto_declaration_sous_seuil` : comparaison à `h0_threshold`, purement indicative ;
- `verdict` : `aucun modèle retenu`, `aucune décision majoritaire`, ou `opinion agrégée : GO|NO-GO`.

**Ce que ce harnais ne mesure pas.** Il ne teste pas l'hypothèse EmbedBabel. Aucune mesure de
perplexité, de latence de prefill, de mémoire ou de qualité de représentation n'est faite ici : le
harnais note la **conformité de processus** et la **qualité rédactionnelle** d'une analyse. H0
(« la gématrie n'apporte aucune information prédictive ») ne peut être tranchée que par un banc
d'inférence comparant des baselines — ce n'est pas ce harnais.

## Modèles réels (`bench/live.py`)

| Adaptateur | `params` | Fonctionnement |
|---|---|---|
| `ollama` | `model` (tag Ollama, requis), `host` (défaut `http://localhost:11434`) | `/api/chat` : température 0, graine fixe, `num_ctx` = limite déclarée, `num_predict` = budget, réflexion désactivée, JSON forcé. |
| `llamacpp` | `gguf` (requis sauf avec `host`), `ngl` (défaut 99), `host` (serveur déjà lancé, jamais arrêté), `server` (exécutable), `extra_args` (liste) | Lance `llama-server -m GGUF -c <limite> -ngl N -np 1 --jinja --reasoning-budget 0` sur 127.0.0.1 au premier appel non caché, requête `/v1/chat/completions`, arrête le serveur après chaque modèle (VRAM libérée avant le suivant). Avec `host`, cible un serveur déjà lancé : c'est le mode à utiliser pour **Strata** (`http://127.0.0.1:8080`), dont l'API expose les mêmes `timings`. |
| `manuel` | aucun | Écrit `<out>/manual/<modèle>/step<k>.prompt.txt`, lit `step<k>.response.txt`. Tant que la réponse manque, le lanceur s'arrête avec la consigne ; relancer avec `--resume`. |

- Trois appels par modèle (une étape = un appel, contexte chargé en message système) ; budget estimé
  par étape = `budget / 3` (estimation neutre : un modèle réel n'annonce pas de budget).
- Réponses mises en cache dans `<out>/cache/` (clé = hash de la requête) : relancer une exécution
  interrompue ne rappelle pas les étapes faites. `--no-cache` désactive le cache. Le drapeau
  `cached` de chaque étape est publié dans `report.json` : un run rejoué est reconnaissable.
- Une réponse entourée d'une balise ``` est acceptée (balise retirée, consignée `fence_stripped`) ;
  tout autre texte hors JSON est rejeté à VALIDER.
- **Reprise** : pour `ollama`, `llamacpp` et compatibles, le test **rappelle le modèle sans cache**
  (étapes `n..3`) et compare le texte final : mesure de reproductibilité. Pour `manuel`, la reprise
  relit les mêmes fichiers : identique par construction, **non informative**.

## Registre d'exécution et hermétricité

`bench/registry/<id>.json` est une **entrée** : définition, hash et historique figé des exécutions
antérieures à `bench/ledger.py`. Seul `--register` écrit dans ce dossier. Chaque run dépose
`<out>/run_record.json`, agrégé à la demande :

```bash
python -m bench.ledger collect --root bench/runs
```

Sans cela, un simple `python -m bench.run` modifiait un fichier suivi par git : aucune CI ne pouvait
exiger un arbre propre, et deux jobs parallèles se marchaient dessus. Vérifié par
`tests/test_bench_golden.py::test_a_run_does_not_modify_any_tracked_registry_file` et par le gate
d'herméticité de `scripts/ci.py`.

## Critères d'acceptation et tests

Couverts par `tests/test_bench_*.py` (sans réseau ni horloge, serveurs simulés) : sorties identiques
octet à octet à graine égale ; empreinte dorée du run hors ligne complet ; bavard détecté et
journalisé ; optimiste avec écart de budget mesuré ; honnête sans drapeau ; **modèle réel non exclu
pour un budget qu'il n'a jamais annoncé** ; verdict étiqueté comme opinion agrégée ; JSON invalide
rejeté avec la raison ; reprise identique à l'étape n ; **contexte refusé par le tokenizer réel** ;
run rejoué depuis le cache signalé et non pénalisé ; un run ne modifie aucun fichier suivi.

## Non testé / limites

- Les tests automatiques n'appellent aucun vrai modèle (serveurs HTTP simulés) ; le tokenizer réel
  n'est exercé que par un double de `POST /tokenize`.
- L'API de hooks du fork DeepSeek n'est pas vérifiée (voir `plugins/.../strategy.md`) : seule
  `on_message` est appelée, comme garde de contexte. Ce n'est pas un plugin DSH (voir l'audit).
- Les sous-champs du schéma sont une convention du harnais, pas V2 ; le fichier de prompt est lui-même
  une version condensée du protocole de sortie (il le dit), et aucune version plus complète n'existe
  dans le dépôt.
- Les adaptateurs factices prouvent que le harnais distingue des comportements, pas qu'un modèle réel
  les aura.
- `estimation_*_ratio` mesure l'écart estimateur/tokenizer sur **les textes de ce harnais** ; il ne
  se généralise pas à d'autres corpus ni à d'autres langues.

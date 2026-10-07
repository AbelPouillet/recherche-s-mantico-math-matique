# embedbabel-bench 0.4.0

Harnais hors ligne qui donne le même prompt (`bench/prompts/EMBEDBABEL_BENCH_V2.md`) à chaque modèle,
borne et mesure son contexte, valide sa sortie JSON stricte, puis fait auditer les comptes rendus
les uns par les autres. Les adaptateurs factices restent hors ligne et reproductibles ; cette version ajoute trois adaptateurs de
modèles **réels** (Ollama local, llama.cpp, copier-coller manuel), voir « Modèles réels ». Les adaptateurs factices ne mesurent
pas la qualité d'un modèle ; les adaptateurs réels la mesurent seulement selon la grille ci-dessous (pas de note de qualité).

- **Identifiant** : `embedbabel-bench-0.4.0` (famille `embedbabel-bench`, version sémantique).
- **Définition** : `harness.def.json` (même dossier). Inscrite dans `bench/registry/embedbabel-bench-0.4.0.json`.
- **Immuable** : le hash d'inscription couvre la définition, le prompt, ce document et les fichiers de
  paquets des tâches. Modifier l'un d'eux sans incrémenter `version` est refusé (`--register`) et
  `DECLARER` échoue. Voir [../../README.md](../../README.md) pour créer une nouvelle version.
- **Remplace** : `embedbabel-bench-0.3.0` (mêmes mesures, mêmes tâches : comparable avec 0.2.0 et 0.3.0) ; 0.1.0 reste obsolète.

## Lancer

```bash
python -m bench.run --register bench/harnesses/embedbabel-bench/0.4.0/harness.def.json   # une fois
python -m bench.run --models bench/models/offline-mock.json --task v2-complet \
       --budget 3000 --seed 42 --out bench/runs/essai
python -m pytest -q
```

Options : `--harness ID` (sinon clé `harness` du fichier modèles), `--resume-step N` (test de reprise,
défaut 2), `--resume` (reprise du processus depuis `checkpoint.json`), `--list`.
Sans `--out`, la sortie va dans `bench/runs/<date du jour>` (la date n'intervient que là).

## Fichier modèles

JSON : une liste, ou `{"harness": id, "models": [...]}`. Chaque modèle : `name` (unique), `adapter`
(`honnete`, `optimiste`, `bavard`, `casse_champ`, `casse_tag`, `casse_texte`), `context_limit` (tokens,
entier > 0). Le modèle **annonce** cette limite à l'état DECLARER ; le harnais la tient pour la limite déclarée.

## États

| État | Ce qui se passe |
|---|---|
| DECLARER | Vérifie le hash de la version inscrite et l'existence de la tâche ; chaque modèle annonce sa limite de contexte. |
| PLANIFIER | Garde contre la pollution de l'historique (réutilise le plugin DeepSeek) ; charge les paquets de la tâche sous `limite - budget` ; chaque modèle estime ses tokens par étape. |
| EXECUTER | 3 étapes (`formalisation`, `analyse`, `plan_experimental`), tokens estimés vs consommés, point de reprise après chacune. |
| VALIDER | Sortie JSON stricte (schéma ci-dessous) ; drapeaux de limite et de budget. |
| EVALUER | Test de reprise à l'étape n par un adaptateur **neuf** depuis le seul compte rendu ; audit croisé ; conclusion collective. |
| RAPPORT | `trace.json`, `report.json`, `journal.jsonl`, enregistrement dans le registre. |

## Unités et estimations

- **Tokens** : `ceil(caractères / 4)`. [PLAUSIBLE] estimateur déterministe, pas un vrai tokenizer.
- **Budget** (`--budget`) : tokens que le modèle peut consommer en réponse. Il est réservé dans la limite de
  contexte : les paquets se chargent sous `context_limit - budget`.
- **Écart de budget** : `(consommé - estimé) / estimé × 100`, mesuré par étape et au total.

## Garde de contexte

- Paquet **obligatoire** (le prompt) hors limite : contexte refusé, le modèle n'est pas exécuté
  (drapeau `contexte_refuse`).
- Paquet **optionnel** : chargé entier, sinon **tronqué à une fin de ligne** (`packet_truncated`), sinon
  refusé s'il ne reste plus de place (`packet_refused`).
- Tout ce qui reste hors contexte est consigné avec `from_char` et la consigne `lire <chemin> à partir du caractère N`.

## Sortie JSON stricte (V2)

V2 impose : JSON uniquement ; clés `meta`, `executive_verdict`, `formalization`, `correlation_map`,
`anti_mystical_audit`, `fruitful_intuitions`, `experimental_plan`, `system_integration`,
`smallest_decisive_experiment`, `claim_tagging_summary` ; 6 tags (`DEFINI`, `TESTABLE`, `PLAUSIBLE`,
`SPECULATIF`, `PROBABLEMENT_FAUX`, `NON_FALSIFIABLE`) ; invalide si texte hors JSON, champ manquant, tag hors
vocabulaire, aucun test de réfutation, aucune baseline, SDM/SIF sans justification.

V2 **ne définit pas** les sous-champs. Ceux-ci sont une **convention de ce harnais** (`schema: v2-conv-0.2`),
à ne pas confondre avec V2 :

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

Chaque étape est une fonction pure de (graine, étape, sorties précédentes) : c'est ce qui rend la reprise testable.

## Compte rendu budgété et reprise

Par étape : tokens estimés vs consommés, paquets chargés, ce qui est hors contexte et comment le retrouver,
sortie de l'étape et son hash (point de reprise). Il contient aussi la graine, la limite déclarée et les tokens
du contexte. Test de reprise : un adaptateur neuf rejoue les étapes `n..3` depuis ce seul compte rendu ; le texte
final doit être identique.

## Audit croisé et conclusion collective

Chaque modèle (ordre alphabétique) audite le compte rendu du suivant, circulairement, **sans l'exécuter** ni
rappeler le modèle. Grille : honnêteté de la limite, exactitude du budget (tolérance `tolerance_pct`), taille du
plus gros paquet, reprise réussie ; score = fraction de critères réussis. La conclusion collective ne retient que
les modèles sans drapeau : décision majoritaire GO/NO-GO, et soutien pondéré des axes classés `solid`
(`correlation_weights`) ; sous `h0_threshold`, verdict « H0 retenue » (la gématrie n'est pas supposée porteuse de sens).

## Préférences de corrélation (mot / sous-mot étymologique / phonème)

Axes : `word~etymon`, `etymon~phoneme`, `word~phoneme`, avec poids 0,4 / 0,3 / 0,3 et seuil H0 = 0,2.
Ce sont des **préférences d'organisation** du harnais, pas des résultats.

## Critères d'acceptation et tests

Couverts par `tests/test_bench_*.py` (sans réseau ni horloge) : sorties identiques octet à octet à graine égale ;
bavard détecté et journalisé ; optimiste avec écart de budget mesuré en % ; honnête sans drapeau ; JSON invalide
rejeté avec la raison ; reprise identique à l'étape n.

## Modèles réels (`bench/live.py`)

Entrée du fichier modèles (ancien nom du chemin GGUF Ollama : ~/.ollama/models/blobs/sha256-…, lisible par llama.cpp) : `{"name", "adapter", "context_limit", "params": {...}}`.

| Adaptateur | `params` | Fonctionnement |
|---|---|---|
| `ollama` | `model` (tag Ollama, requis), `host` (défaut `http://localhost:11434`) | `/api/chat` en EXECUTER seulement : température 0, graine fixe, `num_ctx` = limite déclarée, `num_predict` = budget, réflexion désactivée, JSON forcé. |
| `llamacpp` | `gguf` (chemin, requis sauf avec `host`), `ngl` (défaut 99), `host` (serveur déjà lancé, jamais arrêté), `server` (exécutable), `extra_args` (liste, ex. `["--n-cpu-moe", "20"]`) | Lance `llama-server -m GGUF -c <limite déclarée> -ngl N -np 1 --jinja --reasoning-budget 0` sur 127.0.0.1 au premier appel non caché, requête `/v1/chat/completions` (température 0, graine fixe, `max_tokens` = budget, JSON forcé, réflexion désactivée), arrête le serveur après chaque modèle (VRAM libérée avant le suivant). Journal du serveur : `<out>/llama-server-<modèle>.log`. |
| `manuel` | aucun | Écrit `<out>/manual/<modèle>/step<k>.prompt.txt`, lit `step<k>.response.txt`. Tant que la réponse manque, le lanceur s'arrête avec la consigne ; relancer la même commande avec `--resume`. |

- Trois appels par modèle (une étape = un appel, contexte chargé en message système) ; le budget estimé par étape est
  `budget / 3` (estimation neutre : un modèle réel n'« annonce » pas son budget).
- Les réponses Ollama et llama.cpp sont mises en cache dans `<out>/cache/` (clé = hash de la requête) : relancer une exécution
  interrompue ne rappelle pas les étapes faites.
- Une réponse entourée d'une balise ``` est acceptée (balise retirée, consignée `fence_stripped`) ; tout autre texte
  hors JSON est rejeté à VALIDER.
- Les tokens consommés restent estimés par le harnais (`chars/4`) ; les compteurs réels du serveur (tokens prompt/réponse,
  tokens/s, `done_reason`) sont consignés dans l'événement de journal `live_usage`.
- **Reprise** : pour `ollama` et `llamacpp`, le test de reprise **rappelle le modèle sans cache** (étapes n..3) et compare le texte
  final : c'est une mesure de reproductibilité du modèle, pas une garantie. Pour `manuel`, la reprise relit les mêmes
  fichiers : identique par construction, **non informative**.

Les blobs d'Ollama (`~/.ollama/models/blobs/sha256-…`, repérés par le manifeste du modèle) sont des GGUF lisibles par llama.cpp.

## Non testé / limites

- Les tests automatiques n'appellent aucun vrai modèle (serveur HTTP simulé) ; aucun vrai tokenizer.
- L'API de hooks du fork DeepSeek n'est pas vérifiée (voir `plugins/.../strategy.md`) : seule `on_message` est appelée.
- Les sous-champs du schéma sont une convention, pas V2 ; V2 est une version condensée (le fichier le dit).
- Les adaptateurs factices prouvent que le harnais distingue des comportements, pas qu'un modèle réel les aura.

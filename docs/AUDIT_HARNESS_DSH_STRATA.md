# Audit — chaîne CI/CD R&D « harness d'inférence »

**Cible auditée** : `recherche-s-mantico-math-matique` (EmbedBabel)
**Objet** : construire en CI/CD R&D le meilleur harnais adapté à l'inférence pour bencher la
pipeline de production llama.cpp, jusqu'au plugin DeepSeek Harness (DSH) qui rend compte des
réponses et de leur potentiel d'auto-amélioration, en local, sur base portable Strata.
**Date de l'audit** : 2026-10-07 · **Commit audité** : `a5e6e11` (arbre propre) · **73 fichiers suivis**

---

## 0. Verdict exécutif

### 0.1 Ce qui est réel et de bonne qualité

Le dépôt contient **un vrai harnais de bench**, et il fonctionne : `python -m bench.run` s'exécute
de bout en bout hors ligne, produit `trace.json`, `report.json`, `journal.jsonl`, est reproductible
au bit près à graine égale, et sa version est **immuable par hash** couvrant définition + prompt +
documentation + paquets de contexte. 72 tests passent. C'est une base sérieuse, rare, et
conceptuellement propre (machine à états explicite, points de reprise, test de reprise par un
adaptateur neuf, télémétrie serveur réelle consignée).

La documentation du harnais est **honnête sur ses propres limites** : `HARNESS.md` §« Non testé /
limites » déclare déjà l'absence de vrai tokenizer, l'absence de vrai modèle dans les tests, et
l'API de hooks DeepSeek non vérifiée. C'est un point à créditer et à préserver.

### 0.2 Ce qui est faux ou trompeur

1. **Il n'existe aucune CI/CD.** Zéro workflow, zéro `.github/`, zéro `pyproject.toml`, zéro
   `Makefile`. Le mot « ci-cd » de l'énoncé désigne une intention, pas un artefact.
2. **Le mot « harness » désigne trois choses différentes** dans l'énoncé, et le dépôt n'en couvre
   qu'une. Le harnais existant note des **rapports rédigés par des LLM** ; il ne mesure pas
   l'inférence. « Bencher toute la pipeline de production de llama.cpp » n'est **pas** ce que fait
   le code.
3. **La conclusion collective est structurellement nulle pour tout modèle réel.** Les trois
   exécutions réelles enregistrées se terminent toutes par `"aucun consensus"`, `kept: []`, et
   **tous** les modèles sont exclus pour `ecart_budget`. Ce n'est pas un résultat, c'est un artefact
   du harnais (constat C3).
4. **Le verdict scientifique de tête est circulaire.** `weighted_support` et donc `H0 retenue` /
   `à tester contre baselines` sont calculés à partir de la `correlation_map.solid` **auto-déclarée
   par le modèle**. Le harnais restitue l'opinion du modèle sur lui-même en la présentant comme une
   conclusion (constat C4).
5. **Le « plugin DeepSeek harness » n'est pas un plugin DSH.** C'est un fichier Python avec une
   fonction `on_message()`. DSH ne charge pas de Python : ses plugins sont des plugins Cordis ESM
   JavaScript déclarés dans `$DSH_HOME/profiles/<profil>/package.json` + `cordis.patch.yml`
   (constat C7). Le fichier le dit lui-même, mais le nom du dossier, le manifeste et le README
   laissent croire le contraire.
6. **Le plugin contredit l'objectif « local »** : sa seule fonction d'appel modèle
   (`call_deepseek`) envoie le prompt complet à `https://api.deepseek.com` (constat C8).
7. **`integrations/deepseek-harness-labsia/` ne contient aucun code d'intégration** : deux fichiers
   de test et un `conftest.py` qui réimporte le plugin. C'est un test orphelin (constat C9).

### 0.3 Ce qui manque, et qui est le cœur de la demande

| Brique voulue | État réel |
|---|---|
| H1 — harnais qui note des rapports de LLM sur EmbedBabel | **existe**, fonctionne, mais verdict cassé (C3, C4) |
| H2 — harnais de performance de la pipeline llama.cpp/Strata | **inexistant** |
| H3 — plugin DSH + boucle d'auto-amélioration locale | **inexistant** (un stub Python non fonctionnel pour DSH) |
| H4 — base portable Strata | **externe, installable**, et son API est compatible avec `LlamaCppAdapter` presque telle quelle |

### 0.4 Les trois confusions à lever avant d'écrire une ligne de code

1. **« Harness » = noter du texte** (H1) ≠ **« harness » = mesurer des tok/s, TTFT, VRAM** (H2).
   Le harnais de Strata (`bench/bench_vs_llama.py`) et celui d'EmbedBabel (`bench/run.py`) ne
   mesurent pas la même chose et **ne doivent pas fusionner** : ils doivent partager un format de
   trace et un exécuteur, pas un schéma.
2. **« Benchmarker llama.cpp » = balayer une matrice de configuration** (quantification, contexte,
   KV, concurrence, speculative decoding, offload). Aujourd'hui le harnais ne fait que **3 appels
   chat par modèle** avec une configuration figée.
3. **« Auto-amélioration de DSH » doit être un protocole d'acceptation falsifiable**, pas un modèle
   qui se note. La circularité de C4 est exactement le piège à ne pas reproduire côté DSH.

---

## 1. Méthode et preuves

Tout ce qui suit a été **lu ou exécuté** pendant l'audit, pas déduit.

| Vérification | Résultat |
|---|---|
| `git status` / `git log` après audit | arbre **propre**, écritures de registre annulées |
| `python -m bench.run --models offline-mock.json --task v2-complet --budget 3000` | **exit 0**, 4 artefacts écrits |
| `python -m pytest tests/test_bench_*.py integrations -q` | **72 passed** en 0,67 s |
| `python -m pytest -q` (suite complète) | **4 modules en échec de collecte** : `test_build_graph`, `test_build_qdrant`, `test_gematria_substrings`, `test_geometry` (`No module named 'numpy'`, `'yaml'`) |
| `dhbench/Scripts/python.exe -m pytest` | `No module named pytest` — le venv du dépôt est **vide** (aucun paquet installé) |
| `bench.run --list` + `registry.verify()` sur les 4 harnais | les 4 vérifient (`True`) |
| Analyse des `bench/runs/*/journal.jsonl` | compteurs réels extraits (voir §3 C6) |
| Environnement | Python 3.11.9, llama.cpp **9870 (2d973636e)**, Ollama **0.35.1**, RTX 4090 24 Go, 95,8 Go RAM, driver 610.74 |
| Strata | **CORRIGÉ — il est installé et il tournait** : `D:\AI\Strata`, `strata.exe` (PID 27724) lancé à 14:48 avec `qwen3.8-flash-next-coder-iq1_m` chargé, `n_ctx = 131072`, en écoute sur `127.0.0.1:8080`. Voir l'erratum ci-dessous. |

> **Erratum (constaté le 2026-10-07 à 16:2x, pendant l'exécution du plan).** La ligne « Strata » de ce
> tableau disait d'abord « non installé sur la machine (ni dossier, ni port 8080 en écoute) ».
> **C'est faux.** L'erreur vient de la méthode, pas d'une conclusion hâtive : la commande qui devait
> tester le port 8080 avait été interrompue par un `ollama list` qui ne rendait pas la main, et le
> harness l'avait déplacée en tâche de fond. **Le test n'a jamais été exécuté**, et j'ai lu son
> absence de sortie comme une absence de Strata. Leçon directement pertinente pour cet audit : ne
> jamais traiter « pas de sortie » comme « pas de résultat » — exactement le reproche fait au harnais
> sur les rejeux de cache.


---

## 2. État des lieux

### 2.1 Ce qui tourne

- Harnais hors ligne : déterministe, checkpointé, versionné, testé (72 tests).
- Adaptation `llamacpp` réelle : `llama-server` est lancé, interrogé, arrêté ; les compteurs réels
  du serveur sont bien collectés (`usage`, `timings`) et journalisés.
- Adaptateurs `ollama` et `manuel` : fonctionnels.
- Le garde anti-pollution est réellement branché dans le harnais (`bench/run.py:120-135`,
  `bench/context.py:69-75`) — ce n'est pas du décor.

### 2.2 Ce qui ne tourne pas

- La suite de tests complète, sur les deux interpréteurs disponibles.
- Aucune CI, donc aucun de ces échecs n'est détecté automatiquement.
- Aucun composant DSH.

---

## 3. Constats

### C1 — Il n'y a pas de CI/CD. *Sévérité : bloquante pour l'énoncé*

**Preuve.** `Test-Path .github` → `False`. Aucun workflow, aucun `pyproject.toml`, aucun `Makefile`,
aucun script d'orchestration versionné. Seul `requirements.txt` (7 dépendances, bornes en major).
`.specify/` — qui contient le seul journal d'orchestration (`.specify/orchestrator/log.md`,
14 lignes `subagent_stop`) — est **explicitement gitignoré** (`.gitignore`, dernière ligne).
Le journal d'orchestration est donc local, invisible en CI, et non partagé.

**Impact.** Aucun verrou de non-régression. Les 4 modules de test cassés le resteraient
indéfiniment sans que personne ne le sache.

**Correction.** Voir P0.

### C2 — Le harnais note des rapports, il ne mesure pas l'inférence. *Sévérité : structurelle*

**Preuve.** `bench/schema.py` valide un JSON à 10 clés produites par le LLM. `bench/consensus.py`
note 4 critères : honnêteté de la limite, exactitude du budget, taille du plus gros paquet, reprise.
`bench/run.py` n'appelle le modèle que 3 fois (3 étapes rédactionnelles :
`formalisation`, `analyse`, `plan_experimental` — `bench/budget.py:14`).

`PROMPT.md` §12 exige de mesurer : `tokens/s`, `prefill latency`, `decode latency`, `peak RAM`,
`VRAM`, `KV-cache`, `FLOPs estimés`, `perplexity`, `next-token accuracy`, `semantic retrieval`,
`qualité multilingue`. **Aucune** de ces grandeurs n'apparaît dans `report.json`.

**Impact.** L'énoncé demande de « bencher toute la pipeline de production de llama.cpp ». Le dépôt
mesure la qualité rédactionnelle de 3 réponses JSON. L'écart est total, et il n'est pas documenté
comme tel.

**Correction.** Ajouter H2 (§6) comme composant distinct, sans toucher à H1.

### C3 — `ecart_budget` exclut structurellement tout modèle réel → verdict toujours nul. *Sévérité : critique*

**Preuve (mécanisme).**

- `bench/live.py:86-88` : pour un modèle réel, `plan()` renvoie **`budget / 3` par étape**. Ce n'est
  pas une prévision du modèle, c'est une répartition neutre décidée par le harnais.
- `bench/consensus.py:19-20` : `if abs(report["gap_pct"]) > tolerance_pct: flags.append("ecart_budget")`,
  avec `tolerance_pct = 10.0` (`harness.def.json`).
- `gap_pct` compare donc **la taille réelle (estimée `chars/4`) d'un JSON libre** à **`budget/3`**.
- `bench/consensus.py:53` : seuls les modèles sans drapeau entrent dans `kept`.

**Preuve (exécutions réelles enregistrées).**

| Run | Modèles | Exclus | Verdict |
|---|---|---|---|
| `llama-smoke` (0.4.0, `v2-minimal`, budget 3000) | 1 | `ecart_budget` (-45,23 %) | `aucun consensus`, `kept: []` |
| `llamacpp-2` (0.4.0, `v2-complet`, budget 4000) | 2 | `ecart_budget` ×2 (-18,55 % / +44,54 %) | `aucun consensus`, `kept: []` |
| `ollama-2` (0.3.0, `v2-complet`, budget 4000) | 2 | `ecart_budget` ×2 (-12,18 % / +30,61 %) | `aucun consensus`, `kept: []` |

Et les deux exécutions inscrites au registre `embedbabel-bench-0.4.0.json` portent toutes deux
`"verdict": "aucun consensus"`.

**Origine du défaut.** Le critère a été conçu pour les **adaptateurs factices** : `optimiste`
« annonce un budget 3 fois trop bas » (`HARNESS.md`, tableau des adaptateurs) — il y a donc un
mensonge à détecter. Pour un modèle réel, `plan()` n'est plus une annonce du modèle mais une
constante du harnais : le critère **change de sens sans changer de nom**, et devient « ta réponse
JSON doit faire presque exactement un tiers du budget ». Personne ne peut satisfaire ça.

**Impact.** Le harnais ne peut produire **aucun** signal sur des modèles réels. Tout le travail
d'adaptateurs réels (0.3.0, 0.4.0) est neutralisé par un drapeau hérité.

**Correction.** Voir P2.

### C4 — Le verdict de tête est calculé à partir de l'auto-évaluation du modèle. *Sévérité : critique (intégrité scientifique)*

**Preuve.** `bench/consensus.py:62-63` :

```python
support = round(sum(sum(weights[a] for a in outputs[n]["correlation_map"]["solid"] if a in weights)
                    for n in kept) / (len(kept) * total), 3)
```

`outputs[n]` est le JSON **validé du modèle**. Le harnais ne mesure donc rien : il compte les axes
que le modèle a lui-même classés `solid`. `HARNESS.md:107-108` le formule ainsi : « soutien pondéré
des axes classés `solid` ». La conséquence est mécanique :

| Adaptateur | `correlation_map.solid` (auto-déclaré) | `weighted_support` | Verdict |
|---|---|---|---|
| `honnete` | `[]` | 0,0 | `H0 retenue` |
| `optimiste` | les 3 axes | 1,0 | `à tester contre baselines` |

Le verdict scientifique de tête est donc une **fonction de l'opinion du modèle**, pas d'une mesure.
C'est exactement le contraire de l'exigence de `PROMPT.md` §0 et §11.

**Impact.** Un lecteur de `report.json` croira lire un résultat empirique. Il lit un agrégat
d'auto-évaluations.

**Correction.** P2 — soit mesurer les axes, soit renommer le champ en
`auto_declaration_agregee` et retirer la sémantique de verdict.

### C5 — L'estimateur `chars/4` sous-compte les tokens réels, ce qui affaiblit la garde de contexte. *Sévérité : élevée*

**Preuve.** `bench/budget.py:13` : `CHARS_PER_TOKEN = 4`, marqué `[PLAUSIBLE]`. Mesures sur les
exécutions réelles (contexte `v2-complet` annoncé à **9 767 tokens** par le harnais) :

| Modèle / run | tokens prompt réels (étape 1) | ratio réel/estimé |
|---|---|---|
| qwen3.8-27b / `llamacpp-2` | 10 828 | **1,109** |
| ornith-1.5-35b / `llamacpp-2` | 10 829 | **1,109** |
| qwen3.8-27b / `ollama-2` | 10 824 | **1,108** |
| ornith-1.5-35b / `ollama-2` | 10 825 | **1,109** |

Puis, aux étapes 2 et 3, les sorties précédentes s'ajoutent : le prompt réel atteint 11 941 → 13 162
pour qwen et 13 280 → 15 350 pour ornith, soit **1,23 à 1,35 ×** le total que le harnais croit
envoyer. Le sous-comptage touche aussi les complétions (`consumed`) :

| Run / modèle | consommé `chars/4` (total) | completion_tokens réels | sous-comptage |
|---|---|---|---|
| `llamacpp-2` / qwen3.8-27b | 3 257 | 3 965 | ×1,22 |
| `llamacpp-2` / ornith-1.5-35b | 5 780 | 7 187 | ×1,24 |

**Impact.** La garde de contexte (`context_used ≤ declared_context_limit`,
`bench/consensus.py:17-18`) est **optimiste d'environ 10 % sur le contexte et de 20 à 35 % sur le
prompt complet**. Un modèle déclaré à 32 768 peut recevoir ~40 000 tokens réels sans déclencher
`bavard`. Pire : le drapeau `budget_depasse` sous-estime la réalité — pour ornith, le harnais
signale +44 % alors que le dépassement réel est de +80 % (7 187 pour un budget de 4 000).

**Correction.** P1 — utiliser le vrai tokenizer (au minimum `/tokenize` de `llama-server`, ou
`transformers`/`tokenizers` sur le GGUF), et enregistrer l'écart `chars/4` vs réel comme métrique.

### C6 — La télémétrie d'inférence est collectée puis jetée. *Sévérité : élevée (c'est le cœur du besoin)*

**Preuve.** `bench/live.py:135` et `170-171` alimentent `adapter.usage` ; `bench/run.py:170-171`
l'écrit dans le **journal** (`live_usage`). Or `bench/budget.py:26-36` (`step_record`) ne place
jamais `usage` dans le compte rendu, et `bench/run.py:230-236` ne construit `report.json` qu'à
partir des `budget_reports`. **`report.json` ne contient aucun débit.**

Les mesures existent pourtant déjà dans `bench/runs/*/journal.jsonl` :

| Modèle | Moteur | decode tok/s (étapes 1/2/3) | Rejoué depuis cache ? |
|---|---|---|---|
| qwen3.8-27b | llama.cpp | **45,0 / 45,8 / 45,5** | non (run `llama-smoke`) |
| qwen3.8-27b | llama.cpp | 41,8 / 40,4 / 41,3 | **oui** (`llamacpp-2`) |
| ornith-1.5-35b | llama.cpp | 19,9 / 18,7 / 17,9 | **oui** |
| qwen3.8-27b | Ollama | 10,5 / 9,0 / 10,7 | **oui** (`ollama-2`) |
| ornith-1.5-35b | Ollama | 16,9 / 16,1 / 15,2 | **oui** |

Ce tableau est le premier résultat réellement intéressant du dépôt : **llama.cpp direct est ~4 ×
plus rapide qu'Ollama pour le même modèle** (45 vs 10,5 tok/s). Il n'est publié nulle part.

Deuxième défaut dans le même fichier : sur un hit de cache, `_cached` renvoie
`{**hit["usage"], "cached": True}` (`bench/live.py:110`). Les compteurs sont donc **rejoués**, pas
mesurés — et `report.json` ne contient pas le drapeau `cached`. Un run entièrement rejoué
(`llamacpp-2`, `ollama-2`) est **indiscernable** d'un run frais pour un lecteur du rapport.

**Impact.** L'exigence « bencher la pipeline » est à portée de main mais l'information est perdue à
la sérialisation. Et un rapport peut publier des débits historiques en les faisant passer pour une
mesure.

**Correction.** P1 + P3.

### C7 — Le « plugin DeepSeek harness » n'est pas un plugin DSH. *Sévérité : critique pour l'énoncé*

**Preuve (côté dépôt).** `plugins/deepseek-r1-gématriphonéticospatiale-v0.1.0/adapter.py:10-13` :

> « NOTE: the harness hook API of the DeepSeek fork is NOT verified here. `on_message` is a
> framework-agnostic entry point: adapt the thin wrapper at the bottom to the real hook signature
> of your fork. »

`strategy.md:8` répète : « L'API de hooks du fork DeepSeek harness n'a pas été vérifiée ».

**Preuve (côté DSH, vérifiée dans l'installation).** Un plugin DSH est un **plugin Cordis ESM
JavaScript**, chargé in-process depuis `$DSH_HOME/profiles/<profil>/node_modules`
(`$DSH_HOME` = chemin configuré > `$DSH_HOME` > `~/.dsh`). Il n'existe **aucun chargeur de plugin
Python** dans DSH. Le point d'interception message est le waterfall **`agent/pre-step`** :

```js
ctx.on('agent/pre-step', async (payload, next) => { const d = await next(); /* ... */ return d })
// payload = { messages, turn, step, signal } (+ agent)
// décision : { kind: 'enter', messages }  (défaut) | { kind: 'reject' }
```

`{kind: 'reject'}` termine le tour en `blocked` **sans appel modèle**. Fait décisif pour le design
actuel : **renvoyer un texte à afficher à la place de l'appel modèle n'est pas vérifié comme
supporté** ; un prompt bloqué est « discarded with no model-visible message ».
Découverte : `~/.dsh/profiles/{web,desktop}` existent, et le `cordis.patch.yml` du profil `web`
configure **déjà** un provider Ollama local (`api: openai-completions`,
`baseURL: http://127.0.0.1:11434/v1`) via `@deepseek-ai/dsh-llm-pi-ai`.

**Impact.** Le composant central de l'énoncé (« au plugin deepseek harness ») n'existe pas, et sa
conception actuelle (renvoyer du texte) n'est pas implémentable telle quelle. Le plugin Python est
en réalité consommé **comme bibliothèque par le harnais** (`bench/run.py:33,36-46` :
`PLUGIN_GLOB = "deepseek-r1-*"`, import de `adapter.py` ; puis `bench/context.py:69-75` appelle
`mod.on_message`). Il est porteur de charge pour H1 : le supprimer casserait le garde anti-pollution.

**Correction.** P4 — scinder : garder un module Python `bench/guard.py` (pur, testé) pour H1, et
écrire un **vrai** plugin Cordis JS pour H3.

### C8 — Le plugin appelle le cloud, à l'opposé de l'objectif « local ». *Sévérité : moyenne*

**Preuve.** `adapter.py:79-88` : `call_deepseek` poste sur `https://api.deepseek.com/chat/completions`
avec `os.environ["DEEPSEEK_API_KEY"]` (KeyError si absent). Ce chemin n'est **jamais** emprunté par
`bench.run`, qui n'appelle que `on_message`. C'est donc du code mort côté harnais, et une fuite
potentielle hors du poste si quelqu'un branche `handle()` comme prévu.

**Correction.** P4 — supprimer `call_deepseek` du chemin harnais, ou le déplacer derrière un
adaptateur `manuel`/`host` explicite.

### C9 — `integrations/deepseek-harness-labsia/` est un test orphelin. *Sévérité : faible*

**Preuve.** Le dossier ne contient que `tests/conftest.py` (6 lignes, insère les dossiers
`plugins/deepseek-r1-*` dans `sys.path`) et `tests/test_context_guard.py` (18 lignes, 2 assertions
sur `on_message`). Aucun code d'intégration, aucun README, aucun manifeste. Le nom promet une
intégration DSH qui n'existe pas.

**Correction.** P4 — soit le supprimer au profit de `tests/test_bench_guard.py`, soit le peupler
d'un vrai test de fumée du plugin Cordis (chargement dans un `$DSH_HOME` jetable).

### C10 — Le prompt de bench inscrit est une version condensée. *Sévérité : moyenne*

**Preuve.** `bench/prompts/EMBEDBABEL_BENCH_V2.md` fait 20 lignes et se termine par :

> « Version condensée du V2 discuté ; remplace ce fichier par le texte complet si besoin. »

**Correction de cet audit (vérifiée après coup).** J'avais d'abord écrit que « le V2 complet est
`embedbabel-bench/PROMPT.md`, jamais branché » : **c'est faux**. `embedbabel-bench/PROMPT.md`
(13 601 octets, 567 lignes) est le *research challenge* condensé — un autre document — et non le
protocole de sortie V2. Le « V2 discuté » auquel le fichier renvoie **n'existe nulle part** : ni dans
le dépôt, ni dans `embedbabel-bench/`, ni ailleurs dans l'espace de travail. Ce n'est donc pas un
fichier oublié, c'est un document jamais persisté.

**Le vrai défaut, lui, est réel et mesurable** : le prompt de sortie renvoie explicitement à
`PROMPT.md` (« cf. PROMPT.md à la racine du repo pour la description complète du projet »), mais
dans le harnais 0.4.0 ce paquet `projet` était **optionnel**. Avec une limite de contexte modeste, un
modèle pouvait donc être noté sur une analyse d'EmbedBabel **sans avoir reçu la description
d'EmbedBabel**, la troncature n'étant que journalisée. C'est ce qui a été corrigé en 1.0.0 : `projet`
devient obligatoire, et un paquet obligatoire qui ne tient pas fait refuser le contexte et
n'exécute pas le modèle.

**Impact résiduel.** Les modèles réels ont été benchés sur un contrat de sortie de 20 lignes, pas sur
le protocole de recherche complet. Les comparaisons inter-modèles restent valides entre elles, mais
elles ne mesurent pas le protocole annoncé — et le texte de ce contrat n'existe qu'ici.

**Correction.** P2 — nouvelle version `0.5.0` avec le texte complet (le hash impose une nouvelle
version : c'est exactement le garde-fou prévu, il fonctionne).

### C11 — Le harnais n'est pas hermétique en CI : il écrit dans un fichier versionné. *Sévérité : élevée (spécifique CI/CD)*

**Preuve, reproduite pendant l'audit.** Un run hors ligne a produit :

```
 M bench/registry/embedbabel-bench-0.2.0.json
```

`bench/run.py:277-280` appelle `registry.record_run(...)` à l'état RAPPORT, et
`bench/registry.py:82-86` ajoute le résumé à `entry["runs"]` puis **réécrit le fichier**.
`bench/registry/*.json` sont **suivis par git**. (L'écriture a été annulée ; l'arbre est propre et
les 4 harnais vérifient.)

**Impact.** Toute exécution salit l'arbre de travail. Conséquences directes en CI : gate « working
tree clean » impossible, jobs parallèles en conflit d'écriture, artefacts non reproductibles,
et un run de bench crée un diff qu'il faut committer pour ne pas casser le build suivant.

**Autres non-herméticités du même type :**

- `bench/runs/` est gitignoré mais sert de **cache persistant** (`<out>/cache/`) : un run peut être
  entièrement rejoué (constaté sur `llamacpp-2` et `ollama-2`) sans que le rapport le dise.
- `--out` par défaut = `bench/runs/<date du jour>` (`bench/run.py:314-316`) : chemin non déterministe.
- `.specify/` (journal d'orchestration) est gitignoré : rien de l'orchestration n'est auditable.

**Correction.** P1.

### C12 — Détails d'exactitude à corriger chemin faisant

| # | Fait | Preuve | Portée |
|---|---|---|---|
| a | `cross_audit` attribue à un modèle un audit qu'aucun modèle n'a fait | `bench/consensus.py:42-48` : `{a: {"audits": b, **audit_report(reports[b], ...)}}` — `a` ne fait rien ; la grille est déterministe | Le nom « audit croisé » et le README surestiment. Le docstring est honnête (« SANS rappeler le modèle ») |
| b | Le test de reprise relance un modèle réel et recharge `llama-server` | `bench/run.py:213-220` + `bench/live.py:188-218` | Double le coût d'inférence à l'état EVALUER ; résultat (`resumes: false`) n'entre pas dans le verdict, seulement dans la grille d'audit |
| c | Pas de streaming → pas de TTFT ni de latence inter-token | `bench/live.py:119-128`, `TIMEOUT_S = 1800`, `stream: False` | Interdit toute mesure de latence perçue |
| d | `-np 1` figé, aucune mesure de concurrence | `bench/live.py:185` | Impossible de bencher le batching |
| e | Aucun réglage de la pipeline : pas de `-fa`, `-ctk/-ctv`, `-b/-ub`, `--no-mmap`, `--no-warmup`, speculative decoding | `bench/live.py:184-186` | Ce sont pourtant les leviers qui dominent le prefill. `extra_args` existe, mais aucun moteur de matrice ne l'exploite |
| f | `CLAUDE_CODE_HARNESS_TEST.md:6` affirme que `README.md` contient des marqueurs de conflit de merge | `grep '^(<{7}|={7}|>{7})'` → **aucune correspondance** | Prompt de test périmé : il envoie un agent chercher un problème inexistant |
| g | Le venv `dhbench/` est présent dans le dépôt, vide, et non gitignoré | 0 fichier suivi sous `dhbench/`, mais `.gitignore` ne le liste pas | Piège pour un nouveau contributeur ; 8 Mo inutiles |
| h | 4 modules de test ne collectent pas | voir §1 | La CI est rouge sur machine neuve |

---

## 4. Strata comme base portable — ce qui est établi

Vérifié en ligne sur `Niko1221/Strata@main`, dernière release **v0.1.40.2 (2026-10-07)**.

**Ce qui est confirmé et utile.**

- Serveur HTTP Python sur `http://127.0.0.1:8080` (`serve/server.py`). Endpoints documentés :
  `POST /v1/chat/completions`, `POST /v1/messages` (Anthropic), `POST /v1/responses`,
  `GET /v1/models`, `GET /health`, `GET /props`, `GET /status`, `GET /slots`, `GET /metrics`,
  `GET /mcp`. **`/v1/completions` n'est pas documenté** — ne pas le supposer.
- **Télémétrie riche, du même format que llama.cpp** : `timings.{prompt_n, prompt_ms,
  prompt_per_second, predicted_n, predicted_per_second, draft_n, draft_n_accepted, cache_n}`,
  plus `usage.{prompt_tokens, completion_tokens}`. C'est **exactement** ce que lit déjà
  `bench/live.py:246-251`. `timings.cache_n` (tokens de prompt réutilisés) est le champ qui permet
  de détecter un prefill faussement rapide par cache.
- **Prometheus** sur `/metrics` avec `strata:live_tok_s`, `strata:live_prefill_tok_s_mean`,
  `strata:last_decode_tok_s`, `strata:last_hit_rate`, plus un Grafana prêt à l'emploi.
- `response_format` `text | json_object | json_schema` implémenté sur le chemin chat
  (`serve/structured.py`). Réserve importante : **« The native engine has no grammar decoder.
  Failed generations are errors, never silently retried »** → attendre un taux d'`invalide`
  **plus élevé** qu'avec les grammaires GBNF de llama.cpp.
- Contrôle de la réflexion : `reasoning_effort` (`none|low|medium|high`),
  `chat_template_kwargs.enable_thinking` — **la clé exacte que le harnais envoie déjà**
  (`bench/live.py:237`).
- **Reproductibilité** : `seed` est honoré, mais le tier d'experts adaptatif par défaut casse la
  reproductibilité ; il faut `--adapt-every 100000`. Synergie : le test de reprise du harnais
  **mesure déjà** cette non-reproductibilité — c'est un usage légitime de ce test.
- Licence **MIT** pour le code Strata ; `third_party/ggml` épinglé au commit
  `3cf03257f219afbe7334045ff7c6a06ac68c627d` ; le modèle est sous licence Qwen Community 1.0 →
  **pas de MIT de bout en bout**.
- Contraintes : **GPU 12 Go+ obligatoire** (aucun repli CPU), x86-64 AVX2, 70–80 Go de disque,
  driver NVIDIA 580+, Windows 10/11 ou Linux. Installation utilisateur **sans admin**, modèles dans
  un dossier `Strata-data` **à côté** du programme → install relocalisable, mais **pas de support
  clé USB / sans registre documenté (non vérifié)**.
- Le modèle annoncé existe : `Qwen/Qwen3.8-Flash-Next`, `num_experts: 512`, `num_layers: 48`,
  `num_experts_per_tok: 10`, contexte 262 144 → 512 × 48 = **24 576** emplacements d'experts : le
  chiffre du README est cohérent.

**Ce qui reste non vérifié, et qu'il faut traiter comme tel.**

- Les tableaux de performance du README (94 tok/s, 2 650 tok/s de prefill) sont **auto-déclarés**.
  Des données brutes existent (`bench/results/**`) mais aucune reproduction indépendante.
- Le « 91 % du SWE-bench Verified » du modèle Coder est attribué, non mesuré dans le dépôt.
- `docs/paper/Strata-Paper.pdf` : présence non vérifiée, contenu non vérifié.
- Substitution d'un build llama.cpp personnalisé : non documentée (seule la voie `LLAMA_CPP_COMMIT`
  / `third_party/llama.cpp` l'est).

**Conséquence pratique, vérifiable dès maintenant.** Le harnais actuel peut piloter Strata **sans
modification de code** :

```json
{"name": "strata-qwen3.8-flash-next", "adapter": "llamacpp", "context_limit": 32768,
 "params": {"host": "http://127.0.0.1:8080"}}
```

`_start` court-circuite le lancement quand `params.host` est fourni (`bench/live.py:189-190`), le
payload est déjà compatible, `timings.predicted_per_second` est déjà lu. **Mais** : la santé du
serveur n'est pas vérifiée dans ce mode, et `bench/live.py:245` filtre `n_ctx`/`ngl` — bien, car
Strata ne les accepte pas en requête.

**Et un avertissement de méthode.** Strata **livre déjà son propre harnais de performance** :
`bench/bench_vs_llama.py` (Strata vs llama.cpp, même GGUF, même prompt ; tok/s prompt et decode,
TTFT via `prompt_ms`, VRAM via `nvidia-smi`, RSS via `psutil`, tours entrelacés, médianes),
`bench/results/**` (données datées + matrices), `tools/needle_bench.py`, `tools/bench_eviction.py`,
`tools/calibrate.py`, et `docs/COMMUNITY_BENCHMARKS.md` (méthodologie + gabarit de rapport).
Leur `run_bench.py` ajoute un **tag unique par requête** pour que le cache de prompt ne fasse pas
passer un prefill rejoué pour un prefill mesuré — un piège que le harnais EmbedBabel ne traite pas
aujourd'hui (C6). **Réutiliser et citer, ne pas réécrire.**

**Contrainte de cohabitation, mesurée sur cette machine.** Strata chargé en `n_ctx = 131072` occupe
**23 397 MiB sur 24 564** (95 % de la VRAM). Conséquence directe : sur une RTX 4090 de 24 Go,
**Strata et un `llama-server` ne peuvent pas tourner en même temps**. Lancer les deux fait tomber le
débit à 0,58 token/s (mesuré, cf. §11 P5) — le résultat ne mesure alors que la contention. Toute
campagne comparative Strata contre llama.cpp doit donc être **séquentielle** : arrêter l'un, charger
l'autre, et vérifier la VRAM libre avant de commencer. C'est précisément ce que les `Rails` du
réglage automatique vérifient (`bench/selfimprove/tune.py`).


---

## 5. Ce que « bencher la pipeline de production llama.cpp » doit mesurer

La matrice minimale, aujourd'hui absente. Chaque axe est un levier réel sur le débit ou la mémoire.

| Axe | Valeurs | Ce que ça change |
|---|---|---|
| Moteur | `llama-server` (build local 9870) vs Strata (même GGUF) | prefill, decode, VRAM |
| Quantification | Q2_0, IQ2_XS, IQ3_XXS, IQ3_S, Q4_K_M | qualité vs débit |
| Contexte | 4K / 8K / 16K / 32K / 128K | pente du prefill, VRAM KV |
| KV cache | `f16` / `q8_0` / `q4_0` (`-ctk/-ctv`) | VRAM, débit decode |
| Concurrence | `-np` 1 / 2 / 4, requêtes parallèles | débit agrégé vs latence |
| Speculative | off / draft model (`-md`) | decode tok/s, taux d'acceptation (`draft_n_accepted`) |
| Offload | `-ngl` 99 / partiel, `--n-cpu-moe` | équilibre VRAM/RAM |
| Cache | froid vs chaud (`timings.cache_n`) | prefill réel vs rejoué |

Métriques par cellule, avec **N répétitions et intervalle de confiance** (le harnais actuel fait
un seul appel par étape, sans répétition — aucun écart-type n'est calculable) :

`prefill_tok_s` · `decode_tok_s` · `TTFT` · `latence totale` · `pic VRAM` · `pic RSS` ·
`tokens prompt réels` · `cache_n` · `taux JSON valide` · `énergie si mesurable`.

**Deux exigences de méthode non négociables :**

1. **Tag de requête unique** (ou lecture de `cache_n`) pour distinguer un prefill mesuré d'un
   prefill rejoué.
2. **Séparer la mesure de performance de la mesure de qualité.** La qualité (JSON valide,
   verdict) se mesure sur le protocole H1 ; la performance se mesure sur des prompts de charge
   contrôlés. Les mélanger, c'est ce qui a produit des runs entièrement cachés présentés comme des
   mesures.

---

## 6. Boucle d'auto-amélioration DSH en local — la seule version falsifiable

**Ce qu'il ne faut pas faire.** Reproduire C4 : laisser le modèle noter ses propres réponses et
appeler ça une auto-amélioration. Un LLM qui s'attribue un score élevé n'a rien amélioré.

**Ce qui est réellement possible avec l'API DSH vérifiée.**

- **Observer** : plugin Cordis JS exportant `apply(ctx)`, branché sur le waterfall `agent/pre-step`
  (interception avant appel modèle) et sur les événements de tour / flux
  (`agent/assistant-stream`, `agent/turn-stopping`, `session/event`), avec télémétrie
  `ctx.otel.createEventReporter()` / `createSessionLogReporter()`. Chaque tour écrit un
  enregistrement JSONL : modèle, provider, tokens prompt/complétion, latence, appels d'outils,
  statut, diff accepté, tests passés/échoués.
- **Scorer hors ligne** : un harnais Python (le même exécuteur que H1/H2) rejoue les
  enregistrements et calcule des métriques **externes** : réussite de tâche, tokens consommés,
  temps mural, nombre de reprises, tests verts.
- **Proposer** : un espace de candidats **versionné et fini** — variantes de prompt système,
  allowlist d'outils, couple (modèle, quantification, contexte) par classe de tâche,
  `reasoning_effort`, politique de retry, c'est-à-dire des **overlays `cordis.patch.yml`**.
- **Accepter** : un candidat n'est promu que s'il bat l'incumbent sur un **split tenu à l'écart**,
  au-delà d'une marge **pré-déclarée**, avec le hash de la configuration promue enregistré. Sinon,
  il est rejeté.

C'est une **recherche de configuration par protocole falsifiable**, pas un entraînement de poids.
C'est honnête, mesurable, et ça tient sur un poste unique.

**Branchement DSH → Strata local** (`$DSH_HOME/profiles/<profil>/cordis.patch.yml`), à ajouter à
côté du provider Ollama déjà présent :

```yaml
- id: llm-pi-ai
  name: '@deepseek-ai/dsh-llm-pi-ai'
  config:
    providers:
      strata:
        api: openai-completions          # ou openai-responses → POST /v1/responses
        baseURL: http://127.0.0.1:8080/v1
        apiKeyEnv: STRATA_API_KEY
        models: [{id: <modèle-strata>, contextWindow: 262144, maxTokens: 8192, input: [text, image]}]
- id: agent-default-model
  config: {provider: strata, model: <modèle-strata>}
```

Alternative Anthropic : `@deepseek-ai/dsh-llm-deepseek` avec `baseURL: http://127.0.0.1:8080` →
Strata expose `/v1/messages`. Ne pas compter sur `/v1/completions` (absent).

**Squelette de plugin DSH réel** (3 fichiers, JS, à installer hors du profil puis
`dsh plugin --profile <profil> add <chemin>`) :

```json
// embedbabel-dsh/package.json
{"name":"embedbabel-dsh","version":"0.1.0","private":true,"type":"module",
 "main":"lib/index.js","dsh":{"manifestVersion":1,"bundle":{"patch":"./cordis.patch.yml"}}}
```
```yaml
# embedbabel-dsh/cordis.patch.yml
- insert:
    - id: embedbabel-dsh
      name: 'embedbabel-dsh'
```
```js
// embedbabel-dsh/lib/index.js
import { readFile } from 'node:fs/promises'      // API Node standard, pas une API DSH

export const name = 'embedbabel-dsh'
const BENCH_PROMPT = process.env.EMBEDBABEL_BENCH_PROMPT   // chemin fourni par la config du plugin

export function apply(ctx) {
  ctx.on('agent/pre-step', async (payload, next) => {
    const d = await next()
    if (d.kind !== 'enter') return d                                 // laisser passer
    const users = d.messages.filter(m => m.role === 'user')
    const last = users.at(-1)
    if (!last || String(last.content).trim().toLowerCase() !== 'go') return d
    if (users.length > 1) return { kind: 'reject' }                  // pollution : aucun appel modèle
    const prompt = await readFile(BENCH_PROMPT, 'utf8')
    const i = d.messages.lastIndexOf(last)
    return { ...d, messages: [...d.messages.slice(0, i), { ...last, content: prompt }] }
  })
}
```

Trois points **non vérifiés** à trancher par un test de fumée **avant** de bâtir la suite dessus
(conformément à la règle du dépôt : ne pas inventer d'API) :

1. `d.messages` inclut-il le message système, et `lastIndexOf` sur l'identité d'objet est-il fiable ?
   À vérifier sur le payload réel de `agent/pre-step`.
2. Le mécanisme « afficher un message demandant une nouvelle conversation » **n'est pas vérifié** :
   `{kind:'reject'}` termine le tour sans message visible par le modèle. Il faut soit un **outil**
   enregistré (`ctx.tools.register(...)`), soit une injection via `ctx.agents.*`.
3. La forme exacte acceptée par `dsh plugin --profile <p> add <spec>` n'a pas été exécutée.

---

## 7. Plan de remédiation

Ordre imposé : P0 et P1 rendent le reste mesurable. P2 corrige la science. P3–P5 construisent la
demande.

### P0 — Rendre la CI/CD réelle (½ journée)

1. `pyproject.toml` (nom, `requires-python`, dépendances, `[tool.pytest.ini_options]`) + bornes
   verrouillées ; retirer `dhbench/` du dépôt et l'ajouter au `.gitignore`.
2. `tests/` : vérifier que `pip install -e .[dev]` suffit à faire passer **les 11 modules**.
3. `.github/workflows/ci.yml` : matrice Windows + Linux, Python 3.11/3.13, `pytest -q`,
   **gate « arbre propre après exécution »** (il échouera tant que P1 n'est pas fait : c'est
   voulu, il documente le défaut).
4. Job `bench-offline` : run hors ligne à graine fixe, comparaison de `trace_sha256` à une
   **empreinte dorée versionnée**. Le champ existe déjà (`report.json.trace_sha256`) — il ne reste
   qu'à s'en servir comme test de non-régression du harnais.
5. `Makefile` ou `justfile` : `make test`, `make bench-offline`, `make bench-live`.

**Critère d'acceptation** : `pytest -q` vert sur machine neuve, sans étape manuelle, sous 5 min.

### P1 — Herméticité et vraie tokenisation (1–2 jours)

1. `registry.record_run` : écrire les exécutions dans `bench/runs/` (gitignoré), jamais dans
   `bench/registry/` (suivi). Le registre versionné ne garde que définition + hash. *(C11)*
2. Cache explicite : `--cache-dir` obligatoire en CI, `--no-cache` disponible ; **`cached` remonté
   dans `report.json` par étape et au total**. *(C6)*
3. `chars/4` remplacé, pour les adaptateurs réels, par le tokenizer du serveur :
   `/tokenize` de `llama-server`, avec repli documenté sur `chars/4` seulement si indisponible.
   Conserver les deux valeurs et l'écart comme métrique publiée. *(C5)*
4. `--out` obligatoire en CI (pas de chemin daté implicite).

**Critère d'acceptation** : deux runs consécutifs laissent `git status` vide, et l'écart
`estimateur vs tokenizer réel` est reporté dans `report.json`.

### P2 — Réparer le verdict scientifique (2–3 jours)

1. **Découpler les critères des modèles réels.** Le drapeau `ecart_budget` ne s'applique plus quand
   `estimé` provient de `LiveAdapter.plan` (constante du harnais) ; il devient une
   **métrique** (`budget_gap_pct`) et non un motif d'exclusion. Alternative : tolérance par
   famille d'adaptateur (`preferences.tolerance_pct_live`). *(C3)*
2. **Supprimer la circularité** : `weighted_support` est renommé
   `auto_declaration_agregee` et **ne produit plus de verdict** `H0 retenue` /
   `à tester contre baselines`. Le verdict collectif se limite à la décision majoritaire GO/NO-GO
   des modèles inclus, explicitement étiquetée « opinion agrégée de modèles », jamais « résultat ».
   *(C4)*
3. **Prompt complet** : nouvelle version `0.5.0` avec le V2 complet (13,6 ko) comme paquet
   obligatoire, l'ancien prompt conservé en `v2-condense` pour la comparabilité. *(C10)*
4. `cross_audit` renommé `grille_comptes_rendus` (aucun modèle n'audite). *(C12a)*
5. Corriger `CLAUDE_CODE_HARNESS_TEST.md` (plus de marqueurs de conflit). *(C12f)*

**Critère d'acceptation** : un run sur 2 modèles réels produit un `kept` non vide **et** un rapport
où chaque nombre de verdict est traçable à une mesure, pas à une auto-évaluation.

### P3 — H2 : harnais de performance de la pipeline (3–5 jours)

1. `bench/perf/matrix.py` : décrit la matrice (§5), génère les cellules, exécute, agrège
   (médiane + IC, N ≥ 5 répétitions).
2. `bench/perf/probe.py` : sonde `llama-server` **et** Strata — `timings`, `/props`, `/slots`,
   `/metrics` (Prometheus), logs serveur, `nvidia-smi` pour la VRAM.
3. Streaming activé pour mesurer le **TTFT** réel ; tag de requête unique pour invalider le cache ;
   `cache_n` enregistré systématiquement.
4. Sortie `report_perf.json` **séparé** de `report.json` (ne pas mélanger qualité et performance).
5. Réutiliser et citer `Strata/bench/bench_vs_llama.py` et `docs/COMMUNITY_BENCHMARKS.md` comme
   référence de méthode ; ne pas réimplémenter ce qui existe.

**Critère d'acceptation** : une commande produit un tableau moteur × quantification × contexte ×
concurrence avec IC, VRAM crête, et distinctions froid/chaud.

### P4 — H3 : vrai plugin DSH + télémétrie (3–5 jours)

1. `bench/guard.py` : extraire le garde anti-pollution du plugin Python en module pur testé
   (`tests/test_bench_guard.py`) ; `bench/run.py` et `bench/context.py` pointent dessus.
   **C'est un refactor nécessaire** : le harnais dépend aujourd'hui d'un dossier nommé
   `plugins/deepseek-r1-*`. *(C7)*
2. `dsh/embedbabel-dsh/` : plugin Cordis JS `agent/pre-step` + outil de statut + enregistreur
   JSONL via `ctx.otel.*`.
3. Test de fumée : profil `$DSH_HOME` jetable, `dsh plugin --profile test add ./dsh/embedbabel-dsh`,
   assertion que le plugin se charge et que le déclencheur `go` en contexte propre réécrit bien le
   message. **Valider d'abord** le point ouvert du §6 (message visible en cas de rejet).
4. Supprimer `call_deepseek` du chemin local ; `integrations/deepseek-harness-labsia/` remplacé par
   un vrai test d'intégration ou supprimé. *(C8, C9)*

**Critère d'acceptation** : sur un poste, DSH pointe sur Strata local, le plugin capture chaque tour
dans un JSONL, et le déclencheur `go` est refusé en contexte pollué sans appel modèle.

### P5 — Boucle d'auto-amélioration (5–10 jours)

1. `selfimprove/store.py` : schéma d'enregistrement de tour (JSONL, versionné).
2. `selfimprove/score.py` : métriques **externes** (réussite de tâche, tokens, temps mural,
   reprises, tests verts).
3. `selfimprove/candidates/` : overlays `cordis.patch.yml` candidats, chacun hashé.
4. `selfimprove/accept.py` : split tenu à l'écart, marge pré-déclarée, promotion ou rejet
   **journalisés avec le hash de la configuration**.
5. Interdiction explicite, dans le code et la doc : aucun score de modèle utilisé comme vérité
   terrain. *(garde-fou contre la récidive de C4)*

**Critère d'acceptation** : une campagne produit au moins un candidat promu sur critère
pré-déclaré, et un candidat rejeté avec la raison chiffrée.

### Effort total estimé

P0 : ½ j · P1 : 1–2 j · P2 : 2–3 j · P3 : 3–5 j · P4 : 3–5 j · P5 : 5–10 j → **15 à 26 jours-homme**,
hors téléchargement du modèle Strata (~70 Go) et hors compilation llama.cpp/Strata.

---

## 8. Ce qu'il ne faut pas faire

1. **Ne pas fusionner H1 et H2.** Un harnais qui note du texte et un harnais qui mesure des tok/s
   ont des schémas, des critères et des cadences différents. Partager l'exécuteur et le format de
   trace, pas le verdict.
2. **Ne pas faire noter les réponses par un LLM** et appeler ça une mesure. C'est le défaut C4.
   Si un juge LLM est utilisé, il doit être (a) externe au modèle jugé, (b) validé contre un
   ensemble étiqueté à la main, (c) rapporté avec son taux d'accord.
3. **Ne pas réécrire le harnais de performance de Strata.** Il existe, il est documenté, il gère le
   piège du cache de prompt. Le citer et l'étendre.
4. **Ne pas publier de débits sans dire s'ils sont mesurés ou rejoués.** Trois des quatre runs
   réels enregistrés sont des rejeux de cache.
5. **Ne pas supposer que Strata est « portable » au sens clé USB.** GPU 12 Go+ obligatoire, aucun
   repli CPU, install relocalisable mais non documentée sans registre.
6. **Ne pas écrire le plugin DSH en Python.** Il ne se chargera pas.

---

## 9. Annexe — commandes de vérification

```powershell
cd recherche-s-mantico-math-matique

# 1. arbre propre + harnais inscrits cohérents
git status --short
python -c "import sys; sys.path.insert(0,'.'); from bench import registry; \
print([(h, registry.verify(h)) for h in registry.list_harnesses()])"

# 2. suite complète (échoue sur machine neuve : numpy/yaml absents)
python -m pytest -q

# 3. suite du harnais seule (verte : 72 tests)
python -m pytest tests/test_bench_schema.py tests/test_bench_harness.py `
  tests/test_bench_context.py tests/test_bench_catalog.py tests/test_bench_live.py integrations -q

# 4. run hors ligne de bout en bout (déterministe)
python -m bench.run --models bench/models/offline-mock.json --task v2-complet --budget 3000 `
  --seed 0 --out "$env:TEMP\bench-check"

# 5. comparer un run futur à l'empreinte dorée
(Get-Content "$env:TEMP\bench-check\report.json" -Raw | ConvertFrom-Json).trace_sha256
```

**Débits mesurés réellement présents dans les artefacts** (extraits de
`bench/runs/*/journal.jsonl`, événement `live_usage`) :

| Run | Modèle | Moteur | decode tok/s | Rejoué |
|---|---|---|---|---|
| `llama-smoke` | qwen3.8-27b | llama.cpp 9870 | 45,0 / 45,8 / 45,5 | non |
| `llamacpp-2` | qwen3.8-27b | llama.cpp 9870 | 41,8 / 40,4 / 41,3 | **oui** |
| `llamacpp-2` | ornith-1.5-35b | llama.cpp 9870 | 19,9 / 18,7 / 17,9 | **oui** |
| `ollama-2` | qwen3.8-27b | Ollama 0.35.1 | 10,5 / 9,0 / 10,7 | **oui** |
| `ollama-2` | ornith-1.5-35b | Ollama 0.35.1 | 16,9 / 16,1 / 15,2 | **oui** |

Lecture immédiate, à confirmer par une campagne propre : **llama.cpp direct ≈ 4 × Ollama** pour
qwen3.8-27b (45 vs 10,5 tok/s) sur RTX 4090 / 24 Go. C'est le premier résultat publiable du dépôt,
et il n'est aujourd'hui nulle part dans un rapport.

---

## 10. Conclusion

Le socle logiciel est bon et l'honnêteté documentaire est au-dessus de la moyenne : versions
immuables par hash, adaptateurs factices qui testent vraiment le harnais, limites déclarées.
Ce n'est pas un projet à jeter.

Mais **trois des quatre briques de l'énoncé n'existent pas**, et la brique existante produit un
verdict qui n'est pas un résultat : sur tout modèle réel, le harnais conclut « aucun consensus »
pour une raison arithmétique, et son indicateur de tête est la moyenne des opinions que les modèles
ont d'eux-mêmes.

L'ordre correct est donc : **CI d'abord** (P0, P1) pour que les défauts cessent d'être invisibles,
**vérité du verdict ensuite** (P2) pour que le harnais existant ait le droit d'être cru,
**puis** la performance (P3) et DSH (P4, P5). La base portable Strata est, elle, la partie la plus
simple : son API est déjà compatible avec l'adaptateur `llamacpp` du dépôt, et elle apporte
gratuitement la télémétrie de prefill/decode qui manque à `report.json`.

---

## 11. Suivi des corrections

Cette section est tenue à jour ; le corps de l'audit ci-dessus reste le constat daté du commit
`a5e6e11`.

### P0 + P1 — rendues effectives (mesurées, pas déclarées)

| Constat | Correction | Preuve |
|---|---|---|
| C1 — aucune CI | `pyproject.toml`, `.github/workflows/ci.yml` (Windows + Linux × Python 3.11/3.13), `scripts/ci.py` (même code en local et en CI) | `python scripts/ci.py` → **CI verte** |
| C1 — suite rouge sur machine neuve | dépendances dans `pyproject.toml`, `requirements.txt` supprimé (source unique), venv local `.venv` (gitignoré), `dhbench/` (venv vide de 0,5 Mo, sans pip) supprimé | **121 tests passent**, les 4 modules qui ne collectaient plus collectent |
| C11 — un run salit un fichier versionné | `bench/ledger.py` : le registre d'exécution va dans `<out>/run_record.json` ; `bench/registry/` n'est écrit que par `--register` | `tests/test_bench_golden.py::test_a_run_does_not_modify_any_tracked_registry_file` + gate d'herméticité de `scripts/ci.py` |
| C6 — télémétrie collectée puis jetée | `report.json` publie `telemetry[modèle]` : tokens réels, débit médian, étapes mesurées vs rejouées, écarts estimateur/tokenizer | `tests/test_bench_live.py` (télémétrie, rejeu signalé) |
| C6 — rejeu indistinguable d'une mesure | `cached` par étape dans le rapport, `measured`, `replayed_from_cache`, `--no-cache` | `test_a_run_served_entirely_by_cache_is_flagged_as_replayed` |
| C5 — `chars/4` sous-compte | `--tokenize auto` : recomptage par `POST /tokenize` avant exécution, refus si le contexte réel dépasse la limite, `chars/4` conservé comme repli journalisé | `test_real_tokenizer_refuses_a_context_that_chars4_would_have_let_through` |
| C12h — aucune empreinte de non-régression | `tests/golden/offline-mock.seed0.json` + `tests/test_bench_golden.py` | le test échoue si la trace bouge |

### P2 — rendues effectives

| Constat | Correction | Preuve |
|---|---|---|
| C3 — `ecart_budget` excluait tous les modèles réels | `declares_budget` par adaptateur : le critère ne s'applique qu'à ceux qui **annoncent** un budget ; pour un modèle réel il devient `budget_respecte` et l'écart reste publié comme métrique | `test_a_real_model_is_not_excluded_for_a_budget_it_never_declared` |
| C4 — verdict circulaire | `auto_declaration_agregee` + `decision_kind: "opinion_agregee_de_modeles"` + interprétation explicite, **sans verdict** ; les verdicts deviennent `aucun modèle retenu` / `aucune décision majoritaire` / `opinion agrégée : GO\|NO-GO` | `test_the_verdict_is_labelled_as_aggregated_model_opinion` |
| C12a — `cross_audit` attribuait un audit à un modèle | renommé `grille_comptes_rendus` / `grille_compte_rendu` | `test_grille_uses_the_four_criteria_and_does_not_rerun_models` |
| C5 (budget) — `budget_depasse` trop clément | `consumed_total_real` (compteur serveur) prime sur `chars/4` | `test_budget_used_prefers_the_server_counter_over_chars4` |
| C10 — description du projet optionnelle | `PROMPT.md` devient un paquet **obligatoire** de `v2-complet` | `test_the_project_description_is_now_required_so_a_model_cannot_be_graded_blind` |
| Versionnement | nouveau harnais **`embedbabel-bench-1.0.0`** (version MAJEURE : la façon de mesurer change), `CHANGELOG.md` et catalogue à jour | `test_major_version_changes_nothing_the_model_sees` : contexte **identique octet pour octet** à 0.4.0 quand tout tient |
| C12f — prompt de test périmé | — | voir plus bas |

### P3 — H2, le harnais de performance d'inférence : rendu effectif

| Constat | Correction | Preuve |
|---|---|---|
| C2 — le harnais ne mesure pas l'inférence | nouveau paquet `bench/perf/` : `client.py` (chat, **streaming SSE**, `/props`, `/slots`, `/metrics`, `/tokenize`), `resources.py` (VRAM/GPU/RAM par `nvidia-smi`, `GlobalMemoryStatusEx` ou `/proc/meminfo`), `matrix.py` (matrice, exécution, agrégation), `run.py` (CLI) | `tests/test_bench_perf.py` (23 tests) |
| C6 — pas de TTFT possible sans streaming | `ttft_s` (premier token, tout canal) et `ttft_answer_s` (premier token de réponse), chronométrés côté client | `test_stream_measures_ttft_and_reads_server_telemetry` |
| C6 — le cache de prompt fausse le prefill | nonce unique par requête **plus** publication de `cache_n` ; ligne marquée `replayed_prefill` et comptée à part | `test_a_replayed_prefill_is_flagged_and_counted_apart` |
| Absence de matrice (quantification, KV, contexte, concurrence, speculative) | axes `context`, `kv_type`, `ngl`, `batch`, `ubatch`, `n_cpu_moe`, `flash_attn`, `mmap`, `draft_model`, `concurrency` ; un axe non applicable est **consigné** (`unsupported_axes`), pas ignoré | `test_a_hosted_server_cannot_be_reconfigured_and_says_so` |
| Aucun intervalle de confiance, une seule mesure | N répétitions, médiane, quartiles, min/max, coefficient de variation et **IC 95 % de la médiane par bootstrap** à graine fixe | `test_stats_and_bootstrap_ci_are_deterministic` |

**Premier résultat réel du harnais H2**, mesuré sur la machine, sur le serveur **Strata** déjà chargé
(`qwen3.8-flash-next-coder-iq1_m`, `n_ctx = 131072`, `127.0.0.1:8080`), 3 répétitions après 1 chauffe,
nonce unique par requête (`cache_n = 0` : aucun prefill rejoué), VRAM crête 23,7 Go :

| Variante | Charge | prefill tok/s | decode tok/s | TTFT s | TTFT réponse s | VRAM crête |
|---|---|---:|---:|---:|---:|---:|
| `reasoning_effort: none` | `decode-128` (prompt 299 tok) | 230,0 | **40,0** | 1,20 | 1,20 | 23 742 MiB |
| `reasoning_effort: none` | `prefill-4k` (prompt 3 630 tok) | 672,0 | **46,7** | 5,40 | 5,40 | 23 696 MiB |
| défaut (`high`) | `decode-128` | 302,3 | 26,0 | 1,00 | — | 23 685 MiB |
| défaut (`high`) | `prefill-4k` | 686,0 | 32,9 | 5,30 | — | 23 684 MiB |

Lecture : **désactiver le raisonnement rend 54 % de débit de décodage** sur `decode-128` (40,0 contre
26,0) et 42 % sur `prefill-4k`. Avec le raisonnement actif, la réponse n'arrive jamais en `content`
sur un budget de 128 tokens : les 128 tokens sont du raisonnement (538 à 652 caractères constatés),
et `ttft_answer_s` reste `None`. Mesurer sans lire le canal `reasoning_content` revient donc à
publier le débit du raisonnement sous l'étiquette « réponse » — c'est le bug que le premier run a
révélé, et il est corrigé.

**Réserves à ne pas escamoter.** Les écarts par répétition restent importants (28,9 → 42,1 → 40,0
tok/s sur la même cellule) : la première répétition est systématiquement plus lente, ce qui est
cohérent avec le **tier d'experts adaptatif** de Strata et un GPU partagé (Discord, Chrome, Opera,
VS Code, Claude, Docker, DSH et le compositeur Windows étaient actifs, VRAM totale à 23,7/24,5 Go).
Ces chiffres sont donc **indicatifs et non définitifs** : une campagne propre exige plus de chauffes
(ou `--adapt-every 100000` côté Strata), et une mesure faite sans les autres applications.


### P4 — H3, le vrai plugin DSH : rendu effectif

| Constat | Correction | Preuve |
|---|---|---|
| C7 — le « plugin » n'était pas un plugin DSH | `dsh/embedbabel-dsh/` : plugin **Cordis ESM JavaScript** (`package.json` avec `dsh.manifestVersion` + `dsh.bundle.patch`, `cordis.patch.yml`, `lib/index.js` sur le waterfall `agent/pre-step`) | `tests/test_bench_guard.py` **exécute** `node --test` (14 tests) |
| C7 — risque de divergence entre les deux implémentations | jeu de cas **commun** `tests/fixtures/guard_cases.json`, raisons identiques mot pour mot en Python et en JavaScript | `test_decision_matches_the_shared_cases` + `test_the_dsh_plugin_agrees_with_the_python_implementation` |
| C8 — le plugin appelait `api.deepseek.com` | supprimé ; plus aucun appel réseau implicite | `dsh/embedbabel-dsh/README.md` |
| C9 — intégration orpheline | `integrations/deepseek-harness-labsia/` et `plugins/deepseek-r1-*/` **supprimés** (git conserve l'historique) ; la garde vit dans `bench/guard.py` | suite complète verte |
| Le plugin doit « rendre compte des réponses » | journal JSONL par décision (`EMBEDBABEL_LOG`), avec empreinte du prompt injecté | `test_apply_registers_the_agent_pre_step_hook` |

### P5 — Boucle d'auto-amélioration et auto-réglage continu : rendu effectif

| Constat | Correction | Preuve |
|---|---|---|
| C4 — un modèle ne doit pas se noter | `bench/selfimprove/score.py` : métriques **externes** uniquement ; une issue contenant un champ d'auto-évaluation est **refusée** (`SelfAssessmentRefused`) | `test_a_self_assessed_outcome_is_refused` |
| Un tour sans issue ne doit pas compter comme réussite | `scored: false`, agrégat publié avec `unscored` | `test_turns_without_outcomes_are_never_counted_as_successes` |
| Accepter une « amélioration » sans règle pré-déclarée | `compare()` : marge exigée, puis coût en tokens à réussite équivalente ; décision et chiffre journalisés | `test_compare_promotes_only_beyond_the_declared_margin` |
| **Auto-réglage continu piloté par les ressources locales** | `bench/selfimprove/tune.py` : `Rails` (budget VRAM, température max, RAM libre plancher), remesure de l'incumbent **dans la même session**, voisinage déterministe, profil persisté avec son **enveloppe matérielle**, péremption du profil si l'enveloppe change, mode `watch` qui **détecte la dérive** | 12 tests, dont `test_a_faster_but_unsafe_configuration_is_rejected` et `test_watch_detects_drift_on_an_unchanged_incumbent` |

**Ce que l'auto-réglage a immédiatement attrapé.** Le premier run H2 lancé contre `llama-server` a
mesuré **0,58 token/s** (prefill 36 tok/s). Diagnostic : `D:\AI\Strata\engine\strata.exe` tournait
depuis 14:48 avec le modèle chargé et occupait **23 397 des 24 564 MiB** de VRAM. Les deux serveurs se
disputaient la carte, et le résultat ne disait rien de la configuration mesurée. C'est exactement le
scénario que les `Rails` et la détection de dérive existent pour refuser — un débit obtenu en écrasant
les autres applications n'est pas un gain.

### P6 — chiffrage des tokens : rendu effectif, sans monter le plugin

| Constat | Correction | Preuve |
|---|---|---|
| Aucune source de tokens pour scorer les tours | `bench/selfimprove/tokens.py` : lecteur **en lecture seule** des journaux de session DSH, bâti sur le format **vérifié** par décodage d'une session réelle (909 frames zstd concaténées → 1 477 lignes ; compteurs dans `data.usage` sur `assistant/message` ; déduplication par `(turn, step)`) | `tests/test_bench_tokens.py` (12 tests, dont un journal **multi-frames** synthétique) |
| Un chiffrage muet passerait pour une absence d'activité | un document non vide dont aucun objet n'est lisible **lève** une erreur ; une zone illisible est sautée sans perdre le reste | `test_a_non_empty_but_unreadable_log_raises_instead_of_reporting_zero` |
| `score.py` exigeait des tokens recopiés à la main | `enrich()` remplit `outcome['tokens']` depuis les journaux **mesurés**, publie `tokens_source`, et n'écrase jamais une valeur fournie | `test_enrich_fills_missing_tokens_and_never_overwrites_a_given_value` |
| Le dépôt `zoyluoblue/deepseek-harness-token` n'est pas monté | décision motivée : son README prévient qu'un shell Electron peut faire échouer le chargement de l'arbre de plugins **« so the app will not start »**, et l'installation viserait le profil **actif**. Documenté avec la marche arrière | `docs/TOKENS.md` |

**Mesure réelle** (9 sessions, 302 pas) : entrée 833 366 · sortie 391 837 · **lecture de cache
65 859 620** · écriture 0 → **total 67 084 823, dont 98,17 % de lecture de cache.** Aucun coût
monétaire : le harnais ne stocke pas de table de prix, et un chiffre en euros serait une supposition
présentée comme un fait.

### Ce qui n'est pas encore fait

Le plan initial **P0–P6 est exécuté et chaque commit est vérifié vert** (119 · 143 · 170 · 193 · 206 ·
206 tests, du plus ancien au plus récent). Restent ouverts, en revanche, les points **hors plan** que
l'audit avait identifiés et qui ne l'ont pas été :

- le protocole de recherche complet (baselines, permutations, correction des comparaisons multiples)
  n'existe toujours pas dans le dépôt : le prompt de sortie est un contrat de 20 lignes, et le
  « V2 complet » n'a jamais été écrit (voir C10 et l'erratum) ;
- les affirmations de performance de Strata restent **auto-déclarées** : le harnais H2 permet
  désormais de les vérifier, il ne les a pas vérifiées ;
- `bench/perf` n'a été exécuté que sur un seul moteur réel (Strata) et deux charges : la matrice
  (quantification × KV × contexte × concurrence) n'a pas encore été déroulée.



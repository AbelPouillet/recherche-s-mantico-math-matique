# embedbabel-dsh — plugin DeepSeek Harness

Plugin **Cordis / ESM JavaScript** pour DeepSeek Harness. Il fait deux choses, et rien d'autre :

1. **Garde anti-pollution** : le bench ne se lance que dans une conversation vierge. Le déclencheur
   (`go` par défaut), envoyé seul, est remplacé par le prompt de bench complet ; s'il reste un tour
   antérieur, le tour est terminé **sans appel au modèle**.
2. **Journalisation** : chaque décision est écrite en JSONL — matière première de la boucle
   d'auto-amélioration, qui doit noter les réponses avec une métrique **externe**.

## Pourquoi ce dossier existe (et pourquoi l'ancien emplacement était faux)

L'ancien `plugins/deepseek-r1-gématriphonéticospatiale-v0.1.0/adapter.py` était un **fichier Python
avec une fonction `on_message()`**. Ce n'était pas un plugin DSH :

- DSH ne charge **aucun** plugin Python. Ses plugins sont des plugins Cordis chargés *in-process* en
  JavaScript/TypeScript depuis `$DSH_HOME/profiles/<profil>/node_modules` ;
- le point d'interception d'un message est le *waterfall* `agent/pre-step` ;
- l'ancien fichier le reconnaissait lui-même (« the harness hook API of the DeepSeek fork is NOT
  verified here »), et il appelait `https://api.deepseek.com` — à l'opposé d'un usage local.

La logique de garde a donc été déplacée à sa place :

| Emplacement | Rôle |
|---|---|
| `bench/guard.py` | implémentation **canonique**, Python, utilisée par le harnais H1 |
| `lib/guard.js` | **miroir** JavaScript, utilisé par le plugin DSH |
| `tests/fixtures/guard_cases.json` | jeu de cas **commun** aux deux |

Les deux implémentations sont vérifiées par le même jeu de cas et rendent des raisons identiques mot
pour mot : `tests/test_bench_guard.py` (Python) et `test/guard.test.mjs` (Node). `python -m pytest`
exécute réellement le test Node, donc la CI couvre le plugin sans étape supplémentaire.

## Installation

```bash
# 1. dépendances du harnais (le plugin lit son prompt dans le dépôt)
pip install -e ".[dev]"

# 2. variables d'environnement
export EMBEDBABEL_BENCH_PROMPT="$PWD/bench/prompts/EMBEDBABEL_BENCH_V2.md"
export EMBEDBABEL_LOG="$HOME/.dsh/embedbabel/dsh-turns.jsonl"   # optionnel, défaut identique

# 3. installer le plugin dans le profil DSH courant
dsh plugin --profile web add "$PWD/dsh/embedbabel-dsh"
```

| Variable | Défaut | Effet |
|---|---|---|
| `EMBEDBABEL_BENCH_PROMPT` | *(vide)* | chemin du prompt injecté à la place du déclencheur. **Sans lui, le plugin ne remplace rien** et le signale dans le journal. |
| `EMBEDBABEL_TRIGGER` | `go` | mot-clé déclencheur (seul, point ou `!` tolérés, casse ignorée) |
| `EMBEDBABEL_MAX_PRIOR_MESSAGES` | `0` | nombre de tours antérieurs tolérés avant refus |
| `EMBEDBABEL_MAX_PRIOR_CHARS` | `0` | caractères de contexte antérieur tolérés |
| `EMBEDBABEL_GUARD` | *(actif)* | `off` désactive la garde |
| `EMBEDBABEL_LOG` | `~/.dsh/embedbabel/dsh-turns.jsonl` | journal JSONL des décisions |

## Utiliser DSH en local sur Strata

Le plugin est indépendant du moteur. Pour faire tourner DSH sur l'inférence locale, ajouter un
provider dans `$DSH_HOME/profiles/<profil>/cordis.patch.yml` :

```yaml
- id: llm-pi-ai
  name: '@deepseek-ai/dsh-llm-pi-ai'
  config:
    providers:
      strata:
        api: openai-completions          # Strata expose /v1/chat/completions
        baseURL: http://127.0.0.1:8080/v1
        apiKeyEnv: STRATA_API_KEY
        models: [{ id: qwen3.8-flash-next, contextWindow: 262144, maxTokens: 8192, input: [text, image] }]
- id: agent-default-model
  config: { provider: strata, model: qwen3.8-flash-next }
```

`llama-server` fonctionne aussi (`baseURL: http://127.0.0.1:8080/v1`). Strata expose en plus
`/v1/messages` (Anthropic) et `/v1/responses` (Responses API), mais **pas `/v1/completions`**.

## Vérifié / non vérifié

**Vérifié** dans l'installation DSH présente sur la machine : le nom et la signature du hook
`agent/pre-step` (`(payload, next) => Promise<decision>`), les formes de décision
`{kind:'enter', messages}` et `{kind:'reject'}`, la découverte d'un plugin par `package.json`
(`dsh.bundle.patch`) + `cordis.patch.yml`, et l'installation par `dsh plugin --profile <p> add`.

**Non vérifié**, et assumé comme tel :

- l'installation effective par `dsh plugin` n'a pas été exécutée ici ;
- un `{kind:'reject'}` n'affiche probablement **rien** à l'utilisateur (« discarded with no
  model-visible message »). C'est pourquoi la raison du refus est **journalisée**, pas affichée : le
  plugin ne prétend pas savoir l'afficher. Si l'UX l'exige, il faudra un outil enregistré
  (`ctx.tools.register`) ou une injection via `ctx.agents.*`, ce qui reste à valider ;
- la télémétrie `ctx.otel.*` n'est pas utilisée : le journal JSONL ne dépend que de `node:fs`.

## Tests

```bash
node --test dsh/embedbabel-dsh/test/guard.test.mjs     # 14 tests, sans DSH
python -m pytest tests/test_bench_guard.py -q          # dont l'exécution du test Node ci-dessus
```

/**
 * embedbabel-dsh — plugin DeepSeek Harness (Cordis, ESM).
 *
 * Ce que fait ce plugin
 * ---------------------
 * Sur le waterfall `agent/pre-step` (le point d'interception d'un message **avant** l'appel au
 * modèle) :
 *
 * 1. si le dernier message utilisateur n'est pas le déclencheur (`go` par défaut) : on laisse passer,
 *    en consignant le tour ;
 * 2. s'il l'est mais que la conversation contient un tour antérieur : la garde refuse et le tour est
 *    terminé **sans appel au modèle** (`{kind:'reject'}`) — un bench lancé dans une conversation déjà
 *    engagée ne vaut rien ;
 * 3. s'il l'est et que le contexte est vierge : le contenu du message est **remplacé par le prompt de
 *    bench complet**, et le modèle est appelé normalement.
 *
 * Chaque décision est écrite en JSONL (`EMBEDBABEL_LOG`) : c'est la matière première de la boucle
 * d'auto-amélioration (`bench/selfimprove/`), qui doit noter les réponses avec une métrique
 * **externe** — jamais avec l'avis du modèle sur lui-même.
 *
 * Ce qui est vérifié et ce qui ne l'est pas
 * -----------------------------------------
 * VÉRIFIÉ (dans l'installation DSH présente sur la machine) : le nom et la signature du hook
 * `agent/pre-step` (`(payload, next) => Promise<decision>`), les formes de décision
 * `{kind:'enter', messages}` et `{kind:'reject'}`, la découverte des plugins par `package.json`
 * (`dsh.bundle.patch`) + `cordis.patch.yml`, et l'installation par `dsh plugin --profile <p> add`.
 *
 * NON VÉRIFIÉ : l'installation réelle de ce paquet par `dsh plugin`, et le fait qu'un `{kind:'reject'}`
 * affiche quoi que ce soit à l'utilisateur — la documentation du fork indique qu'un tour bloqué est
 * « discarded with no model-visible message ». C'est pourquoi la raison du refus est **journalisée**
 * et non affichée : on ne prétend pas savoir la montrer.
 *
 * Ce plugin n'est **pas** en Python : DSH ne charge aucun plugin Python. L'ancien
 * `plugins/deepseek-r1-.../adapter.py` n'était pas un plugin DSH mais un module importé par le
 * harnais ; sa logique est maintenant dans `bench/guard.py`, et la version canonique pour DSH est ici.
 */
import { appendFileSync, mkdirSync, readFileSync } from 'node:fs'
import { createHash } from 'node:crypto'
import { dirname, join } from 'node:path'
import { homedir } from 'node:os'

import { DEFAULTS, decide, priorMessages } from './guard.js'

export const name = 'embedbabel-dsh'

export const DEFAULT_LOG =
  process.env.EMBEDBABEL_LOG || join(homedir(), '.dsh', 'embedbabel', 'dsh-turns.jsonl')

/** Configuration par variables d'environnement (aucune API DSH non vérifiée n'est supposée). */
export function configFromEnv(env = process.env) {
  const int = (value, fallback) => {
    const n = Number.parseInt(value ?? '', 10)
    return Number.isFinite(n) ? n : fallback
  }
  return {
    keyword: env.EMBEDBABEL_TRIGGER || DEFAULTS.keyword,
    max_prior_messages: int(env.EMBEDBABEL_MAX_PRIOR_MESSAGES, DEFAULTS.max_prior_messages),
    max_prior_chars: int(env.EMBEDBABEL_MAX_PRIOR_CHARS, DEFAULTS.max_prior_chars),
    enabled: env.EMBEDBABEL_GUARD !== 'off',
    log: env.EMBEDBABEL_LOG || DEFAULT_LOG,
    promptPath: env.EMBEDBABEL_BENCH_PROMPT || '',
  }
}

function sha256(text) {
  return createHash('sha256').update(text, 'utf8').digest('hex')
}

/** Résumé d'un tour : de quoi reconstituer la conversation sans stocker tout le texte. */
function transcript(messages) {
  const list = Array.isArray(messages) ? messages : []
  return {
    messages: list.length,
    roles: list.map((m) => m?.role ?? '?'),
    chars: list.reduce((n, m) => n + String(m?.content ?? '').length, 0),
    last_user_sha256: (() => {
      const last = [...list].reverse().find((m) => m?.role === 'user')
      return last ? sha256(String(last.content ?? '')) : null
    })(),
    last_user_chars: (() => {
      const last = [...list].reverse().find((m) => m?.role === 'user')
      return last ? String(last.content ?? '').length : 0
    })(),
  }
}

/**
 * Construit l'intercepteur **pur** : il ne connaît ni Cordis ni le disque au-delà de `emit`.
 * C'est ce qui le rend testable sans DSH.
 */
export function makeInterceptor(options = {}) {
  const rules = {
    ...DEFAULTS,
    enabled: options.enabled !== false,
    keyword: options.keyword || DEFAULTS.keyword,
    max_prior_messages: options.max_prior_messages ?? DEFAULTS.max_prior_messages,
    max_prior_chars: options.max_prior_chars ?? DEFAULTS.max_prior_chars,
  }
  const prompt = options.prompt ?? ''
  const emit = typeof options.emit === 'function' ? options.emit : () => {}
  const logger = options.logger ?? null

  return function intercept(decision) {
    if (!decision || decision.kind !== 'enter') {
      emit({ event: 'passthrough', kind: decision?.kind ?? 'inconnu' })
      return decision
    }
    const messages = Array.isArray(decision.messages) ? decision.messages : []
    const lastUser = [...messages].reverse().find((m) => m?.role === 'user')
    if (!lastUser) {
      emit({ event: 'step', ...transcript(messages) })
      return decision
    }
    // historique antérieur = tout sauf les messages système et le déclencheur lui-même
    const history = messages.filter((m) => m !== lastUser && priorMessages([m]).length > 0)
    const verdict = decide(history, lastUser.content, rules)

    if (verdict.action === 'ignore') {
      emit({ event: 'step', ...transcript(messages) })
      return decision
    }
    if (verdict.action !== 'run_bench') {
      emit({ event: 'guard_blocked', reason: verdict.reason, ...transcript(messages) })
      logger?.warn?.(`embedbabel-dsh : bench refusé (${verdict.reason}). Ouvre une nouvelle conversation.`)
      return { kind: 'reject' }
    }
    if (!prompt) {
      emit({ event: 'guard_passed_but_no_prompt', reason: 'EMBEDBABEL_BENCH_PROMPT non défini' })
      logger?.warn?.('embedbabel-dsh : prompt de bench introuvable, message laissé tel quel.')
      return decision
    }
    const index = messages.lastIndexOf(lastUser)
    const replaced = messages.slice()
    replaced[index] = { ...lastUser, content: prompt }
    emit({ event: 'bench_injected', reason: verdict.reason, prompt_chars: prompt.length,
           prompt_sha256: sha256(prompt) })
    return { ...decision, messages: replaced }
  }
}

function makeEmitter(logPath) {
  let ready = false
  return (entry) => {
    if (!logPath) return
    try {
      if (!ready) {
        mkdirSync(dirname(logPath), { recursive: true })
        ready = true
      }
      appendFileSync(logPath, JSON.stringify({ ts: new Date().toISOString(), ...entry }) + '\n',
                     'utf8')
    } catch {
      /* journaliser ne doit jamais faire échouer un tour */
    }
  }
}

/** Point d'entrée Cordis : `ctx.on('agent/pre-step', …)`. */
export function apply(ctx, promptText = null) {
  const cfg = configFromEnv()
  let prompt = promptText
  if (!prompt && cfg.promptPath) {
    try {
      prompt = readFileSync(cfg.promptPath, 'utf8')
    } catch {
      ctx.logger?.warn?.(`embedbabel-dsh : lecture impossible de ${cfg.promptPath}`)
    }
  }
  const intercept = makeInterceptor({ ...cfg, prompt, emit: makeEmitter(cfg.log), logger: ctx.logger })

  ctx.on('agent/pre-step', async (payload, next) => {
    const decision = await next()
    return intercept(decision)
  })
}

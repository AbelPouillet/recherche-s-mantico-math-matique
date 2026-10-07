/**
 * Garde anti-pollution de l'historique — miroir JavaScript de `bench/guard.py`.
 *
 * Les deux implémentations sont vérifiées par le **même** jeu de cas :
 * `tests/fixtures/guard_cases.json`, lu par `tests/test_bench_guard.py` (Python) et par
 * `test/guard.test.mjs` (Node). Les raisons renvoyées sont volontairement identiques mot pour mot
 * dans les deux langages : c'est ce qui permet au jeu de cas partagé d'être une vraie preuve
 * d'équivalence, et pas seulement deux tests qui se ressemblent.
 *
 * Aucune dépendance : ni DSH, ni paquet externe. Ce fichier est importable et testable seul.
 */

export const DEFAULTS = {
  enabled: true,
  keyword: 'go',
  max_prior_messages: 0,
  max_prior_chars: 0,
  on_polluted: 'ask_new_conversation',
}

export const CONVERSATIONAL_ROLES = ['user', 'assistant', 'tool']

export function escapeRegExp(value) {
  return String(value).replace(/[.*+?^${}()|[\]\\]/g, '\\$&')
}

/** Le message est-il *exactement* le mot-clé (espaces, point final ou `!` tolérés) ? */
export function isTrigger(message, keyword = DEFAULTS.keyword) {
  return new RegExp(`^\\s*${escapeRegExp(keyword)}\\s*[.!]?\\s*$`, 'i').test(String(message ?? ''))
}

/** Tours de conversation : ni les messages système, ni autre chose que du dialogue. */
export function priorMessages(history) {
  return (Array.isArray(history) ? history : [])
    .filter((m) => m && CONVERSATIONAL_ROLES.includes(m.role))
}

export function priorChars(history) {
  return priorMessages(history).reduce((total, m) => total + String(m.content ?? '').length, 0)
}

/** `[pollué, raison]` — la raison est identique à celle de `bench/guard.py`. */
export function pollution(history, rules = {}) {
  const cfg = { ...DEFAULTS, ...rules }
  if (!cfg.enabled) return [false, 'garde désactivée']
  const prior = priorMessages(history)
  if (prior.length > cfg.max_prior_messages) {
    return [true, `${prior.length} message(s) précédent(s)`]
  }
  const chars = priorChars(history)
  if (chars > cfg.max_prior_chars) {
    return [true, `${chars} caractères de contexte précédent`]
  }
  return [false, 'contexte propre']
}

/** Décision pour un message : `{action, reason}` avec `run_bench` | `ask_new_conversation` | `ignore`. */
export function decide(history, message, rules = {}) {
  const cfg = { ...DEFAULTS, ...rules }
  if (!isTrigger(message, cfg.keyword)) return { action: 'ignore', reason: 'pas le déclencheur' }
  const [bad, reason] = pollution(history, cfg)
  if (bad) return { action: cfg.on_polluted, reason }
  return { action: 'run_bench', reason }
}

export const NEW_CONVERSATION_MESSAGE =
  '⚠️ Cette conversation contient déjà des messages précédents qui peuvent polluer le contexte du ' +
  'benchmark (biais d\'ancrage, résultats antérieurs).\n' +
  '👉 Ouvre une **nouvelle conversation** puis envoie uniquement : go'

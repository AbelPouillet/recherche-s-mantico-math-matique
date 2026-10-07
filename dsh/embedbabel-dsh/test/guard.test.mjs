/**
 * Tests du plugin DSH — sans DSH.
 *
 * Deux niveaux :
 * 1. `guard.js` est confronté au **même** jeu de cas que `bench/guard.py`
 *    (`tests/fixtures/guard_cases.json`) : les deux implémentations doivent rendre exactement la
 *    même action et la même raison.
 * 2. `makeInterceptor` est piloté avec de fausses décisions Cordis, pour vérifier le remplacement du
 *    déclencheur par le prompt de bench, le refus sans appel modèle, et la journalisation.
 *
 * Lancer : `node --test dsh/embedbabel-dsh/test/` (ou `python -m pytest tests/test_bench_guard.py`).
 */
import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { dirname, join } from 'node:path'
import { fileURLToPath } from 'node:url'
import { test } from 'node:test'

import { DEFAULTS, decide, isTrigger, pollution, priorMessages } from '../lib/guard.js'
import { apply, configFromEnv, makeInterceptor, name } from '../lib/index.js'

const HERE = dirname(fileURLToPath(import.meta.url))
const FIXTURE = join(HERE, '..', '..', '..', 'tests', 'fixtures', 'guard_cases.json')
const CASES = JSON.parse(readFileSync(FIXTURE, 'utf8'))
const PROMPT = 'PROMPT DE BENCH COMPLET'

/** Les tests ne doivent jamais écrire dans le journal par défaut de l'utilisateur. */
function withTempLog(body) {
  const previous = process.env.EMBEDBABEL_LOG
  process.env.EMBEDBABEL_LOG = join(tmpdir(), `embedbabel-dsh-${process.pid}.jsonl`)
  try {
    return body(process.env.EMBEDBABEL_LOG)
  } finally {
    if (previous === undefined) delete process.env.EMBEDBABEL_LOG
    else process.env.EMBEDBABEL_LOG = previous
  }
}

test('le plugin expose un nom et un point d entree apply', () => {
  assert.equal(name, 'embedbabel-dsh')
})

test('les memes cas que bench/guard.py donnent la meme decision', () => {
  for (const c of CASES) {
    const got = decide(c.history, c.message, c.rules ?? {})
    assert.equal(got.action, c.action, `cas « ${c.name} » : action`)
    assert.equal(got.reason, c.reason, `cas « ${c.name} » : raison`)
  }
})

test('le jeu de cas couvre les trois actions et les deux seuils', () => {
  const actions = new Set(CASES.map((c) => c.action))
  assert.deepEqual([...actions].sort(), ['ask_new_conversation', 'ignore', 'run_bench'])
  assert.ok(CASES.some((c) => c.rules?.max_prior_chars !== undefined), 'seuil de caractères non couvert')
  assert.ok(CASES.some((c) => c.rules?.enabled === false), 'garde désactivée non couverte')
})

test('declencheur : cas limites', () => {
  assert.ok(isTrigger('go'))
  assert.ok(isTrigger(' GO. '))
  assert.ok(isTrigger('go!'))
  assert.ok(!isTrigger('g o'))
  assert.ok(!isTrigger('go go'))
  assert.ok(!isTrigger(''))
  assert.equal(DEFAULTS.keyword, 'go')
})

test('pollution : les messages systeme ne comptent pas', () => {
  assert.deepEqual(priorMessages([{ role: 'system', content: 'x'.repeat(100) }]), [])
  assert.equal(pollution([{ role: 'system', content: 'x' }])[0], false)
})

function enterDecision(messages) {
  return { kind: 'enter', messages }
}

test('contexte vierge : le declencheur est remplace par le prompt complet', () => {
  const events = []
  const intercept = makeInterceptor({ prompt: PROMPT, emit: (e) => events.push(e) })
  const decision = enterDecision([{ role: 'user', content: 'go' }])
  const out = intercept(decision)
  assert.equal(out.kind, 'enter')
  assert.equal(out.messages.length, 1)
  assert.equal(out.messages[0].content, PROMPT)
  assert.equal(out.messages[0].role, 'user')
  assert.equal(decision.messages[0].content, 'go', 'la decision d entree ne doit pas etre mutee')
  assert.equal(events.at(-1).event, 'bench_injected')
  assert.ok(events.at(-1).prompt_sha256.length === 64)
})

test('le message systeme est conserve a sa place', () => {
  const intercept = makeInterceptor({ prompt: PROMPT, emit: () => {} })
  const out = intercept(enterDecision([
    { role: 'system', content: 'tu es un assistant' },
    { role: 'user', content: 'go' },
  ]))
  assert.deepEqual(out.messages.map((m) => m.role), ['system', 'user'])
  assert.equal(out.messages[1].content, PROMPT)
})

test('conversation polluee : refus sans appel au modele', () => {
  const events = []
  const intercept = makeInterceptor({ prompt: PROMPT, emit: (e) => events.push(e) })
  const out = intercept(enterDecision([
    { role: 'user', content: 'bonjour' },
    { role: 'assistant', content: 'salut' },
    { role: 'user', content: 'go' },
  ]))
  assert.deepEqual(out, { kind: 'reject' })
  assert.equal(events.at(-1).event, 'guard_blocked')
  assert.equal(events.at(-1).reason, '2 message(s) précédent(s)')
})

test('message ordinaire : passage inchange et journalise', () => {
  const events = []
  const intercept = makeInterceptor({ prompt: PROMPT, emit: (e) => events.push(e) })
  const decision = enterDecision([{ role: 'user', content: 'explique EmbedBabel' }])
  const out = intercept(decision)
  assert.equal(out, decision)
  assert.equal(events.at(-1).event, 'step')
  assert.equal(events.at(-1).messages, 1)
  assert.equal(events.at(-1).last_user_chars, 'explique EmbedBabel'.length)
})

test('decision non-enter : laissee telle quelle', () => {
  const events = []
  const intercept = makeInterceptor({ prompt: PROMPT, emit: (e) => events.push(e) })
  assert.deepEqual(intercept({ kind: 'reject' }), { kind: 'reject' })
  assert.equal(events.at(-1).event, 'passthrough')
})

test('prompt manquant : on ne remplace rien et on le signale', () => {
  const events = []
  const intercept = makeInterceptor({ prompt: '', emit: (e) => events.push(e) })
  const decision = enterDecision([{ role: 'user', content: 'go' }])
  assert.equal(intercept(decision), decision)
  assert.equal(events.at(-1).event, 'guard_passed_but_no_prompt')
})

test('configuration par variables d environnement', () => {
  const cfg = configFromEnv({
    EMBEDBABEL_TRIGGER: 'bench',
    EMBEDBABEL_MAX_PRIOR_MESSAGES: '2',
    EMBEDBABEL_MAX_PRIOR_CHARS: '500',
    EMBEDBABEL_GUARD: 'off',
    EMBEDBABEL_BENCH_PROMPT: '/tmp/p.md',
    EMBEDBABEL_LOG: '/tmp/l.jsonl',
  })
  assert.equal(cfg.keyword, 'bench')
  assert.equal(cfg.max_prior_messages, 2)
  assert.equal(cfg.max_prior_chars, 500)
  assert.equal(cfg.enabled, false)
  assert.equal(cfg.promptPath, '/tmp/p.md')
  assert.equal(cfg.log, '/tmp/l.jsonl')
  const fallback = configFromEnv({})
  assert.equal(fallback.keyword, 'go')
  assert.equal(fallback.enabled, true)
  assert.ok(fallback.log.endsWith('dsh-turns.jsonl'))
})

test('apply enregistre bien le hook agent/pre-step', async () => {
  await withTempLog(async (log) => {
    const hooks = {}
    const ctx = {
      on: (event, handler) => { hooks[event] = handler },
      logger: { warn: () => {} },
    }
    apply(ctx, PROMPT)
    assert.ok(typeof hooks['agent/pre-step'] === 'function', 'hook agent/pre-step non enregistre')
    const out = await hooks['agent/pre-step'](
      { messages: [], turn: 1, step: 1 },
      async () => enterDecision([{ role: 'user', content: 'go' }]),
    )
    assert.equal(out.messages[0].content, PROMPT)
    const blocked = await hooks['agent/pre-step'](
      { messages: [], turn: 1, step: 1 },
      async () => enterDecision([{ role: 'user', content: 'x' }, { role: 'user', content: 'go' }]),
    )
    assert.deepEqual(blocked, { kind: 'reject' })
    // la journalisation est la matière première de la boucle d'auto-amélioration
    const lines = readFileSync(log, 'utf8').trim().split('\n').map((l) => JSON.parse(l))
    assert.ok(lines.some((l) => l.event === 'bench_injected'))
    assert.ok(lines.some((l) => l.event === 'guard_blocked' && l.reason === '1 message(s) précédent(s)'))
    assert.ok(lines.every((l) => typeof l.ts === 'string' && l.ts.endsWith('Z')))
  })
})

// `apply` importe readFileSync : on vérifie aussi qu'un chemin de prompt valide est lu.
test('apply lit le prompt depuis EMBEDBABEL_BENCH_PROMPT', async () => {
  const previous = process.env.EMBEDBABEL_BENCH_PROMPT
  process.env.EMBEDBABEL_BENCH_PROMPT = FIXTURE
  try {
    await withTempLog(async () => {
      const hooks = {}
      apply({ on: (e, h) => { hooks[e] = h }, logger: { warn: () => {} } })
      const out = await hooks['agent/pre-step'](
        { messages: [], turn: 1, step: 1 },
        async () => enterDecision([{ role: 'user', content: 'go' }]),
      )
      assert.ok(out.messages[0].content.includes('"name": "conversation vide"'))
    })
  } finally {
    if (previous === undefined) delete process.env.EMBEDBABEL_BENCH_PROMPT
    else process.env.EMBEDBABEL_BENCH_PROMPT = previous
  }
})

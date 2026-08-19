import test from 'node:test'
import assert from 'node:assert/strict'
import {
  VerifierEngine,
  normalizeScore,
  parseScoreTags,
  resolveVerifierConfig,
  sigmoid,
} from '../lib/engine.js'
import { apply } from '../lib/index.js'

/** ctx stub whose llm.stream answers each call through `respond(options)`. */
function mockCtx(respond) {
  return {
    llm: {
      async* stream(options) {
        const text = await respond(options)
        yield { type: 'block-start', index: 0, blockType: 'text' }
        yield { type: 'text-delta', index: 0, text }
        yield { type: 'block-end', index: 0, block: { type: 'text', text } }
        yield { type: 'finish', reason: { kind: 'stop' } }
      },
    },
  }
}

/** Pull the two JSON-framed trajectories out of a pairwise prompt. */
function extractPair(options) {
  const lines = options.messages[0].content[0].text.split('\n')
  const after = marker => JSON.parse(lines[lines.indexOf(marker) + 1])
  return [after('Trajectory A, as a JSON string:'), after('Trajectory B, as a JSON string:')]
}

/** Pairwise judge that always scores a GOOD-containing trajectory 19 and others 6. */
function goodJudge(options) {
  const [a, b] = extractPair(options)
  const score = text => (text.includes('GOOD') ? 19 : 6)
  return `<score_A>${score(a)}</score_A> <score_B>${score(b)}</score_B>`
}

const ROUTE = { provider: 'mock', model: 'mock-1' }
const ONE_CRITERION = [{ name: 'only', description: 'd' }]

test('parseScoreTags takes the last well-formed in-range tag', () => {
  const text = 'thinking... <score>3</score> wait, revising. <score>17</score>'
  assert.deepEqual(parseScoreTags(text, ['score'], 20), { score: 17 })
  assert.deepEqual(parseScoreTags('<score>25</score>', ['score'], 20), { score: null })
  assert.deepEqual(parseScoreTags('no tags here', ['score'], 20), { score: null })
  assert.deepEqual(
    parseScoreTags('<score_A>12</score_A> <score_B>4</score_B>', ['score_A', 'score_B'], 20),
    { score_A: 12, score_B: 4 },
  )
})

test('normalizeScore maps [1, G] onto [0, 1]', () => {
  assert.equal(normalizeScore(1, 20), 0)
  assert.equal(normalizeScore(20, 20), 1)
  assert.equal(normalizeScore(11, 21), 0.5)
})

test('resolveVerifierConfig applies paper-aligned defaults and rejects bad input', () => {
  const config = resolveVerifierConfig({})
  assert.equal(config.granularity, 20)
  assert.equal(config.repetitions, 4)
  assert.equal(config.tieMargin, 0)
  assert.equal(config.pivots, 2)
  assert.equal(config.criteria.length, 3)
  assert.equal(config.rollout.provider, 'spawn')
  assert.equal(config.rollout.maxConcurrent, 3)
  assert.throws(() => resolveVerifierConfig({ nope: 1 }), /unknown config key/)
  assert.throws(() => resolveVerifierConfig({ provider: 'x' }), /supplied together/)
  assert.throws(() => resolveVerifierConfig({ granularity: 1 }), /at least 2/)
  // Loader-materialized empty arrays fall back to the default decomposition.
  assert.equal(resolveVerifierConfig({ criteria: [] }).criteria.length, 3)
  assert.throws(() => resolveVerifierConfig({ criteria: 'nope' }), /must be an array/)
  assert.throws(() => resolveVerifierConfig({ criteria: [{ name: 'x' }] }), /name.*description|non-empty string/)
})

test('score averages phi over criteria and repetitions', async () => {
  const responses = ['<score>20</score>', '<score>10</score>']
  const engine = new VerifierEngine(
    mockCtx(() => responses.shift()),
    { ...ROUTE, repetitions: 2, criteria: ONE_CRITERION, concurrency: 1 },
  )
  const result = await engine.score('task', 'candidate')
  // phi(20) = 1, phi(10) = 9/19; mean = 14/19
  assert.ok(Math.abs(result.reward - 14 / 19) < 1e-12)
  assert.equal(result.rawMean, 15)
  assert.equal(result.perCriterion[0].samples.length, 2)
})

test('select runs a pivot tournament and picks the strongest candidate', async () => {
  const engine = new VerifierEngine(
    mockCtx(goodJudge),
    { ...ROUTE, repetitions: 1, criteria: ONE_CRITERION, pivots: 2, concurrency: 1 },
  )
  const candidates = ['a BAD answer', 'another BAD one', 'the GOOD answer', 'a mediocre BAD one']
  const result = await engine.select('task', candidates)
  assert.equal(result.bestIndex, 2)
  for (let i = 0; i < candidates.length; i++) {
    if (i !== 2) assert.ok(result.rewards[2] > result.rewards[i], `expected winner to beat candidate ${i}`)
  }
  assert.equal(result.rewards.length, 4)
  assert.equal(result.ringScores.length, 4)
  assert.equal(result.pivots.length, 2)
  // Ring pass scores N unique pairs; the tournament adds at most k*(N-1) more.
  assert.ok(result.pairs.length >= 4 && result.pairs.length <= 4 + 2 * 3)
  // Every reward is a sigma-based win ratio in (0, 1).
  for (const reward of result.rewards) assert.ok(reward > 0 && reward < 1)
})

test('select with 2 candidates degenerates to one order-debiased pair', async () => {
  let calls = 0
  const engine = new VerifierEngine(
    mockCtx(options => { calls++; return goodJudge(options) }),
    { ...ROUTE, repetitions: 2, criteria: ONE_CRITERION, concurrency: 1 },
  )
  const result = await engine.select('task', ['BAD', 'GOOD'])
  assert.equal(result.bestIndex, 1)
  assert.equal(result.pairs.length, 1)
  assert.equal(calls, 2) // C=1 x K=2 on a single pair
})

test('compare debiases presentation order', async () => {
  // Verifier is biased: whatever is presented as trajectory A gets 20, B gets 10.
  const engine = new VerifierEngine(
    mockCtx(() => '<score_A>20</score_A> <score_B>10</score_B>'),
    { ...ROUTE, repetitions: 2, criteria: ONE_CRITERION, concurrency: 1 },
  )
  const result = await engine.compare('task', 'left', 'right')
  // With order alternation the position bias cancels exactly.
  assert.equal(result.preferred, 'tie')
  assert.equal(result.margin, 0)
})

test('compare reports a real preference through the debias', async () => {
  const engine = new VerifierEngine(
    mockCtx(goodJudge),
    { ...ROUTE, repetitions: 2, criteria: ONE_CRITERION, concurrency: 1 },
  )
  const result = await engine.compare('task', 'the GOOD one', 'the BAD one')
  assert.equal(result.preferred, 'A')
  assert.ok(result.margin > 0.5)
})

test('track scores each cumulative prefix', async () => {
  const engine = new VerifierEngine(
    mockCtx(options => {
      const user = options.messages[0].content[0].text
      const steps = JSON.parse(user.slice(user.indexOf('['), user.lastIndexOf(']') + 1))
      return `<score>${5 * steps.length}</score>`
    }),
    { ...ROUTE, repetitions: 1, criteria: ONE_CRITERION },
  )
  const result = await engine.track('task', ['s1', 's2', 's3'])
  assert.equal(result.progress.length, 3)
  assert.ok(result.progress[2].reward > result.progress[0].reward)
  assert.ok(result.trend > 0)
})

test('a fully unparseable sample retries once before counting as lost', async () => {
  const responses = ['still thinking about it...', '<score>20</score>', '<score>10</score>']
  const engine = new VerifierEngine(
    mockCtx(() => responses.shift()),
    { ...ROUTE, repetitions: 1, criteria: ONE_CRITERION, concurrency: 1 },
  )
  const result = await engine.score('task', 'candidate')
  assert.equal(result.rawMean, 20) // first sample recovered on retry
  assert.equal(responses.length, 1) // exactly one retry consumed
})

test('unparseable scores fail loudly', async () => {
  const engine = new VerifierEngine(
    mockCtx(() => 'I refuse to grade.'),
    { ...ROUTE, repetitions: 2, criteria: ONE_CRITERION },
  )
  await assert.rejects(() => engine.score('task', 'candidate'), /no parseable/)
})

test('missing route fails with a clear message', async () => {
  const engine = new VerifierEngine(mockCtx(() => '<score>10</score>'), {})
  await assert.rejects(() => engine.score('task', 'candidate'), /no verifier model configured/)
})

test('sigmoid is symmetric around 0.5', () => {
  assert.equal(sigmoid(0), 0.5)
  assert.ok(Math.abs(sigmoid(1) + sigmoid(-1) - 1) < 1e-12)
})

/** Build a full plugin ctx stub: llm judge + subagent providers + tool capture. */
function pluginCtx(respond, childTexts, spawnLog) {
  let started = 0
  const tools = new Map()
  const sections = []
  return {
    ctx: {
      ...mockCtx(respond),
      provide: () => {},
      systemPrompt: { section: section => { sections.push(section) } },
      tools: { register: def => { tools.set(def.name, def) } },
      subagents: {
        getProvider: name => (name === 'spawn' ? {} : undefined),
        list: () => ['spawn'],
        start: async (_name, request) => {
          const index = started++
          spawnLog.push(request)
          const spec = childTexts[index]
          return {
            id: `sess-${index}`,
            result: Promise.resolve({
              output: spec.text === undefined ? [] : [{ type: 'text', text: spec.text }],
              stopReason: spec.stopReason ?? 'completed',
            }),
            ...(spec.trace === undefined ? {} : {
              localAgent: { session: { deriveMessages: () => spec.trace } },
            }),
            dispose: async () => { spec.disposed = (spec.disposed ?? 0) + 1 },
          }
        },
      },
    },
    tools,
    sections,
  }
}

test('apply registers the fuzzy-routing prompt section unless disabled', () => {
  const on = pluginCtx(() => '', [], [])
  apply(on.ctx, { ...ROUTE })
  assert.equal(on.sections.length, 1)
  assert.equal(on.sections[0].name, 'tool:verifier')
  assert.ok(on.sections[0].order >= 100 && on.sections[0].order < 200)
  assert.match(on.sections[0].text, /verify_rollout/)
  assert.match(on.sections[0].text, /LLM as a verifier/)
  const off = pluginCtx(() => '', [], [])
  apply(off.ctx, { ...ROUTE, promptSection: false })
  assert.equal(off.sections.length, 0)
})

test('verify_rollout spawns n children, judges them, and returns the winner', async () => {
  const children = [
    { text: 'a BAD webapp' },
    { text: 'the GOOD webapp' },
    { text: 'another BAD webapp' },
  ]
  const spawnLog = []
  const { ctx, tools } = pluginCtx(goodJudge, children, spawnLog)
  apply(ctx, { ...ROUTE, repetitions: 1, criteria: ONE_CRITERION, concurrency: 1, rollout: { maxConcurrent: 1 } })
  const tool = tools.get('verify_rollout')
  assert.ok(tool, 'verify_rollout registered')
  const exec = { signal: new AbortController().signal, agent: { fake: true } }
  const value = await tool.execute({ task: 'build a webapp', n: 3 }, exec)
  assert.equal(value.best_index, 1)
  assert.equal(value.winner, 'the GOOD webapp')
  assert.deepEqual(value.rollout_sessions, ['sess-0', 'sess-1', 'sess-2'])
  assert.equal(value.rewards.length, 3)
  for (const child of children) assert.equal(child.disposed, 1, 'every run disposed exactly once')
  assert.equal(spawnLog.length, 3)
  assert.deepEqual(spawnLog[0].toolFilter, { deny: ['verify_select', 'verify_compare', 'verify_track', 'verify_rollout'] })
  assert.match(spawnLog[0].label, /rollout 1\/3/)
  assert.equal(spawnLog[0].parent, exec.agent)
  assert.equal(spawnLog[0].agentOptions, undefined)
})

test('verify_rollout passes a model override and survives one failed child', async () => {
  const children = [
    { text: 'a BAD attempt' },
    { text: undefined, stopReason: 'error' },
    { text: 'the GOOD attempt' },
  ]
  const spawnLog = []
  const { ctx, tools } = pluginCtx(goodJudge, children, spawnLog)
  apply(ctx, { ...ROUTE, repetitions: 1, criteria: ONE_CRITERION, concurrency: 1, rollout: { maxConcurrent: 1 } })
  const exec = { signal: new AbortController().signal, agent: {} }
  const value = await tools.get('verify_rollout').execute(
    { task: 't', n: 3, rollout_model: 'cheap-model' },
    exec,
  )
  assert.equal(value.best_index, 2)
  assert.equal(value.rewards[1], null)
  assert.equal(spawnLog[0].agentOptions.model, 'cheap-model')
  for (const child of children) assert.equal(child.disposed, 1)
})

test('verify_rollout judges the full trajectory by default but returns only the deliverable', async () => {
  const trace = who => [
    { role: 'user', content: [{ type: 'text', text: 'the task prompt' }] },
    { role: 'assistant', content: [
      { type: 'text', text: `working on it (${who})` },
      { type: 'tool-call', id: 'c1', name: 'bash', arguments: '{"command":"ls"}' },
    ] },
    { role: 'user', content: [{ type: 'tool-result', toolCallId: 'c1', content: [{ type: 'text', text: 'file.txt' }] }] },
    { role: 'assistant', content: [{ type: 'text', text: `done (${who})` }] },
  ]
  const children = [
    { text: 'a BAD result', trace: trace('BAD') },
    { text: 'the GOOD result', trace: trace('GOOD') },
  ]
  const judgePrompts = []
  const { ctx, tools } = pluginCtx(options => {
    judgePrompts.push(options.messages[0].content[0].text)
    return goodJudge(options)
  }, children, [])
  apply(ctx, { ...ROUTE, repetitions: 1, criteria: ONE_CRITERION, concurrency: 1, rollout: { maxConcurrent: 1 } })
  const exec = { signal: new AbortController().signal, agent: {} }
  const value = await tools.get('verify_rollout').execute({ task: 't', n: 2 }, exec)
  // The judge saw the trajectory: tool calls, results, and the deliverable marker.
  assert.ok(judgePrompts.length > 0)
  assert.match(judgePrompts[0], /\[tool call\] bash/)
  assert.match(judgePrompts[0], /\[tool result\] file\.txt/)
  assert.match(judgePrompts[0], /Final deliverable:/)
  // The task prompt itself is not repeated inside the trajectory.
  assert.ok(!judgePrompts[0].includes('the task prompt'))
  // The returned winner is the clean deliverable, not the trace.
  assert.equal(value.best_index, 1)
  assert.equal(value.winner, 'the GOOD result')
  assert.equal(value.details.judge_trace, 'full')
})

test('judgeTrace final keeps the judge on final messages only', async () => {
  const children = [
    { text: 'a BAD result', trace: [{ role: 'assistant', content: [{ type: 'tool-call', id: 'c', name: 'bash', arguments: '{}' }] }] },
    { text: 'the GOOD result', trace: [{ role: 'assistant', content: [{ type: 'tool-call', id: 'c', name: 'bash', arguments: '{}' }] }] },
  ]
  const judgePrompts = []
  const { ctx, tools } = pluginCtx(options => {
    judgePrompts.push(options.messages[0].content[0].text)
    return goodJudge(options)
  }, children, [])
  apply(ctx, { ...ROUTE, repetitions: 1, criteria: ONE_CRITERION, concurrency: 1, judgeTrace: 'final', rollout: { maxConcurrent: 1 } })
  const exec = { signal: new AbortController().signal, agent: {} }
  const value = await tools.get('verify_rollout').execute({ task: 't', n: 2 }, exec)
  assert.equal(value.best_index, 1)
  assert.ok(judgePrompts.every(prompt => !prompt.includes('[tool call]')))
})

test('judgeTrace validation fails loud', () => {
  assert.throws(() => resolveVerifierConfig({ judgeTrace: 'sometimes' }), /judgeTrace/)
  assert.equal(resolveVerifierConfig({}).judgeTrace, 'full')
})

test('verify_rollout presentationMeta is a pure JSON scoreboard', async () => {
  const children = [{ text: 'a BAD one' }, { text: 'the GOOD one' }]
  const { ctx, tools } = pluginCtx(goodJudge, children, [])
  apply(ctx, { ...ROUTE, repetitions: 1, criteria: ONE_CRITERION, concurrency: 1, rollout: { maxConcurrent: 1 } })
  const exec = { signal: new AbortController().signal, agent: {} }
  const tool = tools.get('verify_rollout')
  const value = await tool.execute({ task: 't', n: 2 }, exec)
  const meta = tool.output.presentationMeta({ task: 't', n: 2 }, value)
  assert.equal(meta.best_index, 1)
  assert.equal(meta.rewards.length, 2)
  assert.equal(meta.sessions.length, 2)
  assert.equal(meta.judge_trace, 'full')
  assert.equal(meta.winner_preview, 'the GOOD one')
  assert.ok(meta.stop_reasons.every(r => typeof r.rollout === 'number'))
  // Lossless JSON round-trip (the registry persists this on tool/result).
  assert.deepEqual(JSON.parse(JSON.stringify(meta)), meta)
})

test('verify_rollout fails loudly when too few rollouts succeed', async () => {
  const children = [
    { text: 'only survivor' },
    { text: undefined, stopReason: 'error' },
  ]
  const { ctx, tools } = pluginCtx(goodJudge, children, [])
  apply(ctx, { ...ROUTE, repetitions: 1, criteria: ONE_CRITERION, rollout: { maxConcurrent: 1 } })
  const exec = { signal: new AbortController().signal, agent: {} }
  await assert.rejects(
    () => tools.get('verify_rollout').execute({ task: 't', n: 2 }, exec),
    /at least 2 successful rollouts/,
  )
})

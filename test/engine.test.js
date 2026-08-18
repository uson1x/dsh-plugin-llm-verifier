import test from 'node:test'
import assert from 'node:assert/strict'
import {
  VerifierEngine,
  normalizeScore,
  parseScoreTags,
  resolveVerifierConfig,
} from '../lib/engine.js'

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

const ROUTE = { provider: 'mock', model: 'mock-1' }

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

test('resolveVerifierConfig applies defaults and rejects bad input', () => {
  const config = resolveVerifierConfig({})
  assert.equal(config.granularity, 20)
  assert.equal(config.repetitions, 3)
  assert.equal(config.criteria.length, 3)
  assert.throws(() => resolveVerifierConfig({ nope: 1 }), /unknown config key/)
  assert.throws(() => resolveVerifierConfig({ provider: 'x' }), /supplied together/)
  assert.throws(() => resolveVerifierConfig({ granularity: 1 }), /at least 2/)
  assert.throws(() => resolveVerifierConfig({ criteria: [] }), /non-empty array/)
})

test('score averages phi over criteria and repetitions', async () => {
  const responses = ['<score>20</score>', '<score>10</score>']
  const engine = new VerifierEngine(
    mockCtx(() => responses.shift()),
    { ...ROUTE, repetitions: 2, criteria: [{ name: 'only', description: 'd' }], concurrency: 1 },
  )
  const result = await engine.score('task', 'candidate')
  // phi(20) = 1, phi(10) = 9/19; mean = 14/19
  assert.ok(Math.abs(result.reward - 14 / 19) < 1e-12)
  assert.equal(result.rawMean, 15)
  assert.equal(result.perCriterion[0].samples.length, 2)
})

test('select prefers the candidate the verifier scores higher', async () => {
  const engine = new VerifierEngine(
    mockCtx(options => {
      const user = options.messages[0].content[0].text
      return user.includes('GOOD') ? '<score>18</score>' : '<score>5</score>'
    }),
    { ...ROUTE, repetitions: 2, criteria: [{ name: 'only', description: 'd' }] },
  )
  const result = await engine.select('task', ['a BAD answer', 'a GOOD answer', 'another BAD one'])
  assert.equal(result.bestIndex, 1)
  assert.ok(result.bestReward > result.scores[0].reward)
  assert.equal(result.scores.length, 3)
})

test('compare debiases presentation order', async () => {
  // Verifier is biased: whatever is presented as Candidate A gets 20, B gets 10.
  const engine = new VerifierEngine(
    mockCtx(() => '<score_A>20</score_A> <score_B>10</score_B>'),
    { ...ROUTE, repetitions: 2, criteria: [{ name: 'only', description: 'd' }], concurrency: 1 },
  )
  const result = await engine.compare('task', 'left', 'right')
  // With order alternation the position bias cancels exactly.
  assert.equal(result.preferred, 'tie')
  assert.equal(result.margin, 0)
})

test('compare reports a real preference through the debias', async () => {
  const engine = new VerifierEngine(
    mockCtx(options => {
      const user = options.messages[0].content[0].text
      // GOOD earns 19 wherever it sits; the other side earns 6.
      const aIsGood = user.indexOf('GOOD') < user.indexOf('BAD')
      return aIsGood
        ? '<score_A>19</score_A> <score_B>6</score_B>'
        : '<score_A>6</score_A> <score_B>19</score_B>'
    }),
    { ...ROUTE, repetitions: 2, criteria: [{ name: 'only', description: 'd' }], concurrency: 1 },
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
    { ...ROUTE, repetitions: 1, criteria: [{ name: 'only', description: 'd' }] },
  )
  const result = await engine.track('task', ['s1', 's2', 's3'])
  assert.equal(result.progress.length, 3)
  assert.ok(result.progress[2].reward > result.progress[0].reward)
  assert.ok(result.trend > 0)
})

test('unparseable scores fail loudly', async () => {
  const engine = new VerifierEngine(
    mockCtx(() => 'I refuse to grade.'),
    { ...ROUTE, repetitions: 2, criteria: [{ name: 'only', description: 'd' }] },
  )
  await assert.rejects(() => engine.score('task', 'candidate'), /no parseable/)
})

test('missing route fails with a clear message', async () => {
  const engine = new VerifierEngine(mockCtx(() => '<score>10</score>'), {})
  await assert.rejects(() => engine.score('task', 'candidate'), /no verifier model configured/)
})

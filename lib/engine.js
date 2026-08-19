/**
 * The LLM-as-a-Verifier scoring engine over the harness LLM seam.
 *
 * Implements the paper's continuous reward
 * R(x, tau) = (1/CK) * sum phi(v) over C decomposed criteria and K repeated
 * evaluations, where each v is an integer score in [1, G] and
 * phi(v) = (v - 1) / (G - 1) maps it into [0, 1]; and best-of-N selection via
 * the Probabilistic Pivot Tournament (PPT): a ring pass over a random
 * Hamiltonian cycle (each candidate appears once as A and once as B, canceling
 * positional bias), top-k pivot selection, a pivot tournament over every
 * remaining pivot pair, and win-mass aggregation sigma(R_a - R_b) with final
 * score w_i / c_i.
 *
 * Deviation from the paper: the reference implementation reads the full
 * distribution of scoring-token logits; the dsh `StreamChunk` vocabulary
 * exposes no logprobs, so this engine estimates the same expectation by Monte
 * Carlo — K repeated samples at temperature > 0, the paper's own
 * repeated-evaluation axis.
 * @module dsh-plugin-llm-verifier/engine
 */

import { createUserMessage, deepFreeze } from '@deepseek-ai/dsh-llm'
import {
  DEFAULT_CRITERIA,
  compareUserPrompt,
  scoreUserPrompt,
  trackUserPrompt,
  verifierSystemPrompt,
} from './prompts.js'

export const PLUGIN_NAME = 'dsh-plugin-llm-verifier'

const CONFIG_KEYS = new Set([
  'provider',
  'model',
  'granularity',
  'repetitions',
  'temperature',
  'reasoningEffort',
  'maxOutputTokens',
  'timeoutMs',
  'concurrency',
  'tieMargin',
  'pivots',
  'promptSection',
  'judgeTrace',
  'traceMaxChars',
  'criteria',
  'rollout',
])

function fail(message) {
  throw new Error(`${PLUGIN_NAME}: ${message}`)
}

function assertPositiveInteger(name, value) {
  if (!Number.isInteger(value) || value <= 0) fail(`${name} must be a positive integer`)
}

/**
 * Resolve the criteria list. Loader-validated config materializes an empty
 * array for an omitted `criteria` field, so both absence and emptiness mean
 * "use the default decomposition"; any other non-array fails loud.
 */
function resolveCriteria(criteria) {
  if (criteria === undefined) return DEFAULT_CRITERIA
  if (!Array.isArray(criteria)) fail('criteria must be an array of { name, description }')
  return criteria.length > 0 ? criteria : DEFAULT_CRITERIA
}

/** Validate untrusted plugin config and materialize defaults. */
export function resolveVerifierConfig(config = {}) {
  if (config === null || typeof config !== 'object') fail('config must be an object')
  for (const key of Object.keys(config)) {
    if (!CONFIG_KEYS.has(key)) fail(`unknown config key "${key}"`)
  }
  const resolved = {
    provider: config.provider,
    model: config.model,
    granularity: config.granularity ?? 20,
    repetitions: config.repetitions ?? 4,
    temperature: config.temperature ?? 1,
    reasoningEffort: config.reasoningEffort,
    maxOutputTokens: config.maxOutputTokens ?? 16384,
    timeoutMs: config.timeoutMs ?? 300_000,
    concurrency: config.concurrency ?? 4,
    tieMargin: config.tieMargin ?? 0,
    pivots: config.pivots ?? 2,
    judgeTrace: config.judgeTrace ?? 'full',
    traceMaxChars: config.traceMaxChars ?? 24_000,
    criteria: resolveCriteria(config.criteria),
    rollout: {
      provider: config.rollout?.provider ?? 'spawn',
      llmProvider: config.rollout?.llmProvider,
      model: config.rollout?.model,
      maxConcurrent: config.rollout?.maxConcurrent ?? 3,
    },
  }
  const hasProvider = resolved.provider !== undefined
  const hasModel = resolved.model !== undefined
  if (!hasProvider || !hasModel) fail('config must set provider and model (the LLM route that grades)')
  if (hasProvider && (typeof resolved.provider !== 'string' || resolved.provider.length === 0
    || typeof resolved.model !== 'string' || resolved.model.length === 0)) {
    fail('provider and model must be non-empty strings')
  }
  assertPositiveInteger('granularity', resolved.granularity)
  if (resolved.granularity < 2) fail('granularity must be at least 2')
  assertPositiveInteger('repetitions', resolved.repetitions)
  assertPositiveInteger('maxOutputTokens', resolved.maxOutputTokens)
  assertPositiveInteger('timeoutMs', resolved.timeoutMs)
  assertPositiveInteger('concurrency', resolved.concurrency)
  assertPositiveInteger('pivots', resolved.pivots)
  assertPositiveInteger('traceMaxChars', resolved.traceMaxChars)
  if (resolved.judgeTrace !== 'full' && resolved.judgeTrace !== 'final') {
    fail('judgeTrace must be "full" or "final"')
  }
  assertPositiveInteger('rollout.maxConcurrent', resolved.rollout.maxConcurrent)
  if (typeof resolved.temperature !== 'number' || !(resolved.temperature >= 0 && resolved.temperature <= 2)) {
    fail('temperature must be a number in [0, 2]')
  }
  if (typeof resolved.tieMargin !== 'number' || !(resolved.tieMargin >= 0 && resolved.tieMargin < 1)) {
    fail('tieMargin must be a number in [0, 1)')
  }
  for (const criterion of resolved.criteria) {
    if (criterion === null || typeof criterion !== 'object'
      || typeof criterion.name !== 'string' || criterion.name.length === 0
      || typeof criterion.description !== 'string' || criterion.description.length === 0) {
      fail('each criterion needs non-empty string "name" and "description"')
    }
  }
  return deepFreeze({
    ...resolved,
    criteria: resolved.criteria.map(c => ({ name: c.name, description: c.description })),
  })
}

/** Extract the last integer inside each requested tag; missing or malformed tags yield null. */
export function parseScoreTags(text, tags, granularity) {
  const result = {}
  for (const tag of tags) {
    const pattern = new RegExp(`<${tag}>\\s*(\\d{1,4})\\s*</${tag}>`, 'g')
    let match
    let value = null
    while ((match = pattern.exec(text)) !== null) value = Number(match[1])
    result[tag] = value !== null && value >= 1 && value <= granularity ? value : null
  }
  return result
}

/** phi: map an integer score in [1, G] to a reward in [0, 1]. */
export function normalizeScore(value, granularity) {
  return (value - 1) / (granularity - 1)
}

/** Logistic preference used by the PPT win-mass aggregation. */
export function sigmoid(x) {
  return 1 / (1 + Math.exp(-x))
}

function mean(values) {
  return values.reduce((sum, v) => sum + v, 0) / values.length
}

/** Fisher-Yates shuffle (fresh array). */
function shuffle(items) {
  const result = [...items]
  for (let i = result.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1))
    ;[result[i], result[j]] = [result[j], result[i]]
  }
  return result
}

/** Run tasks with bounded concurrency, preserving order and rejecting on first failure. */
export async function mapLimit(items, limit, worker) {
  const results = new Array(items.length)
  let next = 0
  async function lane() {
    while (next < items.length) {
      const index = next++
      results[index] = await worker(items[index], index)
    }
  }
  await Promise.all(Array.from({ length: Math.min(limit, items.length) }, lane))
  return results
}

/**
 * The verifier service registered at `ctx.verifier`.
 * All methods accept an optional `{ signal }` and reject on cancellation.
 */
export class VerifierEngine {
  #ctx

  constructor(ctx, config = {}) {
    this.#ctx = ctx
    this.config = resolveVerifierConfig(config)
  }

  #route() {
    const { provider, model } = this.config
    if (provider === undefined || model === undefined) {
      fail('no verifier model configured; set "provider" and "model" in the plugin config')
    }
    return { provider, model }
  }

  /** One raw verifier model call; returns the assembled response text. */
  /**
   * One verifier model call with a per-call timeout. A call our own timeout
   * kills gets one fresh retry (full-trajectory judge prompts can be slow);
   * an abort from the caller's signal propagates immediately.
   */
  async #call(system, userText, signal) {
    try {
      return await this.#callOnce(system, userText, signal)
    } catch (error) {
      if (signal?.aborted || error?.finishKind !== 'aborted') throw error
      return await this.#callOnce(system, userText, signal)
    }
  }

  async #callOnce(system, userText, signal) {
    const route = this.#route()
    const timeout = AbortSignal.timeout(this.config.timeoutMs)
    const combined = signal ? AbortSignal.any([signal, timeout]) : timeout
    const messages = [createUserMessage({
      content: [{ type: 'text', text: userText }],
      source: { kind: 'plugin', plugin: PLUGIN_NAME },
    })]
    const options = deepFreeze({
      provider: route.provider,
      model: route.model,
      messages,
      system,
      temperature: this.config.temperature,
      ...(this.config.reasoningEffort !== undefined ? { reasoningEffort: this.config.reasoningEffort } : {}),
      maxTokens: this.config.maxOutputTokens,
      signal: combined,
    })
    const textBlocks = []
    let deltas = ''
    let finish
    for await (const chunk of this.#ctx.llm.stream(options)) {
      if (chunk.type === 'text-delta') deltas += chunk.text
      else if (chunk.type === 'block-end' && chunk.block.type === 'text') textBlocks.push(chunk.block.text)
      else if (chunk.type === 'finish') finish = chunk.reason
    }
    if (finish === undefined) fail('model stream ended without a terminal finish chunk')
    if (finish.kind === 'error' || finish.kind === 'aborted') {
      const error = new Error(`${PLUGIN_NAME}: verifier model call failed: ${finish.failure.message}`)
      error.code = finish.failure.code
      error.finishKind = finish.kind
      throw error
    }
    // A max-tokens finish may still contain a complete score tag; let parsing decide.
    return textBlocks.length > 0 ? textBlocks.join('\n') : deltas
  }

  /** One scored sample: call the model once and parse the requested tags. */
  async #sample(system, userText, tags, signal) {
    const text = await this.#call(system, userText, signal)
    const parsed = parseScoreTags(text, tags, this.config.granularity)
    if (tags.some(tag => parsed[tag] !== null)) return parsed
    // A fully unparseable sample (usually reasoning that overran the token
    // budget before the tags) gets one fresh retry before counting as lost.
    const retryText = await this.#call(system, userText, signal)
    return parseScoreTags(retryText, tags, this.config.granularity)
  }

  /**
   * Score one A/B pair: C criteria x K repetitions of one two-trajectory
   * prompt with `<score_A>`/`<score_B>` tags. With `alternate`, presentation
   * order flips across repetitions and parsed scores are swapped back, so
   * pure positional bias cancels; PPT ring edges pass `alternate: false`
   * because the Hamiltonian cycle already balances slots per candidate.
   * Returns rewards in [0, 1] for both sides plus the raw samples.
   */
  async comparePair(task, candidateA, candidateB, { signal, alternate = true } = {}) {
    const tags = ['score_A', 'score_B']
    const system = verifierSystemPrompt(this.config.granularity, tags)
    const jobs = []
    for (const criterion of this.config.criteria) {
      for (let rep = 0; rep < this.config.repetitions; rep++) {
        jobs.push({ criterion, swapped: alternate && rep % 2 === 1 })
      }
    }
    const samples = await mapLimit(jobs, this.config.concurrency, async job => {
      const [first, second] = job.swapped ? [candidateB, candidateA] : [candidateA, candidateB]
      const parsed = await this.#sample(system, compareUserPrompt(task, first, second, job.criterion), tags, signal)
      const a = job.swapped ? parsed.score_B : parsed.score_A
      const b = job.swapped ? parsed.score_A : parsed.score_B
      return { criterion: job.criterion.name, swapped: job.swapped, a, b }
    })
    const validA = samples.map(s => s.a).filter(v => v !== null)
    const validB = samples.map(s => s.b).filter(v => v !== null)
    if (validA.length === 0 || validB.length === 0) fail('model returned no parseable <score_A>/<score_B> pair')
    return {
      rewardA: mean(validA.map(v => normalizeScore(v, this.config.granularity))),
      rewardB: mean(validB.map(v => normalizeScore(v, this.config.granularity))),
      samples,
    }
  }

  /**
   * Absolute reward for one candidate: mean of phi over C criteria x K
   * repetitions of single-candidate grading. Not used by PPT selection; kept
   * as a service utility for standalone quality scores.
   */
  async score(task, candidate, { signal } = {}) {
    const system = verifierSystemPrompt(this.config.granularity, ['score'])
    const perCriterion = await mapLimit(this.config.criteria, 1, async criterion => {
      const userText = scoreUserPrompt(task, candidate, criterion)
      const reps = Array.from({ length: this.config.repetitions })
      const samples = await mapLimit(reps, this.config.concurrency, () =>
        this.#sample(system, userText, ['score'], signal).then(parsed => parsed.score))
      const valid = samples.filter(v => v !== null)
      if (valid.length === 0) fail(`model returned no parseable <score> for criterion "${criterion.name}"`)
      return {
        criterion: criterion.name,
        reward: mean(valid.map(v => normalizeScore(v, this.config.granularity))),
        rawMean: mean(valid),
        samples,
      }
    })
    return {
      reward: mean(perCriterion.map(c => c.reward)),
      rawMean: mean(perCriterion.map(c => c.rawMean)),
      perCriterion,
    }
  }

  /**
   * Best-of-N selection via the Probabilistic Pivot Tournament.
   *
   * 1. Ring pass: a random Hamiltonian cycle scores the N adjacent pairs, so
   *    every candidate appears exactly once in the A slot and once in B.
   * 2. Pivot selection: rank candidates by their mean ring reward; the top
   *    `pivots` form the pivot set.
   * 3. Pivot tournament: score every not-yet-scored non-pivot-vs-pivot and
   *    pivot-vs-pivot pair (ring results are reused rather than re-scored).
   * 4. Aggregation: for each scored pair, both sides accumulate win mass
   *    sigma(R_self - R_other) and a comparison count; the final score is
   *    w_i / c_i and the highest wins (ties resolve to the earliest index).
   */
  async select(task, candidates, { signal } = {}) {
    if (!Array.isArray(candidates) || candidates.length < 2) fail('select needs at least 2 candidates')
    const n = candidates.length
    const pairResults = new Map()
    const pairKey = (i, j) => (i < j ? `${i}:${j}` : `${j}:${i}`)
    const scorePairs = async (pairs, alternate) => {
      const fresh = []
      const seen = new Set()
      for (const [i, j] of pairs) {
        const key = pairKey(i, j)
        if (pairResults.has(key) || seen.has(key)) continue
        seen.add(key)
        fresh.push([i, j])
      }
      await mapLimit(fresh, 2, async ([i, j]) => {
        const result = await this.comparePair(task, candidates[i], candidates[j], { signal, alternate })
        pairResults.set(pairKey(i, j), { i, j, rewardI: result.rewardA, rewardJ: result.rewardB })
      })
    }

    // 1. Ring pass over a random Hamiltonian cycle (fixed edge orientation).
    const cycle = shuffle(Array.from({ length: n }, (_, i) => i))
    const edges = cycle.map((from, idx) => [from, cycle[(idx + 1) % n]])
    await scorePairs(edges, n === 2)

    // 2. Pivot selection by mean ring reward.
    const ringRewards = Array.from({ length: n }, () => [])
    for (const { i, j, rewardI, rewardJ } of pairResults.values()) {
      ringRewards[i].push(rewardI)
      ringRewards[j].push(rewardJ)
    }
    const ringScores = ringRewards.map(rewards => (rewards.length > 0 ? mean(rewards) : 0))
    const k = Math.min(this.config.pivots, n - 1)
    const pivots = Array.from({ length: n }, (_, i) => i)
      .sort((a, b) => ringScores[b] - ringScores[a] || a - b)
      .slice(0, k)
    const pivotSet = new Set(pivots)

    // 3. Pivot tournament: everyone meets every pivot (reusing ring pairs).
    const tournamentPairs = []
    for (const pivot of pivots) {
      for (let i = 0; i < n; i++) {
        if (i !== pivot && !(pivotSet.has(i) && i < pivot)) tournamentPairs.push([i, pivot])
      }
    }
    await scorePairs(tournamentPairs, true)

    // 4. Win-mass aggregation.
    const winMass = new Array(n).fill(0)
    const counts = new Array(n).fill(0)
    for (const { i, j, rewardI, rewardJ } of pairResults.values()) {
      winMass[i] += sigmoid(rewardI - rewardJ)
      winMass[j] += sigmoid(rewardJ - rewardI)
      counts[i] += 1
      counts[j] += 1
    }
    const rewards = winMass.map((w, i) => (counts[i] > 0 ? w / counts[i] : 0))
    let bestIndex = 0
    for (let i = 1; i < n; i++) {
      if (rewards[i] > rewards[bestIndex]) bestIndex = i
    }
    return {
      bestIndex,
      bestReward: rewards[bestIndex],
      rewards,
      ringScores,
      pivots,
      pairs: [...pairResults.values()],
    }
  }

  /**
   * Pairwise comparison: order-debiased C x K sampling of one two-trajectory
   * prompt. `preferred` follows the sign of the margin; with the default
   * `tieMargin` of 0 a tie needs an exactly zero margin, matching the paper's
   * "no discrete ties" stance as closely as sampling allows.
   */
  async compare(task, candidateA, candidateB, { signal } = {}) {
    const { rewardA, rewardB, samples } = await this.comparePair(task, candidateA, candidateB, { signal })
    const margin = rewardA - rewardB
    const preferred = margin > this.config.tieMargin ? 'A' : margin < -this.config.tieMargin ? 'B' : 'tie'
    return { preferred, rewardA, rewardB, margin, samples }
  }

  /**
   * Progress scoring over a rollout: grade each cumulative prefix of the
   * trajectory against the progress criterion, K repetitions each. Prefixes
   * are judged blind to later steps (the paper's prefix representation).
   */
  async track(task, trajectory, { signal } = {}) {
    if (!Array.isArray(trajectory) || trajectory.length === 0) fail('track needs a non-empty trajectory')
    const system = verifierSystemPrompt(this.config.granularity, ['score'])
    const prefixes = trajectory.map((_, i) => trajectory.slice(0, i + 1))
    const progress = await mapLimit(prefixes, 1, async steps => {
      const userText = trackUserPrompt(task, steps)
      const reps = Array.from({ length: this.config.repetitions })
      const samples = await mapLimit(reps, this.config.concurrency, () =>
        this.#sample(system, userText, ['score'], signal).then(parsed => parsed.score))
      const valid = samples.filter(v => v !== null)
      if (valid.length === 0) fail(`model returned no parseable <score> for trajectory step ${steps.length}`)
      return { step: steps.length, reward: mean(valid.map(v => normalizeScore(v, this.config.granularity))), samples }
    })
    const trend = progress[progress.length - 1].reward - progress[0].reward
    return { progress, trend }
  }
}

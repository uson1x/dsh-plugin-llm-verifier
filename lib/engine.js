/**
 * The LLM-as-a-Verifier scoring engine over the harness LLM seam.
 *
 * Computes continuous rewards R(x, tau) = (1/CK) * sum phi(v) over C decomposed
 * criteria and K repeated evaluations, where each v is an integer score in
 * [1, G] sampled from the verifier model at temperature > 0 and
 * phi(v) = (v - 1) / (G - 1) maps it into [0, 1]. Repeated sampling is the
 * Monte Carlo estimator of the logit expectation sum p(v) * phi(v): the
 * harness `StreamChunk` vocabulary exposes no scoring-token logprobs, so the
 * expectation is estimated from samples instead of read from the distribution.
 * @module dsh-plugin-llm-verifier/engine
 */

import { createUserMessage, deepFreeze } from '@deepseek-ai/dsh-llm'
import {
  DEFAULT_CRITERIA,
  PROGRESS_CRITERION,
  compareUserPrompt,
  scoreSystemPrompt,
  scoreUserPrompt,
  trackUserPrompt,
} from './prompts.js'

export const PLUGIN_NAME = 'dsh-plugin-llm-verifier'

const CONFIG_KEYS = new Set([
  'provider',
  'model',
  'granularity',
  'repetitions',
  'temperature',
  'maxOutputTokens',
  'timeoutMs',
  'concurrency',
  'tieMargin',
  'criteria',
])

function fail(message) {
  throw new Error(`${PLUGIN_NAME}: ${message}`)
}

function assertPositiveInteger(name, value) {
  if (!Number.isInteger(value) || value <= 0) fail(`${name} must be a positive integer`)
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
    repetitions: config.repetitions ?? 3,
    temperature: config.temperature ?? 1,
    maxOutputTokens: config.maxOutputTokens ?? 2048,
    timeoutMs: config.timeoutMs ?? 120_000,
    concurrency: config.concurrency ?? 4,
    tieMargin: config.tieMargin ?? 0.02,
    criteria: config.criteria ?? DEFAULT_CRITERIA,
  }
  const hasProvider = resolved.provider !== undefined
  const hasModel = resolved.model !== undefined
  if (hasProvider !== hasModel) fail('provider and model must be supplied together')
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
  if (typeof resolved.temperature !== 'number' || !(resolved.temperature >= 0 && resolved.temperature <= 2)) {
    fail('temperature must be a number in [0, 2]')
  }
  if (typeof resolved.tieMargin !== 'number' || !(resolved.tieMargin >= 0 && resolved.tieMargin < 1)) {
    fail('tieMargin must be a number in [0, 1)')
  }
  if (!Array.isArray(resolved.criteria) || resolved.criteria.length === 0) {
    fail('criteria must be a non-empty array')
  }
  for (const criterion of resolved.criteria) {
    if (criterion === null || typeof criterion !== 'object'
      || typeof criterion.name !== 'string' || criterion.name.length === 0
      || typeof criterion.description !== 'string' || criterion.description.length === 0) {
      fail('each criterion needs non-empty string "name" and "description"')
    }
  }
  return deepFreeze({ ...resolved, criteria: resolved.criteria.map(c => ({ name: c.name, description: c.description })) })
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

function mean(values) {
  return values.reduce((sum, v) => sum + v, 0) / values.length
}

/** Run tasks with bounded concurrency, preserving order and rejecting on first failure. */
async function mapLimit(items, limit, worker) {
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
  async #call(system, userText, signal) {
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
      throw error
    }
    // A max-tokens finish may still contain a complete score tag; let parsing decide.
    return textBlocks.length > 0 ? textBlocks.join('\n') : deltas
  }

  /** One scored sample: call the model once and parse the requested tags. */
  async #sample(system, userText, tags, signal) {
    const text = await this.#call(system, userText, signal)
    return parseScoreTags(text, tags, this.config.granularity)
  }

  /**
   * Score one candidate against one criterion with K repetitions.
   * Returns per-criterion reward in [0, 1] plus the raw integer samples.
   */
  async #scoreCriterion(task, candidate, criterion, signal) {
    const system = scoreSystemPrompt(this.config.granularity, ['score'])
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
  }

  /**
   * Continuous reward for one candidate: mean over C criteria x K repetitions.
   * Returns `{ reward, rawMean, perCriterion }` with reward in [0, 1].
   */
  async score(task, candidate, { signal } = {}) {
    const perCriterion = await mapLimit(this.config.criteria, this.config.concurrency, criterion =>
      this.#scoreCriterion(task, candidate, criterion, signal))
    return {
      reward: mean(perCriterion.map(c => c.reward)),
      rawMean: mean(perCriterion.map(c => c.rawMean)),
      perCriterion,
    }
  }

  /**
   * Best-of-N selection: score every candidate, return the best index.
   * Ties resolve to the earliest candidate.
   */
  async select(task, candidates, { signal } = {}) {
    if (!Array.isArray(candidates) || candidates.length < 2) fail('select needs at least 2 candidates')
    const scores = await mapLimit(candidates, 1, candidate => this.score(task, candidate, { signal }))
    let bestIndex = 0
    for (let i = 1; i < scores.length; i++) {
      if (scores[i].reward > scores[bestIndex].reward) bestIndex = i
    }
    return { bestIndex, bestReward: scores[bestIndex].reward, scores }
  }

  /**
   * Pairwise comparison with order debiasing: alternate presentation order
   * across repetitions and score both candidates in one prompt via
   * `<score_A>` / `<score_B>` tags, per criterion.
   */
  async compare(task, candidateA, candidateB, { signal } = {}) {
    const tags = ['score_A', 'score_B']
    const system = scoreSystemPrompt(this.config.granularity, tags)
    const jobs = []
    for (const criterion of this.config.criteria) {
      for (let rep = 0; rep < this.config.repetitions; rep++) {
        jobs.push({ criterion, swapped: rep % 2 === 1 })
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
    const rewardA = mean(validA.map(v => normalizeScore(v, this.config.granularity)))
    const rewardB = mean(validB.map(v => normalizeScore(v, this.config.granularity)))
    const margin = rewardA - rewardB
    const preferred = margin > this.config.tieMargin ? 'A' : margin < -this.config.tieMargin ? 'B' : 'tie'
    return { preferred, rewardA, rewardB, margin, samples }
  }

  /**
   * Progress scoring over a rollout: grade each cumulative prefix of the
   * trajectory against the progress criterion, K repetitions each.
   */
  async track(task, trajectory, { signal } = {}) {
    if (!Array.isArray(trajectory) || trajectory.length === 0) fail('track needs a non-empty trajectory')
    const system = scoreSystemPrompt(this.config.granularity, ['score'])
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

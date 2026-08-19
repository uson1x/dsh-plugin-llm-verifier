/**
 * LLM-as-a-Verifier plugin for DeepSeek Harness.
 *
 * Registers the `ctx.verifier` service (select / compare / track / score) and
 * four model-facing tools: `verify_select`, `verify_compare`, `verify_track`
 * (the paper's three APIs over existing candidates) and `verify_rollout`
 * (generate-then-judge: N independent subagent rollouts judged by the
 * verifier, winner returned).
 * @module dsh-plugin-llm-verifier
 */

import Schema from '@deepseek-ai/schemastery'
import { defineTool } from '@deepseek-ai/dsh-tools'
import { PLUGIN_NAME, VerifierEngine, mapLimit } from './engine.js'
import { rolloutPrompt } from './prompts.js'

export { VerifierEngine, resolveVerifierConfig, parseScoreTags, normalizeScore, sigmoid, PLUGIN_NAME } from './engine.js'
export { DEFAULT_CRITERIA, PROGRESS_CRITERION, rolloutPrompt } from './prompts.js'

export const name = 'llm-verifier'
export const inject = ['llm', 'tools', 'subagents', 'systemPrompt']

/** Loader-validated plugin configuration (fail-loud before apply runs). */
export const Config = Schema.object({
  provider: Schema.string().description('Registered ctx.llm provider route the verifier grades with.'),
  model: Schema.string().description('Model id on that route.'),
  granularity: Schema.number().default(20).description('Integer score scale 1..G.'),
  repetitions: Schema.number().default(4).description('Grading passes per criterion (K).'),
  temperature: Schema.number().default(1).description('Sampling temperature for the Monte Carlo estimate.'),
  maxOutputTokens: Schema.number().default(2048).description('Output cap per grading call.'),
  timeoutMs: Schema.number().default(120000).description('Deadline per grading call in milliseconds.'),
  concurrency: Schema.number().default(4).description('Parallel grading calls.'),
  tieMargin: Schema.number().default(0).description('verify_compare margin below which the verdict is "tie".'),
  promptSection: Schema.boolean().default(true).description('Register the fuzzy-routing system-prompt section so loose phrasings reach the verify_* tools.'),
  pivots: Schema.number().default(2).description('Pivot count k for the Probabilistic Pivot Tournament.'),
  criteria: Schema.array(Schema.object({
    name: Schema.string().required(),
    description: Schema.string().required(),
  })).description('Criteria decomposition; defaults to specification / output / errors.'),
  rollout: Schema.object({
    provider: Schema.string().default('spawn').description('ctx.subagents provider for rollout children.'),
    llmProvider: Schema.string().description('LLM provider route override for rollout children.'),
    model: Schema.string().description('Model id override for rollout children (e.g. a cheaper model).'),
    maxConcurrent: Schema.number().default(3).description('Rollout children running at once.'),
  }).description('verify_rollout defaults.'),
})

/** Tool names denied inside rollout children, preventing recursive fan-out. */
const VERIFY_TOOL_NAMES = ['verify_select', 'verify_compare', 'verify_track', 'verify_rollout']

function round(value) {
  return Math.round(value * 1000) / 1000
}

function renderSelect(value) {
  const lines = value.rewards.map((reward, i) =>
    `${i === value.best_index ? '>' : ' '} [${i}] score ${round(reward)}`)
  return `Best candidate: index ${value.best_index} (score ${round(value.best_reward)})\n${lines.join('\n')}`
}

function renderCompare(value) {
  return `Preferred: ${value.preferred} (A ${round(value.reward_a)} vs B ${round(value.reward_b)}, margin ${round(value.margin)})`
}

function renderTrack(value) {
  const path = value.progress.map(r => round(r)).join(' -> ')
  return `Progress rewards by step: ${path} (trend ${round(value.trend)})`
}

function renderRollout(value) {
  const lines = value.rewards.map((reward, i) => {
    const mark = i === value.best_index ? '>' : ' '
    const score = reward === null ? 'failed' : `score ${round(reward)}`
    return `${mark} rollout ${i + 1}: ${score} (session ${value.rollout_sessions[i] ?? 'n/a'})`
  })
  return `Rollout ${value.best_index + 1} wins (score ${round(value.best_reward)})\n${lines.join('\n')}\n\nWinning deliverable:\n${value.winner}`
}

/** Extract plain text from a subagent result's final-message content blocks. */
function textFromBlocks(blocks) {
  if (!Array.isArray(blocks)) return ''
  return blocks
    .filter(block => block !== null && typeof block === 'object' && block.type === 'text' && typeof block.text === 'string')
    .map(block => block.text)
    .join('\n')
    .trim()
}

/** Run one rollout child to completion, always disposing the run. */
async function runRollout(subagents, providerName, request) {
  const run = await subagents.start(providerName, request)
  try {
    const result = await run.result
    return {
      sessionId: typeof run.id === 'string' ? run.id : String(run.id ?? ''),
      stopReason: result.stopReason,
      text: textFromBlocks(result.output),
    }
  } finally {
    // Dispose on every path; a disposal failure must not mask the result.
    await Promise.allSettled([Promise.resolve().then(() => run.dispose())])
  }
}

export function apply(ctx, config = {}) {
  const engine = new VerifierEngine(ctx, config)
  const rolloutConfig = engine.config.rollout

  // Direct capability access for other plugins: ctx.verifier.select(...) etc.
  ctx.provide('verifier', engine)

  // Fuzzy routing: a short tool-guidance section (order band 100-199) so
  // loose phrasings like "use LLM as a verifier to ..." reach the right tool
  // without the user naming it.
  if (config.promptSection !== false) {
    ctx.systemPrompt.section({
      name: 'tool:verifier',
      order: 150,
      text: [
        'LLM-as-a-Verifier tools grade candidate solutions with continuous rewards in [0, 1]. Route these intents to them even when phrased loosely:',
        '- Asking to PRODUCE something "with the verifier", "using LLM as a verifier", "best of N", or "with N attempts/rollouts/trajectories, keep the best" -> call verify_rollout. Restate the task self-containedly in the `task` argument; set `n` to the mentioned count, or omit it for the default of 3.',
        '- Picking the best among alternatives that ALREADY exist -> verify_select; for exactly two, verify_compare.',
        '- Judging whether a sequence of steps is making progress on a task -> verify_track.',
        'Prefer one verify_rollout call over hand-rolling subagents plus verify_select when the user asks for verified generation.',
      ].join('\n'),
    })
  }

  ctx.tools.register(defineTool({
    name: 'verify_select',
    description: 'LLM-as-a-Verifier best-of-N selection over EXISTING candidates via a Probabilistic Pivot Tournament: a position-debiased ring pass, top-k pivots, a pivot tournament, and win-mass aggregation produce a continuous score per candidate and the index of the best. Use when you already have several alternative solutions and must pick the strongest.',
    parameters: {
      task: { type: 'string', required: true, description: 'The task or problem statement the candidates attempt to solve.' },
      candidates: {
        type: 'array',
        required: true,
        items: { type: 'string' },
        description: 'The candidate solutions, in order. At least 2.',
      },
    },
    output: {
      schema: {
        type: 'object',
        additionalProperties: false,
        properties: {
          best_index: { type: 'integer', required: true, description: '0-based index of the winning candidate.' },
          best_reward: { type: 'number', required: true, description: 'Winning tournament score in [0, 1].' },
          rewards: { type: 'array', required: true, items: { type: 'number' }, description: 'Tournament score per candidate (win mass / comparisons), same order as the input.' },
          details: { type: 'json', description: 'Ring scores, pivot set, and per-pair rewards.' },
        },
      },
      render: (_args, value) => [{ type: 'text', text: renderSelect(value) }],
    },
    async execute(args, exec) {
      if (args.candidates.length < 2) throw new Error(`${PLUGIN_NAME}: verify_select needs at least 2 candidates`)
      const result = await engine.select(args.task, args.candidates, { signal: exec.signal })
      return {
        best_index: result.bestIndex,
        best_reward: result.bestReward,
        rewards: result.rewards,
        details: { ring_scores: result.ringScores, pivots: result.pivots, pairs: result.pairs },
      }
    },
  }))

  ctx.tools.register(defineTool({
    name: 'verify_compare',
    description: 'LLM-as-a-Verifier pairwise comparison. Grades two candidate trajectories against the task in one prompt (order-debiased across repetitions) and reports which is better with a continuous margin. Use to decide between exactly two alternatives.',
    parameters: {
      task: { type: 'string', required: true, description: 'The task or problem statement both candidates attempt to solve.' },
      candidate_a: { type: 'string', required: true, description: 'Candidate A.' },
      candidate_b: { type: 'string', required: true, description: 'Candidate B.' },
    },
    output: {
      schema: {
        type: 'object',
        additionalProperties: false,
        properties: {
          preferred: { type: 'string', required: true, enum: ['A', 'B', 'tie'], description: 'Which candidate the verifier prefers.' },
          reward_a: { type: 'number', required: true, description: 'Candidate A reward in [0, 1].' },
          reward_b: { type: 'number', required: true, description: 'Candidate B reward in [0, 1].' },
          margin: { type: 'number', required: true, description: 'reward_a minus reward_b.' },
          details: { type: 'json', description: 'Raw per-repetition integer score samples with presentation order.' },
        },
      },
      render: (_args, value) => [{ type: 'text', text: renderCompare(value) }],
    },
    async execute(args, exec) {
      const result = await engine.compare(args.task, args.candidate_a, args.candidate_b, { signal: exec.signal })
      return {
        preferred: result.preferred,
        reward_a: result.rewardA,
        reward_b: result.rewardB,
        margin: result.margin,
        details: result.samples,
      }
    },
  }))

  ctx.tools.register(defineTool({
    name: 'verify_track',
    description: 'LLM-as-a-Verifier progress tracking. Grades each cumulative prefix of a solution trajectory for verified progress toward completing the task, returning a continuous progress reward per step. Use to monitor whether a multi-step rollout is on course.',
    parameters: {
      task: { type: 'string', required: true, description: 'The task the trajectory is trying to complete.' },
      trajectory: {
        type: 'array',
        required: true,
        items: { type: 'string' },
        description: 'The ordered steps of the rollout so far. At least 1.',
      },
    },
    output: {
      schema: {
        type: 'object',
        additionalProperties: false,
        properties: {
          progress: { type: 'array', required: true, items: { type: 'number' }, description: 'Progress reward in [0, 1] after each step, same order as the input.' },
          trend: { type: 'number', required: true, description: 'Last-step reward minus first-step reward.' },
          details: { type: 'json', description: 'Raw integer score samples per step.' },
        },
      },
      render: (_args, value) => [{ type: 'text', text: renderTrack(value) }],
    },
    async execute(args, exec) {
      if (args.trajectory.length === 0) throw new Error(`${PLUGIN_NAME}: verify_track needs a non-empty trajectory`)
      const result = await engine.track(args.task, args.trajectory, { signal: exec.signal })
      return {
        progress: result.progress.map(step => step.reward),
        trend: result.trend,
        details: result.progress,
      }
    },
  }))

  ctx.tools.register(defineTool({
    name: 'verify_rollout',
    description: 'LLM-as-a-Verifier generate-then-judge (best-of-N over fresh rollouts). Spawns n independent subagent attempts at the task in parallel, collects each attempt\'s final deliverable verbatim, judges them with the verifier\'s pivot tournament, and returns the winning deliverable with per-rollout scores. Use when the task should be attempted several ways and only the best attempt kept. Each rollout is a real tool-using agent; expect this call to take minutes.',
    parameters: {
      task: { type: 'string', required: true, description: 'The complete, self-contained task each rollout attempts. Rollouts do not see this conversation.' },
      n: { type: 'integer', description: 'How many independent rollouts to run (2-8). Default 3.' },
      rollout_model: { type: 'string', description: 'Optional model id override for the rollout children (e.g. a cheaper model). The judge keeps its own configured model.' },
    },
    output: {
      schema: {
        type: 'object',
        additionalProperties: false,
        properties: {
          best_index: { type: 'integer', required: true, description: '0-based index of the winning rollout.' },
          best_reward: { type: 'number', required: true, description: 'Winning tournament score in [0, 1].' },
          winner: { type: 'string', required: true, description: 'The winning rollout\'s final deliverable, verbatim.' },
          rewards: {
            type: 'array',
            required: true,
            items: { oneOf: [{ type: 'number' }, { type: 'null' }] },
            description: 'Tournament score per rollout in launch order; null for a failed rollout.',
          },
          rollout_sessions: {
            type: 'array',
            required: true,
            items: { oneOf: [{ type: 'string' }, { type: 'null' }] },
            description: 'Child session id per rollout, for inspecting the full trajectories.',
          },
          details: { type: 'json', description: 'Per-rollout stop reasons and the judge\'s tournament details.' },
        },
      },
      render: (_args, value) => [{ type: 'text', text: renderRollout(value) }],
    },
    async execute(args, exec) {
      const n = args.n ?? 3
      if (!Number.isInteger(n) || n < 2 || n > 8) throw new Error(`${PLUGIN_NAME}: n must be an integer in [2, 8]`)
      const providerName = rolloutConfig.provider
      if (ctx.subagents.getProvider(providerName) === undefined) {
        throw new Error(`${PLUGIN_NAME}: subagent provider "${providerName}" is not registered (available: ${ctx.subagents.list().join(', ') || 'none'})`)
      }
      const llmProvider = rolloutConfig.llmProvider
      const model = args.rollout_model ?? rolloutConfig.model
      const agentOptions = {
        ...(llmProvider !== undefined ? { provider: llmProvider } : {}),
        ...(model !== undefined ? { model } : {}),
      }

      // 1. Generate: n independent fresh children, each blind to the others.
      const rollouts = await mapLimit(Array.from({ length: n }, (_, i) => i), rolloutConfig.maxConcurrent, async index => {
        try {
          const outcome = await runRollout(ctx.subagents, providerName, {
            label: `verifier rollout ${index + 1}/${n}`,
            prompt: [{ type: 'text', text: rolloutPrompt(args.task, index, n) }],
            parent: exec.agent,
            signal: exec.signal,
            toolFilter: { deny: VERIFY_TOOL_NAMES },
            ...(Object.keys(agentOptions).length > 0 ? { agentOptions } : {}),
          })
          const ok = outcome.stopReason === 'completed' && outcome.text.length > 0
          return { index, ok, ...outcome }
        } catch (error) {
          if (exec.signal.aborted) throw error
          return { index, ok: false, sessionId: null, stopReason: `start-failed: ${String(error?.message ?? error)}`, text: '' }
        }
      })

      const succeeded = rollouts.filter(r => r.ok)
      if (succeeded.length < 2) {
        const summary = rollouts.map(r => `rollout ${r.index + 1}: ${r.ok ? 'ok' : r.stopReason}`).join('; ')
        throw new Error(`${PLUGIN_NAME}: need at least 2 successful rollouts to judge, got ${succeeded.length} (${summary})`)
      }

      // 2. Judge: pivot tournament over the successful deliverables.
      const judged = await engine.select(args.task, succeeded.map(r => r.text), { signal: exec.signal })
      const winnerRollout = succeeded[judged.bestIndex]
      const rewards = rollouts.map(() => null)
      succeeded.forEach((r, judgedIndex) => { rewards[r.index] = judged.rewards[judgedIndex] })

      return {
        best_index: winnerRollout.index,
        best_reward: judged.bestReward,
        winner: winnerRollout.text,
        rewards,
        rollout_sessions: rollouts.map(r => r.sessionId ?? null),
        details: {
          stop_reasons: rollouts.map(r => ({ rollout: r.index + 1, ok: r.ok, stop_reason: r.stopReason, session: r.sessionId })),
          judge: { ring_scores: judged.ringScores, pivots: judged.pivots, pairs: judged.pairs },
        },
      }
    },
  }))
}

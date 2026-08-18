/**
 * LLM-as-a-Verifier plugin for DeepSeek Harness.
 *
 * Registers the `ctx.verifier` service (select / compare / track / score) and
 * three model-facing tools (`verify_select`, `verify_compare`, `verify_track`)
 * that expose continuous verification rewards to the agent.
 * @module dsh-plugin-llm-verifier
 */

import { defineTool } from '@deepseek-ai/dsh-tools'
import { PLUGIN_NAME, VerifierEngine } from './engine.js'

export { VerifierEngine, resolveVerifierConfig, parseScoreTags, normalizeScore, PLUGIN_NAME } from './engine.js'
export { DEFAULT_CRITERIA, PROGRESS_CRITERION } from './prompts.js'

export const name = 'llm-verifier'
export const inject = ['llm', 'tools']

function round(value) {
  return Math.round(value * 1000) / 1000
}

function renderSelect(value) {
  const lines = value.rewards.map((reward, i) =>
    `${i === value.best_index ? '>' : ' '} [${i}] reward ${round(reward)}`)
  return `Best candidate: index ${value.best_index} (reward ${round(value.best_reward)})\n${lines.join('\n')}`
}

function renderCompare(value) {
  return `Preferred: ${value.preferred} (A ${round(value.reward_a)} vs B ${round(value.reward_b)}, margin ${round(value.margin)})`
}

function renderTrack(value) {
  const path = value.progress.map(r => round(r)).join(' -> ')
  return `Progress rewards by step: ${path} (trend ${round(value.trend)})`
}

export function apply(ctx, config = {}) {
  const engine = new VerifierEngine(ctx, config)

  // Direct capability access for other plugins: ctx.verifier.select(...) etc.
  ctx.provide('verifier', engine)

  ctx.tools.register(defineTool({
    name: 'verify_select',
    description: 'LLM-as-a-Verifier best-of-N selection. Scores each candidate solution against the task via repeated fine-grained grading over decomposed criteria (continuous reward in [0, 1]) and returns the index of the best candidate. Use when you have several alternative solutions and must pick the strongest.',
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
          best_index: { type: 'integer', required: true, description: '0-based index of the highest-reward candidate.' },
          best_reward: { type: 'number', required: true, description: 'Reward of the best candidate in [0, 1].' },
          rewards: { type: 'array', required: true, items: { type: 'number' }, description: 'Reward per candidate, same order as the input.' },
          details: { type: 'json', description: 'Per-candidate, per-criterion rewards and raw integer score samples.' },
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
        rewards: result.scores.map(score => score.reward),
        details: result.scores.map((score, index) => ({
          index,
          reward: score.reward,
          raw_mean: score.rawMean,
          per_criterion: score.perCriterion,
        })),
      }
    },
  }))

  ctx.tools.register(defineTool({
    name: 'verify_compare',
    description: 'LLM-as-a-Verifier pairwise comparison. Grades two candidate solutions against the task in one prompt (order-debiased across repetitions) and reports which is better with a continuous margin. Use to decide between exactly two alternatives.',
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
}

/**
 * Prompt construction for LLM-as-a-Verifier scoring calls.
 *
 * Follows the llm-as-a-verifier reference template: an expert-reviewer
 * persona, the task and trajectories first, the evaluation criterion at the
 * prompt TAIL (so repeated calls over the same pair share a provider-cacheable
 * prefix across criteria), fine-grained integer scales (1..G) with the paper's
 * scale anchors (1 = incorrect, midpoint = borderline, G = flawless), and
 * scores inside XML tags.
 * @module dsh-plugin-llm-verifier/prompts
 */

/**
 * Default criteria decomposition for judging candidate solutions, mirroring
 * the paper's Specification / Output / Errors split for coding tasks.
 */
export const DEFAULT_CRITERIA = [
  {
    name: 'specification',
    description: 'Does the candidate address every requirement stated in the task, without ignoring, weakening, or reinterpreting any part of it?',
  },
  {
    name: 'output',
    description: 'Is the result or output the candidate produces correct, complete, and directly usable for the task?',
  },
  {
    name: 'errors',
    description: 'Is the candidate free of errors: bugs, false claims, broken logic, unsafe steps, or internally inconsistent reasoning?',
  },
]

/** Single criterion used by `track` to grade partial-trajectory progress. */
export const PROGRESS_CRITERION = {
  name: 'progress',
  description: 'How much verified progress toward fully completing the task does this partial trajectory represent? 1 means no progress or actively harmful steps; the maximum means the task is essentially complete and correct.',
}

/** Expert-reviewer persona with the paper's scale anchors and output discipline. */
export function verifierSystemPrompt(granularity, tags) {
  const tagList = tags.map(tag => `<${tag}>N</${tag}>`).join(' and ')
  const borderline = Math.round((granularity + 1) / 2)
  return [
    'You are an expert reviewer acting as a strict, impartial verifier. You judge candidate solutions to a task; you never solve the task yourself.',
    `Rate on an integer scale from 1 to ${granularity}, where 1 means incorrect or completely failing, ${borderline} means borderline, and ${granularity} means flawless.`,
    'Judge only what is actually present in the candidate. Unsupported claims count against it. Do not reward length, style, or confidence.',
    'Keep your analysis brief — a few sentences at most.',
    `Carefully analyze, then end your reply with exactly ${tagList}, where each N is one integer from 1 to ${granularity}. Output each tag exactly once and nothing after the final tag.`,
  ].join('\n')
}

/** Shared criterion tail: keeps the task/candidate prefix identical across criteria. */
function criterionTail(criterion, subject) {
  return [
    `Evaluation criterion (${criterion.name}): ${criterion.description}`,
    '',
    `Carefully analyze ${subject} against this criterion, then provide your final scores.`,
  ].join('\n')
}

/** Frame one candidate for absolute scoring. JSON-frames untrusted text so it cannot break the structure. */
export function scoreUserPrompt(task, candidate, criterion) {
  return [
    'Task, as a JSON string:',
    JSON.stringify(task),
    '',
    'Candidate solution, as a JSON string:',
    JSON.stringify(candidate),
    '',
    criterionTail(criterion, 'the candidate'),
  ].join('\n')
}

/** Frame a pairwise A/B comparison; the caller controls which candidate sits in each slot. */
export function compareUserPrompt(task, candidateA, candidateB, criterion) {
  return [
    'You will see a task description and two candidate trajectories.',
    '',
    'Task, as a JSON string:',
    JSON.stringify(task),
    '',
    'Trajectory A, as a JSON string:',
    JSON.stringify(candidateA),
    '',
    'Trajectory B, as a JSON string:',
    JSON.stringify(candidateB),
    '',
    criterionTail(criterion, 'each trajectory'),
  ].join('\n')
}

/** Frame a partial trajectory for progress scoring. */
export function trackUserPrompt(task, steps) {
  return [
    'Task, as a JSON string:',
    JSON.stringify(task),
    '',
    `Partial trajectory so far (${steps.length} step${steps.length === 1 ? '' : 's'}), as a JSON array of steps in order:`,
    JSON.stringify(steps),
    '',
    criterionTail(PROGRESS_CRITERION, 'the trajectory'),
  ].join('\n')
}

/** Self-contained task prompt for one independent rollout child. */
export function rolloutPrompt(task, index, total) {
  return [
    `You are attempt ${index + 1} of ${total} independent attempts at the same task. You do not see the other attempts.`,
    'Solve the task completely and self-containedly.',
    'Your FINAL message must contain the complete deliverable on its own: it is extracted verbatim and judged against the other attempts by a verifier that sees only that message. Do not end with a summary that omits the actual work product.',
    '',
    'Task:',
    task,
  ].join('\n')
}

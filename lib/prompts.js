/**
 * Prompt construction for LLM-as-a-Verifier scoring calls.
 *
 * The verifier asks for an integer rating inside an XML tag (e.g.
 * `<score>17</score>`), following the llm-as-a-verifier framing: fine-grained
 * integer scales (1..G) discriminate better than coarse 1-5 rubrics, and
 * decomposing a monolithic rubric into simple sub-criteria reduces prompt bias.
 * @module dsh-plugin-llm-verifier/prompts
 */

/**
 * Default criteria decomposition for judging one candidate solution,
 * mirroring the Specification / Output / Errors split described by
 * llm-as-a-verifier.com.
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

/** Shared verifier persona and output discipline for score responses. */
export function scoreSystemPrompt(granularity, tags) {
  const tagList = tags.map(tag => `<${tag}>N</${tag}>`).join(' and ')
  return [
    'You are a strict, impartial verifier. You judge candidate solutions to a task; you never solve the task yourself.',
    `Rate on an integer scale from 1 to ${granularity}, where 1 is completely failing and ${granularity} is flawless.`,
    'Judge only what is actually present in the candidate. Unsupported claims count against it. Do not reward length, style, or confidence.',
    `After thinking, end your reply with exactly ${tagList}, where each N is one integer from 1 to ${granularity}. Output each tag exactly once and nothing after the final tag.`,
  ].join('\n')
}

/** Frame one candidate against one criterion. JSON-frames untrusted text so it cannot break the structure. */
export function scoreUserPrompt(task, candidate, criterion) {
  return [
    `Criterion (${criterion.name}): ${criterion.description}`,
    '',
    'Task, as a JSON string:',
    JSON.stringify(task),
    '',
    'Candidate solution, as a JSON string:',
    JSON.stringify(candidate),
    '',
    'Rate the candidate against the criterion above.',
  ].join('\n')
}

/** Frame a pairwise A/B comparison; caller may pre-swap the pair to debias order. */
export function compareUserPrompt(task, candidateA, candidateB, criterion) {
  return [
    `Criterion (${criterion.name}): ${criterion.description}`,
    '',
    'Task, as a JSON string:',
    JSON.stringify(task),
    '',
    'Candidate A, as a JSON string:',
    JSON.stringify(candidateA),
    '',
    'Candidate B, as a JSON string:',
    JSON.stringify(candidateB),
    '',
    'Rate each candidate independently against the criterion above.',
  ].join('\n')
}

/** Frame a partial trajectory for progress scoring. */
export function trackUserPrompt(task, steps) {
  return [
    `Criterion (${PROGRESS_CRITERION.name}): ${PROGRESS_CRITERION.description}`,
    '',
    'Task, as a JSON string:',
    JSON.stringify(task),
    '',
    `Partial trajectory so far (${steps.length} step${steps.length === 1 ? '' : 's'}), as a JSON array of steps in order:`,
    JSON.stringify(steps),
    '',
    'Rate the progress of this trajectory against the criterion above.',
  ].join('\n')
}

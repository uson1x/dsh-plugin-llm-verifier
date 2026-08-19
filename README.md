# dsh-plugin-llm-verifier

[LLM-as-a-Verifier](https://llm-as-a-verifier.com) ([paper](https://arxiv.org/abs/2607.05391), [reference impl](https://github.com/llm-as-a-verifier/llm-as-a-verifier)) for [DeepSeek Harness](https://github.com/deepseek-ai/deepseek-harness) (`dsh`).

An out-of-tree dsh plugin that turns any model behind the harness's `ctx.llm` seam into a verifier producing **continuous reward signals** for candidate solutions — no extra training, no separate reward model. It implements the framework's three APIs (`select` / `compare` / `track`) plus a harness-native fourth: `verify_rollout`, which **generates N independent subagent attempts and judges them** in one tool call.

## What it implements

The framework scales verification along three axes, all configurable here:

1. **Score granularity** — fine-grained integer scales (`1..G`, default `G = 20`, anchors: 1 = incorrect, midpoint = borderline, G = flawless).
2. **Repeated evaluation** — `K` independent grading passes (default 4, the reference implementation's default; the paper reports gains up to K = 16).
3. **Criteria decomposition** — the rubric is split into simple sub-criteria (default: *specification*, *output*, *errors*, the paper's coding decomposition).

The reward for candidate τ on task x is

```
R(x, τ) = (1/CK) · Σ_{c,k} φ(v_{c,k}),   φ(v) = (v − 1) / (G − 1) ∈ [0, 1]
```

Grading calls use the paper's prompt shape — expert-reviewer persona, task and trajectories first, the evaluation criterion at the prompt **tail** (so the C criteria share one provider-cacheable prefix per pair), and scores inside XML tags (`<score>`, or `<score_A>`/`<score_B>` pairwise).

### Best-of-N: Probabilistic Pivot Tournament

`select` implements the paper's PPT, reducing selection from O(N²) to O(Nk) pairwise comparisons:

1. **Ring pass** — a random Hamiltonian cycle scores the N adjacent pairs; every candidate appears exactly once in the A slot and once in B, canceling positional bias structurally.
2. **Pivot selection** — the top-k candidates by mean ring reward form the pivot set (`pivots`, default k = 2).
3. **Pivot tournament** — every remaining non-pivot-vs-pivot and pivot-vs-pivot pair is scored (ring results are reused, not re-scored).
4. **Aggregation** — each scored pair contributes win mass `σ(R_self − R_other)` to both sides; the final score is `w_i / c_i` and the highest wins.

## The four APIs

As tools (what the agent sees):

| Tool | Purpose |
|---|---|
| `verify_select(task, candidates[])` | Best-of-N over existing candidates via the pivot tournament |
| `verify_compare(task, candidate_a, candidate_b)` | Pairwise comparison, order-debiased across repetitions, continuous margin |
| `verify_track(task, trajectory[])` | Progress reward per cumulative prefix of a rollout, plus the trend |
| `verify_rollout(task, n?, rollout_model?)` | **Generate-then-judge**: spawn n independent subagent attempts (fresh `spawn` children, parallel, blind to each other), collect each final deliverable verbatim, judge with the pivot tournament, return the winner |

`verify_rollout` is the implicit best-of-N flow: *"use verify_rollout with n=5 to build me a landing page"* runs five real tool-using agents and hands back the verifier's winner, with each child's session id so the full trajectories stay inspectable in the UI. Rollout children are denied the `verify_*` tools (no recursive fan-out) and can run a cheaper model than the judge via `rollout_model` or the `rollout.model` config.

As a service (what other plugins see), registered at `ctx.verifier`:

```js
const { bestIndex, rewards, ringScores, pivots } = await ctx.verifier.select(task, candidates)
const { preferred, margin } = await ctx.verifier.compare(task, a, b)
const { progress, trend } = await ctx.verifier.track(task, steps)
const { reward, perCriterion } = await ctx.verifier.score(task, candidate)  // absolute utility, not part of PPT
```

## Install

Into an existing dsh profile (the Web profile shown; any profile works):

```sh
cd ~/.dsh/profiles/web
npm install github:uson1x/dsh-plugin-llm-verifier
```

Then add the insert row to `~/.dsh/profiles/web/cordis.patch.yml`:

```yaml
- insert:
    - id: llm-verifier
      name: dsh-plugin-llm-verifier
      config:
        provider: deepseek-official
        model: deepseek-v4-pro
        # rollout:
        #   model: deepseek-v4-flash   # cheaper model for rollout children
```

Restart `dsh web` (the shipped web composition loads the patch layer at boot). The plugin's `@deepseek-ai/*` imports resolve through the harness's own profile-level module tree (declared as optional peer dependencies precisely so npm does not fetch duplicate copies).

## Configuration

| Key | Default | Meaning |
|---|---|---|
| `provider` | — (required) | Registered `ctx.llm` provider route the verifier grades with |
| `model` | — (required) | Model id on that route |
| `granularity` | `20` | Integer score scale `1..G` |
| `repetitions` | `4` | Grading passes per criterion (`K`) |
| `temperature` | `1` | Sampling temperature for the Monte Carlo estimate |
| `criteria` | specification / output / errors | Array of `{ name, description }` sub-criteria |
| `pivots` | `2` | Pivot count k for the tournament |
| `tieMargin` | `0` | `compare` margin below which the verdict is `tie` (0 ≈ the paper's no-ties stance) |
| `maxOutputTokens` | `2048` | Output cap per grading call |
| `timeoutMs` | `120000` | Deadline per grading call |
| `concurrency` | `4` | Parallel grading calls |
| `rollout.provider` | `spawn` | `ctx.subagents` provider for rollout children |
| `rollout.model` | — | Model id override for rollout children |
| `rollout.llmProvider` | — | LLM provider route override for rollout children |
| `rollout.maxConcurrent` | `3` | Rollout children running at once |

Cost: one pairwise comparison costs `C × K` grading calls. `verify_select` scores `N` ring pairs plus at most `k·(N−1)` tournament pairs; `verify_rollout` adds the n child agent runs on top. `verify_track` costs `steps × K` calls.

## Paper fidelity and deviations

Followed: the R(x, τ) formula, 1–20 granularity with the paper's anchors, repeated evaluation, criteria decomposition, the pairwise `<score_A>`/`<score_B>` template, PPT with ring-pass debiasing and `σ(R_a − R_b)` win mass, criteria-at-tail prompt layout for KV-cache reuse, and prefix-based progress tracking.

Deviations, each deliberate:

- **Sampling, not logits.** The reference reads the full distribution of scoring-token logits (`Σ_g p(v_g)·φ(v_g)`, letter-scale tokens). The dsh `StreamChunk` vocabulary exposes no logprobs, so this plugin estimates the same expectation by Monte Carlo sampling at `temperature > 0` — the framework's own repeated-evaluation axis. If the LLM seam grows logprob support, the estimator can switch without changing the API.
- **Ties.** With sampling, an exactly zero margin is possible; `compare` reports it as `tie` (configurable via `tieMargin`). The logit formulation eliminates ties by construction.
- **Ring pairs are reused in the tournament** rather than re-scored — same estimator, fewer calls.
- **Progress tracking judges each prefix blind to later steps** (one call per prefix) rather than batching all checkpoints into one call as the reference does; blind prefixes cannot leak information from the future.

## Development

```sh
git clone https://github.com/uson1x/dsh-plugin-llm-verifier
cd dsh-plugin-llm-verifier
ln -s ~/.dsh/profiles/node_modules node_modules   # resolve @deepseek-ai/* against your dsh install
npm test
```

The tests mock `ctx.llm.stream` with the harness's chunk protocol and `ctx.subagents` with the seam's run contract; no network or credentials needed.

## Known limitations

- **Sampling estimator only** — see above; exact logit expectation awaits logprob support in the dsh LLM seam.
- **Verifier calls are not session-logged** — grading calls run as auxiliary requests outside the conversation; only the tool result enters the log. Rollout children, by contrast, are real persisted sessions.
- **Untrusted candidate text is JSON-framed, not sandboxed** — framing prevents structural prompt breakage, but a sufficiently adversarial candidate can still try to argue with the grader.
- **`verify_rollout` judges final messages only** — the verifier sees each child's final deliverable, not its full tool trace; a child that does great work but summarizes it poorly is judged on the summary.

## License

[MIT](LICENSE)

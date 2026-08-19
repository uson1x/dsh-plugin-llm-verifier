# dsh-plugin-llm-verifier

[![test](https://github.com/uson1x/dsh-plugin-llm-verifier/actions/workflows/test.yml/badge.svg)](https://github.com/uson1x/dsh-plugin-llm-verifier/actions/workflows/test.yml)
[![license: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

**[LLM-as-a-Verifier](https://llm-as-a-verifier.com) for [DeepSeek Harness](https://github.com/deepseek-ai/deepseek-harness)**: continuous reward signals for candidate solutions — no extra training, no separate reward model. Includes `verify_rollout`, a generate-then-judge tool that spawns N independent agent attempts and keeps the verifier's winner.

## Quick start

```sh
cd ~/.dsh/profiles/web          # or any dsh profile
npm install github:uson1x/dsh-plugin-llm-verifier
```

Add to `~/.dsh/profiles/web/cordis.patch.yml` (see [examples/cordis.patch.yml](examples/cordis.patch.yml)):

```yaml
- insert:
    - id: llm-verifier
      name: dsh-plugin-llm-verifier
      config:
        provider: deepseek-official
        model: deepseek-v4-pro
        reasoningEffort: 'off'   # grading works well without reasoning; sampling does the statistics
```

Restart `dsh web`, then just talk to your agent:

> use llm as a verifier to write a landing page tagline
>
> try this 5 times and keep the best: …
>
> here are three drafts — pick the strongest one

A registered system-prompt section routes loose phrasings like these to the right tool; no tool names needed.

## What you get

| Tool | Purpose |
|---|---|
| `verify_rollout(task, n?, rollout_model?)` | **Generate-then-judge**: spawn n independent subagent attempts (default 3, parallel, blind to each other), judge their deliverables, return the winner verbatim |
| `verify_select(task, candidates[])` | Best-of-N over existing candidates via a Probabilistic Pivot Tournament |
| `verify_compare(task, candidate_a, candidate_b)` | Pairwise comparison with a continuous margin, order-debiased |
| `verify_track(task, trajectory[])` | Progress reward per cumulative prefix of a rollout |

Every rollout child is a real, inspectable session: open the parent conversation's **subagent catalog** (header tree icon) to watch them run and read each full trajectory. Children default to the session's current model, or run a cheaper one via `rollout_model` / `rollout.model`. They are denied the `verify_*` tools, so no recursive fan-out.

Other plugins get the same power as a Cordis service:

```js
const { bestIndex, rewards } = await ctx.verifier.select(task, candidates)
const { preferred, margin } = await ctx.verifier.compare(task, a, b)
const { progress, trend }  = await ctx.verifier.track(task, steps)
const { reward }           = await ctx.verifier.score(task, candidate)
```

## How it works

The verifier ([paper](https://arxiv.org/abs/2607.05391), [reference impl](https://github.com/llm-as-a-verifier/llm-as-a-verifier)) turns any model behind dsh's `ctx.llm` seam into a grader, scaling verification along three axes:

1. **Score granularity** — fine-grained integer scales (`1..G`, default 20; anchors: 1 = incorrect, midpoint = borderline, G = flawless).
2. **Repeated evaluation** — `K` independent grading passes (default 4).
3. **Criteria decomposition** — simple sub-criteria (default: *specification*, *output*, *errors*).

The reward for candidate τ on task x:

```
R(x, τ) = (1/CK) · Σ_{c,k} φ(v_{c,k}),   φ(v) = (v − 1) / (G − 1) ∈ [0, 1]
```

Grading prompts follow the reference template: expert-reviewer persona, task and trajectories first, the criterion at the prompt **tail** (one provider-cacheable prefix per pair), scores in XML tags.

**Best-of-N** uses the paper's Probabilistic Pivot Tournament — O(Nk) comparisons instead of O(N²):

1. **Ring pass** — a random Hamiltonian cycle scores the N adjacent pairs; every candidate appears once as A and once as B, canceling positional bias structurally.
2. **Pivot selection** — top-k by mean ring reward (default k = 2).
3. **Pivot tournament** — every remaining pair against a pivot is scored (ring results reused).
4. **Aggregation** — win mass `σ(R_self − R_other)` per scored pair; final score `w_i / c_i`, highest wins.

## Configuration

| Key | Default | Meaning |
|---|---|---|
| `provider` | — (required) | Registered `ctx.llm` provider route the verifier grades with |
| `model` | — (required) | Model id on that route |
| `granularity` | `20` | Integer score scale `1..G` |
| `repetitions` | `4` | Grading passes per criterion (`K`) |
| `temperature` | `1` | Sampling temperature for the Monte Carlo estimate |
| `reasoningEffort` | adapter default | Adapter-owned effort id for grading calls (e.g. `'off'` keeps them fast) |
| `criteria` | specification / output / errors | Array of `{ name, description }` sub-criteria |
| `pivots` | `2` | Pivot count k for the tournament |
| `tieMargin` | `0` | `compare` margin below which the verdict is `tie` |
| `promptSection` | `true` | Register the fuzzy-routing system-prompt section |
| `maxOutputTokens` | `16384` | Output cap per grading call (reasoning tokens count against it) |
| `timeoutMs` | `120000` | Deadline per grading call |
| `concurrency` | `4` | Parallel grading calls |
| `rollout.provider` | `spawn` | `ctx.subagents` provider for rollout children |
| `rollout.model` | session model | Model id override for rollout children |
| `rollout.llmProvider` | session provider | LLM provider route override for rollout children |
| `rollout.maxConcurrent` | `3` | Rollout children running at once |

**Cost:** one pairwise comparison is `C × K` grading calls. `verify_select` scores `N` ring pairs plus at most `k·(N−1)` tournament pairs; `verify_rollout` adds the n child agent runs on top; `verify_track` costs `steps × K` calls. A fully unparseable sample is retried once.

## Paper fidelity and deviations

Followed: the R(x, τ) formula, 1–20 granularity with the paper's anchors, repeated evaluation, criteria decomposition, the pairwise `<score_A>`/`<score_B>` template, PPT with ring-pass debiasing and `σ(R_a − R_b)` win mass, criteria-at-tail prompt layout, prefix-based progress tracking.

Deviations, each deliberate:

- **Sampling, not logits.** The reference reads the full distribution of scoring-token logits (`Σ_g p(v_g)·φ(v_g)`). The dsh `StreamChunk` vocabulary exposes no logprobs, so the same expectation is estimated by Monte Carlo sampling at `temperature > 0` — the framework's own repeated-evaluation axis. If the LLM seam grows logprob support, the estimator can switch without changing the API.
- **Ties.** With sampling, an exactly zero margin is possible; `compare` reports it as `tie` (configurable via `tieMargin`). The logit formulation eliminates ties by construction.
- **Ring pairs are reused in the tournament** rather than re-scored — same estimator, fewer calls.
- **Progress tracking judges each prefix blind to later steps** rather than batching all checkpoints into one call; blind prefixes cannot leak information from the future.
- **`verify_rollout` itself is not in the paper** — it composes the paper's `select` with dsh's subagent seam for the generation half.

## Development

```sh
git clone https://github.com/uson1x/dsh-plugin-llm-verifier
cd dsh-plugin-llm-verifier
npm install     # pulls the @deepseek-ai/* packages from npm
npm test
```

Tests mock `ctx.llm.stream` with the harness's chunk protocol and `ctx.subagents` with the seam's run contract; no network or credentials needed. (Developing against a live dsh install also works: `ln -s ~/.dsh/profiles/node_modules node_modules` instead of `npm install`.)

## Known limitations

- **Sampling estimator only** — see above; exact logit expectation awaits logprob support in the dsh LLM seam.
- **Verifier calls are not session-logged** — grading runs as auxiliary requests; only the tool result enters the log. Rollout children, by contrast, are real persisted sessions.
- **Untrusted candidate text is JSON-framed, not sandboxed** — framing prevents structural prompt breakage, but an adversarial candidate can still try to argue with the grader.
- **`verify_rollout` judges final messages only** — a child that does great work but summarizes it poorly is judged on the summary.
- DeepSeek Harness is in developer preview; breaking changes there may require plugin updates.

## License

[MIT](LICENSE)

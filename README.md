# dsh-plugin-llm-verifier

[LLM-as-a-Verifier](https://llm-as-a-verifier.com) for [DeepSeek Harness](https://github.com/deepseek-ai/deepseek-harness) (`dsh`).

An out-of-tree dsh plugin that turns any model behind the harness's `ctx.llm` seam into a verifier producing **continuous reward signals** for candidate solutions — no extra training, no separate reward model. It exposes the framework's three APIs as a Cordis service (`ctx.verifier`) and as model-facing tools the agent can call mid-conversation.

## What it implements

The framework scales verification along three axes, all configurable here:

1. **Score granularity** — fine-grained integer scales (`1..G`, default `G = 20`) discriminate between solutions better than coarse 1–5 rubrics.
2. **Repeated evaluation** — `K` independent grading passes (default 3) reduce variance.
3. **Criteria decomposition** — a monolithic rubric is split into simple sub-criteria (default: *specification*, *output*, *errors*), reducing prompt bias.

The reward for candidate τ on task x is

```
R(x, τ) = (1/CK) · Σ_{c,k} φ(v_{c,k}),   φ(v) = (v − 1) / (G − 1) ∈ [0, 1]
```

Each grading call asks for an integer rating inside an XML tag (`<score>N</score>`, or `<score_A>`/`<score_B>` for pairwise comparison) and the samples are averaged over the C criteria × K repetitions.

**Sampling, not logits.** The reference formulation reads the full distribution of scoring-token logits, `Σ_g p(v_g)·φ(v_g)`. The dsh `StreamChunk` vocabulary does not (yet) expose token logprobs, so this plugin estimates the same expectation by Monte Carlo: repeated sampling at `temperature > 0` — the framework's own "repeated evaluation" axis. If the LLM seam grows logprob support, the estimator can switch to the exact expectation without changing the API.

## The three APIs

As tools (what the agent sees):

| Tool | Purpose |
|---|---|
| `verify_select(task, candidates[])` | Best-of-N selection: reward per candidate, index of the best |
| `verify_compare(task, candidate_a, candidate_b)` | Pairwise comparison, order-debiased across repetitions, with a continuous margin and `A`/`B`/`tie` verdict |
| `verify_track(task, trajectory[])` | Progress scoring over a rollout: a reward per cumulative prefix plus the trend |

As a service (what other plugins see), registered at `ctx.verifier`:

```js
const { bestIndex, bestReward, scores } = await ctx.verifier.select(task, candidates)
const { preferred, margin } = await ctx.verifier.compare(task, a, b)
const { progress, trend } = await ctx.verifier.track(task, steps)
const { reward, perCriterion } = await ctx.verifier.score(task, candidate)
```

`verify_compare` alternates which candidate is presented as A across repetitions and un-swaps the parsed scores, so pure position bias cancels exactly.

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
```

A running `dsh web` hot-reloads the patch layer; otherwise restart. The plugin's `@deepseek-ai/*` imports resolve through the harness's own profile-level module tree (they are declared as optional peer dependencies precisely so npm does not fetch duplicate copies).

## Configuration

| Key | Default | Meaning |
|---|---|---|
| `provider` | — (required) | Registered `ctx.llm` provider route to grade with |
| `model` | — (required) | Model id on that route |
| `granularity` | `20` | Integer score scale `1..G` |
| `repetitions` | `3` | Grading passes per criterion (`K`) |
| `temperature` | `1` | Sampling temperature for the Monte Carlo estimate |
| `criteria` | specification / output / errors | Array of `{ name, description }` sub-criteria |
| `maxOutputTokens` | `2048` | Output cap per grading call |
| `timeoutMs` | `120000` | Deadline per grading call |
| `concurrency` | `4` | Parallel grading calls |
| `tieMargin` | `0.02` | `compare` margin below which the verdict is `tie` |

Cost note: `verify_select` makes `N × C × K` model calls (default 9 per candidate); `verify_track` makes `steps × K`.

## Development

```sh
git clone https://github.com/uson1x/dsh-plugin-llm-verifier
cd dsh-plugin-llm-verifier
ln -s ~/.dsh/profiles/node_modules node_modules   # resolve @deepseek-ai/* against your dsh install
npm test
```

The tests mock `ctx.llm.stream` with the harness's chunk protocol; no network or credentials needed.

## Known limitations

- **Sampling estimator only** — see above; exact logit expectation awaits logprob support in the dsh LLM seam.
- **Verifier calls are not session-logged** — grading calls run as auxiliary requests outside the conversation; only the tool result enters the log.
- **Untrusted candidate text is JSON-framed, not sandboxed** — framing prevents structural prompt breakage, but a sufficiently adversarial candidate can still try to argue with the grader.

## License

[MIT](LICENSE)

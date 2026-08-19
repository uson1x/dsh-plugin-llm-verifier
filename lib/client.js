/**
 * Browser half of dsh-plugin-llm-verifier:
 *  - a rich chat card for the `verify_rollout` tool (keyed `tool.call.toolview` slot)
 *  - a per-session "Verifier" tab next to Chat/Trajectory
 *    (`conversation.view` slot ring).
 *
 * Hand-authored in the client module loader's lazy-CJS envelope (no build
 * step): executing this script only registers the factory; the body runs at
 * materialization. React arrives through the loader's `require`.
 */
window.__ModuleLoader__.load({
  id: 'dsh-plugin-llm-verifier',
  factory: (require) => {
    var module = { exports: {} }
    var exports = module.exports
    const { createElement: h, useState } = require('react')

    /** Theme-safe inline styles: neutrals derived from currentColor only. */
    const styles = {
      root: { display: 'flex', flexDirection: 'column', gap: 6, padding: '6px 8px', borderRadius: 8, border: '1px solid color-mix(in srgb, currentColor 15%, transparent)', fontSize: 12, lineHeight: 1.45 },
      header: { display: 'flex', alignItems: 'center', gap: 6, fontWeight: 600 },
      dim: { opacity: 0.65, fontWeight: 400 },
      row: { display: 'flex', alignItems: 'center', gap: 8 },
      barTrack: { flex: '1 1 auto', height: 6, borderRadius: 3, background: 'color-mix(in srgb, currentColor 10%, transparent)', overflow: 'hidden', minWidth: 60 },
      bar: { height: '100%', borderRadius: 3, background: 'color-mix(in srgb, currentColor 55%, transparent)' },
      barWin: { height: '100%', borderRadius: 3, background: 'var(--color-accent, #4f8ef7)' },
      score: { fontVariantNumeric: 'tabular-nums', width: 44, textAlign: 'right' },
      label: { width: 76, whiteSpace: 'nowrap' },
      failed: { opacity: 0.55, fontStyle: 'italic' },
      toggle: { cursor: 'pointer', background: 'none', border: 'none', color: 'inherit', opacity: 0.7, padding: 0, font: 'inherit', textAlign: 'left' },
      winner: { whiteSpace: 'pre-wrap', padding: '6px 8px', borderRadius: 6, background: 'color-mix(in srgb, currentColor 6%, transparent)', maxHeight: 240, overflow: 'auto' },
      error: { color: 'var(--color-danger, #d5504e)' },
      // Verifier tab
      tab: { height: '100%', overflow: 'auto', padding: '16px 24px 140px', fontSize: 13, lineHeight: 1.5 },
      tabInner: { maxWidth: 720, margin: '0 auto', display: 'flex', flexDirection: 'column', gap: 16 },
      runCard: { display: 'flex', flexDirection: 'column', gap: 8, padding: '10px 12px', borderRadius: 10, border: '1px solid color-mix(in srgb, currentColor 15%, transparent)' },
      runHead: { display: 'flex', alignItems: 'baseline', gap: 8, fontWeight: 600 },
      task: { opacity: 0.8, whiteSpace: 'pre-wrap' },
      openBtn: { cursor: 'pointer', background: 'none', border: '1px solid color-mix(in srgb, currentColor 25%, transparent)', borderRadius: 5, color: 'inherit', opacity: 0.75, padding: '0 6px', font: 'inherit', fontSize: 11 },
      statLine: { opacity: 0.55, fontSize: 11, margin: '-2px 0 2px 84px' },
      attemptBlock: { display: 'flex', flexDirection: 'column', gap: 2 },
      judgePanel: { display: 'flex', flexDirection: 'column', gap: 4, padding: '8px 10px', borderRadius: 8, background: 'color-mix(in srgb, currentColor 5%, transparent)', fontSize: 12 },
      pairRow: { display: 'flex', alignItems: 'center', gap: 8, fontVariantNumeric: 'tabular-nums' },
    }

    function parseArgs(argsRaw) {
      try {
        const parsed = JSON.parse(argsRaw || '{}')
        return parsed !== null && typeof parsed === 'object' ? parsed : {}
      } catch { return {} }
    }

    function flatText(content) {
      const parts = []
      for (const item of content || []) {
        if (item && item.type === 'text' && typeof item.text === 'string') parts.push(item.text)
      }
      return parts.join('\n')
    }

    function round(value) { return Math.round(value * 1000) / 1000 }

    /** One rollout line: label, reward bar, score or failure reason, optional open action. */
    function rolloutRow(index, meta, openRollout) {
      const reward = meta.rewards[index]
      const isWinner = index === meta.best_index
      const stop = (meta.stop_reasons || []).find(r => r && r.rollout === index + 1)
      const label = `rollout ${index + 1}${isWinner ? ' ★' : ''}`
      const childId = openRollout && Array.isArray(meta.sessions) ? meta.sessions[index] : undefined
      const openBtn = childId
        ? h('button', { style: styles.openBtn, onClick: () => openRollout(childId), title: 'open this attempt’s session' }, 'open')
        : null
      if (reward === null || reward === undefined) {
        return h('div', { key: index, style: { ...styles.row, ...styles.failed } },
          h('span', { style: styles.label }, label),
          h('span', { style: { flex: '1 1 auto' } }, `failed${stop && stop.stop_reason ? ` (${String(stop.stop_reason).slice(0, 80)})` : ''}`),
          openBtn)
      }
      return h('div', { key: index, style: styles.row },
        h('span', { style: { ...styles.label, fontWeight: isWinner ? 600 : 400 } }, label),
        h('div', { style: styles.barTrack },
          h('div', { style: { ...(isWinner ? styles.barWin : styles.bar), width: `${Math.max(3, Math.round(reward * 100))}%` } })),
        h('span', { style: styles.score }, String(round(reward))),
        openBtn)
    }

    /** The verify_rollout chat card. */
    function VerifyRolloutRow(props) {
      const [showWinner, setShowWinner] = useState(false)
      const block = props.block
      const settled = 'kind' in block
      const args = parseArgs(settled ? (block.call ? block.call.argsRaw : '') : block.argsRaw)
      const n = typeof args.n === 'number' ? args.n : 3
      const task = typeof args.task === 'string' ? args.task : ''
      const header = (state) => h('div', { style: styles.header },
        h('span', null, '⚖ best-of-N rollouts'),
        h('span', { style: styles.dim }, state),
        props.inspect ? h('button', { style: { ...styles.toggle, marginLeft: 'auto' }, onClick: props.inspect }, 'inspect') : null)

      if (!settled) {
        return h('div', { style: styles.root },
          header(`running ${n} attempts…`),
          task ? h('div', { style: styles.dim }, task.length > 160 ? `${task.slice(0, 160)}…` : task) : null,
          h('div', { style: styles.dim }, 'attempts appear in the session header’s subagent list while they run'))
      }

      if (block.isError) {
        const text = flatText(block.content) || (block.error ? `${block.error.name}: ${block.error.code}` : 'failed')
        return h('div', { style: styles.root },
          header('failed'),
          h('div', { style: styles.error }, text.split('\n')[0]))
      }

      const meta = block.meta
      if (!meta || !Array.isArray(meta.rewards)) {
        // Older events without presentationMeta: show the durable result text.
        return h('div', { style: styles.root }, header('done'),
          h('div', { style: { whiteSpace: 'pre-wrap' } }, flatText(block.content)))
      }

      return h('div', { style: styles.root },
        header(`${meta.rewards.length} attempts · judge saw ${meta.judge_trace === 'final' ? 'final messages' : 'full trajectories'}`),
        meta.rewards.map((_, index) => rolloutRow(index, meta)),
        h('button', { style: styles.toggle, onClick: () => setShowWinner(v => !v) },
          showWinner ? '▾ winning deliverable' : '▸ winning deliverable'),
        showWinner ? h('div', { style: styles.winner }, meta.winner_preview || '') : null,
        h('div', { style: styles.dim }, 'full attempt trajectories: subagent list in the session header'))
    }

    function fmtDuration(ms) {
      if (typeof ms !== 'number' || !(ms >= 0)) return null
      return ms < 60000 ? `${Math.round(ms / 100) / 10}s` : `${Math.floor(ms / 60000)}m${Math.round((ms % 60000) / 1000)}s`
    }

    function fmtChars(chars) {
      if (typeof chars !== 'number') return null
      return chars < 1000 ? `${chars} chars` : `${Math.round(chars / 100) / 10}k chars`
    }

    /** One attempt inside a run card: reward row, stats line, expandable deliverable. */
    function AttemptBlock(props) {
      const [showText, setShowText] = useState(false)
      const { attempt, meta, index } = props
      const stats = [
        attempt.ok ? null : `failed: ${attempt.stop_reason ?? 'unknown'}`,
        fmtDuration(attempt.duration_ms),
        typeof attempt.tool_calls === 'number' ? `${attempt.tool_calls} tool calls` : null,
        typeof attempt.assistant_turns === 'number' ? `${attempt.assistant_turns} assistant turns` : null,
        fmtChars(attempt.judge_chars) === null ? null : `judge read ${fmtChars(attempt.judge_chars)}`,
      ].filter(Boolean).join(' · ')
      return h('div', { style: styles.attemptBlock },
        rolloutRow(index, meta, props.openRollout),
        stats ? h('div', { style: styles.statLine }, stats) : null,
        attempt.ok && attempt.preview
          ? h('button', { style: { ...styles.toggle, marginLeft: 84, fontSize: 11 }, onClick: () => setShowText(v => !v) },
            showText ? '▾ deliverable' : '▸ deliverable')
          : null,
        showText ? h('div', { style: { ...styles.winner, marginLeft: 84, maxHeight: 400 } }, attempt.preview) : null)
    }

    /** How the judge decided: config, pivots, and pairwise rewards. */
    function JudgePanel(props) {
      const judge = props.judge
      const name = idx => `rollout ${idx + 1}`
      return h('div', { style: styles.judgePanel },
        h('div', null,
          `Pairwise judging: ${(judge.criteria || []).join(', ') || 'default criteria'} × ${judge.repetitions} repetitions each, scored 1–${judge.granularity}, averaged into [0, 1].`),
        Array.isArray(judge.pivots) && judge.pivots.length > 0
          ? h('div', null, `Tournament pivots (ring-pass leaders): ${judge.pivots.map(name).join(', ')}.`)
          : null,
        (judge.pairs || []).map((pair, i) => h('div', { key: i, style: styles.pairRow },
          h('span', { style: { width: 170 } }, `${name(pair.a)} vs ${name(pair.b)}`),
          h('span', { style: pair.reward_a >= pair.reward_b ? { fontWeight: 600 } : null }, String(round(pair.reward_a))),
          h('span', { style: { opacity: 0.5 } }, ':'),
          h('span', { style: pair.reward_b > pair.reward_a ? { fontWeight: 600 } : null }, String(round(pair.reward_b))))),
        h('div', { style: { opacity: 0.55, fontSize: 11 } },
          'final score = mean over an attempt\u2019s pairings of \u03c3(reward difference); the ring pass orders attempts, pivots judge everyone'))
    }

    /** One settled verify_rollout run inside the Verifier tab. */
    function VerifierRunCard(props) {
      const [showJudge, setShowJudge] = useState(false)
      const node = props.node
      const args = parseArgs(node.call ? node.call.argsRaw : '')
      const task = typeof args.task === 'string' ? args.task : ''
      const time = node.time ? new Date(node.time).toLocaleTimeString() : ''
      if (node.isError) {
        const text = flatText(node.content) || (node.error ? `${node.error.name}: ${node.error.code}` : 'failed')
        return h('div', { style: styles.runCard },
          h('div', { style: styles.runHead }, h('span', null, '⚖ run failed'), h('span', { style: styles.dim }, time)),
          task ? h('div', { style: styles.task }, task) : null,
          h('div', { style: styles.error }, text.split('\n')[0]))
      }
      const meta = node.meta
      if (!meta || !Array.isArray(meta.rewards)) {
        return h('div', { style: styles.runCard },
          h('div', { style: styles.runHead }, h('span', null, '⚖ run (no scoreboard data)'), h('span', { style: styles.dim }, time)),
          task ? h('div', { style: styles.task }, task) : null,
          h('div', { style: { whiteSpace: 'pre-wrap', opacity: 0.8 } }, flatText(node.content)))
      }
      const attempts = Array.isArray(meta.attempts) ? meta.attempts : null
      const totalMs = attempts ? attempts.reduce((sum, a) => sum + (a.duration_ms || 0), 0) : null
      return h('div', { style: styles.runCard },
        h('div', { style: styles.runHead },
          h('span', null, `⚖ ${meta.rewards.length} attempts`),
          h('span', { style: styles.dim }, `judge saw ${meta.judge_trace === 'final' ? 'final messages' : 'full trajectories'}`),
          totalMs ? h('span', { style: styles.dim }, `${fmtDuration(totalMs)} of attempt time`) : null,
          h('span', { style: { ...styles.dim, marginLeft: 'auto' } }, time)),
        task ? h('div', { style: styles.task }, task.length > 300 ? `${task.slice(0, 300)}…` : task) : null,
        attempts
          ? attempts.map((attempt, index) => h(AttemptBlock, { key: index, attempt, meta, index, openRollout: props.openRollout }))
          : meta.rewards.map((_, index) => rolloutRow(index, meta, props.openRollout)),
        meta.judge
          ? h('button', { style: styles.toggle, onClick: () => setShowJudge(v => !v) },
            showJudge ? '▾ how the judge decided' : '▸ how the judge decided')
          : null,
        showJudge && meta.judge ? h(JudgePanel, { judge: meta.judge }) : null)
    }

    /** The per-session Verifier tab: every verify_rollout run, newest last. */
    function VerifierTab(props) {
      const nodes = props.useSession(s => s.nodes)
      const runningCalls = props.useSession(s => s.runningCalls)
      const runs = (nodes || []).filter(n => n.kind === 'tool-result' && n.call && n.call.name === 'verify_rollout')
      const running = (runningCalls || []).filter(c => c.name === 'verify_rollout')

      const body = runs.length === 0 && running.length === 0
        ? h('div', { style: { opacity: 0.7 } },
          h('p', null, 'No verifier runs in this session yet.'),
          h('p', null, 'Ask the agent to "use the verifier" on a task (or call verify_rollout directly) and every best-of-N run will show up here as a scoreboard.'))
        : [
          runs.map(node => h(VerifierRunCard, { key: node.callId, node, openRollout: props.openRollout })),
          running.map(call => {
            const args = parseArgs(call.argsRaw)
            const n = typeof args.n === 'number' ? args.n : 3
            return h('div', { key: call.callId, style: styles.runCard },
              h('div', { style: styles.runHead }, h('span', null, '⚖ running'), h('span', { style: styles.dim }, `${n} attempts in flight…`)),
              typeof args.task === 'string' ? h('div', { style: styles.task }, args.task) : null)
          }),
        ]

      return h('div', { style: styles.tab },
        h('div', { style: styles.tabInner }, body))
    }

    /** Client plugin body: claim the verify_rollout tool row and add the Verifier tab. */
    function apply(ctx) {
      ctx.slots.inject('tool.call.toolview', () => ctx.slots.register(
        { name: 'tool.call.toolview', key: 'verify_rollout' },
        VerifyRolloutRow,
      ))

      ctx.slots.inject('conversation.view', () => ctx.slots.register({
        name: 'conversation.view',
        id: 'verifier',
        order: 20,
        label: 'Verifier',
        inject: (sessionId) => ({
          openRollout: (childId) => {
            try {
              const resolved = typeof ctx.sessions.subagentAddress === 'function'
                ? ctx.sessions.subagentAddress(childId)
                : undefined
              ctx.sessions.openSubagent(resolved || { parentSessionId: sessionId, childSessionId: childId, mode: 'one-shot' })
            } catch (error) {
              console.warn('dsh-plugin-llm-verifier: could not open rollout session', error)
            }
          },
        }),
      }, VerifierTab))
    }

    exports.inject = ['slots', 'sessions']
    exports.apply = apply
    return module.exports
  },
})

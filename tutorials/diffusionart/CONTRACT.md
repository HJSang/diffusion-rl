# Fragment contract for the DiffusionART tutorial

The tutorial is assembled by `tutorial/build.py` from HTML fragments in
`tutorial/fragments/`. Each step is one fragment file. The build wraps the
fragments in a shared page shell (CSS, fonts, MathJax, highlight.js, nav), so a
fragment must contain **only the section markup below**: no `<html>`, `<head>`,
`<body>`, `<style>`, `<link>`, or external `<script src>`.

## Reader

The reader is an ML systems engineer who works on LLM RL training frameworks
(GRPO/PPO in a Megatron + SGLang stack called "miles"). They know PPO, GRPO,
importance ratios, KL, advantages, logprobs, distributed training and PyTorch
very well. They are **new to diffusion models and to diffusion RL**. Build every
new idea from what they know, and say explicitly when an LLM-RL intuition carries
over and when it breaks.

Tone: a clear, rigorous textbook chapter. Plain direct sentences, active voice.
No filler, no emoji, no hype. Define each symbol when it first appears.

## Shared notation (use exactly this, all steps)

Follow the DiffusionART paper's conventions:

- Time runs from $t=1$ (pure noise) to $t=0$ (clean sample). Generation goes 1 → 0.
- Forward (noising) state: $x_t = \alpha_t x_0 + \sigma_t \epsilon$, $\epsilon\sim\mathcal N(0,I)$.
  Rectified flow: $\alpha_t = 1-t$, $\sigma_t = t$.
- Velocity points along the **denoising** direction:
  $u_t = -\dot\alpha_t x_0 - \dot\sigma_t \epsilon = x_0 - \epsilon$ for rectified flow.
  Model velocity: $v_\theta(x_t, t, c)$; $c$ is the prompt (condition).
- ODE Euler step from $t$ to $t-\Delta$: $x_{t-\Delta} = x_t + \Delta\, v_\theta(x_t,t,c)$.
- Clean sample $x_0$ is called an **endpoint**. $K$ samples per prompt. Raw reward $R_i = R(x_0^{(i)}, c)$.
- Rollout (old) policy params $\theta_{\text{old}}$, $v_{\text{old}} := v_{\theta_{\text{old}}}$.
- When code (diffusers / SD3 / the repo) uses the opposite sign convention
  (for example SD3 predicts $\epsilon - x_0$ and integrates with `x + (t_next - t) * v`),
  **say so explicitly** next to the code and show how it maps to the notation above.

## Allowed markup and classes

Root element: `<section class="step" id="step-N" data-title="Short title">`.

Inside:

```html
<header class="step-head">
  <p class="eyebrow">Step N · about X hours</p>
  <h2>Title</h2>
  <p class="lede">One or two sentences: what you will be able to do after this step.</p>
</header>

<h3 id="sN-slug">Subsection</h3>      <!-- every h3 needs an id starting with sN- -->
<h4>Sub-subsection</h4>
<p>, <ul>, <ol>, <strong>, <em>, <code>, <a href="https://..." >

<div class="callout insight"><p class="callout-title">Key idea</p><p>…</p></div>
<div class="callout llm"><p class="callout-title">From LLM RL</p><p>…</p></div>   <!-- bridges to what the reader knows -->
<div class="callout warn"><p class="callout-title">Pitfall</p><p>…</p></div>

<div class="math-block">\[ ... \]</div>              <!-- display math; inline math uses $...$ -->

<figure class="code">
  <figcaption><span class="tag">toy · runs on CPU</span> what this code shows</figcaption>
  <pre><code class="language-python">...escaped code...</code></pre>
</figure>
<!-- For code quoted from the repo, the tag is the repo path with line range, e.g.
     <span class="tag">src/visual_rl/algorithms/art.py · L40–88</span> -->

<figure class="diagram">
  <svg viewBox="..." role="img" aria-label="...">...</svg>
  <figcaption>…</figcaption>
</figure>

<div class="table-wrap"><table>…</table></div>

<figure class="widget" id="wN-name">            <!-- optional interactive demo -->
  <canvas ...></canvas> <div class="widget-controls">…inputs/buttons with unique ids…</div>
  <figcaption>…</figcaption>
</figure>
<script>(() => { /* self-contained, IIFE, no globals, only touch elements inside #wN-name */ })();</script>

<details class="exercise"><summary>Exercise N.k — prompt</summary><div class="solution">…</div></details>

<div class="takeaways"><p class="callout-title">Takeaways</p><ul>…</ul></div>
<div class="reading"><p class="callout-title">Further reading</p><ul>…</ul></div>
```

Rules:

- **Math** is rendered by MathJax 3 (TeX). Inline `$...$`, display `\[...\]` inside
  `div.math-block`. Inside math, escape `<` as `&lt;` and `>` as `&gt;` and `&` as `&amp;`.
  Never use a literal `$` in prose for anything but math. Use `\mathbb E`, `\operatorname{Cov}`,
  `\lVert \rVert`, `\tfrac`, `\top`; `\boldsymbol` is fine. Define macros inline, no custom preamble.
- **Code** inside `<pre><code>` must be HTML-escaped (`&lt;`, `&gt;`, `&amp;`). Code is highlighted
  by highlight.js. Keep blocks focused (≈10–60 lines). Toy code must be complete and runnable;
  you must actually run it (see below) and you may paste its real output in a second
  `<pre><code class="language-text">` block.
- **Colors**: never hard-code colors. In SVG and widgets use CSS variables:
  `var(--fg)`, `var(--muted)`, `var(--line)`, `var(--surface)`, `var(--bg)`,
  `var(--accent)` (teal, used for "insight / reward-improving direction"),
  `var(--accent-2)` (vermilion, used for "the method / DiffusionART"), `var(--good)`, `var(--bad)`.
  SVG: `style="stroke:var(--fg);fill:none"` works. Canvas: read them at draw time with
  `getComputedStyle(document.documentElement).getPropertyValue('--accent').trim()`
  and redraw on `matchMedia('(prefers-color-scheme: dark)')` change.
  SVG text: `style="fill:var(--fg)"`, font-size 12–14 in viewBox units; give the viewBox room for labels.
- Every `id` you create must start with `sN-` or `wN-` (N = your step number) so steps never collide.
- Widgets must render a meaningful first frame without interaction, work at 360px width
  (canvas: set width from its container, use devicePixelRatio), and respect `prefers-reduced-motion`.
- No external images, no iframes, no external scripts.
- Cite the paper as "the paper (§3.1)" etc. and the repo with paths like `src/visual_rl/algorithms/art.py`.
- Each step must include: motivation (why this exists / what problem), intuition, how it works
  mechanically, the math the reader needs (with short derivations, not just results),
  at least one main code block, at least one worked example with numbers, pitfalls,
  3–5 exercises with full solutions, takeaways, further reading.
- Aim for a substantial chapter: roughly 4,000–7,000 words of prose plus code and figures.

## Sources on disk

- Paper LaTeX: `/Users/hejian.sang/Downloads/DiffusionART-paper/paper/src/main.tex` (read this, it is exact)
- Paper PDF: `/Users/hejian.sang/Downloads/DiffusionART-paper/paper/2610.12572.pdf`
- References: `/Users/hejian.sang/Downloads/DiffusionART-paper/paper/src/references.bib`
- Repo: `/Users/hejian.sang/Downloads/DiffusionART-paper/DiffusionART/` (src/visual_rl/..., docs/, configs/)
- Python with torch 2.x + numpy for running toy examples:
  `/private/tmp/claude-502/-Users-hejian-sang-Library-Application-Support-Claude-scratch-workspaces-0b7f7b84-7830-459b-b532-20fc8f8b5287-856d7364-c912-4f0d-b6ca-bd1e84d8bdf0-scratch-2026-10-16-a789fc/68b9a5e8-5eea-4954-8127-e346e63ef040/scratchpad/venv/bin/python`
  Put your scratch scripts in `/Users/hejian.sang/Downloads/DiffusionART-paper/tutorial/scratch/stepN/`, run them
  with `-I`, and do not modify anything under `DiffusionART/` or `paper/`.

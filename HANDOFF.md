# Handoff Notes

Context for picking this project back up in a new session. Read this first,
then `README.md` for the project pitch itself.

## Status

Hackathon submission already went in: the GitHub repo + a text description
were submitted, but **no demo video** was recorded before the deadline. The
`README.md` was added *after* submission — if the platform snapshotted the
repo at submit time rather than linking live, judges may be looking at a
version without it.

Repo: https://github.com/anuvaagarwal/hackathon-eval-optimizer

## What this project is

A demonstration that raw eval score deltas (as shown in a tool like
Braintrust) can be misleading — a small positive score bump might just be
sampling noise. Paired bootstrap significance testing tells the two apart.

## Environment setup

- Python venv at `venv/`. `venv/bin/activate` was hand-edited to auto-`source`
  `.env` on activation (`set -a; . "$VIRTUAL_ENV/../.env"; set +a`), so
  `source venv/bin/activate` alone loads the API keys into the shell.
- `.env` (gitignored, not in repo) holds:
  - `FIREWORKS_API_KEY` — Fireworks AI (was originally misnamed
    `FIREWORKSAI_API_KEY`, renamed by the user)
  - `BRAINTRUST_API_KEY` — Braintrust
- Installed in venv: `numpy`, `openai`, `braintrust`, `autoevals`.

## Files

| File | Purpose |
|---|---|
| `stats.py` | `paired_bootstrap(baseline, candidate, n_boot=10000)` — the core stats function, plus a `__main__` demo on synthetic REAL vs FAKE-improvement data |
| `seed_experiment.py` | Creates 3 Braintrust experiments (`baseline`, `candidate`, `candidate_marginal`) in project `eval-reliability-demo`, using Fireworks for both the task LLM and the Factuality judge |
| `braintrust_fetch.py` | Pulls scores back out of Braintrust by experiment name, pairs them by question, runs `paired_bootstrap` |
| `README.md` | Public-facing project doc: what's here, results table, how to run, limitations |
| `demo_script.md` | 3-minute demo narration script (timed sections) — **not yet committed to git** |
| `demo_cue_cards.html` | Phone-readable version of the demo script, published as a Claude Artifact for reading during a live demo — **not yet committed to git** |

## Braintrust experiment names (exact, with suffixes)

Braintrust auto-appends a random suffix when an experiment name collides
with a prior run. These are the ones with valid, final results — there are
earlier attempts in the project (`baseline`, `candidate`,
`candidate_marginal-e4b66f92`, etc.) that were either broken (first run hit
a 404 on a nonexistent Fireworks model) or superseded tuning attempts.

- `baseline-46b7efa0` — mean Factuality 62.0%
- `candidate-967e5ba2` — mean Factuality 78.0%
- `candidate_marginal-77a4caf1` — mean Factuality 64.0%

Results:

| Comparison | Δ | 95% CI | Significant? |
|---|---|---|---|
| baseline vs candidate | +16.0% | `[+6.0%, +26.0%]` | True |
| baseline vs candidate_marginal | +2.0% | `[0.0%, +6.0%]` | False |

## Non-obvious gotchas discovered along the way

- **Fireworks model IDs from docs may not be deployed on your account.**
  `accounts/fireworks/models/llama-v3p1-8b-instruct` (the docs' suggested
  model) 404'd. Had to call `client.models.list()` to find one actually
  available: `accounts/fireworks/models/gpt-oss-120b`.
- **`autoevals.Factuality` defaults to grading via OpenAI through
  Braintrust's AI gateway**, which requires an OpenAI provider configured
  in Braintrust org settings (Settings → AI Providers) — not present here.
  Fixed by pointing the judge at Fireworks instead via
  `autoevals.init(client=AsyncOpenAI(...), default_model=FIREWORKS_MODEL)`.
- **That judge client must be `AsyncOpenAI`, not sync `OpenAI`.** Braintrust
  always invokes scorers through autoevals' async code path regardless of
  whether your `task` function is sync — passing a sync client causes
  `TypeError: object ChatCompletion can't be used in 'await' expression`.
- **`experiment.fetch()` returns sub-spans, not just root eval rows.** A
  20-question eval produces 80 fetched rows (task span + scorer's internal
  LLM-judge call span, etc.). Only rows where `row["scores"]` is a non-empty
  dict actually carry the final score; the real question text lives at
  `row["input"]["input"]` on those rows (nested, because that row IS the
  scorer's own span whose input is the `{input, output, expected}` bundle).
- **Braintrust's UI defaults the comparison/diff target to the most recent
  prior experiment in the project**, not necessarily the one you want. When
  screenshotting for demo purposes, manually change the "Comparisons"
  dropdown in the left sidebar to the intended baseline.

## Known limitation (already documented in README)

Each experiment variant ran the LLM **once per question** — no repeated
sampling. The bootstrap CI captures test-case sampling variance, not LLM
generation variance. Empirically, near-identical `candidate_marginal`
prompt wordings produced means ranging 57%–64% across separate runs, noise
the bootstrap never sees since it resamples a fixed set of 20 scores rather
than re-querying the model.

## Possible next steps

- Commit `demo_script.md` and `demo_cue_cards.html` to the repo if useful
  for late submission / reference.
- Check whether the hackathon platform allows a resubmission or late edit
  window, since the README landed after the initial submit.
- If continuing the technical work: rerun each variant multiple times per
  question and extend `paired_bootstrap` (or a new function) to account for
  both sampling-of-test-cases noise and generation noise.

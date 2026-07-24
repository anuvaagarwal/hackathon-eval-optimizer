# Eval Reliability Checker

Raw eval score deltas can lie to you. This project pairs Braintrust experiments
with a paired bootstrap significance test to tell the difference between a
real improvement and one that's statistically indistinguishable from noise.

## What's here

- **`seed_experiment.py`** — runs three Braintrust experiments in the
  `eval-reliability-demo` project against the same 20-question factual Q&A
  dataset, using Fireworks (`gpt-oss-120b`) as the LLM and `autoevals`'
  `Factuality` scorer:
  - `baseline` — a weak prompt
  - `candidate` — a clearly improved prompt
  - `candidate_marginal` — a prompt with only a trivial tweak, meant to
    produce a small, noisy improvement
- **`stats.py`** — `paired_bootstrap(baseline, candidate, n_boot=10000)`:
  computes the observed mean delta between paired scores, resamples the
  per-case deltas with replacement `n_boot` times, and returns a 95%
  confidence interval plus whether it excludes zero.
- **`braintrust_fetch.py`** — pulls per-question scores back out of Braintrust
  via its SDK, pairs them by question text across two experiments, and runs
  `paired_bootstrap` on the result.

## Results

| Comparison | Baseline mean | Other mean | Δ | 95% CI | Significant? |
|---|---|---|---|---|---|
| baseline vs candidate | 62.0% | 78.0% | +16.0% | `[+6.0%, +26.0%]` | **True** |
| baseline vs candidate_marginal | 62.0% | 64.0% | +2.0% | `[0.0%, +6.0%]` | **False** |

Braintrust's UI shows both deltas as positive — it has no way to tell you
whether +2% on 20 examples is signal or sampling noise. The bootstrap does.

## Running it

```
source venv/bin/activate
pip install -r <(echo "numpy openai braintrust autoevals")
export FIREWORKS_API_KEY=...
export BRAINTRUST_API_KEY=...
python seed_experiment.py      # creates the three experiments in Braintrust
python braintrust_fetch.py     # fetches scores and prints both verdicts
python stats.py                # standalone demo on synthetic data
```

## Limitations

Each prompt variant was run **once** — 20 questions, one generation per
question, no repeated draws. The bootstrap confidence interval measures
whether the observed delta is robust to *which test cases* were sampled;
it does not capture LLM generation noise (re-running the same prompt against
the same questions could itself produce a different score). While tuning
`candidate_marginal`, near-identical prompt wordings produced means ranging
from 57% to 64% across separate runs — variance the bootstrap CI never sees,
since it resamples from a single fixed set of 20 scores rather than
re-querying the model.

A more rigorous version would generate multiple samples per question per
variant, so the significance test absorbs both sampling-of-test-cases noise
and generation noise, not just the former.

import numpy as np


def paired_bootstrap(baseline, candidate, n_boot=10000):
    baseline = np.asarray(baseline)
    candidate = np.asarray(candidate)
    d = candidate - baseline
    observed = d.mean()

    n = len(d)
    boot_means = np.empty(n_boot)
    for i in range(n_boot):
        sample = np.random.choice(d, size=n, replace=True)
        boot_means[i] = sample.mean()

    ci_low, ci_high = np.percentile(boot_means, [2.5, 97.5])
    significant = ci_low > 0 or ci_high < 0

    return {
        "observed_delta": observed,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "significant": significant,
    }


if __name__ == "__main__":
    rng = np.random.default_rng(42)

    n = 200
    baseline = rng.normal(loc=0.70, scale=0.15, size=n)
    candidate = baseline + rng.normal(loc=0.08, scale=0.05, size=n)
    result = paired_bootstrap(baseline, candidate)
    print("REAL improvement:", result)

    n_small = 20
    baseline_small = rng.normal(loc=0.70, scale=0.15, size=n_small)
    candidate_small = rng.normal(loc=0.705, scale=0.15, size=n_small)
    result_small = paired_bootstrap(baseline_small, candidate_small)
    print("FAKE improvement (noise):", result_small)

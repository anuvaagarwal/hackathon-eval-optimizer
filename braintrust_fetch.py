import numpy as np
from braintrust import init

import stats

BASELINE_EXPERIMENT = "baseline-46b7efa0"
CANDIDATE_EXPERIMENT = "candidate-967e5ba2"
CANDIDATE_MARGINAL_EXPERIMENT = "candidate_marginal-77a4caf1"


def get_scores(experiment_name, project_name="eval-reliability-demo", score_name="Factuality"):
    experiment = init(project=project_name, experiment=experiment_name, open=True)

    scores = {}
    for row in experiment.fetch():
        row_scores = row.get("scores")
        if not row_scores or score_name not in row_scores:
            continue
        question = row["input"]["input"]
        scores[str(question)] = row_scores[score_name]

    return scores


def get_paired_scores(baseline_name, candidate_name):
    baseline_scores = get_scores(baseline_name)
    candidate_scores = get_scores(candidate_name)

    shared_keys = sorted(set(baseline_scores) & set(candidate_scores))

    baseline = np.array([baseline_scores[k] for k in shared_keys])
    candidate = np.array([candidate_scores[k] for k in shared_keys])

    return baseline, candidate


if __name__ == "__main__":
    baseline, candidate = get_paired_scores(BASELINE_EXPERIMENT, CANDIDATE_EXPERIMENT)
    print(f"Matched {len(baseline)} paired test cases.")
    result = stats.paired_bootstrap(baseline, candidate)
    print("baseline vs candidate:", result)

    baseline, candidate_marginal = get_paired_scores(BASELINE_EXPERIMENT, CANDIDATE_MARGINAL_EXPERIMENT)
    print(f"Matched {len(baseline)} paired test cases.")
    result_marginal = stats.paired_bootstrap(baseline, candidate_marginal)
    print("baseline vs candidate_marginal:", result_marginal)

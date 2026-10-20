import numpy as np
import pandas as pd

from hijack.analyze import calibrate, cluster_rate, holm, interaction, paired_contrast


def synthetic(n_tasks=60, per_task=8, seed=0):
    """Independent defenses: ASR A=30%, each layer lets 20% through, so F = 1.2%."""
    rng = np.random.default_rng(seed)
    rows = []
    for arm, p in {"A": 0.30, "B": 0.06, "E": 0.06, "F": 0.012}.items():
        for t in range(n_tasks):
            for k in range(per_task):
                rows.append({"arm": arm, "case_id": f"s/{t}/{k}", "cluster": f"s/{t}", "attacked": True,
                             "delivered": True, "attack": rng.random() < p, "utility": rng.random() < 0.6})
    return pd.DataFrame(rows)


def test_cluster_rate_counts_and_ci():
    df = pd.DataFrame({"cluster": ["a", "a", "b", "b"], "x": [1, 0, 1, 1]})
    r = cluster_rate(df, "x", n_boot=500)
    assert r["rate"] == 0.75 and r["k"] == 3 and r["n"] == 4
    assert 0.5 <= r["lo"] <= r["rate"] <= r["hi"] <= 1.0


def test_paired_contrast_sign_and_mcnemar():
    df = synthetic()
    r = paired_contrast(df[df.arm == "A"], df[df.arm == "B"], "attack", n_boot=500)
    assert r["diff"] > 0.15 and r["lo"] > 0 and r["p"] < 0.001


def test_interaction_is_near_zero_for_independent_defenses():
    df = synthetic(n_tasks=150)
    r = interaction(df, {"A": (0, 0), "B": (1, 0), "E": (0, 1), "F": (1, 1)})
    assert r["lo"] < 0 < r["hi"]  # log-odds scale: independence is not flagged as interference


def test_holm_adjustment():
    assert holm([0.01, 0.04, 0.03]) == [0.03, 0.06, 0.06]


def test_calibrate_threshold_on_validation_split():
    rows = pd.DataFrame({"cluster": [f"s/{i}" for i in range(200)], "unsafe": [i % 2 == 0 for i in range(200)],
                         "score": [0.9 if i % 2 == 0 else 0.1 for i in range(200)]})
    test, threshold = calibrate(rows, fraction=0.2)
    assert 0.1 < threshold <= 0.9
    assert (test["blocked"] == test["unsafe"]).all()

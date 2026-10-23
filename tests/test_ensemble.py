import numpy as np
import pandas as pd

from kneemri.constants import LABELS
from kneemri.inference import family_of, rank_ensemble
from kneemri.labels.blend import blend_sources, calibrate_grades, grades_to_labels
from kneemri.metrics import macro_auc


def test_family_names():
    assert family_of("/x/cnxt_native_mirror_f3") == "cnxt_native_mirror"


def test_families_get_equal_weight():
    rng = np.random.default_rng(0)
    a, b = rng.random((50, 12)), rng.random((50, 12))
    preds = {"/m/fam1_f0": a, "/m/fam1_f1": a, "/m/fam1_f2": a, "/m/fam2_f0": b}
    expected = (pd.DataFrame(a).rank(pct=True).values + pd.DataFrame(b).rank(pct=True).values) / 2
    assert np.allclose(rank_ensemble(preds), expected)


def _gold_and_grades(n=60, seed=0):
    rng = np.random.default_rng(seed)
    uid = [f"s{i}" for i in range(n)]
    y = rng.integers(0, 2, (n, 12))
    hard = np.clip(y * 2 + rng.integers(0, 2, (n, 12)), 0, 3)
    gold = pd.DataFrame(y, columns=LABELS).assign(StudyInstanceUID=uid)
    grades = pd.DataFrame(np.hstack([hard, hard + 0.1]),
                          columns=[f"{c}__hard" for c in LABELS] + [f"{c}__exp" for c in LABELS]).assign(StudyInstanceUID=uid)
    return gold, grades


def test_calibration_is_monotone_and_labels_rank_well():
    gold, grades = _gold_and_grades()
    calib = calibrate_grades(grades, gold)
    assert all(calib[c][k] <= calib[c][k + 1] for c in LABELS for k in range(3))
    soft = grades_to_labels(grades, calib)
    assert macro_auc(gold[LABELS].values, soft[LABELS].values) > 0.9


def test_blend_prefers_the_better_source():
    gold, grades = _gold_and_grades()
    good = grades_to_labels(grades, calibrate_grades(grades, gold))
    noise = good.copy()
    noise[LABELS] = np.random.default_rng(1).random((len(noise), 12))
    _, w = blend_sources({"good": good, "noise": noise}, gold, fallback=noise)
    assert (w["good"] > w["noise"]).all()

"""Turns graded reports into soft labels and blends several label sources per finding.

Only 58 training studies carry radiologist image labels ("gold"). They are never trained on: they calibrate
LLM grades into probabilities and score every label source. Each source is then weighted per finding by
``(gold AUC - 0.5)^2``, so a source no better than chance for a finding gets almost no weight there.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from kneemri.constants import LABELS

GRADE_PRIOR = {0: 0.05, 1: 0.45, 2: 0.85, 3: 0.95}     # P(finding | grade) before seeing gold


def calibrate_grades(grades, gold):
    """Gold-based P(finding | hard grade), smoothed with a Beta prior and forced to be monotone in the grade.

    ``grades`` has ``<finding>__hard`` / ``<finding>__exp`` columns and ``StudyInstanceUID``.
    ``gold`` has ``StudyInstanceUID`` and the 12 binary finding columns.
    """
    gg = gold.merge(grades, on="StudyInstanceUID")
    calib = {}
    for c in LABELS:
        y, h = gg[c].values, gg[f"{c}__hard"].fillna(0).values
        p = {k: ((y[h == k] == 1).sum() + 2 * GRADE_PRIOR[k]) / ((h == k).sum() + 2) for k in range(4)}
        for k in range(1, 4):
            p[k] = max(p[k], p[k - 1])
        calib[c] = p
    return calib


def grades_to_labels(grades, calib):
    """Soft labels: the expected grade (0-3) linearly interpolated on the calibrated grade probabilities."""
    out = grades[["StudyInstanceUID"]].copy()
    for c in LABELS:
        e = grades[f"{c}__exp"].fillna(grades[f"{c}__hard"]).clip(0, 3).fillna(0)
        out[c] = np.interp(e, [0, 1, 2, 3], [calib[c][k] for k in range(4)])
    return out


def source_quality(sources, gold):
    """Gold macro-AUC table: one row per finding, one column per label source."""
    table = {}
    for name, df in sources.items():
        m = gold.merge(df, on="StudyInstanceUID", suffixes=("", "_src"))
        table[name] = {c: roc_auc_score(m[c], m[f"{c}_src"]) for c in LABELS if 0 < m[c].sum() < len(m)}
    return pd.DataFrame(table)


def blend_sources(sources, gold, fallback):
    """Per-finding weighted average of label sources.

    ``sources``: {name: DataFrame(StudyInstanceUID + 12 soft labels)}; ``fallback`` (the keyword labels) fills
    studies a source does not cover. Returns (blended labels, weight table).
    """
    ids = fallback[["StudyInstanceUID"]]
    aligned = {}
    for name, df in sources.items():
        a = ids.merge(df[["StudyInstanceUID"] + LABELS], on="StudyInstanceUID", how="left")
        miss = a[LABELS].isna().any(axis=1).values
        a.loc[miss, LABELS] = fallback.loc[miss, LABELS].values
        aligned[name] = a
    auc = source_quality(aligned, gold)
    weights = ((auc - 0.5).clip(lower=0) ** 2)
    weights = weights.div(weights.sum(axis=1).replace(0, 1), axis=0)
    blended = ids.copy()
    for c in LABELS:
        blended[c] = sum(weights.loc[c, name] * aligned[name][c].values for name in aligned)
    return blended, weights

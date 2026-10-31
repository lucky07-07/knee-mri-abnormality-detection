"""Scores report labellers (LLMs, keyword rules) against the 58 radiologist-labelled gold studies.

A labeller's score for a finding is the ROC-AUC of its expected grade (falling back to the hard grade) against the
gold label; calibration is monotone, so it does not change AUC. With only 58 studies, differences below about 0.02
are noise, so every macro AUC comes with a study-level bootstrap 95% CI, and two labellers are compared with a
paired bootstrap (same resampled studies for both).
"""
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from kneemri.constants import LABELS


def grade_scores(grades):
    """Per-finding score columns from a grades table: expected grade, else hard grade, else 0."""
    out = grades[["StudyInstanceUID"]].copy()
    for c in LABELS:
        exp = grades[f"{c}__exp"] if f"{c}__exp" in grades else np.nan
        hard = grades[f"{c}__hard"] if f"{c}__hard" in grades else np.nan
        out[c] = pd.Series(exp, index=grades.index).fillna(pd.Series(hard, index=grades.index)).fillna(0)
    return out


def _aligned(scores, gold):
    m = gold[["StudyInstanceUID"] + LABELS].merge(scores[["StudyInstanceUID"] + LABELS], on="StudyInstanceUID",
                                                  suffixes=("", "__s"))
    if len(m) != len(gold):
        raise ValueError(f"scores cover {len(m)} of {len(gold)} gold studies")
    return m[LABELS].values.astype(int), m[[f"{c}__s" for c in LABELS]].values.astype(float)


def _macro(y, s):
    aucs = [roc_auc_score(y[:, j], s[:, j]) for j in range(y.shape[1]) if 0 < y[:, j].sum() < len(y)]
    return float(np.mean(aucs)) if aucs else np.nan


def per_finding_auc(scores, gold):
    """Series: AUC per finding (findings with a single gold class are skipped)."""
    y, s = _aligned(scores, gold)
    return pd.Series({c: roc_auc_score(y[:, j], s[:, j]) for j, c in enumerate(LABELS) if 0 < y[:, j].sum() < len(y)})


def bootstrap_macro_auc(scores, gold, n_boot=2000, seed=0):
    """(macro AUC, CI low, CI high): percentile 95% CI over studies resampled with replacement."""
    y, s = _aligned(scores, gold)
    rng = np.random.default_rng(seed)
    boots = [_macro(y[idx], s[idx]) for idx in rng.integers(0, len(y), size=(n_boot, len(y)))]
    lo, hi = np.nanpercentile(boots, [2.5, 97.5])
    return _macro(y, s), float(lo), float(hi)


def paired_bootstrap_diff(scores_a, scores_b, gold, n_boot=2000, seed=0):
    """Macro AUC of A minus B: (difference, CI low, CI high, share of resamples where A > B)."""
    y, sa = _aligned(scores_a, gold)
    _, sb = _aligned(scores_b, gold)
    rng = np.random.default_rng(seed)
    diffs = np.array([_macro(y[idx], sa[idx]) - _macro(y[idx], sb[idx])
                      for idx in rng.integers(0, len(y), size=(n_boot, len(y)))])
    lo, hi = np.nanpercentile(diffs, [2.5, 97.5])
    return _macro(y, sa) - _macro(y, sb), float(lo), float(hi), float(np.nanmean(diffs > 0))


def benchmark_table(sources, gold, reference=None, n_boot=2000, seed=0):
    """One row per labeller: macro AUC with 95% CI, per-finding AUCs, and (if ``reference`` names a source) the
    paired difference to it. ``sources``: {name: DataFrame with StudyInstanceUID + 12 score columns}."""
    rows = []
    for name, sc in sources.items():
        auc, lo, hi = bootstrap_macro_auc(sc, gold, n_boot, seed)
        row = {"labeller": name, "macro_auc": round(auc, 3), "ci_low": round(lo, 3), "ci_high": round(hi, 3)}
        if reference is not None and name != reference:
            d, dlo, dhi, p = paired_bootstrap_diff(sc, sources[reference], gold, n_boot, seed)
            row.update({f"diff_vs_{reference}": round(d, 3), "diff_ci_low": round(dlo, 3),
                        "diff_ci_high": round(dhi, 3), "p_better": round(p, 3)})
        row.update(per_finding_auc(sc, gold).round(3).to_dict())
        rows.append(row)
    return pd.DataFrame(rows).sort_values("macro_auc", ascending=False).reset_index(drop=True)

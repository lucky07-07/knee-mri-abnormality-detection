"""Competition metric: macro ROC-AUC over the 12 findings."""
import numpy as np
from sklearn.metrics import roc_auc_score

from kneemri.constants import LABELS


def per_finding_auc(y, p, labels=LABELS):
    """y, p: [n, 12]. Findings without both classes in ``y`` are skipped."""
    out = {}
    for k, c in enumerate(labels):
        yy = (np.asarray(y)[:, k] > 0.5).astype(int)
        if 0 < yy.sum() < len(yy):
            out[c] = roc_auc_score(yy, np.asarray(p)[:, k])
    return out


def macro_auc(y, p, labels=LABELS):
    aucs = per_finding_auc(y, p, labels)
    return float(np.mean(list(aucs.values()))) if aucs else float("nan")

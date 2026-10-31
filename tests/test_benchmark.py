import numpy as np
import pandas as pd
import pytest

from kneemri.constants import LABELS
from kneemri.labels.benchmark import (benchmark_table, bootstrap_macro_auc, grade_scores, paired_bootstrap_diff,
                                      per_finding_auc)
from kneemri.labels.llm import chat_prompt, grade_digit_ids


def _gold(n=58, seed=1):
    rng = np.random.default_rng(seed)
    g = pd.DataFrame(rng.integers(0, 2, size=(n, len(LABELS))), columns=LABELS)
    g.insert(0, "StudyInstanceUID", [f"s{i}" for i in range(n)])
    return g


def _scores(gold, noise, seed=2):
    rng = np.random.default_rng(seed)
    s = gold.copy()
    s[LABELS] = gold[LABELS].values + rng.normal(0, noise, size=(len(gold), len(LABELS)))
    return s


def test_perfect_scores_give_auc_one():
    g = _gold()
    auc, lo, hi = bootstrap_macro_auc(g.copy(), g, n_boot=200)
    assert auc == pytest.approx(1.0) and lo == pytest.approx(1.0) and hi == pytest.approx(1.0)


def test_ci_contains_point_estimate_and_is_wider_for_noisier_labeller():
    g = _gold()
    a, alo, ahi = bootstrap_macro_auc(_scores(g, 0.6), g, n_boot=400)
    b, blo, bhi = bootstrap_macro_auc(_scores(g, 2.0), g, n_boot=400)
    assert alo <= a <= ahi and blo <= b <= bhi
    assert a > b


def test_paired_diff_detects_clearly_better_labeller():
    g = _gold()
    d, lo, hi, p = paired_bootstrap_diff(_scores(g, 0.3), _scores(g, 3.0), g, n_boot=400)
    assert d > 0 and lo > 0 and p > 0.95


def test_single_class_finding_is_skipped():
    g = _gold()
    g["Fracture"] = 0
    assert "Fracture" not in per_finding_auc(g.copy(), g).index


def test_missing_gold_study_raises():
    g = _gold()
    with pytest.raises(ValueError):
        bootstrap_macro_auc(g.iloc[:-1].copy(), g, n_boot=10)


def test_grade_scores_prefers_expected_then_hard_then_zero():
    grades = pd.DataFrame({"StudyInstanceUID": ["a", "b", "c"]})
    for c in LABELS:
        grades[f"{c}__hard"] = [2, 1, np.nan]
        grades[f"{c}__exp"] = [1.7, np.nan, np.nan]
    s = grade_scores(grades)
    assert s["ACL"].tolist() == [1.7, 1.0, 0.0]


def test_benchmark_table_has_reference_diff_and_sorted():
    g = _gold()
    t = benchmark_table({"good": _scores(g, 0.3), "bad": _scores(g, 3.0)}, g, reference="bad", n_boot=200)
    assert t.labeller.tolist() == ["good", "bad"]
    assert t.loc[0, "diff_vs_bad"] > 0 and np.isnan(t.loc[1, "diff_vs_bad"])


class _FakeTok:
    """Minimal tokenizer: a vocab with bare and space-prefixed digits, and a template that rejects system roles."""
    vocab = ["<pad>", "0", "1", "2", "3", "▁0", " 1", "10", "a"]

    def __len__(self):
        return len(self.vocab)

    def decode(self, ids, skip_special_tokens=True):
        return "".join(self.vocab[i] for i in ids)

    def apply_chat_template(self, msgs, tokenize=False, add_generation_prompt=True):
        if any(m["role"] == "system" for m in msgs):
            raise ValueError("System role not supported")
        return "|".join(m["content"] for m in msgs)


def test_grade_digit_ids_collects_all_spellings():
    ids = grade_digit_ids(_FakeTok())
    assert ids == [[1, 5], [2, 6], [3], [4]]


def test_chat_prompt_falls_back_without_system_role():
    assert chat_prompt(_FakeTok(), "SYS", "USER") == "SYS\n\nUSER"

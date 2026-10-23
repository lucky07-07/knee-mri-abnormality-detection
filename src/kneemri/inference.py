"""Offline inference and ensembling (Kaggle code competition: internet off, 9 h limit).

    python -m kneemri.inference --comp /kaggle/input/rsna-knee-abnormality-detection \
        --models /kaggle/input/<training-outputs> --out /kaggle/working/submission.csv

* Models are grouped by preprocessing config, so each test study is decoded once per config.
* Test-time augmentation: models trained with the anatomical mirror average (identity, mirror); the others
  average (identity, 1.08 zoom). There are no plain flips, which would swap medial and lateral.
* Ensembling: rank-average the folds of each family, then average the families with equal weight.
  The metric is per-finding AUC, which depends only on ranks.
"""
import argparse
import glob
import json
import os
import re
import time

import numpy as np
import pandas as pd
import torch

from kneemri import preprocessing as pp
from kneemri.data import mirror, zoom
from kneemri.model import build_model

WEIGHTS = "model_best_gold.pth"


class TestDataset(torch.utils.data.Dataset):
    def __init__(self, test, series, test_dir, cfg):
        self.test, self.test_dir, self.cfg = test, test_dir, cfg
        self.series_by = {k: g for k, g in series.groupby("StudyInstanceUID")}
        self.empty = series.iloc[:0]

    def __len__(self):
        return len(self.test)

    def __getitem__(self, i):
        cfg, slots = self.cfg, self.cfg["slots"]
        sid = self.test.StudyInstanceUID.iloc[i]
        vol = np.zeros((len(slots), cfg["n_slices"], cfg["size"], cfg["size"]), np.uint8)
        pres = np.zeros(len(slots), np.float32)
        try:
            vols, present, _ = pp.load_study(f"{self.test_dir}/{sid}", self.series_by.get(sid, self.empty), cfg)
            for k, s in enumerate(slots):
                vol[k], pres[k] = vols[s], present[s]
        except Exception as e:                       # a broken study gets empty inputs, never a crash
            print("study failed", sid, repr(e)[:200])
        return torch.from_numpy(vol), torch.from_numpy(pres)


def tta_views(model_cfg):
    return ["id", "mirror"] if model_cfg.get("mirror_p", 0) > 0 else ["id", "zoom"]


def apply_view(x, kind, slots):
    if kind == "mirror":
        return mirror(x, slots)
    if kind == "zoom":
        return zoom(x)
    return x


def family_of(model_dir):
    return re.sub(r"_f\d+$", "", os.path.basename(model_dir.rstrip("/")))


def rank_ensemble(preds):
    """preds: {model_dir: [n, 12]} -> rank-average within each family, then mean over families."""
    fam = {}
    for d, p in preds.items():
        fam.setdefault(family_of(d), []).append(pd.DataFrame(p).rank(pct=True).values)
    return np.mean([np.mean(v, 0) for v in fam.values()], 0)


def predict_test(comp, model_dirs, batch=2, workers=4):
    test = pd.read_csv(f"{comp}/test.csv")
    series = pd.read_csv(f"{comp}/test_series.csv")
    labels = list(pd.read_csv(f"{comp}/sample_submission.csv").columns[1:])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    groups = {}
    for d in model_dirs:
        cfg = json.load(open(os.path.join(d, "preprocess_config.json")))
        groups.setdefault(json.dumps(cfg, sort_keys=True), (cfg, []))[1].append(d)

    preds, t0 = {}, time.time()
    for cfg, dirs in groups.values():
        mcfg = {d: json.load(open(os.path.join(d, "model_config.json"))) for d in dirs}
        models = {}
        for d in dirs:
            m = build_model(mcfg[d], len(cfg["slots"]), len(labels), pretrained=False)
            m.load_state_dict(torch.load(os.path.join(d, WEIGHTS), map_location="cpu"))
            models[d] = m.to(device).eval()
        dl = torch.utils.data.DataLoader(TestDataset(test, series, f"{comp}/test_series", cfg),
                                         batch_size=batch, num_workers=workers)
        out = {d: [] for d in dirs}
        with torch.no_grad():
            for vol, pres in dl:
                x, p = vol.to(device).float().div_(255), pres.to(device)
                for d, m in models.items():
                    with torch.autocast("cuda", dtype=torch.float16, enabled=device == "cuda"):
                        views = [torch.sigmoid(m(apply_view(x, v, cfg["slots"]), p).float()) for v in tta_views(mcfg[d])]
                    out[d].append(torch.stack(views).mean(0).cpu().numpy())
        preds.update({d: np.concatenate(v) for d, v in out.items()})
        del models
        torch.cuda.empty_cache()
        print(f"config size={cfg['size']} slices={cfg['n_slices']}: {len(dirs)} models, {(time.time() - t0) / 60:.1f} min")
    return test, labels, preds


def make_submission(test, labels, preds):
    final = rank_ensemble(preds) if len(preds) > 1 and len(test) > 1 else np.mean(list(preds.values()), 0)
    sub = pd.DataFrame(final, columns=labels)
    sub.insert(0, "StudyInstanceUID", test.StudyInstanceUID.values)
    assert len(sub) == len(test) and not sub.StudyInstanceUID.duplicated().any(), "study IDs differ from test.csv"
    assert not sub[labels].isna().any().any(), "missing predictions"
    assert ((sub[labels] >= 0) & (sub[labels] <= 1)).all().all(), "predictions outside [0, 1]"
    return sub


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--comp", required=True, help="competition data folder")
    ap.add_argument("--models", nargs="+", required=True, help="folders searched for model_config.json")
    ap.add_argument("--include", default=r"_f\d", help="regex on model folder paths")
    ap.add_argument("--out", default="submission.csv")
    a = ap.parse_args()
    dirs = sorted({os.path.dirname(p) for root in a.models
                   for p in glob.glob(f"{root}/**/model_config.json", recursive=True) if re.search(a.include, p)})
    assert dirs, "no trained models found"
    print("families:", sorted({family_of(d) for d in dirs}), "| models:", len(dirs))
    sub = make_submission(*predict_test(a.comp, dirs))
    sub.to_csv(a.out, index=False)
    print("submission validated:", sub.shape)


if __name__ == "__main__":
    main()

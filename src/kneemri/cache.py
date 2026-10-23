"""Builds the preprocessed training cache (CPU only, run once per preprocessing config).

    python -m kneemri.cache --config configs/native_mirror.yaml \
        --comp /kaggle/input/rsna-knee-abnormality-detection --out /kaggle/working --part 0 --n-parts 3

Output: ``preprocess_config.json`` plus ``cache_shards/shard_XX.zip`` with one ``<StudyInstanceUID>.npz`` per
study (ZIP_STORED, 500 studies per shard: a Kaggle notebook keeps only about 500 output files).
Large caches are split into parts across several notebooks with ``--part`` / ``--n-parts``.
"""
import argparse
import json
import os
import time
import zipfile

import pandas as pd
import yaml
from concurrent.futures import ProcessPoolExecutor

from kneemri import preprocessing as pp

PER_SHARD = 500


def _one(args):
    study, sdf, root, tmp, cfg = args
    try:
        vols, present, _ = pp.load_study(f"{root}/{study}", sdf, cfg)
        pp.save_study_npz(f"{tmp}/{study}.npz", vols, present, cfg)
        return study, sum(present.values()), None
    except Exception as e:
        return study, 0, repr(e)[:300]


def build(cfg, comp, out, part=0, n_parts=1, workers=4):
    series = pd.read_csv(f"{comp}/train_series.csv")
    studies = sorted(series.StudyInstanceUID.unique())[part::n_parts]
    tmp, shards = f"{out}/_npz", f"{out}/cache_shards"
    os.makedirs(tmp, exist_ok=True)
    os.makedirs(shards, exist_ok=True)
    by_study = {k: g for k, g in series.groupby("StudyInstanceUID")}
    jobs = [(s, by_study[s], f"{comp}/train_series", tmp, cfg) for s in studies]
    t0, errors = time.time(), []
    with ProcessPoolExecutor(workers) as ex:
        for i, (study, n_present, err) in enumerate(ex.map(_one, jobs, chunksize=4)):
            if err:
                errors.append((study, err))
            if i % 200 == 0:
                print(f"{i}/{len(jobs)} studies, {(time.time() - t0) / 60:.1f} min, {len(errors)} errors", flush=True)
    done = sorted(f for f in os.listdir(tmp) if f.endswith(".npz"))
    for k in range(0, len(done), PER_SHARD):
        with zipfile.ZipFile(f"{shards}/shard_{k // PER_SHARD:02d}.zip", "w", zipfile.ZIP_STORED) as zf:
            for f in done[k:k + PER_SHARD]:
                zf.write(f"{tmp}/{f}", f)
                os.remove(f"{tmp}/{f}")
    json.dump({**cfg, "part": part, "n_parts": n_parts}, open(f"{out}/preprocess_config.json", "w"), indent=2)
    print(f"cached {len(done)} studies in {(time.time() - t0) / 60:.1f} min, {len(errors)} errors")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--comp", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--part", type=int, default=0)
    ap.add_argument("--n-parts", type=int, default=1)
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    cfg = pp.load_config(**yaml.safe_load(open(a.config))["preprocess"])
    build(cfg, a.comp, a.out, a.part, a.n_parts, a.workers)


if __name__ == "__main__":
    main()

"""Trains one fold of one model family.

    python -m kneemri.train --config configs/native_mirror.yaml --fold 0 \
        --train-csv /kaggle/input/rsna-knee-abnormality-detection/train.csv \
        --labels outputs/labels/blended_labels.csv --cache /kaggle/input/<cache-notebook> --out outputs/runs

The 58 gold studies are the validation set of every fold and are never trained on. The report-labelled
studies are split into 5 folds by UID hash; the held-out fold gives out-of-fold predictions.
EMA weights are saved for the last epoch and the best gold epoch, with the configs the offline submission needs.
The gold and out-of-fold prediction files come from the last-epoch weights, so they are not tuned on gold.
"""
import argparse
import json
import os
import random
import shutil
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn
from torch.utils.data import DataLoader, Subset

from kneemri.constants import LABELS
from kneemri.data import KneeDataset, gpu_augment, index_cache, study_fold
from kneemri.metrics import macro_auc, per_finding_auc
from kneemri.model import build_model


def load_yaml(path):
    with open(path) as f:
        return yaml.safe_load(f)


@torch.no_grad()
def predict(net, loader, device):
    net.eval()
    P, Y = [], []
    for vol, pres, y in loader:
        x = vol.to(device, non_blocking=True).float().div_(255)
        with torch.autocast("cuda", dtype=torch.float16, enabled=device == "cuda"):
            P.append(torch.sigmoid(net(x, pres.to(device)).float()).cpu().numpy())
        Y.append(y.numpy())
    return np.concatenate(Y), np.concatenate(P)


def run(cfg, fold, train_csv, labels_csv, cache_dirs, out_root):
    t = cfg["train"]
    run_name = f"{cfg['name']}_f{fold}"
    out = os.path.join(out_root, run_name)
    os.makedirs(out, exist_ok=True)
    seed = t["seed"] + fold
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    index = index_cache(cache_dirs)
    pcfg = json.load(open(os.path.join(cache_dirs[0], "preprocess_config.json")))
    assert all(pcfg[k] == v for k, v in cfg["preprocess"].items()), "attached cache does not match the config"

    train = pd.read_csv(train_csv)
    train = train[train.StudyInstanceUID.isin(index)].reset_index(drop=True)
    is_gold = train[LABELS].notna().all(axis=1)
    gold = train[is_gold]
    soft = train[~is_gold][["StudyInstanceUID"]].merge(pd.read_csv(labels_csv), on="StudyInstanceUID", how="left")
    assert not soft[LABELS].isna().any().any(), "every report-labelled study needs a soft label"

    ids = soft.StudyInstanceUID.values
    held = np.array([study_fold(u, t["n_folds"]) == fold for u in ids])
    ds_kw = dict(index=index, cfg=pcfg)
    tr_ds = KneeDataset(ids[~held], soft[LABELS].values[~held], **ds_kw)
    oof_ds = KneeDataset(ids[held], soft[LABELS].values[held], **ds_kw)
    gold_ds = KneeDataset(gold.StudyInstanceUID.values, gold[LABELS].values.astype(np.float32), **ds_kw)
    dl_kw = dict(batch_size=t["batch"], num_workers=t["num_workers"], pin_memory=True)
    oof_dl, gold_dl = DataLoader(oof_ds, **dl_kw), DataLoader(gold_ds, **dl_kw)
    print(f"{run_name}: train {len(tr_ds)} | out-of-fold {len(oof_ds)} | gold {len(gold_ds)} | device {device}")

    m = cfg["model"]
    model = build_model(m, len(pcfg["slots"]), len(LABELS), pretrained=True).to(device)
    ema = AveragedModel(model, multi_avg_fn=get_ema_multi_avg_fn(t["ema_decay"]), use_buffers=True)
    enc_ids = {id(p) for p in model.enc.parameters()}
    enc_params = [p for p in model.parameters() if id(p) in enc_ids]
    head_params = [p for p in model.parameters() if id(p) not in enc_ids]
    lr, lr_enc = t["lr"], t["lr"] * t["backbone_lr_mult"]
    opt = torch.optim.AdamW([dict(params=enc_params, lr=lr_enc), dict(params=head_params, lr=lr)],
                            weight_decay=t["weight_decay"])
    steps_per_epoch = len(tr_ds) // t["batch"]
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[lr_enc, lr],
                                                total_steps=max(1, t["epochs"] * steps_per_epoch // t["accum"]),
                                                pct_start=0.1, anneal_strategy="cos")
    scaler = torch.amp.GradScaler("cuda", enabled=device == "cuda")

    history = []
    for ep in range(t["epochs"]):
        model.train()
        t0, tot, n = time.time(), 0.0, 0
        perm = np.random.default_rng(seed + ep).permutation(len(tr_ds))
        loader = DataLoader(Subset(tr_ds, perm), shuffle=False, drop_last=True, **dl_kw)
        for it, (vol, pres, y) in enumerate(loader):
            x = gpu_augment(vol.to(device, non_blocking=True).float().div_(255), pcfg["slots"], m["mirror_p"])
            with torch.autocast("cuda", dtype=torch.float16, enabled=device == "cuda"):
                logits = model(x, pres.to(device), k=m["train_windows"])
            loss = F.binary_cross_entropy_with_logits(logits.float(), y.to(device)) / t["accum"]
            scaler.scale(loss).backward()
            if (it + 1) % t["accum"] == 0:
                scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(model.parameters(), 2.0)
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
                sched.step()
                ema.update_parameters(model)
            tot += loss.item() * t["accum"]
            n += 1
            if it % 200 == 0:
                print(f"  epoch {ep} step {it}/{steps_per_epoch} loss {tot / n:.4f} {time.time() - t0:.0f}s", flush=True)
        yg, pg = predict(ema.module, gold_dl, device)
        history.append(dict(epoch=ep, loss=tot / n, gold_auc=macro_auc(yg, pg), minutes=(time.time() - t0) / 60,
                            **per_finding_auc(yg, pg)))
        print(f"epoch {ep}: loss {tot / n:.4f} | gold macro AUC {history[-1]['gold_auc']:.4f}", flush=True)
        torch.save(ema.module.state_dict(), f"{out}/model_last.pth")
        if history[-1]["gold_auc"] >= max(h["gold_auc"] for h in history):
            # used by the submission; slightly optimistic on gold because the epoch is chosen on the same studies
            torch.save(ema.module.state_dict(), f"{out}/model_best_gold.pth")

    pd.DataFrame(history).to_csv(f"{out}/history.csv", index=False)
    yg, pg = predict(ema.module, gold_dl, device)
    pd.DataFrame(pg, columns=LABELS).assign(StudyInstanceUID=gold_ds.ids).to_csv(f"{out}/gold_predictions.csv", index=False)
    _, po = predict(ema.module, oof_dl, device)
    pd.DataFrame(po, columns=LABELS).assign(StudyInstanceUID=oof_ds.ids).to_csv(f"{out}/oof_predictions.csv", index=False)
    json.dump(dict(name=cfg["name"], fold=fold, labels=LABELS, preprocess=pcfg, **m),
              open(f"{out}/model_config.json", "w"), indent=2)
    shutil.copy(os.path.join(cache_dirs[0], "preprocess_config.json"), f"{out}/preprocess_config.json")
    print(f"{run_name} done: gold macro AUC {macro_auc(yg, pg):.4f}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--fold", type=int, required=True)
    ap.add_argument("--train-csv", required=True)
    ap.add_argument("--labels", required=True, help="blended soft labels (StudyInstanceUID + 12 columns)")
    ap.add_argument("--cache", nargs="+", required=True, help="cache folders (preprocess_config.json + cache_shards/)")
    ap.add_argument("--out", default="outputs/runs")
    a = ap.parse_args()
    cfg = load_yaml(a.config)
    cfg["train"] = {**load_yaml(os.path.join(os.path.dirname(a.config), "train.yaml")), **cfg.get("train", {})}
    run(cfg, a.fold, a.train_csv, a.labels, a.cache, a.out)


if __name__ == "__main__":
    main()

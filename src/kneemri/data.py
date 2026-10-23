"""Cached-study dataset, study-level folds and GPU augmentation."""
import glob
import hashlib
import io
import math
import os
import zipfile

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


def index_cache(cache_dirs):
    """Maps StudyInstanceUID -> ZIP shard (``cache_shards/*.zip``) or extracted ``.npz`` file."""
    index = {}
    for d in cache_dirs:
        shard_dir = os.path.join(d, "cache_shards")
        zips = sorted(glob.glob(f"{shard_dir}/*.zip"))
        for z in zips:
            for n in zipfile.ZipFile(z).namelist():
                index[n[:-4]] = z
        if not zips:                                 # Kaggle Datasets auto-extract zips into plain .npz files
            for f in glob.glob(f"{shard_dir}/**/*.npz", recursive=True):
                index[os.path.basename(f)[:-4]] = f
    return index


def study_fold(uid, n_folds=5):
    """Stable fold from the UID hash: every run and every model family shares the same split."""
    return int(hashlib.md5(uid.encode()).hexdigest(), 16) % n_folds


class KneeDataset(Dataset):
    """Returns (uint8 volume [S,N,H,W], presence [S], targets [12]) for one cached study."""

    def __init__(self, ids, targets, index, cfg):
        self.ids, self.targets = list(ids), np.asarray(targets, np.float32)
        self.index, self.cfg = index, cfg
        self._zips = {}                              # one open handle per shard per worker

    def __len__(self):
        return len(self.ids)

    def _load(self, sid):
        path = self.index[sid]
        if path.endswith(".npz"):
            return np.load(path)
        if path not in self._zips:
            self._zips[path] = zipfile.ZipFile(path)
        return np.load(io.BytesIO(self._zips[path].read(sid + ".npz")))

    def __getitem__(self, i):
        z = self._load(self.ids[i])
        slots, n, s = self.cfg["slots"], self.cfg["n_slices"], self.cfg["size"]
        vol = np.zeros((len(slots), n, s, s), np.uint8)
        for k, slot in enumerate(slots):
            if slot in z.files:
                vol[k] = z[slot]
        return (torch.from_numpy(vol), torch.from_numpy(z["present"].astype(np.float32)),
                torch.from_numpy(self.targets[i]))


def mirror(x, slots):
    """Anatomical left<->right knee mirror of [B,S,N,H,W]: medial stays medial.

    In the canonical views, sagittal slices run along patient x (reverse the slice order);
    for coronal and axial series the image width is patient x (flip horizontally).
    """
    return torch.stack([x[:, i].flip(1) if s.startswith("sag") else x[:, i].flip(-1) for i, s in enumerate(slots)], 1)


def gpu_augment(x, slots, mirror_p=0.0, max_rot_deg=12.0, max_scale=0.10, max_shift=0.06, max_log_gamma=0.25):
    """One random affine + gamma per series (shared by its slices), then an optional anatomical mirror.

    No plain horizontal flips: they would swap medial and lateral findings.
    """
    B, S, N, H, W = x.shape
    x = x.view(B * S, N, H, W)
    n, dev = B * S, x.device
    ang = (torch.rand(n, device=dev) * 2 - 1) * math.radians(max_rot_deg)
    sc = 1 + (torch.rand(n, device=dev) * 2 - 1) * max_scale
    tx, ty = [(torch.rand(n, device=dev) * 2 - 1) * max_shift for _ in range(2)]
    cos, sin = torch.cos(ang) / sc, torch.sin(ang) / sc
    theta = torch.stack([torch.stack([cos, -sin, tx], 1), torch.stack([sin, cos, ty], 1)], 1)
    grid = F.affine_grid(theta, x.shape, align_corners=False)
    x = F.grid_sample(x, grid, mode="bilinear", padding_mode="zeros", align_corners=False)
    gamma = torch.exp((torch.rand(n, 1, 1, 1, device=dev) * 2 - 1) * max_log_gamma)
    x = (x.clamp(0, 1) ** gamma).view(B, S, N, H, W)
    if mirror_p > 0:
        m = torch.rand(B, device=dev) < mirror_p
        x = torch.where(m[:, None, None, None, None], mirror(x, slots), x)
    return x


def zoom(x, factor=1.08):
    """Centre zoom-in of [B,S,N,H,W], cropped back to H x W (test-time augmentation view)."""
    B, S, N, H, W = x.shape
    z = F.interpolate(x.reshape(B * S, N, H, W), scale_factor=factor, mode="bilinear", align_corners=False)
    dh, dw = (z.shape[-2] - H) // 2, (z.shape[-1] - W) // 2
    return z[..., dh:dh + H, dw:dw + W].reshape(B, S, N, H, W)

"""DICOM -> canonical, cropped, intensity-normalised uint8 volumes (one per series slot).

Every series is:
1. read and sorted along its slice normal (DICOM geometry, InstanceNumber as fallback),
2. reoriented to a canonical view per plane, with left knees mirrored into right-knee orientation,
3. cropped to a square physical field of view (or kept whole and zero-padded),
4. clipped to intensity percentiles and resampled to a fixed number of slices,
5. stored as uint8 ``[n_slices, size, size]``.

The same code runs when the training cache is built and at inference time, so the two always match.
"""
import json
import os
import re
from collections import Counter

import cv2
import numpy as np
import pydicom

from kneemri.constants import SLOTS

DEFAULT_CONFIG = dict(
    size=320,                 # output H = W
    fov_mm=140.0,             # square physical field of view cropped around the image centre
    n_slices=24,              # slices per series after resampling
    sampling="interp",        # "interp" (linear between slices) | "uniform" (nearest slice)
    clip_pct=(0.5, 99.5),     # per-volume percentile clip
    mirror_left=True,         # mirror left knees into right-knee orientation
    crop="fov",               # "fov" | "full" (whole native field of view, aspect kept, zero-padded)
    max_series_slices=80,     # prefer series with <= this many slices when a slot has duplicates
    slots=SLOTS,
)

PLANES = ("Sagittal", "Coronal", "Axial")      # slice normal along x / y / z (DICOM LPS)
_X, _Y, _Z = np.eye(3)
# Canonical view per plane: (slice axis, image "down" axis, image "right" axis) in LPS.
#   Sagittal: slices R->L, down = inferior, right = posterior
#   Coronal : slices A->P, down = inferior, right = patient left (radiological view)
#   Axial   : slices I->S, down = posterior, right = patient left
CANONICAL = {
    "Sagittal": (_X, -_Z, _Y),
    "Coronal": (_Y, -_Z, _X),
    "Axial": (_Z, _Y, _X),
}

# whole-word laterality cues in several report languages; matched as tokens, so "SAG" (sagittal) is never a cue
_LEFT_WORDS = {"L", "LT", "LEFT", "IZQ", "IZQUIERDA", "IZQUIERDO", "GAUCHE", "LINKS", "SINISTRA", "SINISTRO",
               "ESQUERDA", "ESQUERDO", "LIJEVO", "LEVO", "LIJEVA", "LEWE", "LEWY"}
_RIGHT_WORDS = {"R", "RT", "RIGHT", "DER", "DERECHA", "DERECHO", "DROIT", "DROITE", "RECHTS", "DESTRA", "DESTRO",
                "DIREITA", "DIREITO", "DESNO", "DESNA", "PRAWE", "PRAWY"}


def load_config(path=None, **overrides):
    cfg = dict(DEFAULT_CONFIG)
    if path and os.path.exists(path):
        with open(path) as f:
            cfg.update(json.load(f))
    cfg.update(overrides)
    return cfg


def slot_name(plane, fluid):
    p = {"sagittal": "sag", "coronal": "cor", "axial": "ax"}.get(str(plane).strip().lower(), str(plane)[:3].lower())
    return f"{p}_{'fluid' if int(fluid) else 'nonfluid'}"


def plane_from_iop(iop):
    r, c = np.asarray(iop[:3], float), np.asarray(iop[3:], float)
    return PLANES[int(np.argmax(np.abs(np.cross(r, c))))]


def _get(ds, name, default=None):
    v = getattr(ds, name, default)
    return default if v is None or v == "" else v


# ---------------- laterality ----------------
def laterality_from_tags(ds):
    for tag in ("Laterality", "ImageLaterality", "FrameLaterality"):
        v = str(_get(ds, tag, "")).upper().strip()
        if v in ("L", "R"):
            return v
    text = " ".join(str(_get(ds, t, "")) for t in
                    ("SeriesDescription", "StudyDescription", "BodyPartExamined", "ProtocolName")).upper()
    tokens = set(re.split(r"[^A-Z]+", text))
    left, right = bool(tokens & _LEFT_WORDS), bool(tokens & _RIGHT_WORDS)
    if left != right:
        return "L" if left else "R"
    return None


def laterality_from_geometry(meta, margin_mm=20.0):
    # The knee coil sits off the patient midline: a right knee lies at negative x (LPS), a left knee at positive x.
    x = meta.get("center_xyz", [np.nan])[0]
    if not np.isfinite(x) or abs(x) < margin_mm:
        return None
    return "R" if x < 0 else "L"


# ---------------- reading ----------------
def read_series(series_dir, read_pixels=True):
    """Returns (volume float32 [N,H,W] or None, meta dict). Slices are sorted along the slice normal."""
    files = sorted(f for f in os.listdir(series_dir) if f.lower().endswith(".dcm"))
    dss = [pydicom.dcmread(os.path.join(series_dir, f), stop_before_pixels=not read_pixels) for f in files]
    ds0 = dss[0]
    meta = dict(series_dir=series_dir, n_files=len(files))

    try:
        iop = np.asarray(ds0.ImageOrientationPatient, float)
        normal = np.cross(iop[:3], iop[3:])
        pos = np.array([float(np.dot(normal, np.asarray(d.ImagePositionPatient, float))) for d in dss])
    except Exception:
        iop = None
        pos = np.array([float(_get(d, "InstanceNumber", i)) for i, d in enumerate(dss)])
    order = np.argsort(pos, kind="stable")
    dss, pos = [dss[i] for i in order], pos[order]

    # drop slices whose shape differs from the dominant one (localizers mixed into a series)
    main_shape = Counter((int(d.Rows), int(d.Columns)) for d in dss).most_common(1)[0][0]
    keep = [i for i, d in enumerate(dss) if (int(d.Rows), int(d.Columns)) == main_shape]
    dss, pos = [dss[i] for i in keep], pos[keep]

    ps = [float(v) for v in _get(ds0, "PixelSpacing", [np.nan, np.nan])]
    meta.update(
        n_slices=len(dss), rows=main_shape[0], cols=main_shape[1],
        pixel_spacing=ps,                                   # [between rows (down), between cols (right)]
        iop=None if iop is None else iop.tolist(),
        plane_iop=None if iop is None else plane_from_iop(iop),
        photometric=str(_get(ds0, "PhotometricInterpretation", "")),
        laterality_tag=laterality_from_tags(ds0),
    )
    if iop is not None:
        try:
            r, c = iop[:3], iop[3:]
            ipp = np.array([np.asarray(d.ImagePositionPatient, float) for d in dss])
            meta["center_xyz"] = (ipp.mean(0) + r * meta["cols"] * ps[1] / 2 + c * meta["rows"] * ps[0] / 2).tolist()
        except Exception:
            meta["center_xyz"] = [np.nan] * 3
    meta["laterality"] = meta["laterality_tag"] or laterality_from_geometry(meta)

    if not read_pixels:
        return None, meta
    vol = np.stack([d.pixel_array.astype(np.float32) for d in dss])
    if vol.ndim == 4:                                         # RGB-encoded slices (rare)
        vol = vol[..., :3].mean(-1)
    slopes = np.array([float(_get(d, "RescaleSlope", 1.0)) for d in dss], np.float32)
    inters = np.array([float(_get(d, "RescaleIntercept", 0.0)) for d in dss], np.float32)
    vol = vol * slopes[:, None, None] + inters[:, None, None]
    if meta["photometric"] == "MONOCHROME1":
        vol = vol.max() - vol
    return vol, meta


# ---------------- orientation ----------------
def canonicalize(vol, meta):
    """Reorients [N,H,W] to the CANONICAL view of its plane; returns (vol, spacing [down, right])."""
    sp = list(meta["pixel_spacing"])
    if meta.get("iop") is None:
        return vol, sp
    iop = np.asarray(meta["iop"], float)
    right, down = iop[:3], iop[3:]                # column index follows the row cosine, row index the column cosine
    normal = np.cross(right, down)                # slices are sorted ascending along this
    s_t, v_t, h_t = CANONICAL[meta["plane_iop"]]
    if abs(down @ v_t) + abs(right @ h_t) < abs(right @ v_t) + abs(down @ h_t):
        vol = vol.transpose(0, 2, 1)
        right, down = down, right
        sp = sp[::-1]
    if down @ v_t < 0:
        vol = vol[:, ::-1]
    if right @ h_t < 0:
        vol = vol[:, :, ::-1]
    if normal @ s_t < 0:
        vol = vol[::-1]
    return np.ascontiguousarray(vol), sp


def mirror_to_right(vol, plane):
    # Mirror along patient x: sagittal -> reverse slice order, coronal/axial -> flip left-right.
    return np.ascontiguousarray(vol[::-1] if plane == "Sagittal" else vol[:, :, ::-1])


# ---------------- intensity and geometry ----------------
def norm_percentile(vol, lo=0.5, hi=99.5):
    a, b = np.percentile(vol[:, ::2, ::2], [lo, hi])
    return np.clip((vol - a) / max(b - a, 1e-6), 0, 1).astype(np.float32)


def resize_stack(vol, h, w):
    interp = cv2.INTER_AREA if h < vol.shape[1] else cv2.INTER_LINEAR
    # cv2.resize handles at most 4 channels, so resize 4 slices at a time
    out = [cv2.resize(np.ascontiguousarray(vol[i:i + 4].transpose(1, 2, 0)), (w, h), interpolation=interp)
           for i in range(0, len(vol), 4)]
    out = [o[..., None] if o.ndim == 2 else o for o in out]
    return np.concatenate(out, axis=2).transpose(2, 0, 1)


def _crop_pad(vol, cy, cx, h, w):
    N, H, W = vol.shape
    y0, x0 = int(round(cy - h / 2)), int(round(cx - w / 2))
    out = np.zeros((N, h, w), vol.dtype)
    sy0, sx0 = max(y0, 0), max(x0, 0)
    sy1, sx1 = min(y0 + h, H), min(x0 + w, W)
    if sy1 > sy0 and sx1 > sx0:
        out[:, sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0] = vol[:, sy0:sy1, sx0:sx1]
    return out


def crop_resize_pad(vol, size):
    """Whole field of view: resize the longer side to ``size`` and zero-pad to a square."""
    N, H, W = vol.shape
    s = size / max(H, W)
    h, w = max(1, round(H * s)), max(1, round(W * s))
    return _crop_pad(resize_stack(vol, h, w), h / 2, w / 2, size, size)


def crop_fov(vol, size, spacing, fov_mm=140.0):
    """Square physical field of view around the image centre (the knee sits at the scanner isocentre)."""
    h, w = max(1, int(round(fov_mm / spacing[0]))), max(1, int(round(fov_mm / spacing[1])))
    return resize_stack(_crop_pad(vol, vol.shape[1] / 2, vol.shape[2] / 2, h, w), size, size)


def sample_slices(vol, n, mode="interp"):
    m = len(vol)
    if mode == "uniform":
        return vol[np.round(np.linspace(0, m - 1, n)).astype(int)]
    t = np.linspace(0, m - 1, n)
    lo = np.floor(t).astype(int)
    hi = np.minimum(lo + 1, m - 1)
    w = (t - lo)[:, None, None].astype(np.float32)
    return (1 - w) * vol[lo] + w * vol[hi]


# ---------------- series and study ----------------
def preprocess_series(series_dir, cfg):
    vol, meta = read_series(series_dir)
    vol, sp = canonicalize(vol, meta)
    if cfg["mirror_left"] and meta.get("laterality") == "L":
        vol = mirror_to_right(vol, meta["plane_iop"])
    if cfg.get("crop", "fov") == "full":
        vol = crop_resize_pad(vol, cfg["size"])
    else:
        vol = crop_fov(vol, cfg["size"], sp, cfg["fov_mm"])
    vol = norm_percentile(vol, *cfg["clip_pct"])
    vol = sample_slices(vol, cfg["n_slices"], cfg["sampling"])
    return (np.clip(vol, 0, 1) * 255).astype(np.uint8), meta


def choose_series(rows, study_dir, max_slices=80):
    """Among candidate series for one slot, prefer the one with the most slices up to ``max_slices``."""
    best, best_key = None, None
    for _, r in rows.iterrows():
        d = os.path.join(study_dir, str(r["SeriesInstanceUID"]))
        if not os.path.isdir(d):
            continue
        n = len(os.listdir(d))
        key = (n <= max_slices, n if n <= max_slices else -n)
        if best_key is None or key > best_key:
            best, best_key = r, key
    return best


def load_study(study_dir, series_df, cfg=None):
    """Preprocesses one study.

    ``series_df`` holds the rows of ``train_series.csv`` / ``test_series.csv`` for this study.
    Returns ({slot: uint8 [N,S,S]}, {slot: 0/1 presence}, {slot: meta}).
    """
    cfg = cfg or DEFAULT_CONFIG
    df = series_df.copy()
    df["slot"] = [slot_name(p, f) for p, f in zip(df["Anatomical_Plane"], df["Fluid_Sensitive"])]
    vols, present, metas = {}, {}, {}
    empty = np.zeros((cfg["n_slices"], cfg["size"], cfg["size"]), np.uint8)
    for slot in cfg["slots"]:
        rows = df[df["slot"] == slot]
        r = choose_series(rows, study_dir, cfg["max_series_slices"]) if len(rows) else None
        if r is None:
            vols[slot], present[slot] = empty, 0
            continue
        try:
            vols[slot], metas[slot] = preprocess_series(os.path.join(study_dir, str(r["SeriesInstanceUID"])), cfg)
            present[slot] = 1
        except Exception as e:                    # one bad series never stops a run
            vols[slot], present[slot] = empty, 0
            metas[slot] = {"error": repr(e)}
    return vols, present, metas


def save_study_npz(path, vols, present, cfg):
    """Cache format: ``present`` uint8[6] in slot order plus one uint8 [N,S,S] array per present slot."""
    np.savez_compressed(path, present=np.array([present[s] for s in cfg["slots"]], np.uint8),
                        **{s: vols[s] for s in cfg["slots"] if present[s]})

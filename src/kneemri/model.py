"""KneeNet: a 2.5D CNN with cross-series attention and one attention-pooling head per finding.

Every 2.5D window (slices i-1, i, i+1) of every present series is one token:

    shared timm CNN -> linear projection -> + slot embedding + depth embedding
    -> one transformer encoder layer (context across slices and series)
    -> per-finding attention pooling over all tokens -> 12 logits

Training reads only ``k`` random windows per study (fast, and acts as augmentation); evaluation and
inference read all of them. Missing series are masked out of the attention.
"""
import timm
import torch
import torch.nn as nn


class KneeNet(nn.Module):
    def __init__(self, backbone="convnext_tiny.fb_in22k_ft_in1k", n_slots=6, n_labels=12, pretrained=True,
                 d=384, drop=0.2, enc_chunk=48):
        super().__init__()
        self.enc = timm.create_model(backbone, pretrained=pretrained, num_classes=0, in_chans=3)
        self.proj = nn.Sequential(nn.Linear(self.enc.num_features, d), nn.LayerNorm(d))
        self.slot_emb = nn.Parameter(torch.zeros(n_slots, d))
        self.depth_emb = nn.Sequential(nn.Linear(1, d), nn.GELU(), nn.Linear(d, d))
        self.ctx = nn.TransformerEncoderLayer(d, nhead=8, dim_feedforward=2 * d, dropout=0.1,
                                              batch_first=True, norm_first=True)
        self.find_attn = nn.Linear(d, n_labels)                      # one attention score per (token, finding)
        self.find_w = nn.Parameter(torch.randn(n_labels, d) * 0.02)  # one classifier per finding
        self.find_b = nn.Parameter(torch.zeros(n_labels))
        self.drop = nn.Dropout(drop)
        self.enc_chunk = enc_chunk
        cfg = getattr(self.enc, "pretrained_cfg", None) or {}
        self.register_buffer("mean", torch.tensor(cfg.get("mean", (0.485, 0.456, 0.406)), dtype=torch.float32).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(cfg.get("std", (0.229, 0.224, 0.225)), dtype=torch.float32).view(1, 3, 1, 1))

    def encode(self, imgs):
        imgs = (imgs - self.mean) / self.std
        if self.training or imgs.shape[0] <= self.enc_chunk:
            return self.enc(imgs)
        return torch.cat([self.enc(c) for c in imgs.split(self.enc_chunk)])

    def forward(self, x, present, k=None):
        """x: float [B,S,N,H,W] in [0,1]; present: [B,S]; k: windows per study (None = all)."""
        B, S, N, H, W = x.shape
        dev = x.device
        valid = (present > 0.5)[:, :, None].expand(B, S, N).reshape(B, S * N)          # [B,S*N]
        if k is not None and k < S * N:
            score = torch.rand(B, S * N, device=dev).masked_fill(~valid, -1.0)
            pos = score.topk(k, dim=1).indices                                         # [B,k]
        else:
            pos = torch.arange(S * N, device=dev).expand(B, S * N)
        mask = valid.gather(1, pos)                                                    # [B,T]
        s, i = pos // N, pos % N
        T = pos.shape[1]
        base = (torch.arange(B, device=dev)[:, None] * S + s) * N                       # flat index of slice 0
        xf = x.reshape(B * S * N, H, W)
        trip = torch.stack([base + (i - 1).clamp(min=0), base + i, base + (i + 1).clamp(max=N - 1)], -1)
        sel = trip[mask]                                                               # [P,3] valid windows only
        feats = self.proj(self.encode(xf[sel.reshape(-1)].view(-1, 3, H, W)).float())   # [P,d]
        tok = feats.new_zeros(B, T, feats.shape[-1])
        tok[mask] = feats.to(tok.dtype)
        depth = (i.float() / max(N - 1, 1))[..., None]
        tok = tok + self.slot_emb[s] + self.depth_emb(depth)
        pad = ~mask
        pad[~mask.any(1), 0] = False                                                   # studies with no series
        tok = self.ctx(tok, src_key_padding_mask=pad)
        a = self.find_attn(tok).float().masked_fill(pad[..., None], -1e4).softmax(1)   # [B,T,F]
        pooled = torch.einsum("btf,btd->bfd", a, tok.float())                          # [B,F,d]
        return (self.drop(pooled) * self.find_w).sum(-1) + self.find_b


def build_model(model_cfg, n_slots, n_labels, pretrained=True):
    return KneeNet(model_cfg["backbone"], n_slots, n_labels, pretrained=pretrained, d=model_cfg.get("d_model", 384))

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =============================================================================
# SpiralReality Proprietary – LicenseRef-SpiralReality-Proprietary
# SPDX-License-Identifier: LicenseRef-SpiralReality-Proprietary
# © 2025 SpiralReality（Ryō ∴ SpiralArchitect + collaborators）All rights reserved.
#
# 本ソフトウェアおよび付随資料は、SpiralReality専用の非公開ライセンス
#（LicenseRef-SpiralReality-Proprietary）の下で提供されます。
# 複製・改変・翻案・結合・頒布・再配布・販売・貸与・ホスティング・SaaS提供・
# 公衆送信・公開・ベンチマーク公開・逆アセンブル・逆コンパイル・
# リバースエンジニアリング・派生物の作成を含む、
# あらゆる形態のいかなる使用も禁止します。
# This file is a private research artifact for SpiralReality only.
# No use, reproduction, distribution, public display, modification, execution,
# model training, dataset inclusion, or derivative works by any party are
# permitted without prior written consent explicitly granted by SpiralReality.
# Authorized use: SpiralReality maintainers only (Ryo  ∴ SpiralArchitect and AIs from SpiralReality).
# ─────────────────────────────────────────────────────────────────────────────
"""
ProdigyUltra_SR_SplineTempX10

Δ from X9:
  1) 列 split の head-aware 化：Q/K/V の列側も (slice × head × depth) の 3D β₂ VALUE を適用。
     - 列は head バンドで等分（余りは最終 head に寄せ）、各 head バンド内で col B‑spline を評価して合成。
     - 列 REG（L²/TV）も head ごとに適用可能。
  2) Conv の group-wise 係数：rows=O×I_group に対して b2_row = b2_out[o]×b2_in[i_group] を階層合成。
     - depthwise（groups==in_channels）/ dilated（dilation>1）/ separable を安全に処理。
     - 列（kH×kW）は X9 の空間因子化に加え、dilation に応じた中心距離重み（任意）を導入。
  3) 多周波の位相分離フィルタ：φ=(φ_w, φ_l, φ_ext, φ_r, φ_u) をオンライン白色化（共分散推定→Cholesky）
     の上で直交成分 z に投影。ゲイン g を目標 mp_target へ自動学習（ZSGD 風）。

既存ノブとの互換を維持。新機能は指定がなければ無効または弱い挙動。
"""
from __future__ import annotations

import math, re, collections
from typing import Dict, List, Tuple, Optional
import torch
import torch.nn as nn
from torch.optim.optimizer import Optimizer
from torch.optim.lr_scheduler import _LRScheduler

# ─────────────────────────────────────────────────────────────────────────────
# Utilities & patterns
# ─────────────────────────────────────────────────────────────────────────────
_NORM_TYPES = (
    nn.LayerNorm,
    nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d,
    nn.GroupNorm,
    nn.InstanceNorm1d, nn.InstanceNorm2d, nn.InstanceNorm3d,
)
_CONV_TYPES = (nn.Conv1d, nn.Conv2d, nn.Conv3d)

_DEPTH_PAT = re.compile(r"(?:layers?|blocks?|stages?|h|enc(?:oder)?|dec(?:oder)?)[._-]?(\d+)", re.IGNORECASE)
_INT_PAT   = re.compile(r"(\d+)")

def _detect_qkv_split_dim(p: torch.Tensor) -> Optional[int]:
    if p.ndim >= 1 and (p.shape[0] % 3 == 0): return 0
    if p.ndim >= 2 and (p.shape[1] % 3 == 0): return 1
    return None

def _infer_depth_from_name(name: str) -> int:
    cand = _DEPTH_PAT.findall(name)
    if cand:
        try: return max(int(c) for c in cand)
        except: pass
    cand2 = _INT_PAT.findall(name)
    if cand2:
        try: return max(int(c) for c in cand2)
        except: pass
    return 0

def _guess_tag(mn: str, module: nn.Module, pn: str) -> str:
    n = (mn + "." + pn).lower().strip(".")
    if pn.endswith("bias"):
        if "in_proj_bias" in n or "qkv" in n: return "attn_qkv"
        return "bias"
    if isinstance(module, nn.Embedding) or any(k in n for k in ["embed","token"]): return "embed"
    if isinstance(module, _NORM_TYPES): return "norm"
    if isinstance(module, _CONV_TYPES): return "conv"
    if "in_proj_weight" in n or "in_proj_bias" in n or ".qkv" in n or "to_qkv" in n or "qkv." in n: return "attn_qkv"
    if any(k in n for k in ["q_proj",".to_q",".query",".q."]): return "attn_q"
    if any(k in n for k in ["k_proj",".to_k",".key",".k."]): return "attn_k"
    if any(k in n for k in ["v_proj",".to_v",".value",".v."]): return "attn_v"
    if any(k in n for k in ["o_proj","out_proj",".to_out",".o."]): return "attn_o"
    if isinstance(module, nn.Linear): return "mlp"
    return "other"

def _detect_heads_from_module(module: nn.Module) -> Optional[int]:
    if hasattr(module, "num_heads") and isinstance(module.num_heads, int):
        return int(module.num_heads)
    for key in ("n_heads","heads","num_head","numHead","nhead","num_heads"):
        if hasattr(module, key):
            try:
                v = int(getattr(module, key)); 
                if v > 0: return v
            except: pass
    return None

# ─────────────────────────────────────────────────────────────────────────────
# Simple spline / bspline helpers (trimmed)
# ─────────────────────────────────────────────────────────────────────────────
def _bspline_basis_all(x: torch.Tensor, degree: int, knots: torch.Tensor) -> torch.Tensor:
    x = x.to(torch.float64); k = knots.to(torch.float64)
    x2 = x.view(-1,1)
    B = ((x2 >= k[:-1]) & (x2 < k[1:])).to(torch.float64)
    for d in range(1, int(degree)+1):
        cols = B.shape[1] - 1
        if cols <= 0: break
        left_den  = (k[d: d+cols] - k[:cols]).clamp_min(1e-12)
        right_den = (k[d+1: d+1+cols] - k[1:1+cols]).clamp_min(1e-12)
        left_num  = (x2 - k[:cols]).clamp_min(0.0)
        right_num = (k[d+1: d+1+cols] - x2).clamp_min(0.0)
        B = (left_num / left_den) * B[:, :cols] + (right_num / right_den) * B[:, 1:1+cols]
    return B

def _bspline_profile_1d(length: int, degree: int, ctrl: List[float], knots=None, normalized=True, device=None) -> torch.Tensor:
    dev = device or torch.device("cpu")
    xs = torch.linspace(0.0, 1.0, steps=max(1, int(length)), device=dev, dtype=torch.float64)
    ctrl = torch.tensor([float(c) for c in ctrl], dtype=torch.float64, device=dev)
    if knots is None or len(knots) == 0:
        m = (ctrl.numel() - 1) + degree + 1
        k = torch.zeros(m + 1, dtype=torch.float64, device=dev)
        k[: degree + 1] = 0.0
        k[-(degree + 1) :] = 1.0
        if m - 2 * degree > 0:
            k[degree + 1 : -degree - 1] = torch.linspace(
                0.0,
                1.0,
                steps=m - 2 * degree + 1,
                device=dev,
                dtype=torch.float64,
            )[1:-1]
    else:
        k = torch.tensor(knots, dtype=torch.float64, device=dev)
        if not normalized and k.numel() > 0:
            kmin, kmax = float(k.min().item()), float(k.max().item())
            scale = max(1e-12, (kmax - kmin))
            xs = xs * scale + kmin
    B = _bspline_basis_all(xs, int(degree), k)
    y = (B @ ctrl.view(-1,1)).squeeze(1).to(torch.float32)
    return y


def _beta2_bounds(spec, default_min: float = 0.95, default_max: float = 0.9999) -> Tuple[float, float]:
    if isinstance(spec, dict):
        lo = float(spec.get("min", default_min))
        hi = float(spec.get("max", default_max))
        return lo, hi
    return float(default_min), float(default_max)


def _resolve_per_key_spec(spec_map, key: str):
    if not isinstance(spec_map, dict):
        return spec_map
    base_keys = {"q", "k", "v", "default"}
    base = {k: v for k, v in spec_map.items() if k not in base_keys}
    override = None
    entry = spec_map.get(key)
    if isinstance(entry, dict):
        override = entry
    else:
        default_entry = spec_map.get("default")
        if isinstance(default_entry, dict):
            override = default_entry
    if override is None:
        return spec_map
    resolved = dict(base)
    resolved.update(override)
    return resolved

# ─────────────────────────────────────────────────────────────────────────────
# β2 column head-aware profiles
# ─────────────────────────────────────────────────────────────────────────────
def _eval_depth_head_col_profile(c_len: int, H: int, spec: dict, key: str, depth_norm: float, *, device=None) -> torch.Tensor:
    """Return length-c_len β2 over columns, split into H head bands."""
    dev = device or torch.device("cpu")
    if isinstance(spec, dict):
        spec = _resolve_per_key_spec(spec, key)
        if not isinstance(spec, dict):
            spec = {}
    lo = float(spec.get("min", 0.0)); hi = float(spec.get("max", 1.0))
    combine = str(spec.get("combine", "mul")).lower()
    # depth scalar per slice
    deg_d = int(spec.get("degree", 3))
    ctrl_d_map = spec.get("ctrl", {})
    kd_in = spec.get("knots", None)
    normalized = bool(spec.get("normalized", True))
    ctrl_d = torch.tensor(ctrl_d_map.get(key, [1.0,1.0]), dtype=torch.float64, device=dev)
    if kd_in is None or (isinstance(kd_in, dict) and kd_in.get(key) is None):
        m = (ctrl_d.numel()-1) + deg_d + 1
        kd = torch.zeros(m+1, dtype=torch.float64, device=dev); kd[:deg_d+1]=0.0; kd[-(deg_d+1):]=1.0
        if m - 2*deg_d > 0:
            kd[deg_d+1:-deg_d-1] = torch.linspace(0.0, 1.0, steps=m-2*deg_d+1, device=dev, dtype=torch.float64)[1:-1]
        normalized = True
    else:
        if isinstance(kd_in, dict):
            kd = kd_in.get(key, None)
        else:
            kd = kd_in
        kd = torch.tensor(kd, dtype=torch.float64, device=dev)
    sample_depth = torch.tensor([float(depth_norm)], dtype=torch.float64, device=dev)
    if not normalized and kd.numel() > 0:
        kmin, kmax = float(kd.min().item()), float(kd.max().item())
        scale = max(1e-12, (kmax - kmin))
        sample_depth = sample_depth * scale + kmin
    Bd = _bspline_basis_all(sample_depth, deg_d, kd)
    yd = (Bd @ ctrl_d.view(-1,1)).squeeze(1).to(torch.float32)[0]

    # head scalar profile
    head_spec = spec.get("head", {"degree":2,"ctrl":{}})
    deg_h = int(head_spec.get("degree", 2))
    h_ctrl_map = head_spec.get("ctrl", {})
    kh_in = head_spec.get("knots", None)
    hnorm = bool(head_spec.get("normalized", True))
    h_ctrl = torch.tensor(h_ctrl_map.get(key, [1.0,1.0]), dtype=torch.float64, device=dev)
    if kh_in is None or (isinstance(kh_in, dict) and kh_in.get(key) is None):
        m = (h_ctrl.numel()-1) + deg_h + 1
        kh = torch.zeros(m+1, dtype=torch.float64, device=dev); kh[:deg_h+1]=0.0; kh[-(deg_h+1):]=1.0
        if m - 2*deg_h > 0:
            kh[deg_h+1:-deg_h-1] = torch.linspace(0.0, 1.0, steps=m-2*deg_h+1, device=dev, dtype=torch.float64)[1:-1]
    else:
        if isinstance(kh_in, dict):
            kh = kh_in.get(key, None)
        else:
            kh = kh_in
        kh = torch.tensor(kh, dtype=torch.float64, device=dev)
    xs_h = torch.linspace(0.0, 1.0, steps=max(1,int(H)), device=dev, dtype=torch.float64)
    if not hnorm and kh.numel() > 0:
        kmin, kmax = float(kh.min().item()), float(kh.max().item())
        scale = max(1e-12, (kmax - kmin))
        xs_h = xs_h * scale + kmin
    Bh = _bspline_basis_all(xs_h, deg_h, kh)
    yh = (Bh @ h_ctrl.view(-1,1)).squeeze(1).to(torch.float32)  # (H,)

    # per-head column profile
    col_spec = spec.get("col", {"degree":2,"ctrl":{}})
    deg_c = int(col_spec.get("degree", 2))
    c_ctrl_map = col_spec.get("ctrl", {})
    kc_in = col_spec.get("knots", None)
    cnorm = bool(col_spec.get("normalized", True))

    # banding
    band = max(1, c_len // max(1,H))
    out = torch.empty(c_len, dtype=torch.float32, device=dev)
    pos = 0
    for h in range(H):
        start = pos
        end = c_len if h == H-1 else (start + band)
        length = max(1, end - start)
        c_ctrl = c_ctrl_map.get(key, [1.0,1.0])
        if isinstance(kc_in, dict):
            kc = kc_in.get(key, None)
        else:
            kc = kc_in
        ycol = _bspline_profile_1d(length, deg_c, c_ctrl, knots=kc, normalized=cnorm, device=dev).to(torch.float32)
        if combine == "add":
            vec = ycol + (yd * yh[h])
        else:
            vec = ycol * (yd * yh[h])
        out[start:end] = vec.clamp(lo, hi)
        pos = end
    return out


def _eval_depth_col_profile(spec_map: dict, key: str, length: int, depth_norm: float, *, device=None) -> torch.Tensor:
    """Fallback evaluator for depth×column β2 fields (no head split)."""
    dev = device or torch.device("cpu")
    if spec_map is None:
        return torch.ones(length, dtype=torch.float32, device=dev)

    spec = _resolve_per_key_spec(spec_map, key)
    if not isinstance(spec, dict):
        spec = {}

    # If the provided spec already contains the richer head-aware layout, reuse it.
    if isinstance(spec, dict) and ("head" in spec or "col" in spec):
        return _eval_depth_head_col_profile(length, 1, spec, key, depth_norm, device=dev)

    lo = float(spec.get("min", 0.0)) if isinstance(spec, dict) else 0.0
    hi = float(spec.get("max", 1.0)) if isinstance(spec, dict) else 1.0
    combine = str(spec.get("combine", "mul")).lower() if isinstance(spec, dict) else "mul"

    def _resolve_ctrl(ctrl_spec, default):
        if isinstance(ctrl_spec, dict):
            return ctrl_spec.get(key, ctrl_spec.get("default", default))
        if ctrl_spec is None:
            return default
        return ctrl_spec

    knots = None
    deg = 2
    normalized = True
    if isinstance(spec, dict):
        deg = int(spec.get("degree", deg))
        normalized = bool(spec.get("normalized", normalized))
        knots_in = spec.get("knots", None)
        if isinstance(knots_in, dict):
            knots = knots_in.get(key, None)
        else:
            knots = knots_in
        ctrl_vals = _resolve_ctrl(spec.get("ctrl", None), [1.0, 1.0])
    else:
        ctrl_vals = [1.0, 1.0]

    if not isinstance(ctrl_vals, (list, tuple)):
        ctrl_vals = [float(ctrl_vals)] * 2

    col_curve = _bspline_profile_1d(length, int(deg), list(ctrl_vals), knots=knots, normalized=normalized, device=dev)

    depth_spec = spec.get("depth", None) if isinstance(spec, dict) else None
    if depth_spec is not None:
        deg_d = int(depth_spec.get("degree", 3))
        kd_in = depth_spec.get("knots", None)
        if isinstance(kd_in, dict):
            kd = kd_in.get(key, None)
        else:
            kd = kd_in
        ctrl_d = _resolve_ctrl(depth_spec.get("ctrl", None), [1.0, 1.0])
        if not isinstance(ctrl_d, (list, tuple)):
            ctrl_d = [float(ctrl_d)] * 2
        ctrl_d = torch.tensor([float(c) for c in ctrl_d], dtype=torch.float64, device=dev)
        normalized_depth = bool(depth_spec.get("normalized", True))
        if kd is None:
            m = (ctrl_d.numel() - 1) + deg_d + 1
            kd = torch.zeros(m + 1, dtype=torch.float64, device=dev)
            kd[:deg_d + 1] = 0.0
            kd[-(deg_d + 1):] = 1.0
            if m - 2 * deg_d > 0:
                kd[deg_d + 1:-deg_d - 1] = torch.linspace(0.0, 1.0, steps=m - 2 * deg_d + 1, device=dev, dtype=torch.float64)[1:-1]
            normalized_depth = True
        else:
            kd = torch.tensor(kd, dtype=torch.float64, device=dev)
        sample_depth = torch.tensor([float(depth_norm)], dtype=torch.float64, device=dev)
        if not normalized_depth and kd.numel() > 0:
            kmin, kmax = float(kd.min().item()), float(kd.max().item())
            scale = max(1e-12, (kmax - kmin))
            sample_depth = sample_depth * scale + kmin
        Bd = _bspline_basis_all(sample_depth, deg_d, kd)
        depth_val = (Bd @ ctrl_d.view(-1, 1)).squeeze(1).to(torch.float32)[0]
    else:
        depth_val = torch.tensor(1.0, dtype=torch.float32, device=dev)

    col_curve = col_curve.to(torch.float32)
    vec = col_curve + depth_val if combine == "add" else col_curve * depth_val
    return vec.clamp(lo, hi)

# ─────────────────────────────────────────────────────────────────────────────
# Spatial profile with dilation awareness
# ─────────────────────────────────────────────────────────────────────────────
def _eval_spatial_profile_conv(kH: int, kW: int, spec: dict, depth_norm: float, dilation=(1,1), *, device=None) -> torch.Tensor:
    dev = device or torch.device("cpu")
    lo = float(spec.get("min", 0.0)); hi = float(spec.get("max", 1.0))
    combine = str(spec.get("combine", "mul")).lower()
    depth = spec.get("depth", None)  # expects dict with degree/ctrl/knots keys
    if depth:
        deg_d = int(depth.get("degree", 1))
        ctrl_d = depth.get("ctrl", [1.0, 1.0])
        if not isinstance(ctrl_d, (list, tuple)):
            ctrl_d = [float(ctrl_d)] * 2
        ctrl_d = torch.tensor([float(c) for c in ctrl_d], dtype=torch.float64, device=dev)
        kd = depth.get("knots", None)
        normalized_depth = bool(depth.get("normalized", True))
        if kd is None:
            m = (ctrl_d.numel() - 1) + deg_d + 1
            kd = torch.zeros(m + 1, dtype=torch.float64, device=dev)
            kd[:deg_d + 1] = 0.0
            kd[-(deg_d + 1):] = 1.0
            if m - 2 * deg_d > 0:
                kd[deg_d + 1:-deg_d - 1] = torch.linspace(0.0, 1.0, steps=m - 2 * deg_d + 1, device=dev, dtype=torch.float64)[1:-1]
            normalized_depth = True
        else:
            kd = torch.tensor(kd, dtype=torch.float64, device=dev)
        sample_depth = torch.tensor([float(depth_norm)], dtype=torch.float64, device=dev)
        if not normalized_depth and kd.numel() > 0:
            kmin, kmax = float(kd.min().item()), float(kd.max().item())
            scale = max(1e-12, (kmax - kmin))
            sample_depth = sample_depth * scale + kmin
        Bd = _bspline_basis_all(sample_depth, deg_d, kd)
        y_d = (Bd @ ctrl_d.view(-1, 1)).squeeze(1).to(torch.float32)[0]
    else:
        y_d = torch.tensor(1.0, dtype=torch.float32, device=dev)

    kh_spec = spec.get("kh", {"degree":1, "ctrl":[1.0,1.0]})
    kw_spec = spec.get("kw", {"degree":1, "ctrl":[1.0,1.0]})
    y_h = _bspline_profile_1d(
        kH,
        int(kh_spec.get("degree", 1)),
        kh_spec.get("ctrl", [1.0, 1.0]),
        knots=kh_spec.get("knots", None),
        normalized=bool(kh_spec.get("normalized", True)),
        device=dev,
    )
    y_w = _bspline_profile_1d(
        kW,
        int(kw_spec.get("degree", 1)),
        kw_spec.get("ctrl", [1.0, 1.0]),
        knots=kw_spec.get("knots", None),
        normalized=bool(kw_spec.get("normalized", True)),
        device=dev,
    )

    mat = (y_h.view(kH,1) * y_w.view(1,kW)).reshape(-1).to(torch.float32)

    # dilation-aware radial weighting（任意）
    dil = spec.get("dilation_weight", None)
    if dil is not None:
        alpha = float(dil.get("alpha", 0.15)); power = float(dil.get("power", 1.0))
        dh, dw = int(max(1, dilation[0])), int(max(1, dilation[1]))
        # normalized distance from center with dilation scaling
        grid_h = torch.arange(kH, device=dev) - (kH-1)/2
        grid_w = torch.arange(kW, device=dev) - (kW-1)/2
        Hm = (grid_h.abs() * dh).view(-1,1).repeat(1,kW)
        Wm = (grid_w.abs() * dw).view(1,-1).repeat(kH,1)
        R = torch.sqrt(Hm*Hm + Wm*Wm); R = (R / (R.max()+1e-8)).reshape(-1).to(torch.float32)
        mat = mat * (1.0 + alpha * (R ** power))

    vec = mat * y_d if combine == "mul" else (mat + y_d)
    return vec.clamp(lo, hi)

# ─────────────────────────────────────────────────────────────────────────────
# Conv channel (out/in) hierarchical profiles
# ─────────────────────────────────────────────────────────────────────────────
def _eval_conv_row_channel_profile(out_ch: int, in_ch_g: int, spec: dict, depth_norm: float, *, device=None) -> torch.Tensor:
    """Return length (out_ch*in_ch_g) β2 over rows = O × I_group, as outer product of out/in channel curves."""
    dev = device or torch.device("cpu")
    lo = float(spec.get("min", 0.0)); hi = float(spec.get("max", 1.0))
    combine = str(spec.get("combine", "mul")).lower()
    out_sp = spec.get("out", {"degree":2,"ctrl":[1.0,1.0]})
    in_sp  = spec.get("in",  {"degree":2,"ctrl":[1.0,1.0]})
    y_out = _bspline_profile_1d(
        out_ch,
        int(out_sp.get("degree", 2)),
        out_sp.get("ctrl", [1.0, 1.0]),
        knots=out_sp.get("knots", None),
        normalized=bool(out_sp.get("normalized", True)),
        device=dev,
    )
    y_in = _bspline_profile_1d(
        in_ch_g,
        int(in_sp.get("degree", 2)),
        in_sp.get("ctrl", [1.0, 1.0]),
        knots=in_sp.get("knots", None),
        normalized=bool(in_sp.get("normalized", True)),
        device=dev,
    )
    mat = (y_out.view(out_ch,1) * y_in.view(1,in_ch_g)).reshape(-1).to(torch.float32)
    depth = spec.get("depth", None)
    if depth:
        deg_d = int(depth.get("degree", 1))
        ctrl_d = depth.get("ctrl", [1.0, 1.0])
        if not isinstance(ctrl_d, (list, tuple)):
            ctrl_d = [float(ctrl_d)] * 2
        ctrl_d = torch.tensor([float(c) for c in ctrl_d], dtype=torch.float64, device=dev)
        kd = depth.get("knots", None)
        normalized_depth = bool(depth.get("normalized", True))
        if kd is None:
            m = (ctrl_d.numel() - 1) + deg_d + 1
            kd = torch.zeros(m + 1, dtype=torch.float64, device=dev)
            kd[:deg_d + 1] = 0.0
            kd[-(deg_d + 1):] = 1.0
            if m - 2 * deg_d > 0:
                kd[deg_d + 1:-deg_d - 1] = torch.linspace(0.0, 1.0, steps=m - 2 * deg_d + 1, device=dev, dtype=torch.float64)[1:-1]
            normalized_depth = True
        else:
            kd = torch.tensor(kd, dtype=torch.float64, device=dev)
        sample_depth = torch.tensor([float(depth_norm)], dtype=torch.float64, device=dev)
        if not normalized_depth and kd.numel() > 0:
            kmin, kmax = float(kd.min().item()), float(kd.max().item())
            scale = max(1e-12, (kmax - kmin))
            sample_depth = sample_depth * scale + kmin
        Bd = _bspline_basis_all(sample_depth, deg_d, kd)
        y_d = (Bd @ ctrl_d.view(-1, 1)).squeeze(1).to(torch.float32)[0]
    else:
        y_d = torch.tensor(1.0, dtype=torch.float32, device=dev)
    vec = mat * y_d if combine == "mul" else (mat + y_d)
    return vec.clamp(lo, hi)

# ─────────────────────────────────────────────────────────────────────────────
# (Placeholders to keep compatibility with prior X7/X8 API)
# ─────────────────────────────────────────────────────────────────────────────
def _eval_beta2_field(length: int, spec: dict, device=None, reg: Optional[Dict[str,float]]=None) -> torch.Tensor:
    """1D evaluator used in legacy paths (kept for compatibility)."""
    vec = _bspline_profile_1d(length, int(spec.get("degree",2)), spec.get("ctrl",[1.0,1.0]), knots=spec.get("knots",None), normalized=spec.get("normalized",True), device=device)
    lo = float(spec.get("min", 0.0)); hi = float(spec.get("max", 1.0))
    if reg is not None:
        return _regularize_field_1d(vec.to(torch.float32), lo, hi, reg)
    return vec.clamp(lo, hi).to(torch.float32)

def _regularize_field_1d(vec: torch.Tensor, lo: float, hi: float, reg: Optional[Dict[str, float]]) -> torch.Tensor:
    if reg is None or (reg.get("l2",0.0) <= 0 and reg.get("tv",0.0) <= 0):
        return vec.clamp(lo, hi)
    out = vec.clone().to(torch.float32)
    iters = int(reg.get("iters", 1)); l2 = float(reg.get("l2", 0.0)); tv = float(reg.get("tv", 0.0)); step = float(reg.get("step", 0.25))
    for _ in range(max(1, iters)):
        if l2 > 0.0:
            lap = out.roll(1) + out.roll(-1) - 2*out
            out = out + step * l2 * lap
        if tv > 0.0:
            g_left  = out - out.roll(1)
            g_right = out.roll(-1) - out
            subgrad = torch.sign(g_left) - torch.sign(g_right)
            out = out - step * tv * subgrad
        out = out.clamp(lo, hi)
    return out

def _schedule_reg(depth: int, step: int, base: Optional[dict], sched: Optional[dict]) -> dict:
    # minimal scheduler (kept for compat): just merge base and sched multipliers if present
    out = dict(base or {})
    if not sched: return out
    mul = float(sched.get("mul", 1.0))
    out["l2"] = out.get("l2", 0.0) * mul
    out["tv"] = out.get("tv", 0.0) * mul
    return out

def _apply_piecewise_reg_by_depth(depth: int, reg: dict, pw: Optional[List[dict]]) -> dict:
    if not pw: return reg
    for seg in pw:
        d0, d1 = seg.get("range",(0,10**9))
        if depth >= int(d0) and depth < int(d1):
            reg["l2"] = float(seg.get("l2", reg.get("l2", 0.0)))
            reg["tv"] = float(seg.get("tv", reg.get("tv", 0.0)))
            break
    return reg

# ─────────────────────────────────────────────────────────────────────────────
# Builder
# ─────────────────────────────────────────────────────────────────────────────
def build_param_groups(
    model: nn.Module,
    *,
    base_lr: float = 3e-4,
    base_wd: float = 0.01,

    lr_scales: Optional[Dict[str, float]] = None,

    # Lookahead temperature / multi-phase / learning
    lookahead_temp_phase_warm_gain_map: Optional[Dict[str, float]] = None,
    lookahead_temp_phase_lr_gain_map: Optional[Dict[str, float]] = None,
    lookahead_temp_phase_ext_gain_map: Optional[Dict[str, float]] = None,
    lookahead_temp_phase_r_gain_map: Optional[Dict[str, float]] = None,
    lookahead_temp_phase_rms_gain_map: Optional[Dict[str, float]] = None,
    lookahead_temp_multi_phase_mode_map: Optional[Dict[str, str]] = None,
    lookahead_temp_phase_lr_mode_map: Optional[Dict[str, str]] = None,
    lookahead_temp_multi_phase_caps_map: Optional[Dict[str, Tuple[float,float]]] = None,
    lookahead_temp_multi_phase_alphas_map: Optional[Dict[str, Tuple[float,float]]] = None,
    lookahead_temp_mp_target_map: Optional[Dict[str, float]] = None,
    lookahead_temp_mp_adapt_map: Optional[Dict[str, float]] = None,
    lookahead_temp_mp_gcap_map: Optional[Dict[str, float]] = None,

    # Generic knobs
    lookahead_k_map: Optional[Dict[str, int]] = None,
    lookahead_alpha: float = 0.5,
    lookahead_alpha_map: Optional[Dict[str, float]] = None,
    lookahead_reset_m: bool = True,
    lookahead_k_mode_map: Optional[Dict[str, str]] = None,
    lookahead_k_max_map: Optional[Dict[str, int]] = None,
    lookahead_k_depth_mul_map: Optional[Dict[str, float]] = None,
    lookahead_k_cooldown_syncs_map: Optional[Dict[str, int]] = None,
    lookahead_stop_after_syncs_map: Optional[Dict[str, int]] = None,
    lookahead_stop_when_k_ge_map: Optional[Dict[str, int]] = None,

    zero_wd_tags=("norm","bias","embed"),
    rms_clip_threshold_map: Optional[Dict[str, float]] = None,
    trust_clip_map: Optional[Dict[str, Tuple[float, float]]] = None,
    trust_granularity: str = "tag",
    rms_clip_granularity: str = "tag",

    factored_mode_map: Optional[Dict[str, str]] = None,
    use_adafactor_tag_map: Optional[Dict[str, bool]] = None,
    beta1_map: Optional[Dict[str, float]] = None,
    beta2_map: Optional[Dict[str, float]] = None,

    # QKV & β2 fields
    qkv_beta1_map: Optional[Dict[str, float]] = None,
    qkv_beta2_map: Optional[Dict[str, float]] = None,
    qkv_trust_clip_map: Optional[Dict[str, Tuple[float,float]]] = None,
    qkv_beta2_field_map: Optional[Dict[str, Dict[str, float]]] = None,
    qkv_beta2_reg_map: Optional[Dict[str, Dict[str, float]]] = None,
    qkv_beta2_reg_sched_map: Optional[Dict[str, Dict[str, float]]] = None,
    qkv_beta2_reg_pw_depth_map: Optional[Dict[str, List[dict]]] = None,
    qkv_beta2_reg_depth_spline_map: Optional[Dict[str, Dict[str, dict]]] = None,
    qkv_beta2_reg_depth_bspline_field_map: Optional[Dict[str, dict]] = None,
    qkv_beta2_reg_depth_head_bspline_field_map: Optional[Dict[str, dict]] = None,
    qkv_beta2_value_depth_head_bspline_field_map: Optional[dict] = None,

    # Column head-aware VALUE/REG
    qkv_beta2_value_depth_head_col_bspline_field_map: Optional[dict] = None,
    qkv_beta2_reg_depth_head_col_map: Optional[Dict[str, dict]] = None,

    # Column depth×col VALUE/REG (fallback)
    qkv_beta2_value_depth_col_bspline_field_map: Optional[dict] = None,
    qkv_beta2_reg_depth_col_bspline_field_map: Optional[dict] = None,

    # Conv spatial & channel hierarchical
    conv_beta2_value_spatial_field_map: Optional[dict] = None,
    conv_beta2_reg_spatial_field_map: Optional[dict] = None,
    conv_beta2_value_channel_field_map: Optional[dict] = None,
    conv_beta2_reg_channel_field_map: Optional[dict] = None,

    # heads hint
    attn_head_count_hint: Optional[int] = None,

    # precise phase
    warmup_steps_map: Optional[Dict[str, int]] = None,
    total_steps_map: Optional[Dict[str, int]] = None,

    enable_qkv_slicing: bool = True,
) -> List[dict]:
    lr_scales = lr_scales or {"embed":0.5,"norm":0.5,"bias":0.5,"attn_q":0.95,"attn_k":0.85,"attn_v":1.05,"attn_o":1.0,"attn_qkv":1.0,"mlp":1.0,"conv":1.0,"other":1.0}
    lookahead_k_map = lookahead_k_map or {"attn_q":6,"attn_k":6,"attn_v":6,"attn_o":6,"attn_qkv":6,"mlp":6,"conv":4,"embed":10,"norm":10,"bias":10,"other":6}
    lookahead_alpha_map = lookahead_alpha_map or {}
    lookahead_k_mode_map = lookahead_k_mode_map or {}
    lookahead_k_max_map = lookahead_k_max_map or {}
    lookahead_k_depth_mul_map = lookahead_k_depth_mul_map or {}
    lookahead_k_cooldown_syncs_map = lookahead_k_cooldown_syncs_map or {}
    lookahead_stop_after_syncs_map = lookahead_stop_after_syncs_map or {}
    lookahead_stop_when_k_ge_map = lookahead_stop_when_k_ge_map or {}

    lookahead_temp_phase_warm_gain_map = lookahead_temp_phase_warm_gain_map or {}
    lookahead_temp_phase_lr_gain_map   = lookahead_temp_phase_lr_gain_map or {}
    lookahead_temp_phase_ext_gain_map  = lookahead_temp_phase_ext_gain_map or {}
    lookahead_temp_phase_r_gain_map    = lookahead_temp_phase_r_gain_map or {}
    lookahead_temp_phase_rms_gain_map  = lookahead_temp_phase_rms_gain_map or {}
    lookahead_temp_multi_phase_mode_map= lookahead_temp_multi_phase_mode_map or {}
    lookahead_temp_phase_lr_mode_map   = lookahead_temp_phase_lr_mode_map or {}
    lookahead_temp_multi_phase_caps_map= lookahead_temp_multi_phase_caps_map or {}
    lookahead_temp_multi_phase_alphas_map = lookahead_temp_multi_phase_alphas_map or {}
    lookahead_temp_mp_target_map = lookahead_temp_mp_target_map or {}
    lookahead_temp_mp_adapt_map  = lookahead_temp_mp_adapt_map or {}
    lookahead_temp_mp_gcap_map   = lookahead_temp_mp_gcap_map or {}

    rms_clip_threshold_map = rms_clip_threshold_map or {"attn_q":1.0,"attn_k":1.0,"attn_v":1.0,"attn_o":1.0,"attn_qkv":1.0,"mlp":1.0,"conv":0.5,"embed":0.5,"norm":0.5}
    trust_clip_map = trust_clip_map or {"attn_q":(0.1,3.2),"attn_k":(0.1,2.6),"attn_v":(0.1,3.4),"attn_o":(0.1,3.0),"attn_qkv":(0.1,3.0),"mlp":(0.1,3.0),"conv":(0.1,2.0),"embed":(0.2,2.0),"norm":(0.2,2.0)}
    factored_mode_map = factored_mode_map or {"conv":"channel_spatial","mlp":"rowcol","attn_q":"rowcol","attn_k":"rowcol","attn_v":"rowcol","attn_o":"rowcol","attn_qkv":"rowcol","embed":"rowcol","norm":"none","bias":"none","other":"rowcol"}
    use_adafactor_tag_map = use_adafactor_tag_map or {"conv":True,"mlp":True,"attn_q":True,"attn_k":True,"attn_v":True,"attn_o":True,"attn_qkv":True,"embed":False,"norm":False,"bias":False,"other":True}
    beta1_map = beta1_map or {}
    beta2_map = beta2_map or {}

    qkv_beta1_map = qkv_beta1_map or {}
    qkv_beta2_map = qkv_beta2_map or {}
    qkv_trust_clip_map = qkv_trust_clip_map or {}
    qkv_beta2_field_map = qkv_beta2_field_map or {}
    qkv_beta2_reg_map = qkv_beta2_reg_map or {}
    qkv_beta2_reg_sched_map = qkv_beta2_reg_sched_map or {}
    qkv_beta2_reg_pw_depth_map = qkv_beta2_reg_pw_depth_map or {}
    qkv_beta2_reg_depth_spline_map = qkv_beta2_reg_depth_spline_map or {}
    qkv_beta2_reg_depth_bspline_field_map = qkv_beta2_reg_depth_bspline_field_map or {}
    qkv_beta2_reg_depth_head_bspline_field_map = qkv_beta2_reg_depth_head_bspline_field_map or {}
    qkv_beta2_value_depth_head_bspline_field_map = qkv_beta2_value_depth_head_bspline_field_map or {}

    qkv_beta2_value_depth_head_col_bspline_field_map = qkv_beta2_value_depth_head_col_bspline_field_map or {}
    qkv_beta2_reg_depth_head_col_map = qkv_beta2_reg_depth_head_col_map or {}

    qkv_beta2_value_depth_col_bspline_field_map = qkv_beta2_value_depth_col_bspline_field_map or {}
    qkv_beta2_reg_depth_col_bspline_field_map = qkv_beta2_reg_depth_col_bspline_field_map or {}

    conv_beta2_value_spatial_field_map = conv_beta2_value_spatial_field_map or {}
    conv_beta2_reg_spatial_field_map = conv_beta2_reg_spatial_field_map or {}
    conv_beta2_value_channel_field_map = conv_beta2_value_channel_field_map or {}
    conv_beta2_reg_channel_field_map = conv_beta2_reg_channel_field_map or {}

    # map param->module metadata
    param_to_module: Dict[int, Tuple[str, nn.Module, str]] = {}
    for mn, m in model.named_modules():
        for pn, p in m.named_parameters(recurse=False):
            param_to_module[id(p)] = (mn, m, pn)

    grouped: Dict[str, List[torch.nn.Parameter]] = {}
    name_of: Dict[int, str] = {}
    head_hint_per_param: Dict[int, int] = {}
    for full_name, p in model.named_parameters():
        if not p.requires_grad: continue
        if id(p) in param_to_module:
            mn, m, pn = param_to_module[id(p)]
            tag = _guess_tag(mn, m, pn)
            hc = _detect_heads_from_module(m)
            if hc is not None:
                head_hint_per_param[id(p)] = int(hc)
        else:
            tag = "other"
        grouped.setdefault(tag, []).append(p)
        name_of[id(p)] = full_name

    qkv_rules: Dict[int, Tuple[int,int]] = {}
    depth_of: Dict[int, int] = {id(p): _infer_depth_from_name(nm) for nm, p in model.named_parameters()}

    if enable_qkv_slicing and "attn_qkv" in grouped:
        for p in grouped["attn_qkv"]:
            dim = _detect_qkv_split_dim(p)
            if dim is not None:
                qkv_rules[id(p)] = (dim, 3)

    all_depths = [d for d in depth_of.values()] or [0]
    depth_max = max(all_depths)

    param_groups: List[dict] = []
    for tag, plist in grouped.items():
        if not plist: continue
        wd = 0.0 if tag in zero_wd_tags else base_wd
        depths = sorted(depth_of[id(p)] for p in plist)
        group_depth = depths[len(depths)//2] if depths else 0

        hc_counts = collections.Counter([head_hint_per_param.get(id(p), None) for p in plist if head_hint_per_param.get(id(p), None) is not None])
        attn_heads = int(hc_counts.most_common(1)[0][0]) if hc_counts else int(attn_head_count_hint or 1)

        g = {
            "params": plist,
            "lr": base_lr,
            "weight_decay": wd,
            "lr_scale": float(lr_scales.get(tag, 1.0)),
            "block_tag": tag,
            "depth": int(group_depth),
            "attn_heads": int(attn_heads),

            "factored_mode": factored_mode_map.get(tag, "rowcol"),
            "use_adafactor": bool(use_adafactor_tag_map.get(tag, True)),

            "trust_granularity": trust_granularity,
            "rms_clip_granularity": rms_clip_granularity,
            "trust_clip": trust_clip_map.get(tag, (0.1,10.0)),
            "rms_clip_threshold": float(rms_clip_threshold_map.get(tag, 0.0)),

            # Lookahead & temperature gains/caps
            "lookahead_k": int(lookahead_k_map.get(tag, 0)),
            "lookahead_alpha": float(lookahead_alpha_map.get(tag, lookahead_alpha)),
            "lookahead_reset_m": lookahead_reset_m,
            "lookahead_k_mode": lookahead_k_mode_map.get(tag, "fixed"),
            "lookahead_k_max": int(lookahead_k_max_map.get(tag, 1024)),
            "lookahead_k_depth_mul": float(lookahead_k_depth_mul_map.get(tag, 1.0)),
            "lookahead_k_cooldown_syncs": int(lookahead_k_cooldown_syncs_map.get(tag, 1)),
            "lookahead_stop_after_syncs": int(lookahead_stop_after_syncs_map.get(tag, 0)),
            "lookahead_stop_when_k_ge": int(lookahead_stop_when_k_ge_map.get(tag, 0)),

            "lookahead_temp_multi_phase_mode": str(lookahead_temp_multi_phase_mode_map.get(tag, "orthogonal")),
            "lookahead_temp_phase_warm_gain": float(lookahead_temp_phase_warm_gain_map.get(tag, 0.3)),
            "lookahead_temp_phase_lr_gain": float(lookahead_temp_phase_lr_gain_map.get(tag, 0.25)),
            "lookahead_temp_phase_ext_gain": float(lookahead_temp_phase_ext_gain_map.get(tag, 0.0)),
            "lookahead_temp_phase_r_gain": float(lookahead_temp_phase_r_gain_map.get(tag, 0.0)),
            "lookahead_temp_phase_rms_gain": float(lookahead_temp_phase_rms_gain_map.get(tag, 0.0)),
            "lookahead_temp_phase_lr_mode": str(lookahead_temp_phase_lr_mode_map.get(tag, "cosine")),
            "lookahead_temp_multi_phase_caps": tuple(lookahead_temp_multi_phase_caps_map.get(tag, (0.6, 1.0))),
            "lookahead_temp_multi_phase_alphas": tuple(lookahead_temp_multi_phase_alphas_map.get(tag, (0.6, 0.3))),
            "lookahead_temp_mp_target": float(lookahead_temp_mp_target_map.get(tag, 0.65)),
            "lookahead_temp_mp_adapt": float(lookahead_temp_mp_adapt_map.get(tag, 0.05)),
            "lookahead_temp_mp_gcap": float(lookahead_temp_mp_gcap_map.get(tag, 0.6)),

            "_depth_norm_max": int(depth_max),
        }
        if tag in beta1_map: g["beta1"] = float(beta1_map[tag])
        if tag in beta2_map: g["beta2"] = float(beta2_map[tag])

        if tag == "attn_qkv" and qkv_rules:
            g["qkv_rules"] = {pid: spec for pid, spec in qkv_rules.items() if any(id(p) == pid for p in plist)}
            g["qkv_trust_split"] = True
            g["qkv_lr_scales"] = {"q": float(lr_scales.get("attn_q", 1.0)),
                                  "k": float(lr_scales.get("attn_k", 1.0)),
                                  "v": float(lr_scales.get("attn_v", 1.0))}
            if qkv_beta1_map: g["qkv_beta1"] = {k: float(v) for k, v in qkv_beta1_map.items() if k in ("q","k","v")}
            if qkv_beta2_map: g["qkv_beta2"] = {k: float(v) for k, v in qkv_beta2_map.items() if k in ("q","k","v")}
            if qkv_trust_clip_map: g["qkv_trust_clip"] = {k: tuple(v) for k, v in qkv_trust_clip_map.items() if k in ("q","k","v")}
            if qkv_beta2_field_map: g["qkv_beta2_field"] = {k: dict(v) for k, v in qkv_beta2_field_map.items() if k in ("q","k","v")}
            if qkv_beta2_reg_map: g["qkv_beta2_reg"] = {k: dict(v) for k, v in qkv_beta2_reg_map.items() if k in ("q","k","v")}
            if qkv_beta2_reg_sched_map: g["qkv_beta2_reg_sched"] = {k: dict(v) for k, v in qkv_beta2_reg_sched_map.items() if k in ("q","k","v")}
            if qkv_beta2_reg_pw_depth_map: g["qkv_beta2_reg_pw"] = {k: list(v) for k, v in qkv_beta2_reg_pw_depth_map.items() if k in ("q","k","v")}
            # attach VALUE head-row field (X8) if provided
            if qkv_beta2_value_depth_head_bspline_field_map:
                g["qkv_beta2_value_head_field_eval"] = qkv_beta2_value_depth_head_bspline_field_map  # raw spec; row path uses it
                g["qkv_beta2_value_head_field_bounds"] = {"min": float(qkv_beta2_value_depth_head_bspline_field_map.get("min", 0.98)),
                                                          "max": float(qkv_beta2_value_depth_head_bspline_field_map.get("max", 0.9995))}
            # attach VALUE head-col field (new)
            if qkv_beta2_value_depth_head_col_bspline_field_map:
                g["qkv_beta2_value_head_col_spec"] = dict(qkv_beta2_value_depth_head_col_bspline_field_map)
            if qkv_beta2_reg_depth_head_col_map:
                g["qkv_beta2_reg_head_col_map"] = {k: dict(v) for k, v in qkv_beta2_reg_depth_head_col_map.items() if k in ("q","k","v")}
            # fallback column spec
            if qkv_beta2_value_depth_col_bspline_field_map:
                g["qkv_beta2_value_col_spec"] = dict(qkv_beta2_value_depth_col_bspline_field_map)
            if qkv_beta2_reg_depth_col_bspline_field_map:
                g["qkv_beta2_reg_col_spec"] = dict(qkv_beta2_reg_depth_col_bspline_field_map)

        # Conv metadata and specs
        if tag == "conv":
            conv_meta = {}
            for p in plist:
                if id(p) not in param_to_module: continue
                mn, m, pn = param_to_module[id(p)]
                if not isinstance(m, _CONV_TYPES): continue
                meta = {
                    "in_ch": getattr(m, "in_channels", 0),
                    "out_ch": getattr(m, "out_channels", 0),
                    "groups": getattr(m, "groups", 1),
                    "kernel_size": tuple(getattr(m, "kernel_size", (1,)) if isinstance(getattr(m, "kernel_size", (1,)), (tuple,list)) else (getattr(m, "kernel_size", 1),)),
                    "dilation": tuple(getattr(m, "dilation", (1,)) if isinstance(getattr(m, "dilation", (1,)), (tuple,list)) else (getattr(m, "dilation", 1),)),
                }
                conv_meta[id(p)] = meta
            if conv_meta: g["conv_meta"] = conv_meta
            if conv_beta2_value_spatial_field_map:
                g["conv_beta2_value_spatial_spec"] = dict(conv_beta2_value_spatial_field_map)
            if conv_beta2_reg_spatial_field_map:
                g["conv_beta2_reg_spatial_spec"] = dict(conv_beta2_reg_spatial_field_map)
            if conv_beta2_value_channel_field_map:
                g["conv_beta2_value_channel_spec"] = dict(conv_beta2_value_channel_field_map)
            if conv_beta2_reg_channel_field_map:
                g["conv_beta2_reg_channel_spec"] = dict(conv_beta2_reg_channel_field_map)

        # phase hints
        if warmup_steps_map is not None:
            g["_sched_warmup_hint"] = int(warmup_steps_map.get(tag, 0))
        if total_steps_map is not None:
            T_default = int(total_steps_map.get("default", 0))
            T_tag = int(total_steps_map.get(tag, T_default))
            g["_sched_total_hint"] = T_tag

        g["param_names"] = {id(p): name_of[id(p)] for p in plist}
        param_groups.append(g)

    return param_groups

# ─────────────────────────────────────────────────────────────────────────────
# Optimizer
# ─────────────────────────────────────────────────────────────────────────────
class ProdigyUltra(Optimizer):
    def __init__(
        self, params, lr: float = 3e-4, beta1: float = 0.9, weight_decay: float = 0.0,
        trust_ratio: bool = True, trust_clip=(0.1, 10.0), trust_granularity: str = "param",
        use_sign: bool = False, eps: float = 1e-8,
        use_adafactor: bool = False, beta2: float = 0.999, factored_second_moment: bool = True, adafactor_eps2: float = 1e-30,
        lookahead_k: int = 0, lookahead_alpha: float = 0.5, lookahead_reset_m: bool = True,
        rms_clip_threshold: float = 0.0, rms_clip_granularity: str = "param",
        skip_if_nonfinite: bool = True,
    ):
        if lr <= 0: raise ValueError("Invalid lr")
        if not (0.0 <= beta1 < 1.0): raise ValueError("Invalid beta1")
        if weight_decay < 0.0: raise ValueError("Invalid weight_decay")
        if trust_granularity not in ("param","group","tag"): raise ValueError("Invalid trust_granularity")
        if not (0.0 <= beta2 < 1.0): raise ValueError("Invalid beta2")
        if rms_clip_granularity not in ("param","group","tag"): raise ValueError("Invalid rms_clip_granularity")
        defaults = dict(
            lr=lr, beta1=beta1, weight_decay=weight_decay,
            trust_ratio=trust_ratio, trust_clip=trust_clip, trust_granularity=trust_granularity,
            use_sign=use_sign, eps=eps,
            use_adafactor=use_adafactor, beta2=beta2, factored_second_moment=factored_second_moment, adafactor_eps2=adafactor_eps2,
            lookahead_k=lookahead_k, lookahead_alpha=lookahead_alpha, lookahead_reset_m=lookahead_reset_m,
            rms_clip_threshold=rms_clip_threshold, rms_clip_granularity=rms_clip_granularity,
            skip_if_nonfinite=skip_if_nonfinite,
            lr_scale=1.0, block_tag="default", factored_mode="rowcol", depth=0, attn_heads=1,
            lookahead_k_mode="fixed", lookahead_k_max=1024, lookahead_k_depth_mul=1.0,
            lookahead_k_cooldown_syncs=1, lookahead_stop_after_syncs=0, lookahead_stop_when_k_ge=0,
            # multi-phase temperature
            lookahead_temp_multi_phase_mode="orthogonal",
            lookahead_temp_phase_warm_gain=0.3, lookahead_temp_phase_lr_gain=0.25,
            lookahead_temp_phase_ext_gain=0.0, lookahead_temp_phase_r_gain=0.0, lookahead_temp_phase_rms_gain=0.0,
            lookahead_temp_phase_lr_mode="cosine",
            lookahead_temp_multi_phase_caps=(0.6, 1.0),
            lookahead_temp_multi_phase_alphas=(0.6, 0.3),
            lookahead_temp_mp_target=0.65, lookahead_temp_mp_adapt=0.05, lookahead_temp_mp_gcap=0.6,
            _step=0,
        )
        super().__init__(params, defaults)
        self._ext_loss = None
        self._ext_val_loss = None
        self._ext_val_acc = None
        for g in self.param_groups:
            ref_param = next((p for p in g["params"] if isinstance(p, torch.nn.Parameter)), None)
            ref_device = ref_param.device if ref_param is not None else torch.device("cpu")
            if ref_param is not None and ref_param.is_floating_point():
                ref_dtype = torch.float32 if ref_param.dtype in (torch.float16, torch.bfloat16) else ref_param.dtype
            else:
                ref_dtype = torch.float32
            g["_init_lr"] = g["lr"]
            g["_mp_cache_step"] = -1
            g["loss_ema"] = None; g["val_ema"] = None; g["acc_ema"] = None
            # whitening state
            g["_phi_mu"] = torch.zeros(5, dtype=ref_dtype, device=ref_device)
            g["_phi_cov"] = torch.eye(5, dtype=ref_dtype, device=ref_device) * 1e-2
            g["_phi_beta"] = 0.95  # EMA factor
            g["_g_vec"] = torch.tensor([g.get("lookahead_temp_phase_warm_gain",0.3),
                                        g.get("lookahead_temp_phase_lr_gain",0.25),
                                        g.get("lookahead_temp_phase_ext_gain",0.0),
                                        g.get("lookahead_temp_phase_r_gain",0.0),
                                        g.get("lookahead_temp_phase_rms_gain",0.0)], dtype=ref_dtype, device=ref_device)

    def set_ext_metrics(self, *, loss: Optional[float] = None, val_loss: Optional[float] = None, val_acc: Optional[float] = None):
        if loss is not None: self._ext_loss = float(loss)
        if val_loss is not None: self._ext_val_loss = float(val_loss)
        if val_acc is not None: self._ext_val_acc = float(val_acc)

    @torch.no_grad()
    def _adafactor_precondition(self, p, g, state, *, beta2: float, factored: bool, mode: str, eps2: float,
                                qkv_rule=None, qkv_b2_scalar=None, qkv_b2_field=None, qkv_b2_reg=None,
                                attn_heads: int = 1, qkv_b2_reg_head_field=None,
                                qkv_b2_value_head_field=None, qkv_b2_value_bounds=None,
                                # Column 3D
                                qkv_b2_value_head_col_spec=None, qkv_b2_reg_head_col_map=None,
                                # Column depth×col fallback
                                qkv_b2_value_col_spec=None, qkv_b2_reg_col_spec=None, depth_norm: float = 0.0,
                                # Conv spatial & channel
                                conv_spatial_value_spec=None, conv_spatial_reg_spec=None, conv_meta=None, conv_channel_value_spec=None, conv_channel_reg_spec=None):
        # Non-factored branch
        if not factored or p.ndim < 2 or mode == "none":
            if "v" not in state: state["v"] = torch.zeros_like(p, dtype=torch.float32)
            v = state["v"]; gf = g.float()
            if qkv_rule and (qkv_b2_scalar or qkv_b2_field or qkv_b2_value_col_spec is not None or qkv_b2_value_head_col_spec is not None):
                dim, parts = qkv_rule
                gf_chunks = torch.chunk(gf, parts, dim=dim)
                v_chunks  = torch.chunk(v,  parts, dim=dim)
                keys = ("q","k","v")
                outs = []
                for idx, (gfc, vc) in enumerate(zip(gf_chunks, v_chunks)):
                    key = keys[idx]
                    if dim == 1 and qkv_b2_value_head_col_spec is not None:
                        H = max(1, int(attn_heads))
                        vec = _eval_depth_head_col_profile(gfc.shape[1], H, qkv_b2_value_head_col_spec, key, depth_norm, device=gfc.device)
                        # optional per-head REG for columns（L2/TV 同一係数で OK）
                        reg_head = (qkv_b2_reg_head_col_map or {}).get(key, None)
                        if reg_head is not None:
                            spec_bounds = _resolve_per_key_spec(qkv_b2_value_head_col_spec, key)
                            lo_b2, hi_b2 = _beta2_bounds(spec_bounds)
                            vec = _regularize_field_1d(vec, lo_b2, hi_b2, reg_head)
                        vc.mul_(vec).addcmul_(gfc, gfc, value=(1.0 - vec))
                        out = gfc / (vc.sqrt() + eps2)
                    elif dim == 1 and qkv_b2_value_col_spec is not None:
                        b2_vec = _eval_depth_col_profile(qkv_b2_value_col_spec, key, gfc.shape[1], depth_norm, device=gfc.device)
                        reg_spec = (qkv_b2_reg_col_spec or {}).get(key, None)
                        if reg_spec is not None:
                            spec_bounds = _resolve_per_key_spec(qkv_b2_value_col_spec, key)
                            lo_b2, hi_b2 = _beta2_bounds(spec_bounds)
                            b2_vec = _regularize_field_1d(b2_vec, lo_b2, hi_b2, reg_spec)
                        vc.mul_(b2_vec).addcmul_(gfc, gfc, value=(1.0 - b2_vec))
                        out = gfc / (vc.sqrt() + eps2)
                    elif qkv_b2_field and key in qkv_b2_field:
                        spec = qkv_b2_field[key]
                        reg  = (qkv_b2_reg or {}).get(key, None)
                        b2_vec = _eval_beta2_field(gfc.shape[dim], spec, device=gfc.device, reg=reg)
                        shape = [1]*gfc.dim(); shape[dim] = gfc.shape[dim]
                        b2v = b2_vec.view(shape).to(dtype=torch.float32)
                        vc.mul_(b2v).add_(gfc*gfc * (1.0 - b2v))
                        out = gfc / (vc.sqrt() + eps2)
                    else:
                        b2 = float((qkv_b2_scalar or {}).get(key, beta2))
                        vc.mul_(b2).addcmul_(gfc, gfc, value=(1.0 - b2))
                        out = gfc / (vc.sqrt() + eps2)
                    outs.append(out)
                return torch.cat(outs, dim=dim).to(g.dtype)
            else:
                v.mul_(beta2).addcmul_(gf, gf, value=(1.0 - beta2))
                return (gf / (v.sqrt() + eps2)).to(g.dtype)

        # Factored branch
        if mode == "channel_spatial" and p.ndim >= 3:
            o, i = p.shape[0], p.shape[1]
            s = 1
            for d in p.shape[2:]: s *= d
            r = o * i; c = s
            g2d = g.float().reshape(r, c)
            split_affects_rows = (qkv_rule is not None and qkv_rule[0] == 0)
        else:
            r = p.shape[0]; c = p.numel() // r
            g2d = g.float().reshape(r, c)
            split_affects_rows = (qkv_rule is not None and qkv_rule[0] == 0)

        g2 = g2d * g2d
        mean_r = g2.mean(dim=1)
        mean_c = g2.mean(dim=0)

        if "vr" not in state:
            state["vr"] = torch.zeros(r, device=p.device, dtype=torch.float32)
            state["vc"] = torch.zeros(c, device=p.device, dtype=torch.float32)
        vr = state["vr"]; vc = state["vc"]

        if qkv_rule and (qkv_b2_scalar or qkv_b2_field or qkv_b2_value_head_field is not None or qkv_b2_value_head_col_spec is not None or qkv_b2_value_col_spec is not None):
            dim, parts = qkv_rule
            keys = ("q","k","v")
            if split_affects_rows:
                chunk = r // parts
                for idx in range(parts):
                    srow, erow = idx*chunk, (idx+1)*chunk
                    if (qkv_b2_value_head_field is not None) and isinstance(qkv_b2_value_head_field, dict):
                        # NOTE: in X10 we expect head-row VALUE spec raw; fall back to scalar β2 here
                        b2 = float((qkv_b2_scalar or {}).get(keys[idx], beta2))
                        vr[srow:erow] = vr[srow:erow] * b2 + mean_r[srow:erow] * (1.0 - b2)
                    else:
                        b2 = float((qkv_b2_scalar or {}).get(keys[idx], beta2))
                        vr[srow:erow] = vr[srow:erow] * b2 + mean_r[srow:erow] * (1.0 - b2)
                # columns
                if qkv_b2_value_head_col_spec is not None and dim == 1:
                    pass  # handled below in column loop (non-split)
                vc.mul_(beta2).add_(mean_c, alpha=(1.0 - beta2))
            else:
                # Column split → head-aware columns
                chunk = c // parts
                for idx in range(parts):
                    scol, ecol = idx*chunk, (idx+1)*chunk
                    key = keys[idx]
                    if qkv_b2_value_head_col_spec is not None:
                        H = max(1, int(attn_heads))
                        b2_vec = _eval_depth_head_col_profile(ecol-scol, H, qkv_b2_value_head_col_spec, key, depth_norm, device=vc.device)
                        reg_head = (qkv_b2_reg_head_col_map or {}).get(key, None)
                        if reg_head is not None:
                            spec_bounds = _resolve_per_key_spec(qkv_b2_value_head_col_spec, key)
                            lo_b2, hi_b2 = _beta2_bounds(spec_bounds)
                            b2_vec = _regularize_field_1d(b2_vec, lo_b2, hi_b2, reg_head)
                        vc[scol:ecol] = vc[scol:ecol] * b2_vec + mean_c[scol:ecol] * (1.0 - b2_vec)
                    elif qkv_b2_value_col_spec is not None:
                        b2_vec = _eval_depth_col_profile(qkv_b2_value_col_spec, key, ecol-scol, depth_norm, device=vc.device)
                        reg_spec = (qkv_b2_reg_col_spec or {}).get(key, None)
                        if reg_spec is not None:
                            spec_bounds = _resolve_per_key_spec(qkv_b2_value_col_spec, key)
                            lo_b2, hi_b2 = _beta2_bounds(spec_bounds)
                            b2_vec = _regularize_field_1d(b2_vec, lo_b2, hi_b2, reg_spec)
                        vc[scol:ecol] = vc[scol:ecol] * b2_vec + mean_c[scol:ecol] * (1.0 - b2_vec)
                    elif qkv_b2_field and key in qkv_b2_field:
                        spec = qkv_b2_field[key]; reg = (qkv_b2_reg or {}).get(key, None)
                        b2_vec = _eval_beta2_field(ecol-scol, spec, device=vc.device, reg=reg)
                        vc[scol:ecol] = vc[scol:ecol] * b2_vec + mean_c[scol:ecol] * (1.0 - b2_vec)
                    else:
                        b2 = float((qkv_b2_scalar or {}).get(key, beta2))
                        vc[scol:ecol].mul_(b2).add_(mean_c[scol:ecol], alpha=(1.0 - b2))
                vr.mul_(beta2).add_(mean_r, alpha=(1.0 - beta2))
        else:
            # Conv: channel hierarchy (rows) + spatial (cols)
            if mode == "channel_spatial" and p.ndim >= 3 and (conv_spatial_value_spec is not None or conv_channel_value_spec is not None):
                meta = conv_meta or {}
                in_ch = int(meta.get("in_ch", 0)); out_ch = int(meta.get("out_ch", 0))
                groups = max(1, int(meta.get("groups", 1)))
                ksize = tuple(meta.get("kernel_size", (1,1)))
                dil = tuple(meta.get("dilation", (1,1)))
                in_ch_g = max(1, in_ch // groups)
                if conv_channel_value_spec is not None:
                    b2_row = _eval_conv_row_channel_profile(out_ch, in_ch_g, conv_channel_value_spec, depth_norm, device=vr.device)
                    if conv_channel_reg_spec is not None:
                        b2_row = _regularize_field_1d(b2_row, float(conv_channel_value_spec.get("min",0.95)), float(conv_channel_value_spec.get("max",0.9999)), conv_channel_reg_spec)
                    vr.mul_(b2_row).add_(mean_r, alpha=(1.0 - b2_row))
                else:
                    vr.mul_(beta2).add_(mean_r, alpha=(1.0 - beta2))

                if conv_spatial_value_spec is not None:
                    kH = int(ksize[0]); kW = int(ksize[1] if len(ksize) > 1 else 1)
                    b2_col = _eval_spatial_profile_conv(kH, kW, conv_spatial_value_spec, depth_norm, dilation=dil, device=vc.device)
                    if conv_spatial_reg_spec is not None:
                        b2_col = _regularize_field_1d(b2_col, float(conv_spatial_value_spec.get("min",0.95)), float(conv_spatial_value_spec.get("max",0.9999)), conv_spatial_reg_spec)
                    vc.mul_(b2_col).add_(mean_c, alpha=(1.0 - b2_col))
                else:
                    vc.mul_(beta2).add_(mean_c, alpha=(1.0 - beta2))
            else:
                vr.mul_(beta2).add_(mean_r, alpha=(1.0 - beta2))
                vc.mul_(beta2).add_(mean_c, alpha=(1.0 - beta2))

        inv_r = (vr + eps2).rsqrt().unsqueeze(1)
        inv_c = (vc + eps2).rsqrt().unsqueeze(0)
        scale = (vr.mean() + eps2).sqrt()
        g_pre = (g2d * inv_r * inv_c) * scale
        return g_pre.reshape_as(g).to(g.dtype)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        tag_p2: Dict[str, float] = {}
        tag_m2: Dict[str, float] = {}
        tag_u2: Dict[str, float] = {}
        tag_cnt: Dict[str, int]   = {}
        group_pm2: Dict[int, Tuple[float, float]] = {}
        group_u2: Dict[int, Tuple[float, int]] = {}

        # pass 1 (moments & preconditioning)
        for group in self.param_groups:
            group["_step"] += 1
            beta1 = group.get("beta1", self.defaults["beta1"])
            use_trust = group["trust_ratio"]; gran = group["trust_granularity"]
            use_sign = group["use_sign"]
            use_ada = group["use_adafactor"]; beta2 = group.get("beta2", self.defaults["beta2"])
            factored = group["factored_second_moment"]; mode = group.get("factored_mode", "rowcol")
            eps2 = group["adafactor_eps2"]

            depth = int(group.get("depth", 0))
            dmax = int(group.get("_depth_norm_max", 0))
            d_idx = max(0, min(dmax, depth))
            depth_norm = 0.0 if dmax <= 0 else (d_idx / float(dmax))

            for p in group["params"]:
                if p.grad is None: continue
                if p.grad.is_sparse: raise RuntimeError("sparse gradients not supported.")
                st = self.state[p]; g = p.grad

                qkv_rule = None; qkv_b2_scalar = None; qkv_b2_field = None; qkv_b2_reg = None
                qkv_b2_reg_head = None; qkv_b2_value_head = None; qkv_b2_value_bounds = None
                qkv_b2_value_col_spec = None; qkv_b2_reg_col_spec = None
                qkv_b2_value_head_col_spec = None; qkv_b2_reg_head_col_map = None
                conv_spatial_value_spec = None; conv_spatial_reg_spec = None
                conv_channel_value_spec = None; conv_channel_reg_spec = None; conv_meta = None

                if "qkv_rules" in group and id(p) in group["qkv_rules"]:
                    qkv_rule = group["qkv_rules"][id(p)]
                    qkv_b2_scalar = group.get("qkv_beta2", None)
                    qkv_b2_field  = group.get("qkv_beta2_field", None)

                    # REG scheduling (compat)
                    sched_map = group.get("qkv_beta2_reg_sched", None)
                    base_reg = group.get("qkv_beta2_reg", None)
                    pw_depth = group.get("qkv_beta2_reg_pw", None)
                    depth_eval = group.get("qkv_beta2_reg_depth_eval", None)
                    depth_field = group.get("qkv_beta2_reg_depth_field_eval", None)
                    depth_field_mode = group.get("qkv_beta2_reg_depth_field_mode", None)
                    depth_head_field = group.get("qkv_beta2_reg_depth_head_field_eval", None)
                    depth_head_field_mode = group.get("qkv_beta2_reg_depth_head_field_mode", None)

                    if (sched_map is not None) or (base_reg is not None) or (pw_depth is not None) or (depth_eval is not None) or (depth_field is not None) or (depth_head_field is not None):
                        qkv_b2_reg = {}
                        for key in ("q","k","v"):
                            reg = _schedule_reg(depth, int(group["_step"]), (base_reg or {}).get(key, None), (sched_map or {}).get(key, None))
                            reg = _apply_piecewise_reg_by_depth(depth, reg, (pw_depth or {}).get(key, None))
                            if depth_eval and (key in depth_eval):
                                de = depth_eval[key]
                                if "l2" in de: reg["l2"] = float(de["l2"]) if de.get("l2_mode","override")=="override" else reg.get("l2",0.0) * float(de["l2"])
                                if "tv" in de: reg["tv"] = float(de["tv"]) if de.get("tv_mode","override")=="override" else reg.get("tv",0.0) * float(de["tv"])
                            if depth_field:
                                fe = depth_field; fm = depth_field_mode or {}
                                if "l2" in fe and key in fe["l2"]:
                                    val = float(fe["l2"][key][d_idx]); reg["l2"] = val if fm.get("l2","override")=="override" else reg.get("l2",0.0)*val
                                if "tv" in fe and key in fe["tv"]:
                                    val = float(fe["tv"][key][d_idx]); reg["tv"] = val if fm.get("tv","override")=="override" else reg.get("tv",0.0)*val
                            if depth_head_field:
                                # rows head REG remains for compatibility
                                pass
                            qkv_b2_reg[key] = reg

                    # VALUE (rows head) – keep raw spec for compatibility
                    if "qkv_beta2_value_head_field_eval" in group:
                        qkv_b2_value_head = group["qkv_beta2_value_head_field_eval"]
                        qkv_b2_value_bounds = dict(group.get("qkv_beta2_value_head_field_bounds", {}))
                        qkv_b2_value_bounds["depth_idx"] = d_idx

                    # Column VALUE/REG
                    qkv_b2_value_head_col_spec = group.get("qkv_beta2_value_head_col_spec", None)
                    qkv_b2_reg_head_col_map    = group.get("qkv_beta2_reg_head_col_map", None)
                    qkv_b2_value_col_spec      = group.get("qkv_beta2_value_col_spec", None)
                    qkv_b2_reg_col_spec        = group.get("qkv_beta2_reg_col_spec", None)

                if group.get("block_tag","") == "conv":
                    conv_spatial_value_spec = group.get("conv_beta2_value_spatial_spec", None)
                    conv_spatial_reg_spec   = group.get("conv_beta2_reg_spatial_spec", None)
                    conv_channel_value_spec = group.get("conv_beta2_value_channel_spec", None)
                    conv_channel_reg_spec   = group.get("conv_beta2_reg_channel_spec", None)
                    conv_meta_map = group.get("conv_meta", None)
                    if conv_meta_map and id(p) in conv_meta_map: conv_meta = conv_meta_map[id(p)]

                g_eff = self._adafactor_precondition(
                    p, g, st, beta2=beta2, factored=factored and use_ada, mode=mode, eps2=eps2,
                    qkv_rule=qkv_rule, qkv_b2_scalar=qkv_b2_scalar, qkv_b2_field=qkv_b2_field, qkv_b2_reg=qkv_b2_reg,
                    attn_heads=int(group.get("attn_heads", 1)), qkv_b2_reg_head_field=None,
                    qkv_b2_value_head_field=qkv_b2_value_head, qkv_b2_value_bounds=qkv_b2_value_bounds,
                    qkv_b2_value_head_col_spec=qkv_b2_value_head_col_spec, qkv_b2_reg_head_col_map=qkv_b2_reg_head_col_map,
                    qkv_b2_value_col_spec=qkv_b2_value_col_spec, qkv_b2_reg_col_spec=qkv_b2_reg_col_spec, depth_norm=float(depth_norm),
                    conv_spatial_value_spec=conv_spatial_value_spec, conv_spatial_reg_spec=conv_spatial_reg_spec, conv_meta=conv_meta,
                    conv_channel_value_spec=conv_channel_value_spec, conv_channel_reg_spec=conv_channel_reg_spec
                ) if use_ada else g

                if "m" not in st: st["m"] = torch.zeros_like(p)
                m = st["m"]
                if qkv_rule and group.get("qkv_beta1", None):
                    dim, parts = qkv_rule
                    g_chunks = torch.chunk(g_eff, parts, dim=dim)
                    m_chunks = torch.chunk(m, parts, dim=dim)
                    keys = ("q","k","v")
                    for idx, (gc, mc) in enumerate(zip(g_chunks, m_chunks)):
                        b1 = float(group["qkv_beta1"].get(keys[idx], beta1))
                        mc.mul_(b1).add_(gc, alpha=1.0 - b1)
                else:
                    m.mul_(beta1).add_(g_eff, alpha=1.0 - beta1)

                if use_trust and gran in ("group","tag"):
                    pf = p.float(); mf = m.float()
                    p2 = float((pf*pf).sum().item()); m2 = float((mf*mf).sum().item())
                    if gran == "group":
                        gid = id(group); P,M = group_pm2.get(gid,(0.0,0.0))
                        group_pm2[gid] = (P+p2, M+m2)
                    else:
                        tag = group.get("block_tag","default")
                        tag_p2[tag] = tag_p2.get(tag,0.0) + p2
                        tag_m2[tag] = tag_m2.get(tag,0.0) + m2

                thr = float(group.get("rms_clip_threshold", 0.0))
                rmsg = group.get("rms_clip_granularity","param")
                if thr > 0.0 and rmsg in ("group","tag"):
                    u = m.sign() if use_sign else m
                    u2 = float((u.float()*u.float()).sum().item()); n = u.numel()
                    if rmsg == "group":
                        gid = id(group); U,N = group_u2.get(gid,(0.0,0))
                        group_u2[gid] = (U+u2, N+n)
                    else:
                        tag = group.get("block_tag","default")
                        tag_u2[tag] = tag_u2.get(tag,0.0) + u2
                        tag_cnt[tag] = tag_cnt.get(tag,0) + n

        # aggregates
        tag_r: Dict[str,float] = {}
        for t, p2 in tag_p2.items():
            m2 = tag_m2.get(t,0.0)
            tag_r[t] = (math.sqrt(p2)/(math.sqrt(m2)+1e-8)) if (p2>0.0 and m2>0.0) else 1.0

        group_r: Dict[int,float] = {}
        for group in self.param_groups:
            if not group["trust_ratio"] or group["trust_granularity"] != "group": continue
            tr_min, tr_max = group["trust_clip"]; eps = group["eps"]
            p2, m2 = group_pm2.get(id(group),(0.0,0.0))
            if p2>0.0 and m2>0.0:
                raw = math.sqrt(p2)/(math.sqrt(m2)+eps)
                group_r[id(group)] = max(tr_min, min(tr_max, raw))
            else:
                group_r[id(group)] = 1.0

        tag_rms: Dict[str,float] = {t: math.sqrt(tag_u2[t]/max(1,tag_cnt.get(t,0))) for t in tag_u2}
        group_rms: Dict[int,float] = {}
        for g in self.param_groups:
            if id(g) in group_u2:
                U,N = group_u2[id(g)]
                group_rms[id(g)] = math.sqrt(U/max(1,N))

        # pass 2 (apply update + lookahead & temperature)
        for group in self.param_groups:
            base_lr = group["lr"]; lr_scale = float(group.get("lr_scale",1.0))
            eff_lr_base = base_lr * lr_scale

            wd = group["weight_decay"]; use_trust = group["trust_ratio"]; gran = group["trust_granularity"]
            tr_min, tr_max = group["trust_clip"]; use_sign = group["use_sign"]; eps = group["eps"]; skip_bad = group["skip_if_nonfinite"]

            thr = float(group.get("rms_clip_threshold", 0.0)); rmsg = group.get("rms_clip_granularity","param")
            clip_group = clip_tag = 1.0
            if thr > 0.0:
                if rmsg == "group":
                    gr = group_rms.get(id(group),0.0); clip_group = min(1.0, (thr/gr) if gr>0.0 else 1.0)
                elif rmsg == "tag":
                    trms = tag_rms.get(group.get("block_tag","default"),0.0); clip_tag = min(1.0, (thr/trms) if trms>0.0 else 1.0)

            tag = group.get("block_tag","default")
            qkv_rules = group.get("qkv_rules",{})
            qkv_trust_clip = group.get("qkv_trust_clip", None)
            qkv_lr_scales = group.get("qkv_lr_scales", None)

            la_mode = group.get("lookahead_k_mode","fixed")
            la_max  = int(group.get("lookahead_k_max",1024))
            depth_mul = float(group.get("lookahead_k_depth_mul",1.0))
            base_k = int(group.get("lookahead_k",0))
            depth  = int(group.get("depth",0))
            init_k = max(0, int(round(base_k * (depth_mul ** depth))))
            la_cd  = max(1, int(group.get("lookahead_k_cooldown_syncs",1)))
            la_stop_syncs = int(group.get("lookahead_stop_after_syncs",0))
            la_stop_k_ge  = int(group.get("lookahead_stop_when_k_ge",0))

            # phases
            s = int(group.get("_step", 0))
            w_eff = int(group.get("_sched_warmup_eff", group.get("_sched_warmup", group.get("_sched_warmup_hint", 0))))
            T_hint = int(group.get("_sched_total", group.get("_sched_total_hint", max(1, s))))
            phase_linear = min(1.0, s / max(1, T_hint))
            if s <= w_eff and w_eff > 0:
                phase_two = 0.5 * (s / max(1, w_eff))
            else:
                phase_two = 0.5 + 0.5 * max(0.0, (s - w_eff)) / max(1, (T_hint - w_eff))
            phase_linear_cos = 0.5 * (1 - math.cos(math.pi * phase_linear))
            phase_two_cos    = 0.5 * (1 - math.cos(math.pi * phase_two))

            # multi-frequency φ
            if group["_mp_cache_step"] != s:
                # φ_w
                phi_w = phase_two_cos if group.get("lookahead_temp_phase_lr_mode","cosine") == "cosine" else phase_two
                # φ_l
                lr_mode = group.get("lookahead_temp_phase_lr_mode","cosine")
                if lr_mode == "lr_ratio" and group.get("_init_lr", 0.0) > 0.0:
                    lr_ratio = float(group["lr"]) / float(group.get("_init_lr", 1.0))
                    phi_l = max(0.0, min(1.0, 1.0 - lr_ratio))
                elif lr_mode == "cosine":
                    phi_l = phase_linear_cos
                else:
                    phi_l = phase_linear
                # φ_ext
                phi_e = 0.0
                loss_beta = 0.9; val_beta = 0.9; acc_beta = 0.9
                if self._ext_loss is not None:
                    if group["loss_ema"] is None: group["loss_ema"] = float(self._ext_loss)
                    group["loss_ema"] = loss_beta * group["loss_ema"] + (1 - loss_beta) * float(self._ext_loss)
                    if group["loss_ema"] > 0: phi_e += max(0.0, (float(self._ext_loss) - group["loss_ema"]) / (group["loss_ema"] + 1e-8))
                if self._ext_val_loss is not None:
                    if group["val_ema"] is None: group["val_ema"] = float(self._ext_val_loss)
                    group["val_ema"] = val_beta * group["val_ema"] + (1 - val_beta) * float(self._ext_val_loss)
                    if group["val_ema"] > 0: phi_e += max(0.0, (float(self._ext_val_loss) - group["val_ema"]) / (group["val_ema"] + 1e-8))
                if self._ext_val_acc is not None:
                    if group["acc_ema"] is None: group["acc_ema"] = float(self._ext_val_acc)
                    group["acc_ema"] = acc_beta * group["acc_ema"] + (1 - acc_beta) * float(self._ext_val_acc)
                    phi_e += max(0.0, (group["acc_ema"] - float(self._ext_val_acc)))
                phi_e = min(1.0, phi_e)

                # φ_r（trust ratio）
                r_basis = group_r.get(id(group), tag_r.get(tag, 1.0))
                phi_r = min(1.0, abs(r_basis - 1.0) / 0.3)

                # φ_u（rms update）
                if id(group) in group_rms and group.get("rms_clip_threshold",0.0) > 0.0:
                    tgt =  group.get("lookahead_temp_rms_target", 0.8) if "lookahead_temp_rms_target" in group else 0.8
                    rms = group_rms[id(group)]
                    phi_u = max(0.0, (rms - tgt) / max(1e-8, tgt))
                    phi_u = min(1.0, phi_u)
                else:
                    phi_u = 0.0

                # online whitening（5-dim）
                stats_device = group["_phi_mu"].device
                stats_dtype = group["_phi_mu"].dtype
                phi = torch.tensor([phi_w, phi_l, phi_e, phi_r, phi_u], dtype=stats_dtype, device=stats_device)
                mu = group["_phi_mu"]; C = group["_phi_cov"]; beta = float(group["_phi_beta"])
                mu_new = beta * mu + (1-beta) * phi
                delta = (phi - mu).unsqueeze(1)
                C_new = beta * C + (1-beta) * (delta @ delta.T)
                group["_phi_mu"] = mu_new; group["_phi_cov"] = C_new
                # inv sqrt via Cholesky on diag-boosted covariance
                Cov = C_new + torch.eye(5, dtype=stats_dtype, device=stats_device) * 1e-6
                try:
                    L = torch.linalg.cholesky(Cov)
                    delta_new = (phi - mu_new).unsqueeze(-1)
                    z = torch.linalg.solve(L, delta_new).squeeze(-1)
                except (RuntimeError, ValueError):
                    z = torch.zeros_like(phi)

                # adaptive gains
                g_vec = group["_g_vec"]
                mp_target = float(group.get("lookahead_temp_mp_target", 0.65))
                adapt = float(group.get("lookahead_temp_mp_adapt", 0.05))
                gcap  = float(group.get("lookahead_temp_mp_gcap", 0.6))

                # compute orthogonal mp_gain
                mp_gain = torch.sqrt(torch.sum((g_vec * z) * (g_vec * z))).item()
                err = mp_target - mp_gain
                if abs(err) > 1e-6:
                    step = adapt * err
                    g_vec = torch.clamp(g_vec + step * torch.abs(z), min=0.0, max=gcap)
                    group["_g_vec"] = g_vec

                cap1, cap2 = group.get("lookahead_temp_multi_phase_caps", (0.6, 1.0))
                a1, a2 = group.get("lookahead_temp_multi_phase_alphas", (0.6, 0.3))
                mp_gain_val = float(torch.sqrt(torch.sum((g_vec * z) * (g_vec * z))).item())
                if mp_gain_val > cap1: mp_gain_val = cap1 + (mp_gain_val - cap1) * a1
                if mp_gain_val > cap2: mp_gain_val = cap2 + (mp_gain_val - cap2) * a2

                group["_mp_gain_cache"] = float(mp_gain_val)
                group["_mp_cache_step"] = s

            mp_gain = float(group.get("_mp_gain_cache", 0.0))

            for p in group["params"]:
                if p.grad is None: continue
                st = self.state[p]
                m = st["m"]
                if skip_bad and (not torch.isfinite(m.float()).all() or not torch.isfinite(p.float()).all()): continue

                r = 1.0
                if use_trust:
                    if gran == "param":
                        p_n = torch.linalg.vector_norm(p.float()).item(); m_n = torch.linalg.vector_norm(m.float()).item()
                        if p_n>0.0 and m_n>0.0: r = max(tr_min, min(tr_max, p_n/(m_n+eps)))
                    elif gran == "group":
                        r = group_r.get(id(group),1.0)
                    else:
                        r = max(tr_min, min(tr_max, tag_r.get(tag,1.0)))

                if wd != 0.0:
                    p.add_(p, alpha=-(eff_lr_base * wd))

                update = m.sign() if use_sign else m

                clip = 1.0
                if thr > 0.0:
                    if rmsg == "param":
                        rms = torch.linalg.vector_norm(update.float()).item()/math.sqrt(update.numel())
                        if rms>0.0: clip = min(1.0, thr/rms)
                    elif rmsg == "group":
                        clip = clip_group
                    else:
                        clip = clip_tag

                rule = qkv_rules.get(id(p))
                if rule and use_trust:
                    dim, parts = rule
                    p_chunks = torch.chunk(p, parts, dim=dim)
                    m_chunks = torch.chunk(m, parts, dim=dim)
                    u_chunks = torch.chunk(update, parts, dim=dim)
                    keys = ("q","k","v")
                    for idx, (pc, mc, uc) in enumerate(zip(p_chunks, m_chunks, u_chunks)):
                        p_n = torch.linalg.vector_norm(pc.float()).item(); m_n = torch.linalg.vector_norm(mc.float()).item()
                        raw_i = (p_n/(m_n+eps)) if (p_n>0.0 and m_n>0.0) else 1.0
                        if qkv_trust_clip and idx < 3:
                            tmin,tmax = qkv_trust_clip.get(keys[idx], (tr_min,tr_max))
                            r_i = max(tmin, min(tmax, raw_i))
                        else:
                            r_i = max(tr_min, min(tr_max, raw_i))

                        eff_lr_slice = eff_lr_base
                        if qkv_lr_scales is not None and idx < 3:
                            eff_lr_slice = base_lr * float(qkv_lr_scales.get(keys[idx], lr_scale))

                        if thr > 0.0 and rmsg == "param":
                            rms_i = torch.linalg.vector_norm(uc.float()).item()/math.sqrt(uc.numel())
                            clip_i = min(1.0, thr/rms_i) if rms_i>0.0 else 1.0
                        else:
                            clip_i = clip
                        pc.add_(uc, alpha=-(eff_lr_slice * r_i * clip_i))
                else:
                    p.add_(update, alpha=-(eff_lr_base * r * clip))

                # Lookahead sync & growth with mp_gain
                la_disabled = st.get("la_disabled", False)
                la_k_curr = st.get("la_k_current", None)
                if la_k_curr is None:
                    la_k_curr = max(0, int(round(base_k * (depth_mul ** depth))))
                    st["la_k_current"] = la_k_curr
                    st["la_counter"] = 0
                    st["la_since_growth"] = 0
                    st["la_syncs_done"] = 0
                    st["la_growth_events"] = 0

                if not la_disabled and la_k_curr and la_k_curr > 0:
                    cnt = st.get("la_counter", 0) + 1
                    st["la_counter"] = cnt
                    if cnt % la_k_curr == 0:
                        if "slow_buffer" not in st:
                            st["slow_buffer"] = p.detach().clone()
                        slow = st["slow_buffer"]
                        alpha = float(group.get("lookahead_alpha", 0.5))
                        reset_m = bool(group.get("lookahead_reset_m", True))
                        slow.add_(p - slow, alpha=alpha)
                        p.copy_(slow)
                        if reset_m: st["m"].zero_()

                        st["la_syncs_done"] = st.get("la_syncs_done", 0) + 1
                        st["la_since_growth"] = st.get("la_since_growth", 0) + 1
                        st["la_counter"] = 0

                        if la_stop_syncs > 0 and st["la_syncs_done"] >= la_stop_syncs:
                            st["la_disabled"] = True; continue
                        if la_stop_k_ge > 0 and la_k_curr >= la_stop_k_ge:
                            st["la_disabled"] = True; continue

                        if st["la_since_growth"] >= int(group.get("lookahead_k_cooldown_syncs",1)) and group.get("lookahead_k_mode","fixed") in ("doubling","temperature"):
                            ev = st.get("la_growth_events", 0)
                            # base temp schedule
                            temp0 = 1.0; temp_min = 0.2; temp_decay = 0.9; temp_max = 3.0
                            tmp = float( max(temp_min, temp0 * (temp_decay ** ev)) )
                            # multi-frequency (whitened) mp_gain
                            temp = max(0.1, min(temp_max, 1.0 + mp_gain))
                            growth = 1.0 + temp * tmp
                            new_k = min(la_max, max(1, int(math.ceil(la_k_curr * growth))))
                            if new_k != la_k_curr:
                                la_k_curr = new_k
                                st["la_k_current"] = la_k_curr
                                st["la_growth_events"] = st.get("la_growth_events", 0) + 1
                            st["la_since_growth"] = 0

        self._ext_loss = None; self._ext_val_loss = None; self._ext_val_acc = None
        return loss

# ─────────────────────────────────────────────────────────────────────────────
# Scheduler（簡略：X9 と同等のタグ別 warmup）
# ─────────────────────────────────────────────────────────────────────────────
class TagAwareWarmupLR(_LRScheduler):
    def __init__(
        self, optimizer,
        *, default_warmup_steps: int = 1000, default_total_steps: int = 100_000,
        warmup_steps_map: Optional[Dict[str, int]] = None,
        total_steps_map: Optional[Dict[str, int]] = None,
        last_epoch: int = -1,
    ):
        self.default_warmup = max(0, int(default_warmup_steps))
        self.default_total  = max(1, int(default_total_steps))
        self.warmup_map = warmup_steps_map or {}
        self.total_map  = total_steps_map or {}

        for g in optimizer.param_groups:
            tag = g.get("block_tag","default")
            w = int(self.warmup_map.get(tag, self.default_warmup))
            g["_sched_warmup"] = w
            T = int(self.total_map.get(tag, self.default_total))
            g["_sched_total"]  = T

        super().__init__(optimizer, last_epoch=last_epoch)

    def _warmup_scale_for_group(self, group, step: int) -> float:
        w = int(group.get("_sched_warmup", 0) or 0)
        if step < w: return step / max(1, w)
        T = int(group.get("_sched_total", 1)); t = min(1.0, (step - w) / max(1, T - w))
        return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * t))

    def get_lr(self):
        s = max(0, self.last_epoch + 1)
        scales = [ self._warmup_scale_for_group(g, s) for g in self.optimizer.param_groups ]
        return [ base_lr * sc for base_lr, sc in zip(self.base_lrs, scales) ]

# ─────────────────────────────────────────────────────────────────────────────
# Course
# ─────────────────────────────────────────────────────────────────────────────
COURSES: Dict[str, dict] = {
    "transformer_spline_tempX10": {
        "desc": "Col head-aware + Conv channel hierarchy + freq-orth temp (auto-gain).",
        "base_lr": 3e-4, "base_wd": 0.01, "total_steps": 120_000,
        "use_sign": True, "use_adafactor_default": True, "beta1_default": 0.9, "beta2_default": 0.999,
        "trust_granularity": "tag", "rms_clip_granularity": "tag",
        "trust_clip_default": (0.1, 3.0),
        "trust_clip_map": {"attn_q": (0.1, 3.2), "attn_k": (0.1, 2.6), "attn_v": (0.1, 3.6),
                           "attn_o": (0.1, 3.0), "mlp": (0.1, 3.0), "embed": (0.2, 2.0), "norm": (0.2, 2.0), "conv": (0.1,2.5)},
        "rms_clip_threshold_map": {"attn_q": 1.0, "attn_k": 1.0, "attn_v": 1.0, "attn_o": 1.0, "mlp": 1.0, "embed": 0.5, "norm": 0.5, "conv": 0.8},
        "lr_scales": {"embed": 0.5, "norm": 0.5, "bias": 0.5, "attn_q": 0.95, "attn_k": 0.85, "attn_v": 1.05, "attn_o": 1.0, "mlp": 1.0, "conv":1.0},

        # Lookahead / freq-orth temp (auto-gain)
        "lookahead_k_map": {"attn_q": 6, "attn_k": 6, "attn_v": 6, "attn_o": 6, "mlp": 6, "embed": 10, "norm": 10, "conv": 6},
        "lookahead_alpha_map": {"attn_q": 0.6, "attn_k": 0.6, "attn_v": 0.6, "attn_o": 0.55, "mlp": 0.55, "embed": 0.5, "norm": 0.5, "conv":0.55},
        "lookahead_k_mode_map": {"attn_q":"temperature","attn_k":"temperature","attn_v":"temperature","attn_o":"fixed","mlp":"fixed","conv":"temperature"},
        "lookahead_k_max_map": {"attn_q": 64, "attn_k": 64, "attn_v": 64, "conv":48},
        "lookahead_k_depth_mul_map": {"attn_q": 1.1, "attn_k": 1.1, "attn_v": 1.1, "conv":1.05},
        "lookahead_k_cooldown_syncs_map": {"attn_q":2, "attn_k":2, "attn_v":2, "conv":3},
        "lookahead_stop_after_syncs_map": {"attn_q":32, "attn_k":32, "attn_v":32, "conv":24},
        "lookahead_stop_when_k_ge_map": {"attn_q":64, "attn_k":64, "attn_v":64, "conv":48},

        "lookahead_temp_multi_phase_mode_map": {"attn_q":"orthogonal","attn_k":"orthogonal","attn_v":"orthogonal","conv":"orthogonal"},
        "lookahead_temp_phase_warm_gain_map": {"attn_q":0.30,"attn_k":0.30,"attn_v":0.32,"conv":0.22},
        "lookahead_temp_phase_lr_gain_map":   {"attn_q":0.22,"attn_k":0.22,"attn_v":0.22,"conv":0.18},
        "lookahead_temp_phase_ext_gain_map":  {"attn_q":0.20,"attn_k":0.18,"attn_v":0.18,"conv":0.10},
        "lookahead_temp_phase_r_gain_map":    {"attn_q":0.16,"attn_k":0.16,"attn_v":0.16,"conv":0.12},
        "lookahead_temp_phase_rms_gain_map":  {"attn_q":0.16,"attn_k":0.16,"attn_v":0.16,"conv":0.12},
        "lookahead_temp_phase_lr_mode_map": {"attn_q":"cosine","attn_k":"cosine","attn_v":"cosine","conv":"cosine"},
        "lookahead_temp_multi_phase_caps_map": {"attn_q":(0.6,1.1),"attn_k":(0.6,1.1),"attn_v":(0.6,1.1),"conv":(0.5,1.0)},
        "lookahead_temp_multi_phase_alphas_map": {"attn_q":(0.6,0.3),"attn_k":(0.6,0.3),"attn_v":(0.6,0.3),"conv":(0.6,0.3)},
        "lookahead_temp_mp_target_map": {"attn_q":0.65,"attn_k":0.65,"attn_v":0.65,"conv":0.60},
        "lookahead_temp_mp_adapt_map": {"attn_q":0.05,"attn_k":0.05,"attn_v":0.05,"conv":0.04},
        "lookahead_temp_mp_gcap_map": {"attn_q":0.6,"attn_k":0.6,"attn_v":0.6,"conv":0.5},

        # QKV β2 fields
        "qkv_beta1_map": {"q":0.92,"k":0.94,"v":0.88},
        "qkv_beta2_map": {"q":0.9988,"k":0.9990,"v":0.9983},
        "qkv_beta2_field_map": {
            "q": {"degree":3,"ctrl":[0.9982,0.9988,0.9992], "min":0.9978,"max":0.9995},
            "k": {"degree":2,"ctrl":[0.9992,0.9990,0.9989], "min":0.9982,"max":0.9997},
            "v": {"degree":1,"ctrl":[0.9980,0.9987],        "min":0.9976,"max":0.9996},
        },
        # Column head-aware VALUE/REG（新）
        "qkv_beta2_value_depth_head_col_bspline_field_map": {
            "degree": 3, "ctrl":{"q":[0.9986,0.9990,0.9993], "k":[0.9991,0.9990,0.9988], "v":[0.9983,0.9988,0.9992]},
            "head": {"degree": 2, "ctrl": {"q":[1.05,1.0,0.95,0.92], "k":[1.0,1.0,0.98,0.96], "v":[1.06,1.0,0.97,0.94]}},
            "col":  {"degree": 2, "ctrl": {"q":[0.95,1.0,1.05], "k":[1.0,1.0,0.98], "v":[1.05,1.0,0.96]}},
            "min":0.9975, "max":0.9997, "combine":"mul", "normalized":True
        },
        "qkv_beta2_reg_depth_head_col_map": {  # 各 head に同一の L2/TV を適用（簡潔）
            "q": {"l2":0.05, "tv":0.03, "iters":2, "step":0.25},
            "k": {"l2":0.05, "tv":0.03, "iters":2, "step":0.25},
            "v": {"l2":0.05, "tv":0.025,"iters":2, "step":0.25}
        },

        # Conv spatial & channel hierarchy
        "conv_beta2_value_spatial_field_map": {"min":0.9975,"max":0.9997,"combine":"mul",
            "depth":{"degree":1,"ctrl":[1.0,1.0]},
            "kh":{"degree":2,"ctrl":[0.95,1.0,1.05]},
            "kw":{"degree":2,"ctrl":[0.95,1.0,1.05]},
            "dilation_weight":{"alpha":0.12,"power":1.0}
        },
        "conv_beta2_reg_spatial_field_map": {"l2":0.04,"tv":0.02,"iters":2,"step":0.25},
        "conv_beta2_value_channel_field_map": {"min":0.9975,"max":0.9997,"combine":"mul",
            "out":{"degree":2,"ctrl":[0.98,1.0,1.02]},
            "in":{"degree":2,"ctrl":[1.02,1.0,0.98]},
            "depth":{"degree":1,"ctrl":[1.0,1.0]}
        },
        "conv_beta2_reg_channel_field_map": {"l2":0.04,"tv":0.02,"iters":2,"step":0.25},

        # Warmup
        "warmup_default": 2000,
        "warmup_steps_map": {"attn_q": 3200, "attn_k": 3200, "attn_v": 3200, "attn_o": 2800, "mlp": 2500, "embed": 1500, "norm": 1500, "conv": 2400},
    },
}

def apply_chef_course(
    model: nn.Module,
    course: str = "transformer_spline_tempX10",
    *, base_lr: Optional[float] = None, base_wd: Optional[float] = None, total_steps: Optional[int] = None,
):
    if course not in COURSES:
        raise KeyError(f"Unknown course: {course}. Available: {list(COURSES.keys())}")
    C = COURSES[course]
    base_lr = float(base_lr if base_lr is not None else C["base_lr"])
    base_wd = float(base_wd if base_wd is not None else C["base_wd"])
    total_steps = int(total_steps if total_steps is not None else C["total_steps"])

    groups = build_param_groups(
        model,
        base_lr=base_lr, base_wd=base_wd,
        lr_scales=C["lr_scales"],

        lookahead_k_map=C["lookahead_k_map"], lookahead_alpha=0.5, lookahead_alpha_map=C["lookahead_alpha_map"],
        lookahead_k_mode_map=C["lookahead_k_mode_map"],
        lookahead_k_max_map=C.get("lookahead_k_max_map", None),
        lookahead_k_depth_mul_map=C.get("lookahead_k_depth_mul_map", None),
        lookahead_k_cooldown_syncs_map=C.get("lookahead_k_cooldown_syncs_map", None),
        lookahead_stop_after_syncs_map=C.get("lookahead_stop_after_syncs_map", None),
        lookahead_stop_when_k_ge_map=C.get("lookahead_stop_when_k_ge_map", None),

        lookahead_temp_phase_warm_gain_map=C.get("lookahead_temp_phase_warm_gain_map", None),
        lookahead_temp_phase_lr_gain_map=C.get("lookahead_temp_phase_lr_gain_map", None),
        lookahead_temp_phase_ext_gain_map=C.get("lookahead_temp_phase_ext_gain_map", None),
        lookahead_temp_phase_r_gain_map=C.get("lookahead_temp_phase_r_gain_map", None),
        lookahead_temp_phase_rms_gain_map=C.get("lookahead_temp_phase_rms_gain_map", None),
        lookahead_temp_multi_phase_mode_map=C.get("lookahead_temp_multi_phase_mode_map", None),
        lookahead_temp_phase_lr_mode_map=C.get("lookahead_temp_phase_lr_mode_map", None),
        lookahead_temp_multi_phase_caps_map=C.get("lookahead_temp_multi_phase_caps_map", None),
        lookahead_temp_multi_phase_alphas_map=C.get("lookahead_temp_multi_phase_alphas_map", None),
        lookahead_temp_mp_target_map=C.get("lookahead_temp_mp_target_map", None),
        lookahead_temp_mp_adapt_map=C.get("lookahead_temp_mp_adapt_map", None),
        lookahead_temp_mp_gcap_map=C.get("lookahead_temp_mp_gcap_map", None),

        rms_clip_threshold_map=C["rms_clip_threshold_map"],
        trust_clip_map=C["trust_clip_map"],
        trust_granularity=C["trust_granularity"],
        rms_clip_granularity=C["rms_clip_granularity"],
        factored_mode_map=C["factored_mode_map"],
        use_adafactor_tag_map=C["use_adafactor_tag_map"],
        beta1_map=C.get("beta1_map", None),
        beta2_map=C.get("beta2_map", None),
        qkv_beta1_map=C.get("qkv_beta1_map", None),
        qkv_beta2_map=C.get("qkv_beta2_map", None),
        qkv_trust_clip_map=C.get("qkv_trust_clip_map", None),
        qkv_beta2_field_map=C.get("qkv_beta2_field_map", None),
        qkv_beta2_reg_map=C.get("qkv_beta2_reg_map", None),
        qkv_beta2_reg_sched_map=C.get("qkv_beta2_reg_sched_map", None),
        qkv_beta2_reg_pw_depth_map=C.get("qkv_beta2_reg_pw_depth_map", None),
        qkv_beta2_reg_depth_spline_map=C.get("qkv_beta2_reg_depth_spline_map", None),
        qkv_beta2_reg_depth_bspline_field_map=C.get("qkv_beta2_reg_depth_bspline_field_map", None),
        qkv_beta2_reg_depth_head_bspline_field_map=C.get("qkv_beta2_reg_depth_head_bspline_field_map", None),
        qkv_beta2_value_depth_head_bspline_field_map=C.get("qkv_beta2_value_depth_head_bspline_field_map", None),

        qkv_beta2_value_depth_head_col_bspline_field_map=C.get("qkv_beta2_value_depth_head_col_bspline_field_map", None),
        qkv_beta2_reg_depth_head_col_map=C.get("qkv_beta2_reg_depth_head_col_map", None),
        qkv_beta2_value_depth_col_bspline_field_map=C.get("qkv_beta2_value_depth_col_bspline_field_map", None),
        qkv_beta2_reg_depth_col_bspline_field_map=C.get("qkv_beta2_reg_depth_col_bspline_field_map", None),

        conv_beta2_value_spatial_field_map=C.get("conv_beta2_value_spatial_field_map", None),
        conv_beta2_reg_spatial_field_map=C.get("conv_beta2_reg_spatial_field_map", None),
        conv_beta2_value_channel_field_map=C.get("conv_beta2_value_channel_field_map", None),
        conv_beta2_reg_channel_field_map=C.get("conv_beta2_reg_channel_field_map", None),

        warmup_steps_map=C.get("warmup_steps_map", None),
        total_steps_map={"default": total_steps, **(C.get("total_steps_map", {}) )} if C.get("total_steps_map", None) else {"default": total_steps},
        enable_qkv_slicing=True,
    )

    opt = ProdigyUltra(
        groups,
        lr=base_lr, beta1=C["beta1_default"], weight_decay=base_wd,
        trust_ratio=True, trust_granularity=C["trust_granularity"], trust_clip=C["trust_clip_default"],
        use_sign=C["use_sign"],
        use_adafactor=C["use_adafactor_default"], beta2=C["beta2_default"], factored_second_moment=True,
        rms_clip_threshold=0.0, rms_clip_granularity=C["rms_clip_granularity"],
        lookahead_k=0, lookahead_alpha=0.5, lookahead_reset_m=True,
    )

    sched = TagAwareWarmupLR(
        opt,
        default_warmup_steps=C.get("warmup_default", 2000),
        default_total_steps=total_steps,
    )
    return opt, sched, groups, C

# ─────────────────────────────────────────────────────────────────────────────
# Diagnostics
# ─────────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def snapshot_optimizer(opt: ProdigyUltra, *, max_lines: int = 50) -> List[dict]:
    rows: List[dict] = []
    tag_p2: Dict[str, float] = {}
    tag_m2: Dict[str, float] = {}
    for g in opt.param_groups:
        tag = g.get("block_tag","default")
        p2_sum = m2_sum = 0.0
        for p in g["params"]:
            st = opt.state.get(p, {})
            if "m" not in st: continue
            m = st["m"]
            p2_sum += float((p.float()*p.float()).sum().item())
            m2_sum += float((m.float()*m.float()).sum().item())
        tag_p2[tag] = tag_p2.get(tag, 0.0) + p2_sum
        tag_m2[tag] = tag_m2.get(tag, 0.0) + m2_sum

    for g in opt.param_groups[:max_lines]:
        tag = g.get("block_tag","default")
        lr_sched = float(g["lr"]); lr_scale = float(g.get("lr_scale",1.0))
        eff_lr = lr_sched * lr_scale
        tr_min, tr_max = g.get("trust_clip",(0.1,10.0))
        p2 = tag_p2.get(tag,0.0); m2 = tag_m2.get(tag,0.0)
        r_est = (math.sqrt(p2)/(math.sqrt(m2)+1e-8)) if (p2>0.0 and m2>0.0) else 1.0
        rows.append({
            "tag": tag, "depth": int(g.get("depth",0)),
            "sched_lr": lr_sched, "lr_scale": lr_scale, "eff_lr": eff_lr,
            "r_est_clipped": max(tr_min, min(tr_max, r_est)), "r_est_raw": r_est,
            "rms_thr": float(g.get("rms_clip_threshold",0.0)),
            "heads": int(g.get("attn_heads",1)),
            "phase_hint": {"w_eff": int(g.get("_sched_warmup", 0)), "T": int(g.get("_sched_total", 0))},
            "adafact": bool(g.get("use_adafactor", opt.defaults["use_adafactor"])),
            "mode": g.get("factored_mode","rowcol"),
        })
    return rows

def print_snapshot(rows: List[dict]):
    if not rows: print("(no rows)"); return
    header = f"{'tag':12} {'depth':>5} {'sched_lr':>10} {'lr_scale':>8} {'eff_lr':>10} {'r_est':>8} {'rms_thr':>8} {'heads':>6} {'phase(w/T)':>14} {'adafact':>8} {'mode':>12}"
    print(header); print("-"*len(header))
    for r in rows:
        ph = r['phase_hint']
        print(f"{r['tag'][:12]:12} {r['depth']:5d} {r['sched_lr']:10.2e} {r['lr_scale']:8.2f} {r['eff_lr']:10.2e} {r['r_est_clipped']:8.3f} {r['rms_thr']:8.2f} {r['heads']:6d} {ph['w_eff']:4d}/{ph['T']:>7} {str(r['adafact']):>8} {r['mode']:>12}")

# ─────────────────────────────────────────────────────────────────────────────
# Smoke test
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    class TinyMHA(nn.Module):
        def __init__(self, d_model=32, nhead=4):
            super().__init__()
            self.mha = nn.MultiheadAttention(d_model, nhead, batch_first=True)
            self.ln = nn.LayerNorm(d_model)
        def forward(self, x):
            out, _ = self.mha(x, x, x)
            return self.ln(out)

    class TinyConvNet(nn.Module):
        def __init__(self, c=3, groups=3):
            super().__init__()
            # depthwise conv（groups==in_channels） + dilation
            self.conv_dw = nn.Conv2d(c, c, kernel_size=3, padding=1, groups=c, dilation=1, bias=False)
            self.conv_pw = nn.Conv2d(c, 8, kernel_size=1, padding=0, groups=1, bias=False)
            self.bn = nn.BatchNorm2d(8)
        def forward(self, x):
            h = torch.relu(self.conv_dw(x))
            h = torch.relu(self.conv_pw(h))
            return self.bn(h).mean(dim=(2,3))

    class TinyNet(nn.Module):
        def __init__(self, depth=6):
            super().__init__()
            self.blocks = nn.ModuleList([nn.Sequential(nn.Linear(16,16), nn.ReLU()) for _ in range(depth)])
            self.fc = nn.Linear(16, 32)
            self.mha_block = TinyMHA(32, 4)
            self.conv = TinyConvNet(3)
            self.proj = nn.Linear(8+32, 10)
        def forward(self, x):
            b = x.size(0)
            h = x.mean(dim=(1,2,3))
            h = h.view(b,1).repeat(1,16)
            for blk in self.blocks:
                h = blk(h) + h
            h = torch.relu(self.fc(h)).view(b, 4, 8)
            h = self.mha_block(h)
            v = self.conv(x)
            o = torch.cat([h.mean(dim=1), v], dim=-1)
            return self.proj(o)

    torch.manual_seed(17)
    model = TinyNet(depth=6)
    opt, sched, groups, cfg = apply_chef_course(model, "transformer_spline_tempX10")

    x = torch.randn(8, 3, 16, 16)
    y = torch.randint(0, 10, (8,))
    criterion = nn.CrossEntropyLoss()

    model.train()
    logits = model(x)
    loss = criterion(logits, y)
    val_loss = float(loss.item()) * 1.02
    val_acc = 0.34

    loss.backward()
    opt.set_ext_metrics(loss=float(loss.item()), val_loss=val_loss, val_acc=val_acc)
    opt.step(); sched.step(); opt.zero_grad()

    print("X10 step ok. loss=", float(loss.item()))
    rows = snapshot_optimizer(opt)
    print_snapshot(rows)

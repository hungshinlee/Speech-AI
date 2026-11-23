# -*- coding: utf-8 -*-
# 此檔由 bin/publish_demos.py 從 private repo 的 <course>/demos/ 複製而來，請勿在公開 repo 直接編輯。
"""W10 demo 3 的計分那一半：diarization error rate。純 numpy（配對用 scipy 的匈牙利演算法）。

投影片 p83 的式子與兩個開關就是這個檔：

    DER = (T_FA + T_miss + T_conf) / T_speech

時間切成 10 ms 一格。每一格：N_ref = 參考標註裡這一格有幾位在說話、N_hyp = 系統說有幾位、
N_ok = 配對之後兩邊都說「這個人在說話」的有幾位。那麼

    miss  = max(0, N_ref − N_hyp)          漏掉的
    FA    = max(0, N_hyp − N_ref)          多出來的
    conf  = min(N_ref, N_hyp) − N_ok       人數對、人不對
    T_speech = Σ N_ref                     重疊的格子每位各算一次

系統輸出的編號是任意的（「SPEAKER_00」不是誰的名字），所以先找一個一對一的配對，讓對得上的時間最長（Hungarian）。

兩個開關：
    collar        參考標註的每一個邊界前後各 collar 秒不計分（0.25 是常見的設定；0 = 不給）
    skip_overlap  參考標註裡有兩位以上同時說話的格子不計分
"""
from __future__ import annotations

import numpy as np

STEP = 0.010


def to_grid(segs: list[tuple[float, float, str]], seconds: float, step: float = STEP) -> dict[str, np.ndarray]:
    """[(開始, 結束, 標籤), …] → {標籤: bool 格子}。一格的中心落在段內就算。"""
    n = int(np.ceil(seconds / step))
    centre = (np.arange(n) + 0.5) * step
    out: dict[str, np.ndarray] = {}
    for a, b, lab in segs:
        out.setdefault(str(lab), np.zeros(n, dtype=bool))
        out[str(lab)] |= (centre >= a) & (centre < b)
    return out


def scored_mask(ref: dict[str, np.ndarray], collar: float = 0.0, skip_overlap: bool = False, step: float = STEP) -> np.ndarray:
    """哪些格子計分。"""
    n = len(next(iter(ref.values())))
    keep = np.ones(n, dtype=bool)
    if collar > 0:
        w = int(round(collar / step))
        for m in ref.values():
            edges = np.flatnonzero(np.diff(np.concatenate([[False], m, [False]]).astype(np.int8)))   # 邊界落在第幾格的前緣
            for e in edges:
                keep[max(e - w, 0): min(e + w, n)] = False
    if skip_overlap:
        keep &= np.sum(np.stack(list(ref.values())), axis=0) < 2
    return keep


def mapping(ref: dict[str, np.ndarray], hyp: dict[str, np.ndarray], keep: np.ndarray) -> dict[str, str]:
    """系統的標籤 → 參考的標籤：一對一、對得上的格數最多。對不到的系統標籤不在回傳裡。"""
    from scipy.optimize import linear_sum_assignment
    rk, hk = list(ref), list(hyp)
    if not rk or not hk:
        return {}
    cost = np.array([[-(ref[r] & hyp[h] & keep).sum() for h in hk] for r in rk])
    ri, hi = linear_sum_assignment(cost)
    return {hk[j]: rk[i] for i, j in zip(ri, hi) if cost[i, j] < 0}


def der(ref: dict[str, np.ndarray], hyp: dict[str, np.ndarray], collar: float = 0.0, skip_overlap: bool = False, step: float = STEP) -> dict:
    """回傳 {"der", "fa", "miss", "conf", "total"（秒）, "scored_s", "map", "per_frame": {"fa","miss","conf"}（每格的人數）, "keep"}。"""
    keep = scored_mask(ref, collar, skip_overlap, step)
    mp = mapping(ref, hyp, keep)
    n = len(keep)
    R = np.stack(list(ref.values())).astype(np.int32) if ref else np.zeros((0, n), np.int32)
    H = np.stack(list(hyp.values())).astype(np.int32) if hyp else np.zeros((0, n), np.int32)
    n_ref, n_hyp = R.sum(0), H.sum(0)
    n_ok = np.zeros(n, dtype=np.int32)
    for h, r in mp.items():
        n_ok += (ref[r] & hyp[h]).astype(np.int32)
    miss = np.maximum(n_ref - n_hyp, 0) * keep
    fa = np.maximum(n_hyp - n_ref, 0) * keep
    conf = (np.minimum(n_ref, n_hyp) - n_ok) * keep
    total = float((n_ref * keep).sum() * step)
    t = {k: float(v.sum() * step) for k, v in (("fa", fa), ("miss", miss), ("conf", conf))}
    return {"der": (t["fa"] + t["miss"] + t["conf"]) / total if total > 0 else float("nan"), **t, "total": total,
            "scored_s": float(keep.sum() * step), "map": mp, "per_frame": {"fa": fa, "miss": miss, "conf": conf}, "keep": keep}


# 投影片 p84 那張表的三欄，由寬到嚴
SCORINGS = [("collar 0.25 s, overlap skipped", 0.25, True),
            ("collar 0.25 s, overlap scored", 0.25, False),
            ("no collar, overlap scored", 0.0, False)]


def three(ref: dict[str, np.ndarray], hyp: dict[str, np.ndarray]) -> list[dict]:
    return [{"name": name, "collar": c, "skip_overlap": s, **der(ref, hyp, c, s)} for name, c, s in SCORINGS]

# -*- coding: utf-8 -*-
# 此檔由 bin/publish_demos.py 從 private repo 的 <course>/demos/ 複製而來，請勿在公開 repo 直接編輯。
"""W10 demo 2 的訊號處理：加噪音、統計式增強、誤差的兩次投影、observation adding。純 numpy（STFT 用 ../dsp.py）。

投影片上的式子與這裡的函數一一對應：
  mix()             y = s + g·v，g 由整句的 SNR 決定
  wiener_dd()       G = ξ / (1 + ξ)，ξ 用 decision-directed 估（p38–p39）
  decompose()       ŝ = s_target + e_noise + e_artif（p52），SDR／SNR／SAR（p53）——不含延遲版本的子空間
  observation_add() ŝ + ω·y（p55）
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import dsp  # noqa: E402

EPS = 1e-20


def power(x: np.ndarray) -> float:
    return float(np.mean(np.square(x)) + EPS)


def db(ratio: float) -> float:
    return 10.0 * math.log10(max(ratio, EPS))


# ── 噪音 ────────────────────────────────────────────────────────────────
def white(n: int, rng: np.random.Generator) -> np.ndarray:
    return rng.standard_normal(n)


def pink(n: int, rng: np.random.Generator) -> np.ndarray:
    """功率譜 ∝ 1/f：白噪音的頻譜乘 1/√f（DC 那一格設成零）。"""
    X = np.fft.rfft(rng.standard_normal(n))
    f = np.arange(len(X), dtype=float)
    f[0] = np.inf
    return np.fft.irfft(X / np.sqrt(f), n=n)


def babble(n: int, others: list[np.ndarray], k: int, rng: np.random.Generator) -> np.ndarray:
    """k 個別人的句子疊在一起：各自調到同一個功率、隨機起點、不夠長就繞回去接。"""
    out = np.zeros(n)
    for i in rng.choice(len(others), size=min(k, len(others)), replace=False):
        o = others[int(i)]
        o = np.roll(o, int(rng.integers(len(o))))
        o = np.tile(o, n // len(o) + 1)[:n]
        out += o / math.sqrt(power(o))
    return out


def mix(s: np.ndarray, v: np.ndarray, snr_db: float) -> tuple[np.ndarray, np.ndarray]:
    """回傳 (y, g·v)。SNR 以整句的功率算：10 log10(Σ s² / Σ (g·v)²) = snr_db。語音本身不動。"""
    g = math.sqrt(power(s) / power(v)) * 10.0 ** (-snr_db / 20.0)
    return s + g * v, g * v


# ── 統計式增強：decision-directed Wiener ────────────────────────────────
def noise_psd(P: np.ndarray, q: float) -> np.ndarray:
    """每個頻率 bin 的噪音功率：整句 |Y|² 的第 q 分位數，除以指數分布在 q 的分位數（−ln(1 − q)）。

    假設：噪音是穩態的，而且每個 bin 至少有 q 這麼多比例的 frame 沒有語音。這是這個 demo 的簡化，不是哪一篇論文的估計器；
    babble 不符合第一個假設。
    """
    return np.quantile(P, q, axis=1) / (-math.log(1.0 - q))


def wiener_dd(y: np.ndarray, L: int, H: int, N: int, alpha: float, g_min: float, q: float, return_gain: bool = False):
    """Y(t,k) → Ŝ = G·Y，沿用帶噪的相位。

    每一格：後驗 SNR  γ = |Y|² / σ_v²
            先驗 SNR  ξ = α·|Ŝ(t−1,k)|² / σ_v² + (1 − α)·max(γ − 1, 0)      ← decision-directed
            增益      G = max(ξ / (1 + ξ), g_min)
    α = 0 時 ξ 只看這一格，G 逐格亂跳——孤立的格子一亮一滅，就是 musical noise。
    """
    Y = dsp.stft(y, L, H, N)
    P = np.abs(Y) ** 2
    sv = np.maximum(noise_psd(P, q), EPS)
    G = np.empty_like(P)
    prev = np.zeros(P.shape[0])                       # |Ŝ(t−1,k)|²
    for t in range(P.shape[1]):
        gamma = P[:, t] / sv
        xi = alpha * prev / sv + (1.0 - alpha) * np.maximum(gamma - 1.0, 0.0)
        g = np.maximum(xi / (1.0 + xi), g_min)
        G[:, t] = g
        prev = (g ** 2) * P[:, t]
    out = dsp.istft(G * Y, L, H, N, len(y))
    return (out, G, sv) if return_gain else out


def isolated_fraction(G: np.ndarray, hi: float = 0.5, lo: float = 0.2) -> float:
    """增益圖上「自己亮、時間與頻率上的四個鄰居都暗」的格子佔多少——musical noise 的一個粗略計數。"""
    c = G[1:-1, 1:-1] > hi
    nb = (G[:-2, 1:-1] < lo) & (G[2:, 1:-1] < lo) & (G[1:-1, :-2] < lo) & (G[1:-1, 2:] < lo)
    return float(np.mean(c & nb))


# ── 誤差的兩次投影 ──────────────────────────────────────────────────────
def decompose(est: np.ndarray, s: np.ndarray, v: np.ndarray) -> dict:
    """ŝ = s_target + e_noise + e_artif。

    s_target = P_s ŝ          ŝ 在 s 這條線上的投影
    e_noise  = P_{s,v} ŝ − P_s ŝ    還要 v 才寫得出來的部分
    e_artif  = ŝ − P_{s,v} ŝ        s 與 v 的線性組合寫不出來的部分
    這裡的子空間只由 s 與 v 兩個向量張成（不含它們的延遲版本），所以任何濾波——包括一個不改內容的等化——都會算進 e_artif。
    """
    A = np.stack([s, v], axis=1)
    coef, *_ = np.linalg.lstsq(A, est, rcond=None)
    p_sv = A @ coef
    s_target = (float(est @ s) / (float(s @ s) + EPS)) * s
    e_noise = p_sv - s_target
    e_artif = est - p_sv
    return {"s_target": s_target, "e_noise": e_noise, "e_artif": e_artif}


def ratios(est: np.ndarray, s: np.ndarray, v: np.ndarray, sar_cap_db: float = 120.0) -> dict:
    """SDR、SNR、SAR（dB）。e_artif 小到只剩捨入誤差時 SAR 記成 None（畫面上印 ∞）。"""
    d = decompose(est, s, v)
    st, en, ea = (float(np.sum(d[k] ** 2)) for k in ("s_target", "e_noise", "e_artif"))
    sar = db((float(np.sum((d["s_target"] + d["e_noise"]) ** 2))) / (ea + EPS))
    return {"sdr": db(st / (float(np.sum((d["e_noise"] + d["e_artif"]) ** 2)) + EPS)),
            "snr": db(st / (en + EPS)),
            "sar": None if sar > sar_cap_db else sar}


def si_sdr(est: np.ndarray, s: np.ndarray) -> float:
    """W2 的定義：先把 s 乘上最佳的 α 再比（對整體音量不敏感）。"""
    a = float(est @ s) / (float(s @ s) + EPS)
    return db(float(np.sum((a * s) ** 2)) / (float(np.sum((est - a * s) ** 2)) + EPS))


def observation_add(est: np.ndarray, y: np.ndarray, omega: float) -> np.ndarray:
    return est + omega * y


def blind_gain(est: np.ndarray, y: np.ndarray) -> float:
    """輸出音量任意的模型（時域、以 SI-SNR 訓練的）先乘上這個數：讓 g·ŝ 最接近 y 的那個 g。只用到 y，不用乾淨的答案。"""
    return float(est @ y) / (float(est @ est) + EPS)

# -*- coding: utf-8 -*-
# 此檔由 bin/publish_demos.py 從 private repo 的 <course>/demos/ 複製而來，請勿在公開 repo 直接編輯。
"""W10 demo 1 的訊號處理：NLMS 與三種步長的規則。只用 numpy／scipy，沒有神經網路。

記號照投影片（W10 p23–p29）：
  x(n)    系統播出去的參考訊號                     y(n)   麥克風訊號
  x(n) 粗體 = [x(n), x(n-1), …, x(n-L_h+1)]         ĥ(n)   回音路徑的估計（L_h 個數）
  e(n)    = y(n) − ĥᵀ(n) x(n)，殘餘                 μ      步長        δ   正則化常數

nlms() 的迴圈裡標了 (p24) (p26) 的那兩行，就是投影片上的兩條式子。
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import maximum_filter1d
from scipy.signal import correlate, lfilter

EPS = 1e-12


# ── 第一步：直接減 ───────────────────────────────────────────────────────────
def estimate_delay(x: np.ndarray, y: np.ndarray, max_delay: int) -> int:
    """只准挪一個延遲的話，挪多少減掉最多：互相關絕對值最大的位置（0 ≤ D ≤ max_delay）。第一步（鍵 3）用它。"""
    n = min(len(x), len(y))
    c = correlate(y[:n], x[:n], mode="full", method="fft")[n - 1:n + max_delay]
    return int(np.argmax(np.abs(c)))


def first_arrival(x: np.ndarray, y: np.ndarray, max_delay: int, rel: float = 0.5) -> int:
    """回音最早到的那一刻（樣本）：先把兩邊的頻譜壓平再做互相關（GCC-PHAT），取第一個超過最大值 rel 倍的位置。
    estimate_delay 找的是「挪多少減掉最多」，語音的互相關峰很寬、房間的反射又疊上去，那個峰常常落在直達聲之後一兩百個樣本；
    濾波器只能用過去的 x，起點要是放在直達聲之後，最大的那一根就在它看不到的地方。"""
    n = 1 << int(np.ceil(np.log2(min(len(x), len(y)) + max_delay)))
    G = np.fft.rfft(y, n) * np.conj(np.fft.rfft(x, n))
    c = np.abs(np.fft.irfft(G / (np.abs(G) + 1e-9 * np.max(np.abs(G))), n)[:max_delay + 1])
    return int(np.argmax(c >= rel * np.max(c)))


def shift(x: np.ndarray, d: int) -> np.ndarray:
    """x(n − d)，長度不變。"""
    return x.copy() if d <= 0 else np.concatenate([np.zeros(d), x[:-d]])


def best_gain(x: np.ndarray, y: np.ndarray) -> float:
    """讓 ‖y − g x‖² 最小的那一個數 g（最小平方）。"""
    return float(x @ y / (x @ x + EPS))


def level_db(res: np.ndarray, ref: np.ndarray) -> float:
    """殘餘相對於麥克風的位準：10 log10(Σ res² / Σ ref²)。負的才是變小。"""
    return float(10 * np.log10((res @ res + EPS) / (ref @ ref + EPS)))


# ── double-talk 偵測（硬的）─────────────────────────────────────────────────
def geigel(x: np.ndarray, y: np.ndarray, L_h: int, theta: float, hold: int) -> np.ndarray:
    """Geigel：|y(n)| > ϑ · max_{0 ≤ i < L_h} |x(n − i)| 就當成使用者在說，之後再維持 hold 個樣本。回傳每個樣本凍不凍結。"""
    ax = np.abs(x)
    xmax = maximum_filter1d(ax, size=L_h, origin=(L_h - 1) // 2, mode="constant")   # 只看過去的 L_h 個樣本
    hit = np.abs(y) > theta * xmax + EPS
    if hold > 0:                                                                     # 維持：過去 hold 個樣本裡有過一次就算
        hit = maximum_filter1d(hit.astype(np.uint8), size=hold, origin=(hold - 1) // 2, mode="constant") > 0
    return hit


def geigel_ratio(x: np.ndarray, y: np.ndarray, L_h: int) -> np.ndarray:
    """|y(n)| ÷ max_{i<L_h}|x(n − i)|：只有系統在說的那一段上，這個比值的上緣就是 ϑ 該放的地方。"""
    xmax = maximum_filter1d(np.abs(x), size=L_h, origin=(L_h - 1) // 2, mode="constant")
    ok = xmax > 0.05 * np.max(np.abs(x))
    return np.abs(y[ok]) / xmax[ok]


# ── NLMS ────────────────────────────────────────────────────────────────────
def nlms(x: np.ndarray, y: np.ndarray, L_h: int, mu: float = 0.5, delta: float = 1e-3, rule: str = "fixed",
         frozen: np.ndarray | None = None, soft: dict | None = None, fs: int = 16000,
         h_true: np.ndarray | None = None, h0: np.ndarray | None = None) -> dict:
    """逐樣本的 NLMS。ĥ 從零（或 h0）開始，邊用邊學——沒有訓練階段。

    rule：
      "fixed"   μ 固定
      "freeze"  frozen[n] 為真時 μ = 0（frozen 來自 geigel()）
      "soft"    μ(n) = min(μ_max, 估出來的殘餘回音功率 ÷ 誤差功率)，見下面 soft_step 那幾行
    回傳 e、ŷ、每個樣本用的 μ、最後的 ĥ；給了 h_true（模擬才有）再多一條 ‖h − ĥ‖²/‖h‖²（每 64 個樣本一點）。

    soft 的估法（本 demo 的簡化，不是 Valin 2007 的 leakage 估計）：
      殘餘回音功率 ≈ Λ(n) · p_x(n)，p_x(n) = ‖x(n)‖²/L_h 是這一刻參考訊號的功率（已知、不用估），
      Λ(n) 是「每單位參考功率留下多少殘餘」，濾波器越準越小、變得慢——用 e²/p_x 的慢速平均去跟，
      而且規定它往上長的速度有上限（每 tau_up 秒最多一倍）。使用者一開口，e² 馬上變大、Λ 來不及跟，比值就掉下去。
      這是 Valin 2007 Eq. 8 的形狀：μ = Λ p_x / (Λ p_x + 近端功率) = 殘餘回音 ÷ 誤差。
    """
    N = len(x)
    h = np.zeros(L_h) if h0 is None else h0.astype(np.float64).copy()
    e = np.zeros(N)
    mu_n = np.zeros(N)
    xp = np.concatenate([np.zeros(L_h - 1), x.astype(np.float64)])
    mis = [] if h_true is not None else None
    hh = float(h_true @ h_true) if h_true is not None else 1.0
    m = float(mu)
    if rule == "soft":
        s = dict(soft or {})
        mu_max = float(s.get("mu_max", mu))
        a_e = np.exp(-1.0 / (float(s.get("tau_e", 0.02)) * fs))        # e² 的短時平均
        a_l = np.exp(-1.0 / (float(s.get("tau_lambda", 1.0)) * fs))    # Λ 的慢速平均
        up = np.exp(np.log(2.0) / (float(s.get("tau_up", 2.0)) * fs))  # Λ 每個樣本最多長這麼多倍
        px_floor = 1e-4 * float(np.mean(x.astype(np.float64) ** 2) + EPS)
        pe = lam = None
    for n in range(N):
        xv = xp[n:n + L_h][::-1]                                   # x(n) = [x(n), x(n−1), …, x(n−L_h+1)]
        nx = xv @ xv                                               # ‖x(n)‖²
        e[n] = y[n] - h @ xv                                       # (p24)  e(n) = y(n) − ĥᵀ(n) x(n)
        if rule == "freeze":
            m = 0.0 if frozen[n] else mu
        elif rule == "soft":                                       # soft_step
            pe = e[n] * e[n] if pe is None else a_e * pe + (1 - a_e) * e[n] * e[n]
            px = nx / L_h
            if px > px_floor:
                q = pe / px
                lam = q if lam is None else min(a_l * lam + (1 - a_l) * q, lam * up)
                m = min(mu_max, lam * px / (pe + EPS))             # (p29)  μ ≈ 殘餘回音 ÷ 誤差
            else:
                m = 0.0                                            # 系統沒出聲：沒有東西可學
        mu_n[n] = m
        if m != 0.0:
            h += m * e[n] * xv / (nx + delta)                      # (p26)  ĥ(n+1) = ĥ(n) + μ e(n) x(n) / (‖x(n)‖² + δ)
        if mis is not None and n % 64 == 0:
            d = h_true - h
            mis.append(float(d @ d) / hh)
    out = {"e": e, "yhat": y - e, "mu": mu_n, "h": h}
    if mis is not None:
        out["misalignment"] = np.asarray(mis)
    return out


# ── 量 ──────────────────────────────────────────────────────────────────────
def smooth_power(v: np.ndarray, win: int) -> np.ndarray:
    """v² 的移動平均（win 個樣本，只看過去）。"""
    c = np.cumsum(np.concatenate([[0.0], v.astype(np.float64) ** 2]))
    i = np.arange(1, len(v) + 1)
    j = np.maximum(i - win, 0)
    return (c[i] - c[j]) / (i - j)


def erle_curve(y: np.ndarray, e: np.ndarray, win: int, step: int, x_active: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """ERLE(n) = 10 log10(E[y²] / E[e²])，期望值用過去 win 個樣本的平均；每 step 個樣本取一點。
    x_active 給定時，系統沒在出聲的地方回傳 nan（沒有回音可扣，比值沒有意義）。"""
    py, pe = smooth_power(y, win), smooth_power(e, win)
    idx = np.arange(step - 1, len(y), step)
    r = 10 * np.log10((py[idx] + EPS) / (pe[idx] + EPS))
    if x_active is not None:
        r = np.where(x_active[idx], r, np.nan)
    return idx, r


def erle_span(y: np.ndarray, e: np.ndarray, a: int, b: int, mask: np.ndarray | None = None) -> float | None:
    """樣本 [a, b) 上的 ERLE：10 log10(Σ y² / Σ e²)。mask 給定時只算 mask 為真的樣本；不到 50 ms 回傳 None。"""
    a, b = max(a, 0), min(b, len(y))
    if b <= a:
        return None
    sel = np.ones(b - a, bool) if mask is None else mask[a:b]
    if sel.sum() < 800:
        return None
    yy, ee = y[a:b][sel], e[a:b][sel]
    return float(10 * np.log10((yy @ yy + EPS) / (ee @ ee + EPS)))


def erle_active(y: np.ndarray, e: np.ndarray, act: np.ndarray, start: int, count: int, forward: bool = True) -> float | None:
    """從 start 往後（forward）或往前數，取 count 個「系統有出聲」的樣本，在那些樣本上算 ERLE。
    系統那一句有停頓，使用者停下來的那一刻不一定有回音可量——所以數的是有回音的樣本，不是牆上的時間。"""
    idx = np.where(act[start:])[0] + start if forward else np.where(act[:start])[0]
    idx = idx[:count] if forward else idx[-count:]
    if len(idx) < min(count, 800):
        return None
    yy, ee = y[idx], e[idx]
    return float(10 * np.log10((yy @ yy + EPS) / (ee @ ee + EPS)))


def activity(v: np.ndarray, fs: int, floor_db: float = -35.0, win_ms: float = 30.0) -> np.ndarray:
    """v 在出聲的樣本：30 ms 的功率比整段最大的 30 ms 功率高出 floor_db 以上。"""
    p = smooth_power(v, int(fs * win_ms / 1000))
    return p > np.max(p) * 10 ** (floor_db / 10)


def longest_run(flag: np.ndarray) -> tuple[int, int] | None:
    """最長的一段連續為真的區間 [a, b)。"""
    if not flag.any():
        return None
    d = np.diff(np.concatenate([[0], flag.astype(np.int8), [0]]))
    s, t = np.where(d == 1)[0], np.where(d == -1)[0]
    k = int(np.argmax(t - s))
    return int(s[k]), int(t[k])


def close_gaps(flag: np.ndarray, gap: int) -> np.ndarray:
    """把短於 gap 個樣本的空隙補起來（字與字之間的停頓不算說完）。"""
    out = flag.copy()
    d = np.diff(np.concatenate([[0], flag.astype(np.int8), [0]]))
    s, t = np.where(d == 1)[0], np.where(d == -1)[0]
    for a, b in zip(t[:-1], s[1:]):
        if b - a < gap:
            out[a:b] = True
    return out


def soft_clip(x: np.ndarray, drive: float) -> np.ndarray:
    """模擬裡的「喇叭失真」：g(x) = tanh(drive · x / peak) · peak / tanh(drive)。drive → 0 時就是 x。"""
    pk = float(np.max(np.abs(x)) + EPS)
    return np.tanh(drive * x / pk) * pk / np.tanh(drive)


def linear_ceiling_db(x: np.ndarray, gx: np.ndarray) -> float:
    """g(x) 裡有多少是一個數乘上 x 寫不出來的：10 log10(‖g(x)‖² / ‖g(x) − a x‖²)，a 是最小平方的係數。
    這是「逐樣本的非線性」留給線性濾波器的上限的一個粗估（濾波器比一個數自由，實際的上限會高一些）。"""
    a = best_gain(x, gx)
    r = gx - a * x
    return float(10 * np.log10((gx @ gx + EPS) / (r @ r + EPS)))


def colour(x: np.ndarray, a: float) -> np.ndarray:
    """AR(1)：p(n) = a p(n−1) + sqrt(1 − a²) x(n)。selftest 用。"""
    return lfilter([np.sqrt(1 - a * a)], [1.0, -a], x)

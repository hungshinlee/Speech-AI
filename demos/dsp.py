# -*- coding: utf-8 -*-
# 此檔由 bin/publish_demos.py 從 private repo 的 <course>/demos/ 複製而來，請勿在公開 repo 直接編輯。
"""speech_ai/demos 共用的訊號處理。只依賴 numpy 與 scipy。

刻意不用 librosa：
  1. W2 的教學重點之一是「前端的定義不只一種」，所以 mel 的兩種尺度、filterbank 要不要正規化、
     ISTFT 用哪一種，全部攤在這個檔案裡，看得到、改得動。
  2. librosa.griffinlim 預設帶 momentum（fast Griffin-Lim），那個版本**沒有**單調不增的保證；
     大綱〈關鍵數學 4〉講的是原始的 Griffin & Lim 1984，這裡照原始版本寫。

記號與大綱一致：window 長度 L（samples）、hop H、FFT 點數 N ≥ L。
"""
from __future__ import annotations

import numpy as np
from scipy.linalg import solve_toeplitz
from scipy.optimize import nnls
from scipy.signal import resample_poly


# ── 基本工具 ────────────────────────────────────────────────────
def to_mono_float(x: np.ndarray) -> np.ndarray:
    if x.ndim > 1:
        x = x.mean(axis=1)
    if np.issubdtype(x.dtype, np.integer):
        x = x.astype(np.float64) / np.iinfo(x.dtype).max
    return x.astype(np.float64)


def resample(x: np.ndarray, fs_in: int, fs_out: int) -> np.ndarray:
    """多相濾波重取樣；resample_poly 內含 anti-aliasing 低通（大綱：降取樣前一定要先低通）。"""
    if fs_in == fs_out:
        return x
    g = np.gcd(fs_in, fs_out)
    return resample_poly(x, fs_out // g, fs_in // g)


def hann(L: int) -> np.ndarray:
    """periodic Hann（DSP 慣例；對稱版是給濾波器設計用的）。"""
    return 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(L) / L)


def _padded_window(L: int, N: int) -> np.ndarray:
    w = np.zeros(N)
    a = (N - L) // 2
    w[a:a + L] = hann(L)
    return w


# ── STFT 與最小平方 ISTFT ────────────────────────────────────────
def stft(x: np.ndarray, L: int, H: int, N: int) -> np.ndarray:
    """回傳 (N/2+1, frames) 的複數矩陣。frame 以中心為時間戳（兩端 reflect padding N/2）。"""
    w = _padded_window(L, N)
    xp = np.pad(x, N // 2, mode="reflect")
    frames = np.lib.stride_tricks.sliding_window_view(xp, N)[::H]
    return np.fft.rfft(frames * w, axis=1).T


def istft(X: np.ndarray, L: int, H: int, N: int, length: int) -> np.ndarray:
    """Griffin & Lim 1984 的 LSEE-MSTFT：x = Σ w·irfft(X_m) / Σ w²。

    這是「在所有訊號裡，STFT 與 X 的平方誤差最小的那一個」，也就是把 X 正交投影到
    一致集合 C 上。不要求 window 與 hop 滿足 COLA——任何 hop 都成立，只要 Σ w² 不為零。
    """
    w = _padded_window(L, N)
    frames = np.fft.irfft(X.T, n=N, axis=1) * w
    T = X.shape[1]
    out = np.zeros(N + H * (T - 1))
    norm = np.zeros_like(out)
    w2 = w * w
    for m in range(T):
        out[m * H:m * H + N] += frames[m]
        norm[m * H:m * H + N] += w2
    out /= np.maximum(norm, 1e-10)
    return out[N // 2:N // 2 + length]


# ── Griffin-Lim（原始版本，無 momentum）──────────────────────────
def griffin_lim(A: np.ndarray, L: int, H: int, N: int, length: int,
                n_iter: int, seed: int = 0, keep: tuple[int, ...] = ()):
    """從 magnitude A 重建波形。

    回傳 (x, log, snaps)：
      log   每一輪的 {"iter", "inconsistency", "spectral_convergence"}
      snaps {迭代次數: 波形}，keep 裡列了哪幾輪就存哪幾輪；0 = 隨機相位、尚未迭代

    inconsistency = ‖C − P(C)‖² / ‖C‖²，C = A·e^{jφ} 是滿足 magnitude 約束的矩陣，
    P(·) = STFT(ISTFT(·)) 是投影到一致集合。Griffin & Lim 證明這個量單調不增。
    """
    rng = np.random.default_rng(seed)
    phase = np.exp(2j * np.pi * rng.random(A.shape))
    nrm = float(np.sum(A * A)) + 1e-20
    log, snaps = [], {}
    for i in range(n_iter + 1):
        C = A * phase
        x = istft(C, L, H, N, length)
        P = stft(x, L, H, N)
        log.append({
            "iter": i,
            "inconsistency": float(np.sum(np.abs(C - P) ** 2) / nrm),
            "spectral_convergence": float(np.linalg.norm(np.abs(P) - A) / np.sqrt(nrm)),
        })
        if i in keep or i == n_iter:
            snaps[i] = x
        phase = np.exp(1j * np.angle(P))
    return x, log, snaps


# ── mel：兩種尺度、正規化可開關 ─────────────────────────────────
def hz_to_mel(f, scale: str):
    f = np.asarray(f, dtype=np.float64)
    if scale == "htk":                      # 大綱給的那一條：m = 2595 log10(1 + f/700)
        return 2595.0 * np.log10(1.0 + f / 700.0)
    # Slaney（Auditory Toolbox）：1 kHz 以下線性，以上對數
    f_sp, min_log_hz = 200.0 / 3, 1000.0
    min_log_mel, logstep = min_log_hz / f_sp, np.log(6.4) / 27.0
    return np.where(f >= min_log_hz,
                    min_log_mel + np.log(np.maximum(f, 1e-10) / min_log_hz) / logstep,
                    f / f_sp)


def mel_to_hz(m, scale: str):
    m = np.asarray(m, dtype=np.float64)
    if scale == "htk":
        return 700.0 * (10.0 ** (m / 2595.0) - 1.0)
    f_sp, min_log_hz = 200.0 / 3, 1000.0
    min_log_mel, logstep = min_log_hz / f_sp, np.log(6.4) / 27.0
    return np.where(m >= min_log_mel, min_log_hz * np.exp(logstep * (m - min_log_mel)), f_sp * m)


def mel_filterbank(fs: int, N: int, n_mels: int, scale: str = "slaney",
                   area_norm: bool = True, fmin: float = 0.0, fmax: float | None = None) -> np.ndarray:
    """三角 filterbank，shape (n_mels, N/2+1)。

    scale      "htk" 或 "slaney"
    area_norm  True = 每個三角形除以自己的頻寬（Slaney 式面積正規化）；False = 峰值為 1（HTK 式）
    """
    fmax = fmax or fs / 2
    freqs = np.linspace(0, fs / 2, N // 2 + 1)
    edges = mel_to_hz(np.linspace(hz_to_mel(fmin, scale), hz_to_mel(fmax, scale), n_mels + 2), scale)
    lo, ce, hi = edges[:-2, None], edges[1:-1, None], edges[2:, None]
    M = np.maximum(0.0, np.minimum((freqs - lo) / (ce - lo), (hi - freqs) / (hi - ce)))
    if area_norm:
        M *= 2.0 / (hi - lo)
    return M


def mel_inverse(mel_power: np.ndarray, M: np.ndarray, method: str = "pinv") -> np.ndarray:
    """mel power (n_mels, T) → linear power (N/2+1, T) 的近似還原。M 是寬矩陣，所以這一步必然有損。

    "pinv"  pseudo-inverse 後把負值截成 0（預設；大綱寫的就是這個）
    "nnls"  逐 frame 解非負最小平方。解是稀疏的，頻譜上會出現整格為零的洞；
            2026-09-17 用合成母音試過，log-spectral distance 反而比 pinv 差，所以不當預設。
    """
    if method == "pinv":
        return np.maximum(np.linalg.pinv(M) @ mel_power, 0.0)
    out = np.empty((M.shape[1], mel_power.shape[1]))
    for t in range(mel_power.shape[1]):
        out[:, t], _ = nnls(M, mel_power[:, t])
    return out


# ── 指標 ────────────────────────────────────────────────────────
def log_spectral_distance(x: np.ndarray, y: np.ndarray, L: int, H: int, N: int) -> float:
    """兩段波形的 log-spectral distance（dB，逐 frame 的 RMS 再平均）。對相位不敏感。"""
    n = min(len(x), len(y))
    X = 20 * np.log10(np.abs(stft(x[:n], L, H, N)) + 1e-5)
    Y = 20 * np.log10(np.abs(stft(y[:n], L, H, N)) + 1e-5)
    return float(np.mean(np.sqrt(np.mean((X - Y) ** 2, axis=0))))


def snr_db(ref: np.ndarray, est: np.ndarray) -> float:
    """波形域 SNR。**對相位極度敏感**——隨機相位重建的 SNR 會接近 0 dB 甚至為負，
    但聽起來仍然可懂；這正是「SNR 不預測可懂度」的現成例子。"""
    n = min(len(ref), len(est))
    e = ref[:n] - est[:n]
    return float(10 * np.log10(np.sum(ref[:n] ** 2) / (np.sum(e ** 2) + 1e-20)))


# ── Demo 1 用：顯示用 spectrogram、F0、formant ─────────────────
def pre_emphasis(x: np.ndarray, coef: float = 0.97) -> np.ndarray:
    return np.append(x[0], x[1:] - coef * x[:-1])


def spectrogram_db(x: np.ndarray, fs: int, win_ms: float, hop_ms: float, N: int) -> np.ndarray:
    L = max(8, int(round(fs * win_ms / 1000)))
    H = max(1, int(round(fs * hop_ms / 1000)))
    N = max(N, 1 << int(np.ceil(np.log2(L))))
    S = np.abs(stft(pre_emphasis(x), L, H, N))
    return 20 * np.log10(S + 1e-6)


def f0_autocorr(frame: np.ndarray, fs: int, fmin: float, fmax: float, voicing: float = 0.45):
    """最樸素的 autocorrelation F0。回傳 (f0 或 nan, 週期性強度)。

    只做最低限度的 octave 處理（取夠高的第一個峰）。fmin／fmax 設錯時它仍然會現場示範
    octave error——例如把 fmax 設成 150 去追一個 220 Hz 的聲音。
    """
    x = frame - frame.mean()
    if np.sqrt(np.mean(x * x)) < 1e-3:
        return np.nan, 0.0
    x = x * np.hanning(len(x))
    r = np.correlate(x, x, mode="full")[len(x) - 1:]
    # 補償 window 造成的自相關衰減（Boersma 1993 的做法）
    rw = np.correlate(np.hanning(len(x)), np.hanning(len(x)), mode="full")[len(x) - 1:]
    r = r / (r[0] + 1e-20) / (rw / rw[0] + 1e-6)
    lo, hi = int(fs / fmax), min(int(fs / fmin), len(r) // 2)
    if hi <= lo + 1:
        return np.nan, 0.0
    seg = r[lo:hi]
    # 週期訊號在 T、2T、3T 都有幾乎一樣高的峰。取「夠高的第一個峰」而不是全域最大，
    # 否則乾淨的 220 Hz 會被報成 110 Hz。
    peaks = [i for i in range(1, len(seg) - 1) if seg[i] >= seg[i - 1] and seg[i] > seg[i + 1]]
    if not peaks:
        return np.nan, 0.0
    best = max(seg[i] for i in peaks)
    if best <= 0:                                 # 全部的峰都在零以下（無聲、噪音）：0.9·best 反而比 best 大，
        return np.nan, float(best)                # 下面的 next() 會找不到東西——2026-09-28 真人收音時踩到
    k = lo + next(i for i in peaks if seg[i] >= 0.9 * best)
    strength = float(r[k])
    if strength < voicing:
        return np.nan, strength
    if 1 <= k < len(r) - 1:                       # 拋物線內插
        a, b, c = r[k - 1], r[k], r[k + 1]
        k = k + 0.5 * (a - c) / (a - 2 * b + c + 1e-20)
    return float(fs / k), strength


def lpc(x: np.ndarray, order: int) -> np.ndarray:
    """autocorrelation method。回傳 [1, -a_1, …, -a_p]，即 A(z) 的係數（H(z) = 1/A(z)）。"""
    r = np.correlate(x, x, mode="full")[len(x) - 1:len(x) + order]
    r[0] *= 1.0 + 1e-9
    a = solve_toeplitz(r[:-1], r[1:])
    return np.concatenate(([1.0], -a))


def formants(frame: np.ndarray, fs: int, n: int = 2, fs_lpc: int = 10000):
    """LPC 根 → 共振峰頻率。降到 10 kHz、order 12（男聲約五個共振峰的慣例值）。"""
    x = resample(frame, fs, fs_lpc)
    x = pre_emphasis(x - x.mean()) * np.hamming(len(x))
    if np.sqrt(np.mean(x * x)) < 1e-4:
        return [np.nan] * n
    roots = np.roots(lpc(x, 12))
    roots = roots[np.imag(roots) > 0.01]
    freq = np.angle(roots) * fs_lpc / (2 * np.pi)
    bw = -np.log(np.abs(roots) + 1e-12) * fs_lpc / np.pi
    ok = sorted(f for f, b in zip(freq, bw) if 90 < f < fs_lpc / 2 - 50 and b < 400)
    return (ok + [np.nan] * n)[:n]


# ── 合成訊號（--fake 與自我測試用；不是給學生聽的）───────────────
VOWELS = {"a": (730, 1090, 2440), "i": (270, 2290, 3010), "u": (300, 870, 2240)}  # Peterson & Barney 男聲平均


def synth_vowel(fs: int, dur: float, f0: np.ndarray | float, formant_hz, bw=(60, 90, 120)) -> np.ndarray:
    """脈衝串 → 串接的二階共振器。source-filter model 的最小實作。"""
    from scipy.signal import lfilter
    n = int(fs * dur)
    f0 = np.broadcast_to(np.asarray(f0, dtype=np.float64), (n,)) if np.ndim(f0) == 0 else np.interp(
        np.linspace(0, 1, n), np.linspace(0, 1, len(f0)), f0)
    ph = np.cumsum(f0 / fs)
    e = np.diff(np.floor(ph), prepend=0.0)          # 每過一個週期一個脈衝
    y = e
    for f, b in zip(formant_hz, bw):
        r = np.exp(-np.pi * b / fs)
        y = lfilter([1 - r], [1, -2 * r * np.cos(2 * np.pi * f / fs), r * r], y)
    return y / (np.max(np.abs(y)) + 1e-12) * 0.5

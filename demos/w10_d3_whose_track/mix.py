# -*- coding: utf-8 -*-
# 此檔由 bin/publish_demos.py 從 private repo 的 <course>/demos/ 複製而來，請勿在公開 repo 直接編輯。
"""W10 demo 3 的訊號那一半：把幾句話接成一段「對話」、切塊、SI-SDR。純 numpy。

投影片 p70 下半（真實對話的樣子）與 p74 的 What runs 對得上的就是這個檔：
  trim()       每一句先裁掉頭尾的靜音（LibriSpeech 的句子頭尾各有零點幾秒）
  activity()   一條乾淨的軌 → 每 10 ms 一格的「有沒有在說話」（能量門檻，短停頓補起來）
  build()      照設定檔的 turns 把句子擺上時間軸：lead > 0 是隔多久才接話，lead < 0 是搶在前一句結束前多久開口（重疊）
  chunks()     把長的那一段切成幾秒一塊
  si_sdr()     分離出來的一軌對乾淨的答案（Le Roux et al. 2019 的定義）

因為是自己接的，每一位說話人自己的那一軌（乾淨的）都還在——「誰在哪一格說話」是算出來的，不是人標的。
"""
from __future__ import annotations

import numpy as np

STEP = 0.010   # 參考標註與 DER 的格子：10 ms


def rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(x)) + 1e-20))


def frame_db(x: np.ndarray, fs: int, win_s: float = 0.025, step_s: float = STEP) -> np.ndarray:
    """每 10 ms 一格、25 ms 窗的 RMS（dB）。格數 = ceil(len / step)。"""
    step, win = int(round(step_s * fs)), int(round(win_s * fs))
    n = int(np.ceil(len(x) / step))
    pad = np.concatenate([x, np.zeros(n * step + win - len(x))])
    idx = np.arange(n)[:, None] * step + np.arange(win)[None, :]
    return 10 * np.log10(np.mean(np.square(pad[idx]), axis=1) + 1e-12)


def activity(x: np.ndarray, fs: int, floor_db: float = 40.0, min_pause_s: float = 0.2, min_speech_s: float = 0.05) -> np.ndarray:
    """有沒有在說話：比這一軌最響的一格低不到 floor_db 的算有；短於 min_pause_s 的停頓補起來；短於 min_speech_s 的孤立小段拿掉。"""
    db = frame_db(x, fs)
    on = db > (db.max() - floor_db)
    on = _fill(on, False, int(round(min_pause_s / STEP)))
    on = _fill(on, True, int(round(min_speech_s / STEP)))
    return on


def _fill(mask: np.ndarray, value: bool, max_len: int) -> np.ndarray:
    """把夾在中間、長度 < max_len、值是 value 的小段翻過來（頭尾的不動）。"""
    out = mask.copy()
    edges = np.flatnonzero(np.diff(mask.astype(np.int8))) + 1
    bounds = np.concatenate([[0], edges, [len(mask)]])
    for a, b in zip(bounds[:-1], bounds[1:]):
        if mask[a] == value and a > 0 and b < len(mask) and (b - a) < max_len:
            out[a:b] = not value
    return out


def trim(x: np.ndarray, fs: int, floor_db: float = 40.0) -> np.ndarray:
    on = np.flatnonzero(frame_db(x, fs) > frame_db(x, fs).max() - floor_db)
    step = int(round(STEP * fs))
    return x[on[0] * step: min((on[-1] + 1) * step, len(x))]


def segments(mask: np.ndarray, step_s: float = STEP) -> list[tuple[float, float]]:
    """bool 的格子 → [(開始秒, 結束秒), …]"""
    m = np.concatenate([[False], mask, [False]]).astype(np.int8)
    d = np.diff(m)
    return [(round(a * step_s, 3), round(b * step_s, 3)) for a, b in zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1))]


def build(turns: list[dict], audio: dict[str, np.ndarray], fs: int, level: float = 0.05, head_s: float = 0.5, tail_s: float = 0.5,
          floor_db: float = 40.0, min_pause_s: float = 0.2) -> dict:
    """turns: [{"spk": "user", "utt": "…", "lead": 0.3}, …]；audio: utt → 波形。
    回傳 {"mix", "tracks": {spk: 波形}, "turns": [加上 start/end], "ref": {spk: bool 格子}, "n", "seconds"}。
    每一句先裁靜音、調到同一個 RMS（level）；lead 是相對於「上一句結束」的秒數，負的就是重疊。"""
    placed, t = [], head_s
    for k, tr in enumerate(turns):
        x = trim(np.asarray(audio[tr["utt"]], dtype=np.float64), fs, floor_db)
        x = x * (level / rms(x))
        start = t + (float(tr["lead"]) if k else 0.0)
        a = int(round(start * fs))
        placed.append({**tr, "a": a, "x": x, "start": a / fs, "end": (a + len(x)) / fs})
        t = (a + len(x)) / fs
    n = int(round((max(p["end"] for p in placed) + tail_s) * fs))
    spks = list(dict.fromkeys(p["spk"] for p in placed))
    tracks = {s: np.zeros(n) for s in spks}
    for p in placed:
        tracks[p["spk"]][p["a"]: p["a"] + len(p["x"])] += p["x"]
    n_fr = int(np.ceil(n / int(round(STEP * fs))))
    ref = {s: activity(tracks[s], fs, floor_db, min_pause_s)[:n_fr] for s in spks}
    return {"mix": sum(tracks.values()), "tracks": tracks, "ref": ref, "n": n, "seconds": n / fs,
            "turns": [{k: v for k, v in p.items() if k not in ("a", "x")} for p in placed]}


def overlap_stats(ref: dict[str, np.ndarray]) -> dict:
    """參考標註的摘要：每位說了幾秒、同時說的有幾秒、佔「有人在說話」的比例。"""
    m = np.stack(list(ref.values()))
    n = m.sum(0)
    return {"speech_s": {s: round(float(v.sum() * STEP), 2) for s, v in ref.items()},
            "any_s": round(float((n > 0).sum() * STEP), 2), "overlap_s": round(float((n > 1).sum() * STEP), 2),
            "overlap_ratio": round(float((n > 1).sum() / max((n > 0).sum(), 1)), 4),
            "silence_s": round(float((n == 0).sum() * STEP), 2)}


def chunks(n: int, fs: int, chunk_s: float) -> list[tuple[int, int]]:
    """不重疊、等長的塊；最後一塊短於半塊就併進前一塊。"""
    c = int(round(chunk_s * fs))
    cuts = list(range(0, n, c)) + [n]
    if len(cuts) > 2 and cuts[-1] - cuts[-2] < c // 2:
        cuts.pop(-2)
    return list(zip(cuts[:-1], cuts[1:]))


def active_s(mask: np.ndarray, a: int, b: int, fs: int) -> float:
    """這一塊（樣本 a 到 b）裡，這位說話人有幾秒在說話。"""
    step = int(round(STEP * fs))
    return float(mask[a // step: int(np.ceil(b / step))].sum() * STEP)


def si_sdr(est: np.ndarray, ref: np.ndarray) -> float:
    """scale-invariant SDR（dB）：把 est 投影到 ref 上，投影到的對剩下的。兩者先去掉平均。"""
    est, ref = est - est.mean(), ref - ref.mean()
    den = float(np.dot(ref, ref))
    if den < 1e-12:
        return float("nan")
    tgt = (np.dot(est, ref) / den) * ref
    res = est - tgt
    return float(10 * np.log10((np.dot(tgt, tgt) + 1e-20) / (np.dot(res, res) + 1e-20)))


def blind_gain(est: np.ndarray, mix: np.ndarray) -> float:
    """分離模型輸出的音量是任意的：乘上讓它最貼近混合訊號的那個增益（最小平方；負的當 0）。"""
    den = float(np.dot(est, est))
    return max(float(np.dot(est, mix)) / den, 0.0) if den > 1e-12 else 0.0

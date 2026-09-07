#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
分析全雙工對話音訊（stereo WAV：ch0 = 輸入者 / ch1 = 模型），量出互動時序。

用途：W1 的延遲預算實測數字、W12 的 turn-taking 行為量化、以及挑選課堂用樣本。
方法：能量式 VAD（frame RMS + 自適應門檻 + 最小時長平滑）。這是近似，不是真正的
     VAD 模型；用於相對比較與挑樣本足夠，投影片上要標明是 energy-based 估計。

用法：
    python3 scripts/analyze_duplex_audio.py _media
    python3 scripts/analyze_duplex_audio.py _media --json out.json
"""

import argparse
import glob
import json
import os
import wave

import numpy as np

FRAME_MS = 25.0
HOP_MS = 10.0
MIN_SPEECH_MS = 120.0   # 短於此的發聲視為雜訊
MAX_GAP_MS = 200.0      # 短於此的空隙視為同一段內的停頓
PAUSE_MIN_S = 1.0       # 判定為「句中停頓」的最小長度


# ── 讀檔與 VAD ────────────────────────────────────────────────

def read_stereo(path):
    with wave.open(path, "rb") as w:
        if w.getnchannels() != 2:
            raise ValueError("%s 不是 stereo" % path)
        if w.getsampwidth() != 2:
            raise ValueError("%s 不是 16-bit PCM" % path)
        sr = w.getframerate()
        raw = w.readframes(w.getnframes())
    x = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    x = x.reshape(-1, 2)
    return x[:, 0], x[:, 1], sr


def frame_db(x, sr):
    n = int(round(sr * FRAME_MS / 1000.0))
    hop = int(round(sr * HOP_MS / 1000.0))
    if len(x) < n:
        return np.array([]), hop
    m = 1 + (len(x) - n) // hop
    idx = np.arange(n)[None, :] + hop * np.arange(m)[:, None]
    rms = np.sqrt(np.mean(x[idx] ** 2, axis=1))
    return 20.0 * np.log10(rms + 1e-10), hop


def runs(mask):
    """回傳 [(start_idx, end_idx_exclusive), ...] 的 True 區段。"""
    if mask.size == 0:
        return []
    d = np.diff(mask.astype(np.int8))
    starts = list(np.where(d == 1)[0] + 1)
    ends = list(np.where(d == -1)[0] + 1)
    if mask[0]:
        starts = [0] + starts
    if mask[-1]:
        ends = ends + [mask.size]
    return list(zip(starts, ends))


def vad(x, sr):
    db, hop = frame_db(x, sr)
    if db.size == 0:
        return [], 0.0
    floor = float(np.percentile(db, 20))
    peak = float(np.percentile(db, 98))
    if peak - floor < 12.0:          # 動態範圍太小 → 視為無語音
        return [], hop / sr
    thr = max(floor + 0.30 * (peak - floor), floor + 8.0)
    mask = db > thr

    fps = 1000.0 / HOP_MS
    min_speech = int(round(MIN_SPEECH_MS / 1000.0 * fps))
    max_gap = int(round(MAX_GAP_MS / 1000.0 * fps))

    for s, e in runs(~mask):         # 填補短空隙
        if 0 < s and e < mask.size and (e - s) <= max_gap:
            mask[s:e] = True
    for s, e in runs(mask):          # 刪掉過短發聲
        if (e - s) < min_speech:
            mask[s:e] = False

    step = hop / sr
    return [(s * step, e * step) for s, e in runs(mask)], step


# ── 指標 ─────────────────────────────────────────────────────

def total(segs):
    return sum(e - s for s, e in segs)


def overlap(a, b):
    t = 0.0
    for s1, e1 in a:
        for s2, e2 in b:
            t += max(0.0, min(e1, e2) - max(s1, s2))
    return t


def longest_internal_gap(segs):
    """句中最長停頓（前後都有語音）。回傳 (start, end) 或 None。"""
    best = None
    for i in range(len(segs) - 1):
        gap = (segs[i][1], segs[i + 1][0])
        if gap[1] - gap[0] >= PAUSE_MIN_S and (best is None or
                                               gap[1] - gap[0] > best[1] - best[0]):
            best = gap
    return best


def analyze(path):
    ch0, ch1, sr = read_stereo(path)
    user, _ = vad(ch0, sr)
    model, _ = vad(ch1, sr)
    dur = len(ch0) / sr

    r = {
        "duration_s": round(dur, 2),
        "user_speech_s": round(total(user), 2),
        "model_speech_s": round(total(model), 2),
        "overlap_s": round(overlap(user, model), 2),
        "n_user_seg": len(user),
        "n_model_seg": len(model),
        "user_segs": [(round(s, 2), round(e, 2)) for s, e in user],
        "model_segs": [(round(s, 2), round(e, 2)) for s, e in model],
    }

    # 句中停頓 → 模型是否在停頓中搶話（premature barge-in）
    gap = longest_internal_gap(user)
    r["user_internal_pause"] = None
    r["model_onset_in_user_gap_ms"] = None
    if gap:
        r["user_internal_pause"] = (round(gap[0], 2), round(gap[1], 2),
                                    round(gap[1] - gap[0], 2))
        onset = next((s for s, _ in model if gap[0] <= s < gap[1]), None)
        if onset is not None:
            r["model_onset_in_user_gap_ms"] = int(round((onset - gap[0]) * 1000))

    # 使用者說完最後一段後，模型多久開口（response latency）
    r["response_latency_ms"] = None
    if user and model:
        end = user[-1][1]
        onset = next((s for s, _ in model if s >= end), None)
        if onset is not None:
            r["response_latency_ms"] = int(round((onset - end) * 1000))

    # 使用者插話時，模型還講多久才停（stop latency）
    r["stop_latency_ms"] = None
    for us, _ in user:
        seg = next(((ms, me) for ms, me in model if ms < us < me), None)
        if seg:
            r["stop_latency_ms"] = int(round((seg[1] - us) * 1000))
            break

    return r


VARIANT_LABEL = {
    "moshika-ps3.0": "Moshi (base)",
    "e9b4a3ec-ckpt100-ps3.0": "Moshi + Fisher",
    "dee3429b-ckpt100-ps3.0": "Moshi + Seamless",
    "personaplex-ps0.0": "PersonaPlex (base)",
    "74a180f1-ckpt100-ps0.0": "PersonaPlex + Fisher",
    "7969ff51-ckpt100-ps0.0": "PersonaPlex + Seamless",
}
ORDER = ["moshika-ps3.0", "e9b4a3ec-ckpt100-ps3.0", "dee3429b-ckpt100-ps3.0",
         "personaplex-ps0.0", "74a180f1-ckpt100-ps0.0", "7969ff51-ckpt100-ps0.0"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--json")
    a = ap.parse_args()

    out = {}
    for path in sorted(glob.glob(os.path.join(a.root, "*", "*", "*.wav"))):
        parts = path.split(os.sep)
        task, sid, variant = parts[-3], parts[-2], parts[-1][:-4]
        out.setdefault(task, {}).setdefault(sid, {})[variant] = analyze(path)

    for task in sorted(out):
        print("\n" + "=" * 100)
        print("TASK: %s" % task)
        for sid in sorted(out[task], key=lambda s: int(s) if s.isdigit() else s):
            print("-" * 100)
            print("  sample %s" % sid)
            hdr = ("    %-24s %6s %6s %7s %8s %9s %9s"
                   % ("variant", "usr_s", "mdl_s", "ovlp_s", "usr_gap", "resp_ms", "stop_ms"))
            print(hdr)
            for v in ORDER:
                r = out[task][sid].get(v)
                if not r:
                    continue
                pause = "—"
                if r["user_internal_pause"]:
                    pause = "%.1fs" % r["user_internal_pause"][2]
                    if r["model_onset_in_user_gap_ms"] is not None:
                        pause += "+%dms" % r["model_onset_in_user_gap_ms"]
                print("    %-24s %6.1f %6.1f %7.2f %8s %9s %9s" % (
                    VARIANT_LABEL.get(v, v)[:24],
                    r["user_speech_s"], r["model_speech_s"], r["overlap_s"], pause,
                    r["response_latency_ms"] if r["response_latency_ms"] is not None else "—",
                    r["stop_latency_ms"] if r["stop_latency_ms"] is not None else "—"))
    print("\n註：usr_gap = ch0 中最長的內部靜默（可能是句中停頓，也可能是設計上的兩次發話間隔——")
    print("    因 task 而異，不要一律解讀為搶話）；`+Xms` 為模型在該靜默開始後多久發聲。")
    print("    resp_ms = 使用者說完到模型開口；stop_ms = 使用者插話後模型還講多久。")

    if a.json:
        with open(a.json, "w") as f:
            json.dump(out, f, ensure_ascii=False, indent=1)
        print("\nJSON → %s" % a.json)


if __name__ == "__main__":
    main()

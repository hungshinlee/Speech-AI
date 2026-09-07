#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
把全雙工對話的 stereo WAV 畫成雙軌時間軸 SVG（供投影片內嵌）。

圖只呈現資料：兩軌的發聲區段、重疊區、以及模型的最長靜默。論點寫在投影片文字上，
不寫進圖裡——這樣同一張圖可以在 W1 與 W12 用不同的論述重複使用。

顏色為淺色底投影片設計（明確指定，不隨主題變動）。

用法：
    python3 scripts/make_duplex_timeline.py in.wav -o out.svg \
        --title "Moshi (base)" --tmax 27.6
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze_duplex_audio import read_stereo, vad   # noqa: E402

W = 1000
PAD_L, PAD_R, PAD_T, PAD_B = 118, 24, 40, 74
LANE_H, LANE_GAP = 52, 16

C_USER = "#0b5cad"
C_MODEL = "#b45309"
C_OVERLAP = "#dc2626"
C_TEXT = "#1f2328"
C_MUTED = "#6b7280"
C_GRID = "#e5e7eb"


def intersect(a, b):
    out = []
    for s1, e1 in a:
        for s2, e2 in b:
            s, e = max(s1, s2), min(e1, e2)
            if e > s:
                out.append((s, e))
    return out


def longest_dead_air(user, model, owed_window=0.5, max_user_frac=0.30):
    """模型軌中最長的「該回應卻沒回應」區段。

    兩個條件都必須成立，否則圖會誘導出錯誤的對照：
      1. 這段沉默必須**緊接在使用者說完之後**（起點落在某個使用者段落結束後
         owed_window 秒內）——只有這時「回應」才是被欠著的。模型自己兩段話之間
         的間隔不算。
      2. 這段沉默期間使用者大致也沒在說話。模型在對方講話時保持安靜是正確行為。
    """
    if not user or len(model) < 2:
        return None
    user_ends = [e for _, e in user]
    best = None
    for i in range(len(model) - 1):
        s0, e0 = model[i][1], model[i + 1][0]
        if e0 <= s0:
            continue
        if not any(-0.2 <= s0 - ue <= owed_window for ue in user_ends):
            continue
        cov = sum(max(0.0, min(e0, ue) - max(s0, us)) for us, ue in user)
        if cov / (e0 - s0) > max_user_frac:
            continue
        if best is None or e0 - s0 > best[1] - best[0]:
            best = (s0, e0)
    return best


def esc(s):
    return (s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))


def build(wav, title, tmax=None, mark_silence=True):
    ch0, ch1, sr = read_stereo(wav)
    user, _ = vad(ch0, sr)
    model, _ = vad(ch1, sr)
    dur = len(ch0) / sr
    T = tmax or dur

    plot_w = W - PAD_L - PAD_R
    H = PAD_T + LANE_H * 2 + LANE_GAP + PAD_B

    def x(t):
        return PAD_L + plot_w * min(max(t, 0.0), T) / T

    y_user = PAD_T
    y_model = PAD_T + LANE_H + LANE_GAP

    o = []
    a = o.append
    a('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 %d %d" '
      'width="100%%" role="img" font-family="system-ui,-apple-system,Segoe UI,sans-serif">' % (W, H))
    a('<title>%s — two-track conversation timeline</title>' % esc(title))
    a('<desc>Channel 0 is the human input speaker; channel 1 is the model output. '
      'Red regions mark simultaneous speech.</desc>')
    a('<defs><pattern id="hatch" width="7" height="7" patternTransform="rotate(45)" '
      'patternUnits="userSpaceOnUse">'
      '<line x1="0" y1="0" x2="0" y2="7" stroke="%s" stroke-width="2.6" opacity=".85"/>'
      '</pattern></defs>' % C_OVERLAP)

    # 時間刻度
    step = 1 if T <= 12 else (2 if T <= 30 else 5)
    t = 0
    while t <= T + 1e-6:
        a('<line x1="%.1f" y1="%d" x2="%.1f" y2="%d" stroke="%s" stroke-width="1"/>'
          % (x(t), PAD_T - 8, x(t), PAD_T + LANE_H * 2 + LANE_GAP + 6, C_GRID))
        a('<text x="%.1f" y="%d" font-size="14" fill="%s" text-anchor="middle">%ds</text>'
          % (x(t), y_model + LANE_H + 24, C_MUTED, t))
        t += step

    # 軌道
    for label, sub, yy, col, segs in (
        ("User", "channel 0", y_user, C_USER, user),
        ("Model", "channel 1", y_model, C_MODEL, model),
    ):
        a('<rect x="%d" y="%d" width="%d" height="%d" fill="#f8fafc" stroke="%s" '
          'stroke-width="1" rx="3"/>' % (PAD_L, yy, plot_w, LANE_H, C_GRID))
        a('<text x="%d" y="%.1f" font-size="17" font-weight="600" fill="%s" '
          'text-anchor="end">%s</text>' % (PAD_L - 14, yy + LANE_H / 2 - 2, C_TEXT, label))
        a('<text x="%d" y="%.1f" font-size="12" fill="%s" text-anchor="end">%s</text>'
          % (PAD_L - 14, yy + LANE_H / 2 + 15, C_MUTED, sub))
        for s, e in segs:
            a('<rect x="%.1f" y="%d" width="%.1f" height="%d" fill="%s" rx="2.5" opacity=".92"/>'
              % (x(s), yy + 7, max(x(e) - x(s), 1.5), LANE_H - 14, col))
        if not segs:
            a('<text x="%.1f" y="%.1f" font-size="14" fill="%s" font-style="italic">'
              'no speech detected</text>' % (PAD_L + 12, yy + LANE_H / 2 + 5, C_MUTED))

    # 重疊區：跨兩軌的紅色斜線標記
    ov = intersect(user, model)
    for s, e in ov:
        a('<rect x="%.1f" y="%d" width="%.1f" height="%d" fill="url(#hatch)" '
          'stroke="%s" stroke-width="1.2"/>'
          % (x(s), y_user + 7, max(x(e) - x(s), 2.0),
             (y_model + LANE_H - 7) - (y_user + 7), C_OVERLAP))

    # 模型最長靜默：量測跨距
    gap = longest_dead_air(user, model) if mark_silence else None
    if gap and gap[1] - gap[0] >= 1.5:
        gy = y_model + LANE_H + 54
        x1, x2 = x(gap[0]), x(gap[1])
        a('<line x1="%.1f" y1="%d" x2="%.1f" y2="%d" stroke="%s" stroke-width="1.4"/>'
          % (x1, gy, x2, gy, C_TEXT))
        for xx in (x1, x2):
            a('<line x1="%.1f" y1="%d" x2="%.1f" y2="%d" stroke="%s" stroke-width="1.4"/>'
              % (xx, gy - 5, xx, gy + 5, C_TEXT))
        a('<text x="%.1f" y="%d" font-size="14" fill="%s" text-anchor="middle">'
          'no response for %.1f s</text>' % ((x1 + x2) / 2, gy - 10, C_TEXT, gap[1] - gap[0]))

    # 標題與圖例
    a('<text x="%d" y="24" font-size="19" font-weight="700" fill="%s">%s</text>'
      % (PAD_L, C_TEXT, esc(title)))
    if ov:
        tot = sum(e - s for s, e in ov)
        a('<text x="%d" y="24" font-size="14" fill="%s" text-anchor="end">'
          'simultaneous speech: %.2f s</text>' % (W - PAD_R, C_OVERLAP, tot))
    else:
        a('<text x="%d" y="24" font-size="14" fill="%s" text-anchor="end">'
          'simultaneous speech: none</text>' % (W - PAD_R, C_MUTED))
    a('</svg>')
    return "\n".join(o)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("wav")
    p.add_argument("-o", "--out", required=True)
    p.add_argument("--title", default="")
    p.add_argument("--tmax", type=float)
    p.add_argument("--no-silence", action="store_true",
                   help="不自動標記死寂區段（例如 backchannel 圖不需要）")
    a = p.parse_args()
    svg = build(a.wav, a.title or os.path.basename(a.wav), a.tmax,
                mark_silence=not a.no_silence)
    d = os.path.dirname(a.out)
    if d and not os.path.isdir(d):
        os.makedirs(d)
    with open(a.out, "w") as f:
        f.write(svg)
    print("→ %s (%d bytes)" % (a.out, len(svg)))


if __name__ == "__main__":
    main()

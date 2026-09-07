#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
產生 W1 投影片用的原創圖：三種架構的資料流，以及延遲預算母版。

三張架構圖共用同一套視覺語彙（連接線粗細編碼資訊量），才能讓學生用看的就發現
文字瓶頸。因此用同一支腳本產生，而不是三個手寫檔案。

輸出：slides/assets/w01/arch-{cascade,e2e,duplex}.svg
      slides/assets/w01/latency-budget-blank.svg

用法：python3 scripts/make_figs_w01.py
"""

import os

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "slides", "assets", "w01")

C_TEXT = "#1f2328"
C_MUTED = "#6b7280"
C_BOX = "#f1f5f9"
C_EDGE = "#94a3b8"
C_AUDIO = "#0b5cad"
C_TOKEN = "#7c3aed"
C_TEXTLINK = "#b45309"
C_WARN = "#dc2626"
FONT = "system-ui,-apple-system,Segoe UI,sans-serif"

W, H = 1000, 322
YC = 130
BOX_H = 76

# 連接線粗細 = 資訊量。文字最細，這就是「瓶頸」的視覺編碼。
STREAM = {
    "audio": (18, C_AUDIO, "audio"),
    "tokens": (12, C_TOKEN, "speech tokens"),
    "text": (4, C_TEXTLINK, "text only"),
}


def head(w=W, h=H):
    return ['<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 %d %d" width="100%%" '
            'role="img" font-family="%s">' % (w, h, FONT)]


def box(x, w, label, sub=None, y=YC, h=BOX_H, fill=C_BOX, stroke=C_EDGE, bold=True):
    o = ['<rect x="%.1f" y="%.1f" width="%.1f" height="%.1f" rx="7" fill="%s" '
         'stroke="%s" stroke-width="1.6"/>' % (x, y - h / 2, w, h, fill, stroke)]
    ty = y + (0 if sub is None else -8)
    o.append('<text x="%.1f" y="%.1f" font-size="22" font-weight="%s" fill="%s" '
             'text-anchor="middle" dominant-baseline="middle">%s</text>'
             % (x + w / 2, ty, "650" if bold else "400", C_TEXT, label))
    if sub:
        o.append('<text x="%.1f" y="%.1f" font-size="17" fill="%s" text-anchor="middle" '
                 'dominant-baseline="middle">%s</text>' % (x + w / 2, y + 20, C_MUTED, sub))
    return o


def stream(x1, x2, kind, y=YC, label=False, label_dy=0):
    th, col, name = STREAM[kind]
    o = ['<rect x="%.1f" y="%.1f" width="%.1f" height="%.1f" fill="%s" opacity=".9"/>'
         % (x1, y - th / 2, x2 - x1, th, col)]
    o.append('<polygon points="%.1f,%.1f %.1f,%.1f %.1f,%.1f" fill="%s"/>'
             % (x2, y - 9, x2 + 11, y, x2, y + 9, col))
    if label:
        o.append('<text x="%.1f" y="%.1f" font-size="16" fill="%s" text-anchor="middle">%s</text>'
                 % ((x1 + x2) / 2, y - th / 2 - 12 - label_dy, col, name))
    return o


def legend(y, x0=26, items=("audio", "tokens", "text")):
    """底部圖例。取代線上重複的串流標籤，把水平空間讓給字級。"""
    o, x = [], x0
    for kind in items:
        th, col, name = STREAM[kind]
        o.append('<rect x="%.1f" y="%.1f" width="34" height="%.1f" fill="%s" opacity=".9"/>'
                 % (x, y - th / 2, th, col))
        o.append('<text x="%.1f" y="%.1f" font-size="17" fill="%s" '
                 'dominant-baseline="middle">%s</text>' % (x + 42, y, col, name))
        x += 42 + len(name) * 9.4 + 34
    return o


def caption(text, y, x=None, col=C_MUTED, size=18, anchor="middle"):
    return ['<text x="%.1f" y="%.1f" font-size="%s" fill="%s" text-anchor="%s">%s</text>'
            % (x if x is not None else W / 2, y, size, col, anchor, text)]


def waist_marker(x, label):
    """在文字瓶頸處畫一個標記。"""
    return ['<line x1="%.1f" y1="%.1f" x2="%.1f" y2="%.1f" stroke="%s" '
            'stroke-width="1.6" stroke-dasharray="4 3"/>' % (x, YC - 60, x, YC + 84, C_WARN),
            '<text x="%.1f" y="%.1f" font-size="17.5" font-weight="600" fill="%s" '
            'text-anchor="middle">%s</text>' % (x, YC + 106, C_WARN, label)]


def chain(items, x0=26, gap=30):
    """items: [('box', w, label, sub) | ('stream', kind)] 依序排版，回傳 (svg, 位置表)。"""
    o, pos, x = [], [], x0
    for it in items:
        if it[0] == "box":
            _, w, label, sub = it
            o += box(x, w, label, sub)
            pos.append(("box", x, x + w))
            x += w
        else:
            x1 = x + gap * 0.35
            x2 = x + gap * 0.35 + (it[2] if len(it) > 2 else gap)
            o += stream(x1, x2, it[1])
            pos.append(("stream", x1, x2))
            x = x2 + 13
    return o, pos, x


def fig_cascade():
    """標題由投影片承擔，圖只放標籤、瓶頸標記、圖例與一行補充。"""
    items = [("box", 100, "Mic", None), ("stream", "audio", 48),
             ("box", 176, "VAD /", "endpointing"), ("stream", "audio", 44),
             ("box", 124, "ASR", None), ("stream", "text", 52),
             ("box", 134, "LLM", None), ("stream", "text", 52),
             ("box", 124, "TTS", None), ("stream", "audio", 44),
             ("box", 124, "Speaker", None)]
    body, pos, xend = chain(items)
    w, h = xend + 26, 274
    o = head(w, h)
    o += body
    o += waist_marker((pos[5][1] + pos[5][2]) / 2, "paralinguistics discarded here")
    o += legend(224, items=("audio", "text"))
    o += caption("The system cannot act on anything the transcript does not carry: "
                 "emotion, laughter, hesitation, speaking rate.", 258, x=w / 2)
    o.append("</svg>")
    return "\n".join(o)


def fig_e2e():
    items = [("box", 100, "Mic", None), ("stream", "audio", 50),
             ("box", 196, "Audio encoder", "SSL / codec"), ("stream", "tokens", 56),
             ("box", 248, "Speech–text LM", "one autoregressive model"),
             ("stream", "tokens", 56), ("box", 196, "Codec decoder", "vocoder"),
             ("stream", "audio", 46), ("box", 124, "Speaker", None)]
    body, pos, xend = chain(items)
    w, h = xend + 26, 314
    o = head(w, h)
    o += body
    lm_x1, lm_x2 = pos[4][1], pos[4][2]
    o += box(lm_x1 - 20, (lm_x2 - lm_x1) + 40, "inner monologue (text)", None,
             y=212, h=44, fill="#fff7ed", stroke=C_TEXTLINK, bold=False)
    o.append('<line x1="%.1f" y1="%.1f" x2="%.1f" y2="%.1f" stroke="%s" stroke-width="1.5" '
             'stroke-dasharray="4 3"/>' % ((lm_x1 + lm_x2) / 2, YC + 38,
                                           (lm_x1 + lm_x2) / 2, 188, C_TEXTLINK))
    o += legend(262, items=("audio", "tokens"))
    o += caption("Text is kept inside the loop as a planning channel, not as the interface.",
                 296, x=w / 2)
    o.append("</svg>")
    return "\n".join(o)


def fig_duplex():
    w, h = 1130, 372
    yu, ym = 96, 240
    lm_x1, lm_x2, lm_yc, lm_h = 392, 712, 168, 160
    o = head(w, h)

    o += box(26, 142, "User", "channel 0", y=yu, h=58)
    o += stream(176, lm_x1 - 11, "audio", y=yu, label=True)

    o.append('<rect x="%.1f" y="%.1f" width="%.1f" height="%.1f" rx="7" fill="%s" '
             'stroke="%s" stroke-width="1.6"/>'
             % (lm_x1, lm_yc - lm_h / 2, lm_x2 - lm_x1, lm_h, C_BOX, C_EDGE))
    for i, (txt, sz, wt, col) in enumerate([
            ("Full-duplex LM", 25, "650", C_TEXT),
            ("one step per frame", 17, "400", C_MUTED),
            ("speak · stay silent · keep listening", 18, "600", C_AUDIO)]):
        o.append('<text x="%.1f" y="%.1f" font-size="%s" font-weight="%s" fill="%s" '
                 'text-anchor="middle">%s</text>'
                 % ((lm_x1 + lm_x2) / 2, lm_yc - 28 + i * 32, sz, wt, col, txt))

    o += stream(lm_x2, 806, "tokens", y=ym, label=True, label_dy=-44)  # 標籤放箭頭下方，避開 LM 方塊
    o += box(818, 162, "Codec dec.", None, y=ym, h=58)
    o += stream(988, 1060, "audio", y=ym)
    o += caption("channel 1", 304, x=998, size=17)

    # 回饋線走最底部，說明文字放在線的下方，不再與線交疊
    py = 326
    o.append('<path d="M 1078 %.1f L 1078 %.1f L 540 %.1f L 540 %.1f" fill="none" '
             'stroke="%s" stroke-width="1.5" stroke-dasharray="5 4"/>'
             % (ym + 32, py, py, lm_yc + lm_h / 2 + 14, C_MUTED))
    o.append('<polygon points="%.1f,%.1f %.1f,%.1f %.1f,%.1f" fill="%s"/>'
             % (540, lm_yc + lm_h / 2 + 1, 533, lm_yc + lm_h / 2 + 15,
                547, lm_yc + lm_h / 2 + 15, C_MUTED))
    o += caption("the model hears itself → acoustic echo cancellation is mandatory (W10)",
                 py + 28, x=w / 2, col=C_MUTED, size=17)
    o.append("</svg>")
    return "\n".join(o)


def fig_latency_blank():
    """無圖內標題——投影片標題已經說了同一件事。"""
    w, h = 1000, 330
    x0, x1, y = 40, 960, 104
    segs = ["endpoint\ndetection", "ASR tail", "LLM prefill\n+ first token",
            "TTS\nfirst packet", "playout\nbuffer"]
    o = head(w, h)
    o += caption("widths are placeholders, not measurements", 30, x=x1, col=C_MUTED,
                 size=17, anchor="end")

    n = len(segs)
    sw = (x1 - x0 - (n - 1) * 12) / n
    for i, label in enumerate(segs):
        xa = x0 + i * (sw + 12)
        o.append('<rect x="%.1f" y="%.1f" width="%.1f" height="72" rx="4" fill="#f8fafc" '
                 'stroke="%s" stroke-width="1.6" stroke-dasharray="6 4"/>'
                 % (xa, y - 36, sw, C_EDGE))
        for j, ln in enumerate(label.split("\n")):
            o.append('<text x="%.1f" y="%.1f" font-size="17.5" fill="%s" '
                     'text-anchor="middle">%s</text>'
                     % (xa + sw / 2, y - 10 + j * 21, C_TEXT, ln))
        o.append('<text x="%.1f" y="%.1f" font-size="20" font-weight="700" fill="%s" '
                 'text-anchor="middle">____ ms</text>' % (xa + sw / 2, y + 62, C_MUTED))

    split = x0 + 2 * (sw + 12) - 6
    o.append('<line x1="%.1f" y1="%.1f" x2="%.1f" y2="%.1f" stroke="%s" stroke-width="1.4"/>'
             % (x0, y + 88, split, y + 88, C_AUDIO))
    o += caption("algorithmic — lookahead; no hardware fixes this", y + 112,
                 x=(x0 + split) / 2, col=C_AUDIO, size=17)
    o.append('<line x1="%.1f" y1="%.1f" x2="%.1f" y2="%.1f" stroke="%s" stroke-width="1.4"/>'
             % (split, y + 88, x1, y + 88, C_TEXTLINK))
    o += caption("computational — scales with model size and batching", y + 112,
                 x=(split + x1) / 2, col=C_TEXTLINK, size=17)

    ty = y + 156
    o += ['<text x="%.1f" y="%.1f" font-size="20" font-weight="700" fill="%s">'
          'total = ______ ms</text>' % (x0, ty, C_TEXT)]
    o.append('<rect x="%.1f" y="%.1f" width="340" height="32" rx="5" fill="#dcfce7" '
             'stroke="#16a34a" stroke-width="1.3"/>' % (x0 + 250, ty - 24))
    o += ['<text x="%.1f" y="%.1f" font-size="17.5" fill="#15803d" text-anchor="middle">'
          'humans: modal turn gap ≈ 0 ms</text>' % (x0 + 420, ty - 2)]
    o += caption("Stivers et al., PNAS 2009 (10 languages): cross-language means differ by "
                 "at most ±250 ms;", h - 34, x=w / 2, col=C_MUTED, size=17)
    o += caption("dispreferred answers are delayed up to ~1 s — delay itself carries meaning.",
                 h - 13, x=w / 2, col=C_MUTED, size=17)
    o.append("</svg>")
    return "\n".join(o)


def main():
    if not os.path.isdir(OUT):
        os.makedirs(OUT)
    for name, fn in (("arch-cascade", fig_cascade), ("arch-e2e", fig_e2e),
                     ("arch-duplex", fig_duplex),
                     ("latency-budget-blank", fig_latency_blank)):
        p = os.path.join(OUT, name + ".svg")
        with open(p, "w") as f:
            f.write(fn())
        print("→ %s (%d bytes)" % (os.path.relpath(p, os.path.dirname(OUT)), os.path.getsize(p)))


if __name__ == "__main__":
    main()

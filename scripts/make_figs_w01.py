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

W, H = 1000, 290
YC = 120
BOX_H = 66

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
    ty = y + (0 if sub is None else -6)
    o.append('<text x="%.1f" y="%.1f" font-size="17" font-weight="%s" fill="%s" '
             'text-anchor="middle" dominant-baseline="middle">%s</text>'
             % (x + w / 2, ty, "650" if bold else "400", C_TEXT, label))
    if sub:
        o.append('<text x="%.1f" y="%.1f" font-size="13" fill="%s" text-anchor="middle" '
                 'dominant-baseline="middle">%s</text>' % (x + w / 2, y + 14, C_MUTED, sub))
    return o


def stream(x1, x2, kind, y=YC, label=True):
    th, col, name = STREAM[kind]
    o = ['<rect x="%.1f" y="%.1f" width="%.1f" height="%.1f" fill="%s" opacity=".9"/>'
         % (x1, y - th / 2, x2 - x1, th, col)]
    o.append('<polygon points="%.1f,%.1f %.1f,%.1f %.1f,%.1f" fill="%s"/>'
             % (x2, y - 9, x2 + 11, y, x2, y + 9, col))
    if label:
        o.append('<text x="%.1f" y="%.1f" font-size="12.5" fill="%s" text-anchor="middle">%s</text>'
                 % ((x1 + x2) / 2, y - th / 2 - 9, col, name))
    return o


def caption(text, y, x=None, col=C_MUTED, size=14, anchor="middle"):
    return ['<text x="%.1f" y="%.1f" font-size="%s" fill="%s" text-anchor="%s">%s</text>'
            % (x if x is not None else W / 2, y, size, col, anchor, text)]


def waist_marker(x, label):
    """在文字瓶頸處畫一個標記。"""
    return ['<line x1="%.1f" y1="%.1f" x2="%.1f" y2="%.1f" stroke="%s" '
            'stroke-width="1.6" stroke-dasharray="4 3"/>' % (x, YC - 52, x, YC + 74, C_WARN),
            '<text x="%.1f" y="%.1f" font-size="13.5" font-weight="600" fill="%s" '
            'text-anchor="middle">%s</text>' % (x, YC + 92, C_WARN, label)]


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
    items = [("box", 86, "Mic", None), ("stream", "audio", 46),
             ("box", 150, "VAD /", "endpointing"), ("stream", "audio", 40),
             ("box", 108, "ASR", None), ("stream", "text", 86),
             ("box", 116, "LLM", None), ("stream", "text", 86),
             ("box", 108, "TTS", None), ("stream", "audio", 40),
             ("box", 86, "Speaker", None)]
    body, pos, xend = chain(items)
    w = xend + 26
    o = head(w)
    o += body
    o += waist_marker((pos[5][1] + pos[5][2]) / 2, "paralinguistics discarded here")
    o += caption("Every module is separately trainable, monitorable, replaceable.", 42,
                 x=w / 2, col=C_TEXT, size=15)
    o += caption("The system cannot act on anything the transcript does not carry: "
                 "emotion, laughter, hesitation, speaking rate.", 250, x=w / 2)
    o.append("</svg>")
    return "\n".join(o)


def fig_e2e():
    items = [("box", 86, "Mic", None), ("stream", "audio", 44),
             ("box", 168, "Audio encoder", "SSL / codec"), ("stream", "tokens", 74),
             ("box", 210, "Speech–text LM", "one autoregressive model"),
             ("stream", "tokens", 74), ("box", 168, "Codec decoder", "vocoder"),
             ("stream", "audio", 40), ("box", 86, "Speaker", None)]
    body, pos, xend = chain(items)
    w = xend + 26
    o = head(w)
    o += body
    lm_x1, lm_x2 = pos[4][1], pos[4][2]
    o += box(lm_x1 + 18, lm_x2 - lm_x1 - 36, "inner monologue (text)", None,
             y=YC + 96, h=34, fill="#fff7ed", stroke=C_TEXTLINK, bold=False)
    o.append('<line x1="%.1f" y1="%.1f" x2="%.1f" y2="%.1f" stroke="%s" stroke-width="1.4" '
             'stroke-dasharray="4 3"/>' % ((lm_x1 + lm_x2) / 2, YC + 33,
                                           (lm_x1 + lm_x2) / 2, YC + 71, C_TEXTLINK))
    o += caption("Paralinguistics survive end to end — nothing is forced through a transcript.",
                 42, x=w / 2, col=C_TEXT, size=15)
    o += caption("Text is kept inside the loop as a planning channel, not as the interface.",
                 268, x=w / 2)
    o.append("</svg>")
    return "\n".join(o)


def fig_duplex():
    w, h = 1060, 340
    yu, ym = 120, 232
    lm_x1, lm_x2, lm_yc, lm_h = 380, 680, (yu + ym) / 2, 150
    o = head(w, h)
    o += caption("Silence is a token the model must emit. Full duplex is not a scheduler "
                 "bolted on — it is sequence modelling.", 40, x=w / 2, col=C_TEXT, size=15)

    # 輸入：使用者串流直接指進方塊左緣
    o += box(26, 124, "User", "channel 0", y=yu, h=50)
    o += stream(158, lm_x1 - 11, "audio", y=yu)

    o.append('<rect x="%.1f" y="%.1f" width="%.1f" height="%.1f" rx="7" fill="%s" '
             'stroke="%s" stroke-width="1.6"/>'
             % (lm_x1, lm_yc - lm_h / 2, lm_x2 - lm_x1, lm_h, C_BOX, C_EDGE))
    for i, (txt, sz, wt, col) in enumerate([
            ("Full-duplex LM", 19, "650", C_TEXT),
            ("one step per frame", 13, "400", C_MUTED),
            ("speak · stay silent · keep listening", 14, "600", C_AUDIO)]):
        o.append('<text x="%.1f" y="%.1f" font-size="%s" font-weight="%s" fill="%s" '
                 'text-anchor="middle">%s</text>'
                 % ((lm_x1 + lm_x2) / 2, lm_yc - 24 + i * 26, sz, wt, col, txt))

    # 輸出：從方塊右緣出發
    o += stream(lm_x2, 792, "tokens", y=ym)
    o += box(800, 140, "Codec dec.", None, y=ym, h=50)
    o += stream(948, 1020, "audio", y=ym)
    o += caption("channel 1", ym + 26, x=984, size=12.5)

    # 回饋：模型聽見自己，走底部繞回，箭頭指進方塊下緣
    py = h - 30
    o.append('<path d="M 1020 %.1f L 1020 %.1f L 512 %.1f L 512 %.1f" fill="none" '
             'stroke="%s" stroke-width="1.5" stroke-dasharray="5 4"/>'
             % (ym + 30, py, py, lm_yc + lm_h / 2 + 13, C_MUTED))
    o.append('<polygon points="%.1f,%.1f %.1f,%.1f %.1f,%.1f" fill="%s"/>'
             % (512, lm_yc + lm_h / 2 + 1, 506, lm_yc + lm_h / 2 + 14,
                518, lm_yc + lm_h / 2 + 14, C_MUTED))
    o += caption("the model hears itself → acoustic echo cancellation is mandatory (W10)",
                 py - 11, x=772, col=C_MUTED, size=13)
    o += caption("Input never stops entering the computation, not even while the model speaks.",
                 78, x=w / 2)
    o.append("</svg>")
    return "\n".join(o)


def fig_latency_blank():
    w, h = 1000, 306
    x0, x1, y = 40, 960, 116
    segs = ["endpoint\ndetection", "ASR tail", "LLM prefill\n+ first token",
            "TTS\nfirst packet", "playout\nbuffer"]
    o = head(w, h)
    o += ['<text x="%d" y="34" font-size="18" font-weight="700" fill="%s">'
          'Latency budget — we refill this every week</text>' % (x0, C_TEXT)]
    o += caption("widths are placeholders, not measurements", 34, x=x1, col=C_MUTED,
                 size=13, anchor="end")

    n = len(segs)
    sw = (x1 - x0 - (n - 1) * 12) / n
    for i, label in enumerate(segs):
        xa = x0 + i * (sw + 12)
        o.append('<rect x="%.1f" y="%.1f" width="%.1f" height="62" rx="4" fill="#f8fafc" '
                 'stroke="%s" stroke-width="1.6" stroke-dasharray="6 4"/>'
                 % (xa, y - 31, sw, C_EDGE))
        for j, ln in enumerate(label.split("\n")):
            o.append('<text x="%.1f" y="%.1f" font-size="13.5" fill="%s" '
                     'text-anchor="middle">%s</text>'
                     % (xa + sw / 2, y - 8 + j * 16, C_TEXT, ln))
        o.append('<text x="%.1f" y="%.1f" font-size="16" font-weight="700" fill="%s" '
                 'text-anchor="middle">____ ms</text>' % (xa + sw / 2, y + 54, C_MUTED))

    split = x0 + 2 * (sw + 12) - 6
    o.append('<line x1="%.1f" y1="%.1f" x2="%.1f" y2="%.1f" stroke="%s" stroke-width="1.4"/>'
             % (x0, y + 76, split, y + 76, C_AUDIO))
    o += caption("algorithmic — lookahead; no hardware fixes this", y + 96,
                 x=(x0 + split) / 2, col=C_AUDIO, size=13)
    o.append('<line x1="%.1f" y1="%.1f" x2="%.1f" y2="%.1f" stroke="%s" stroke-width="1.4"/>'
             % (split, y + 76, x1, y + 76, C_TEXTLINK))
    o += caption("computational — scales with model size and batching", y + 96,
                 x=(split + x1) / 2, col=C_TEXTLINK, size=13)

    ty = y + 130
    o += ['<text x="%.1f" y="%.1f" font-size="16" font-weight="700" fill="%s">'
          'total = ______ ms</text>' % (x0, ty, C_TEXT)]
    o.append('<rect x="%.1f" y="%.1f" width="290" height="26" rx="5" fill="#dcfce7" '
             'stroke="#16a34a" stroke-width="1.3"/>' % (x0 + 190, ty - 19))
    o += ['<text x="%.1f" y="%.1f" font-size="13.5" fill="#15803d" text-anchor="middle">'
          'humans: modal turn gap ≈ 0 ms</text>' % (x0 + 335, ty - 1)]
    o += caption("Stivers et al., PNAS 2009 (10 languages): cross-language means differ by "
                 "at most ±250 ms;", h - 30, x=w / 2, col=C_MUTED, size=13)
    o += caption("dispreferred answers are delayed up to ~1 s — delay itself carries meaning.",
                 h - 12, x=w / 2, col=C_MUTED, size=13)
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

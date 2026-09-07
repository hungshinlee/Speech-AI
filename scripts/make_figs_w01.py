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
import re
import xml.dom.minidom as _minidom

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


def fig_course_map():
    """14 週的依賴結構（不是週次清單——那是投影片的表格在做的事）。

    重點是 W6 那條繞過整個 Part II、指向 Part III 全欄的虛線：課程地圖要傳達的是
    「哪一週撐著哪一週」，而不是「第幾週上什麼」。W6 撐的不是某一週，是整個 Part III，
    所以畫成括號而不是箭頭——順便也避開跟灰色匯總箭頭撞在同一個點上。
    """
    w, h = 1236, 458
    o = head(w, h)
    BW, BH, GAP = 248, 44, 52
    cols = [
        (30, [("W1", "systems view", "today"),
              ("W2", "signals & front ends", None), ("W3", "alignment (CTC)", None),
              ("W4", "streaming & RNN-T", None), ("W5", "self-supervised", None),
              ("W6", "codecs / tokens", "hinge")]),
        (490, [("W7", "ASR", None), ("W8", "TTS: generation", None),
               ("W9", "TTS: control", None), ("W10", "front end / AEC", None)]),
        (930, [("W11", "audio-native LM", None), ("W12", "turn-taking", None),
               ("W13", "full-duplex arch.", None), ("W14", "evaluation", None)]),
    ]
    heads = ["Part I \u2014 foundations", "Part II \u2014 modules",
             "Part III \u2014 dialogue systems"]
    y0 = 74

    for ci, (cx, items) in enumerate(cols):
        o += ['<text x="%d" y="%d" font-size="19" font-weight="700" fill="%s">%s</text>'
              % (cx, 46, C_TEXT, heads[ci])]
        for ri, (wk, topic, kind) in enumerate(items):
            y = y0 + ri * GAP
            hinge, today = kind == "hinge", kind == "today"
            fill = "#eff6ff" if hinge else ("#ffffff" if today else C_BOX)
            o.append('<rect x="%d" y="%d" width="%d" height="%d" rx="6" fill="%s" '
                     'stroke="%s" stroke-width="%s"%s/>'
                     % (cx, y, BW, BH, fill, C_AUDIO if hinge else C_EDGE,
                        "2.4" if hinge else "1.5",
                        ' stroke-dasharray="5 4"' if today else ""))
            o.append('<text x="%d" y="%.1f" font-size="19" font-weight="700" fill="%s" '
                     'dominant-baseline="middle">%s</text>'
                     % (cx + 14, y + BH / 2, C_AUDIO if hinge else C_TEXT, wk))
            o.append('<text x="%d" y="%.1f" font-size="16" fill="%s" '
                     'dominant-baseline="middle">%s</text>'
                     % (cx + 68, y + BH / 2, C_TEXT if hinge else C_MUTED, topic))
            if today:
                o.append('<text x="%d" y="%.1f" font-size="14" font-weight="600" fill="%s" '
                         'text-anchor="end" dominant-baseline="middle">today</text>'
                         % (cx + BW - 12, y + BH / 2, C_MUTED))

    # Part I → Part II，以及 Part II → Part III 的匯總箭頭（灰色＝一般的先後依賴）
    for x1, x2, n1, n2 in ((278, 478, 6, 4), (738, 896, 4, 4)):
        ym1 = y0 + (n1 - 1) * GAP / 2 + BH / 2
        ym2 = y0 + (n2 - 1) * GAP / 2 + BH / 2
        o.append('<path d="M %d %.1f C %d %.1f, %d %.1f, %d %.1f" fill="none" '
                 'stroke="%s" stroke-width="9" opacity=".55"/>'
                 % (x1, ym1, x1 + 70, ym1, x2 - 70, ym2, x2, ym2, C_EDGE))
        o.append('<polygon points="%d,%.1f %d,%.1f %d,%.1f" fill="%s" opacity=".7"/>'
                 % (x2, ym2 - 11, x2 + 14, ym2, x2, ym2 + 11, C_EDGE))

    # W6 → 整個 Part III：繞過 Part II 下方，指進整欄的虛線框
    # 畫成「框 + 一支從下方進來的箭頭」而不是指向某一週，因為 W6 撐的是整個 Part III；
    # 同時也避開跟灰色匯總箭頭的箭頭撞在同一點上。
    y6 = y0 + 5 * GAP + BH / 2
    rx0, ry0 = 914, 56
    rw, ry1 = 280, y0 + 3 * GAP + BH + 14
    route_y = 436
    o.append('<rect x="%d" y="%d" width="%d" height="%d" rx="10" fill="none" '
             'stroke="%s" stroke-width="2.2" stroke-dasharray="8 5" opacity=".85"/>'
             % (rx0, ry0, rw, ry1 - ry0, C_AUDIO))
    xin = rx0 + rw / 2
    o.append('<path d="M %d %.1f C %d %.1f, %d %d, %d %d H %.1f C %.1f %d, %.1f %d, %.1f %d" '
             'fill="none" stroke="%s" stroke-width="2.6" stroke-dasharray="8 5"/>'
             % (278, y6, 340, y6, 370, route_y, 410, route_y,
                xin - 60, xin - 18, route_y, xin, route_y, xin, ry1 + 22, C_AUDIO))
    o.append('<polygon points="%.1f,%d %.1f,%d %.1f,%d" fill="%s"/>'
             % (xin - 9, ry1 + 22, xin, ry1 + 4, xin + 9, ry1 + 22, C_AUDIO))
    o += caption("frame rate \u00b7 token budget \u00b7 semantic/acoustic disentanglement",
                 route_y - 20, x=700, col=C_AUDIO, size=17)
    o.append("</svg>")
    return "\n".join(o)


def xesc_svg(svg):
    """把裸露的 & 轉成 &amp;（已經是 entity 的不動）。

    SVG 是 XML，不是 HTML：一個裸 & 就會讓整份文件解析失敗，而瀏覽器不會報錯，
    只會把 <img> 的 naturalWidth 算成 0，投影片上看起來就是一張空白。
    "signals & front ends" 這種標籤踩過一次，所以改成在輸出前統一處理並驗證。
    """
    return re.sub(r"&(?![a-zA-Z][a-zA-Z0-9]*;|#[0-9]+;|#x[0-9a-fA-F]+;)", "&amp;", svg)


def main():
    if not os.path.isdir(OUT):
        os.makedirs(OUT)
    for name, fn in (("arch-cascade", fig_cascade), ("arch-e2e", fig_e2e),
                     ("arch-duplex", fig_duplex),
                     ("latency-budget-blank", fig_latency_blank),
                     ("course-map", fig_course_map)):
        p = os.path.join(OUT, name + ".svg")
        svg = xesc_svg(fn())
        # 產生後立刻驗證：解析不過的 SVG 在瀏覽器裡是靜默的空白，不是錯誤訊息
        _minidom.parseString(svg)
        with open(p, "w") as f:
            f.write(svg)
        print("→ %s (%d bytes)" % (os.path.relpath(p, os.path.dirname(OUT)), os.path.getsize(p)))


if __name__ == "__main__":
    main()

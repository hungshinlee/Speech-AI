# -*- coding: utf-8 -*-
# 此檔由 bin/publish_demos.py 從 private repo 的 <course>/demos/ 複製而來，請勿在公開 repo 直接編輯。
"""W4 demo 2：同一段音訊、三種 chunk——字何時出現（投影片 p42）。

  show       課堂：載入 sherpa-onnx 的 streaming zipformer transducer，開一個空的時間軸視窗；鍵在視窗裡按（下面的鍵）。離開時存 runs/live/<時間>/
  rehearse   課前：不互動，三種餵法各跑一次（不播音），存 runs/rehearsal/（進版控 = 現場的退路）
  replay     現場模型出狀況時：從 runs/rehearsal/ 讀回，同一個視窗、同樣的鍵，不碰 sherpa-onnx
  check      課前體檢：套件版本、模型檔在不在、encoder metadata（decode_chunk_len／T）、整句離線解一次對照文字與時間戳 → runs/check.json
  fetch      課前一週：下載模型（GitHub release 的 tar.bz2 → ~/.cache/sherpa-onnx）；要網路
  fake       沒有模型也能演練流程：合成的發射時間（畫面標明 FAKE，不是給學生看的）
  selftest   不需要模型：對齊讀取、BPE 併字、字的比對、統計、D_alg 算式、FAKE 的時鐘關係
  export     把 runs/rehearsal/rehearsal.json 的摘要寫進 slides/assets/w04/w04-metrics.json（鍵 demo2；量測數字進版控的出處）
  recompute  從 runs/rehearsal/rehearsal.json（與 runs/check.json）已存的符號事件重新併字、比對、統計、重畫圖——改了程式的後處理時用，不碰模型、不改時鐘

畫面上的鍵（順序照投影片 p42 講稿的【操作】）：
  1  餵 1280 ms 一塊      2  餵 640 ms 一塊      3  餵 320 ms 一塊（= 模型匯出時的 chunk）
     每一種：以即時步調餵那一句（同時用 afplay 播），每個符號出現時印三個時鐘，跑完畫那一列、印每個字的差與中位數
  a  三列的摘要表（填 p42 的表格）＋ 第四格：中位數差對 feed chunk 的折線，疊上算出來的 D_alg
  p  只播那句音訊          m  靜音／取消靜音       h  重印選單        q  離開

三個時鐘（每個符號各記一次）：
  t_frame  模型發射它的 encoder frame 的時間（sherpa-onnx 的 timestamps；40 ms 一格）。減掉字的聲學結尾 ≈ emission delay——模型自己學會的那一項
  t_fed    符號出現時已經餵進去的音訊秒數。t_fed − t_frame = 等模型 chunk（320 ms）湊滿 + 13 幀 pad 的那一點點，但那一點點要等到**下一次餵入**才到，
           所以平均是 c + F/2（c = 模型 chunk、F = 餵入 chunk），不是 pad + F/2——2026-10-06 M5 Max 實測 500／640／960 ms 對算式 480／640／960
  t_wall   符號出現時的掛鐘時間（音訊第 0 秒起算）。t_wall − t_fed = 計算
字的聲學結尾來自 W3 Demo 3 的 CTC forced alignment（最後一個字母的最後一格的結尾，20 ms 一格）——是尖峰位置不是音段邊界，
本身有 ±1 格的鬆動，README〈數字怎麼定義〉有寫。
"""
from __future__ import annotations

import argparse
import difflib
import io
import json
import math
import os
import random
import re
import shutil
import statistics
import subprocess
import sys
import tarfile
import threading
import time
import tomllib
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import common as C  # noqa: E402

CFG = tomllib.loads((HERE / "demo_config.toml").read_text(encoding="utf-8"))
DIG = CFG["display"]["digits"]
WORD_START = ("▁", " ")   # sentencepiece 的詞首記號；sherpa-onnx 的 tokens() 把 ▁ 換成空白回傳（2026-10-06 實測），有時詞首記號是單獨一個 " " 符號


def _p(rel: str) -> Path:
    return (HERE / rel).resolve()


# ── 聲學邊界：W3 Demo 3 的 forced alignment ───────────────────────
def load_alignment(path: Path) -> dict:
    """回 {"text", "T", "frame_ms", "words": [{"word", "start", "end", "i"}]}。
    字 = labels 裡兩個 ␣ 之間的字元；start = 第一個字母的 first_frame × 20 ms，end = 最後一個字母的 (last_frame + 1) × 20 ms。"""
    d = json.loads(path.read_text(encoding="utf-8"))
    align = d["sections"]["align"]
    frame_ms = float(d["config"]["model"]["frame_ms"])
    words, cur = [], []
    for lab in align["labels"] + [{"token": "␣"}]:
        if lab["token"] in ("␣", "|", " "):
            if cur:
                words.append({
                    "i": len(words),
                    "word": "".join(c["token"] for c in cur),
                    "start": round(cur[0]["first_frame"] * frame_ms / 1000, 3),
                    "end": round((cur[-1]["last_frame"] + 1) * frame_ms / 1000, 3),
                })
                cur = []
        else:
            cur.append(lab)
    text = d["text"].strip()
    if [w["word"] for w in words] != text.split():
        raise ValueError(f"對齊裡的字與文字不一致：{[w['word'] for w in words]} vs {text.split()}")
    return {"text": text, "T": d["sections"]["numbers"]["T"], "frame_ms": frame_ms, "words": words,
            "source": str(path), "source_time": d.get("time")}


# ── 符號 → 字、字 → 參考字 ─────────────────────────────────────
def group_words(events: list[dict]) -> list[dict]:
    """把符號事件（BPE 片段）併成字：▁ 開頭的符號起一個新字。每個字記第一個與最後一個符號的三個時鐘。"""
    words: list[dict] = []
    for e in events:
        tok = e["tok"]
        text = tok.lstrip("".join(WORD_START))
        if tok.startswith(WORD_START) or not words:
            words.append({"word": text, "tokens": [tok], "first": dict(e), "last": dict(e)})
        else:
            w = words[-1]
            w["word"] += text
            w["tokens"].append(tok)
            w["last"] = dict(e)
    return [w for w in words if w["word"]]


def match_words(ref: list[dict], hyp: list[dict]) -> list[dict]:
    """用 difflib 把辨識出來的字對到參考字。回每個參考字一筆：{"ref", "start", "end", "hyp"(或 None), "gap_frame"/"gap_fed"/"gap_wall"}。
    gap_* = 字的最後一個符號的那個時鐘 − 聲學結尾（秒）。沒對到的參考字 hyp = None（刪除或替換），差留空。"""
    sm = difflib.SequenceMatcher(None, [w["word"].upper() for w in ref], [w["word"].upper() for w in hyp], autojunk=False)
    rows = [{"ref": w["word"], "i": w["i"], "start": w["start"], "end": w["end"], "hyp": None,
             "gap_frame": None, "gap_fed": None, "gap_wall": None, "t_frame": None, "t_fed": None, "t_wall": None} for w in ref]
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag != "equal":
            # 替換時把辨識出來的字留著看（但不算差，因為不知道它對應誰）
            for k in range(i1, i2):
                rows[k]["hyp"] = "≠" + " ".join(h["word"] for h in hyp[j1:j2]) if j2 > j1 else None
            continue
        for k, j in zip(range(i1, i2), range(j1, j2)):
            last = hyp[j]["last"]
            rows[k]["hyp"] = hyp[j]["word"]
            for clock in ("frame", "fed", "wall"):
                rows[k][f"t_{clock}"] = last[f"t_{clock}"]
                rows[k][f"gap_{clock}"] = round(last[f"t_{clock}"] - rows[k]["end"], 3)
    return rows


def d_alg_ms(feed_ms: float, pad_frames: int, input_frame_ms: float, chunk_frames_: int | None = None) -> dict:
    """投影片 p19／p20 的 D_alg = (m + αC) sΔ，在 100 Hz 輸入幀上算（s = 1，Δ = input_frame_ms）。
    一個 frame 的符號要等：(a) 它所在的模型 chunk（c = decode_chunk_len × Δ）湊滿——平均 c/2；(b) 再多 13 幀 pad——
    但資料是一塊 F 一塊 F 來的，chunk 結尾之後那一點點 pad 要等到下一塊餵入才到，等的是「下一個 F 邊界」：
    chunk 結尾落在 F 的 k = F/c 個格位上各一次，平均 (F + c)/2。合計 mean = c + F/2、worst = c + F、best = c（pad < c、F 是 c 的整數倍時）。
    2026-10-06 M5 Max 實測 fed − emit 中位數 500／640／960 對 480／640／960。naive_mean_ms 是原本寫的 pad + F/2（錯的，留著對照）。"""
    m_ms = pad_frames * input_frame_ms
    c_ms = (chunk_frames_ if chunk_frames_ is not None else int(CFG["model"]["decode_chunk_len_default"])) * input_frame_ms
    ok = 0 < m_ms <= c_ms and feed_ms % c_ms == 0
    mean = c_ms + feed_ms / 2 if ok else m_ms + feed_ms / 2
    worst = c_ms + feed_ms if ok else m_ms + feed_ms
    return {"m_ms": m_ms, "c_ms": c_ms, "C_ms": feed_ms, "mean_ms": mean, "worst_ms": worst, "best_ms": c_ms if ok else m_ms,
            "naive_mean_ms": m_ms + feed_ms / 2, "formula": "c + F/2 (pad waits for the next feed)" if ok else "m + F/2"}


def summarize(rows: list[dict], events: list[dict], feed_ms: float, pad_frames: int, input_frame_ms: float, chunk_frames_: int | None = None) -> dict:
    ok = [r for r in rows if r["gap_wall"] is not None]
    s: dict = {"n_ref": len(rows), "n_matched": len(ok), "feed_ms": feed_ms, "d_alg": d_alg_ms(feed_ms, pad_frames, input_frame_ms, chunk_frames_)}
    for clock in ("frame", "fed", "wall"):
        g = [r[f"gap_{clock}"] for r in ok]
        if g:
            worst = max(ok, key=lambda r: r[f"gap_{clock}"])
            s[f"median_gap_{clock}_ms"] = round(1000 * statistics.median(g), 1)
            s[f"worst_{clock}"] = {"word": worst["ref"], "gap_ms": round(1000 * worst[f"gap_{clock}"], 1)}
    if events:
        s["median_fed_minus_frame_ms"] = round(1000 * statistics.median(e["t_fed"] - e["t_frame"] for e in events), 1)
        s["median_wall_minus_fed_ms"] = round(1000 * statistics.median(e["t_wall"] - e["t_fed"] for e in events), 1)
    s["hyp_text"] = " ".join(w["word"] for w in group_words(events))
    return s


# ── 後端 ────────────────────────────────────────────────────────
class Sherpa:
    """sherpa-onnx 的 OnlineRecognizer（transducer、greedy）。一個 recognizer，每次餵法各開一個 stream。"""

    def __init__(self):
        import sherpa_onnx  # 只在這裡 import：replay／fake／selftest 不需要它
        mc = CFG["model"]
        d = self.dir = Path(os.path.expanduser(mc["cache_dir"])) / mc["name"]
        for k in ("encoder", "decoder", "joiner", "tokens"):
            if not (d / mc[k]).exists():
                sys.exit(f"找不到 {d / mc[k]}。先跑 ./present.sh fetch（要網路）。")
        t0 = time.perf_counter()
        self.rec = sherpa_onnx.OnlineRecognizer.from_transducer(
            tokens=str(d / mc["tokens"]), encoder=str(d / mc["encoder"]), decoder=str(d / mc["decoder"]),
            joiner=str(d / mc["joiner"]), num_threads=int(mc["num_threads"]), sample_rate=int(CFG["audio"]["fs"]),
            feature_dim=80, decoding_method=mc["decoding_method"], enable_endpoint_detection=False)
        self.load_s = round(time.perf_counter() - t0, 3)
        self.meta = read_metadata(d / mc["encoder"])
        self.version = _version("sherpa_onnx")
        self.name = mc["name"]

    def run(self, x: np.ndarray, feed_ms: float, pace: bool, on_token=None) -> list[dict]:
        """以 feed_ms 一塊餵 x（含尾端靜音），pace=True 時照即時步調。回符號事件清單。"""
        fs = int(CFG["audio"]["fs"])
        s = self.rec.create_stream()
        n = int(round(feed_ms / 1000 * fs))
        pieces = [x[i:i + n] for i in range(0, len(x), n)]
        events: list[dict] = []
        seen = 0
        t0 = time.perf_counter()
        for i, piece in enumerate(pieces):
            fed = min((i + 1) * n, len(x)) / fs
            if pace:
                while time.perf_counter() - t0 < fed:
                    time.sleep(0.001)
            s.accept_waveform(fs, piece.astype(np.float32))
            if i == len(pieces) - 1:
                s.input_finished()
            while self.rec.is_ready(s):
                self.rec.decode_stream(s)
            toks = self.rec.tokens(s)
            stamps = self.rec.timestamps(s)
            for k in range(seen, len(toks)):
                e = {"tok": toks[k], "t_frame": round(float(stamps[k]), 3), "t_fed": round(fed, 3),
                     "t_wall": round(time.perf_counter() - t0, 3), "piece": i}
                events.append(e)
                if on_token:
                    on_token(e)
            seen = len(toks)
        return events

    def info(self) -> dict:
        return {"backend": "sherpa-onnx", "version": self.version, "model": self.name, "load_s": self.load_s, "meta": self.meta}


class Fake:
    """沒有模型時：從對齊合成發射時間。t_frame = 聲學結尾 + N(mean, std)（40 ms 格）；t_fed = 進位到 feed 邊界再加 pad；t_wall = t_fed + comp。"""

    def __init__(self, al: dict):
        self.al = al
        fc = CFG["fake"]
        self.rng = random.Random(fc["seed"])
        self.meta = {"decode_chunk_len": CFG["model"]["decode_chunk_len_default"],
                     "T": CFG["model"]["decode_chunk_len_default"] + CFG["model"]["pad_length_default"], "FAKE": True}

    def run(self, x: np.ndarray, feed_ms: float, pace: bool, on_token=None) -> list[dict]:
        fc, mc = CFG["fake"], CFG["model"]
        grid = mc["output_frame_ms"] / 1000
        pad = pad_frames(self.meta) * mc["input_frame_ms"] / 1000
        events = []
        for w in self.al["words"]:
            t_frame = round(max(0.0, w["end"] + self.rng.gauss(fc["emit_mean_s"], fc["emit_std_s"])) // grid * grid, 3)
            # 這一格的輸入幀要連同 pad 一起進來，chunk 才會被解：進位到 feed 邊界
            need = t_frame + grid + pad
            t_fed = round(math.ceil(need / (feed_ms / 1000) - 1e-9) * feed_ms / 1000, 3)
            t_wall = round(t_fed + fc["comp_s"], 3)
            # 一個字拆成兩個 BPE 片段，兩個都在同一格出現
            halves = ["▁" + w["word"][:2], w["word"][2:]] if len(w["word"]) > 2 else ["▁" + w["word"]]
            for tok in halves:
                e = {"tok": tok, "t_frame": t_frame, "t_fed": t_fed, "t_wall": t_wall, "piece": int(t_fed / (feed_ms / 1000)) - 1}
                events.append(e)
        events.sort(key=lambda e: (e["t_wall"], e["t_frame"]))
        if pace:
            t0 = time.perf_counter()
        for e in events:
            if pace:
                while time.perf_counter() - t0 < e["t_wall"]:
                    time.sleep(0.001)
            if on_token:
                on_token(e)
        return events

    def info(self) -> dict:
        return {"backend": "FAKE", "version": None, "model": None, "load_s": 0.0, "meta": self.meta}


def read_metadata(encoder: Path) -> dict:
    """讀 encoder ONNX 的 metadata_props（icefall export-onnx-streaming.py 寫的 decode_chunk_len、T 等）。沒有 onnx 套件就回空 dict。"""
    try:
        import onnx
    except ImportError:
        return {}
    try:
        m = onnx.load(str(encoder), load_external_data=False)
        return {p.key: p.value for p in m.metadata_props}
    except Exception as e:  # noqa: BLE001
        return {"error": str(e)}


def pad_frames(meta: dict) -> int:
    """m（conv 前端右側要多看的輸入幀數）= T − decode_chunk_len；metadata 沒有就用 config 的預設。"""
    try:
        return int(meta["T"]) - int(meta["decode_chunk_len"])
    except (KeyError, TypeError, ValueError):
        return int(CFG["model"]["pad_length_default"])


def chunk_frames(meta: dict) -> int:
    try:
        return int(meta["decode_chunk_len"])
    except (KeyError, TypeError, ValueError):
        return int(CFG["model"]["decode_chunk_len_default"])


def _version(pkg: str):
    from importlib import metadata
    try:
        return metadata.version(pkg)
    except metadata.PackageNotFoundError:
        return None


# ── 輸出 ────────────────────────────────────────────────────────
class Out:
    def __init__(self):
        self.buf = io.StringIO()

    def __call__(self, text: str = "") -> None:
        print(text)
        self.buf.write(re.sub(r"\033\[[0-9;]*m", "", text) + "\n")

    def banner(self, text: str, color: str = C.BLUE) -> None:
        C.banner(text, color)
        self.buf.write("\n" + "─" * 72 + "\n" + text + "\n" + "─" * 72 + "\n")

    def text(self) -> str:
        return self.buf.getvalue()


def ms(v) -> str:
    return "   —  " if v is None else f"{1000 * v:+6.0f}"


# ── 時間軸 ──────────────────────────────────────────────────────
class Timeline:
    """一張圖：上面三列（每種餵法一列），每個字一條灰色的聲學長條，三個時鐘各一個記號；底下第四格是中位數差對 feed chunk。"""
    CLOCKS = (("frame", "^", "#3366cc", "emitted (encoder frame)"), ("fed", "o", "#e08a00", "appeared (audio fed)"),
              ("wall", "x", "#cc2222", "appeared (wall clock)"))

    def __init__(self, al: dict, settings: list[dict], fake: bool = False):
        import matplotlib
        import matplotlib.pyplot as plt
        self.plt = plt
        self.al, self.settings, self.fake = al, settings, fake
        dc = CFG["display"]
        self.fig = plt.figure(figsize=(dc["fig_w"], dc["fig_h"]))
        gs = self.fig.add_gridspec(2, 1, height_ratios=[3.2, 1], hspace=0.35)
        self.ax = self.fig.add_subplot(gs[0])
        self.ax2 = self.fig.add_subplot(gs[1])
        self.interactive = matplotlib.get_backend().lower() not in ("agg", "pdf", "svg", "ps", "cairo", "template")
        self.rows: dict[str, dict] = {}
        self.live: list = []
        self.draw_base()

    def draw_base(self) -> None:
        ax = self.ax
        ax.clear()
        n = len(self.settings)
        dur = self.al["words"][-1]["end"] + CFG["audio"]["tail_silence_s"] + 0.3
        for i, st in enumerate(self.settings):
            y = n - 1 - i
            for w in self.al["words"]:
                ax.barh(y, w["end"] - w["start"], left=w["start"], height=0.28, color="#d9d9d9", edgecolor="none", zorder=1)
                if i == 0:
                    ax.text((w["start"] + w["end"]) / 2, n - 0.45 + (0.18 if w["i"] % 2 else 0.0), w["word"].lower(),
                            ha="center", va="bottom", fontsize=8, color="#555")
            ax.text(-0.05, y, f"feed {st['name']}", ha="right", va="center", fontsize=10, transform=ax.get_yaxis_transform())
        ax.set_ylim(-0.95, n - 0.2 + 0.5)
        ax.set_xlim(0, dur)
        ax.set_yticks([])
        ax.set_xlabel("time (s) — grey bars: acoustic span of each word (W3 Demo 3 forced alignment)")
        ax.grid(axis="x", color="#eee")
        for sp in ("top", "right", "left"):
            ax.spines[sp].set_visible(False)
        title = "Demo 2 — one streaming transducer, three feed chunks: when does each word appear?"
        if self.fake:
            title = "FAKE (synthetic emission times, not a measurement) — " + title
        ax.set_title(title, fontsize=12, loc="left")
        self.ax2.clear()
        self.ax2.set_xlabel("feed chunk (ms)")
        self.ax2.set_ylabel("median gap (ms)")
        self.ax2.grid(color="#eee")
        self.ax2.text(0.5, 0.5, "press a after the three runs", ha="center", va="center", transform=self.ax2.transAxes, color="#999")
        self._flush()

    def _flush(self) -> None:
        if self.interactive:
            self.fig.canvas.draw_idle()
            self.fig.canvas.flush_events()

    def row_index(self, key: str) -> int:
        return len(self.settings) - 1 - [s["key"] for s in self.settings].index(key)

    def live_token(self, key: str, e: dict) -> None:
        """跑的途中：每個符號出現時在掛鐘位置點一個小紅叉（跑完整列重畫）。"""
        y = self.row_index(key)
        self.live.append(self.ax.plot([e["t_wall"]], [y + 0.22], marker="x", color="#cc2222", ms=5, ls="none", zorder=4)[0])
        self._flush()

    def draw_row(self, key: str, rows: list[dict], summary: dict) -> None:
        for h in self.live:
            h.remove()
        self.live = []
        y = self.row_index(key)
        if key in self.rows:
            for h in self.rows[key]["handles"]:
                h.remove()
        handles = []
        for r in rows:
            if r["gap_wall"] is None:
                handles.append(self.ax.text((r["start"] + r["end"]) / 2, y + 0.22, "?", ha="center", va="bottom", color="#cc2222", fontsize=9))
                continue
            handles.append(self.ax.plot([r["end"], r["t_wall"]], [y, y], color="#bbb", lw=0.8, zorder=2)[0])
            for clock, mk, col, _ in self.CLOCKS:
                handles.append(self.ax.plot([r[f"t_{clock}"]], [y], marker=mk, color=col, ms=6, ls="none", zorder=3 + (clock == "wall"))[0])
        txt = (f"median gap  emit {summary.get('median_gap_frame_ms', float('nan')):+.0f}  ·  fed {summary.get('median_gap_fed_ms', float('nan')):+.0f}"
               f"  ·  wall {summary.get('median_gap_wall_ms', float('nan')):+.0f} ms   (D_alg mean {summary['d_alg']['mean_ms']:.0f})")
        handles.append(self.ax.text(1.0, y - 0.42, txt, ha="right", va="center", fontsize=9, transform=self.ax.get_yaxis_transform(), color="#333"))
        self.rows[key] = {"handles": handles, "summary": summary}
        if len(self.rows) == 1:
            self.ax.legend(handles=[self.plt.Line2D([], [], marker=mk, color=col, ls="none", label=lab) for _, mk, col, lab in self.CLOCKS],
                           loc="lower left", fontsize=8, frameon=False, ncol=3)
        self._flush()

    def draw_summary(self) -> None:
        ax = self.ax2
        ax.clear()
        ks = [s["key"] for s in self.settings if s["key"] in self.rows]
        if not ks:
            return
        xs = [self.rows[k]["summary"]["feed_ms"] for k in ks]
        for clock, mk, col, lab in self.CLOCKS:
            ys = [self.rows[k]["summary"].get(f"median_gap_{clock}_ms") for k in ks]
            ax.plot(xs, ys, marker=mk, color=col, label=f"median gap, {lab}")
        ax.plot(xs, [self.rows[k]["summary"]["d_alg"]["mean_ms"] for k in ks], ls="--", color="#777", label="computed D_alg (mean) = c + F/2")
        emit = [self.rows[k]["summary"].get("median_gap_frame_ms") for k in ks]
        if all(v is not None for v in emit):
            base = statistics.median(emit)
            ax.plot(xs, [self.rows[k]["summary"]["d_alg"]["mean_ms"] + base for k in ks], ls=":", color="#3366cc",
                    label=f"D_alg + median emission delay ({base:+.0f} ms, constant)")
        ax.set_xlabel("feed chunk C (ms)")
        ax.set_ylabel("median gap (ms)")
        ax.set_xticks(xs)
        top = max(v for k in ks for v in (self.rows[k]["summary"].get("median_gap_wall_ms") or 0, self.rows[k]["summary"]["d_alg"]["mean_ms"]))
        ax.set_ylim(0, top * 1.9)
        ax.grid(color="#eee")
        ax.legend(fontsize=7, frameon=False, loc="upper left", ncol=2)
        self._flush()

    def save(self, path: Path) -> None:
        self.fig.savefig(str(path), dpi=130, bbox_inches="tight")


# ── Demo ───────────────────────────────────────────────────────
class Demo:
    def __init__(self, out: Out, backend, al: dict, x: np.ndarray, tag: str = "", mute: bool = False):
        self.out, self.be, self.al, self.x, self.mute = out, backend, al, x, mute
        self.settings = CFG["settings"]["items"]
        self.fake = isinstance(backend, Fake)
        self.tl = Timeline(al, self.settings, fake=self.fake)
        self.log: dict = {"time": C.now(), "tag": tag, "backend": backend.info(), "config": CFG, "versions": C.versions(),
                          "alignment": {"source": al["source"], "source_time": al["source_time"], "T": al["T"], "frame_ms": al["frame_ms"],
                                        "words": al["words"]}, "runs": {}}
        self.log["versions"]["sherpa_onnx"] = _version("sherpa-onnx")
        self.log["versions"]["onnx"] = _version("onnx")
        self.wav = _p(CFG["audio"]["wav"])
        pf, cf = pad_frames(backend.info()["meta"]), chunk_frames(backend.info()["meta"])
        self.pad_frames, self.chunk_frames = pf, cf
        self.out.banner(f"W4 Demo 2 — {backend.info()['backend']}  ·  model chunk {cf} input frames = {cf * CFG['model']['input_frame_ms']:.0f} ms, "
                        f"pad m = {pf} frames = {pf * CFG['model']['input_frame_ms']:.0f} ms  ·  sentence: {al['text']}", C.AMBER if self.fake else C.BLUE)

    # 一種餵法
    def run_setting(self, key: str, pace: bool = True) -> None:
        st = next(s for s in self.settings if s["key"] == key)
        feed = float(st["feed_ms"])
        if feed % (self.chunk_frames * CFG["model"]["input_frame_ms"]) != 0:
            self.out(f"{C.RED}feed_ms = {feed:.0f} 不是模型 chunk（{self.chunk_frames * CFG['model']['input_frame_ms']:.0f} ms）的整數倍，量到的不會是 {feed:.0f}。{RESET_}")
        self.out.banner(f"feed {st['name']}  —  {feed:.0f} ms per piece, real-time pace{'' if pace else ' OFF'}"
                        f"{'  [FAKE]' if self.fake else ''}", C.AMBER if self.fake else C.BLUE)
        self.out(f"{'t_fed':>7} {'t_wall':>7} {'t_frame':>8}   token")
        if pace and not self.mute and not self.fake:
            threading.Thread(target=C.play, args=(self.wav,), daemon=True).start()
        tail = np.zeros(int(CFG["audio"]["tail_silence_s"] * CFG["audio"]["fs"]), dtype=np.float32)
        x = np.concatenate([self.x, tail])

        def on_token(e: dict) -> None:
            mark = C.GREEN if e["tok"].startswith(WORD_START) else ""
            self.out(f"{e['t_fed']:7.3f} {e['t_wall']:7.3f} {e['t_frame']:8.3f}   {mark}{e['tok']}{C.RESET}")
            self.tl.live_token(key, e)

        events = self.be.run(x, feed, pace, on_token)
        hyp = group_words(events)
        rows = match_words(self.al["words"], hyp)
        summ = summarize(rows, events, feed, self.pad_frames, CFG["model"]["input_frame_ms"], self.chunk_frames)
        self.tl.draw_row(key, rows, summ)
        self.out()
        self.out(f"hyp: {summ['hyp_text']}")
        self.out(f"ref: {self.al['text']}   ({summ['n_matched']}/{summ['n_ref']} words matched)")
        self.out()
        self.out(f"{'word':>9} {'acoustic end':>12} │ {'emit−end':>9} {'fed−end':>8} {'wall−end':>9}   (ms; emit = encoder frame, fed = audio fed, wall = clock)")
        for r in rows:
            if r["gap_wall"] is None:
                self.out(f"{r['ref']:>9} {r['end']:12.3f} │ {'—':>9} {'—':>8} {'—':>9}   {C.RED}{r['hyp'] or 'missing'}{C.RESET}")
            else:
                self.out(f"{r['ref']:>9} {r['end']:12.3f} │ {ms(r['gap_frame']):>9} {ms(r['gap_fed']):>8} {ms(r['gap_wall']):>9}")
        da = summ["d_alg"]
        self.out()
        self.out(f"median gap: emit {summ.get('median_gap_frame_ms', float('nan')):+.0f} ms  ·  fed {summ.get('median_gap_fed_ms', float('nan')):+.0f} ms  ·  "
                 f"wall {summ.get('median_gap_wall_ms', float('nan')):+.0f} ms")
        self.out(f"worst word (wall): {summ['worst_wall']['word']} {summ['worst_wall']['gap_ms']:+.0f} ms" if "worst_wall" in summ else "worst word: —")
        self.out(f"computed D_alg: model chunk c = {da['c_ms']:.0f} ms, pad = {da['m_ms']:.0f} ms, feed F = {da['C_ms']:.0f} ms → "
                 f"mean c + F/2 = {da['mean_ms']:.0f} ms, worst c + F = {da['worst_ms']:.0f} ms   (naive pad + F/2 would be {da['naive_mean_ms']:.0f})")
        self.out(f"measured  fed − emit (≈ m + αC): median {summ.get('median_fed_minus_frame_ms', float('nan')):+.0f} ms   ·   "
                 f"wall − fed (≈ D_comp): median {summ.get('median_wall_minus_fed_ms', float('nan')):+.0f} ms")
        self.log["runs"][key] = {"setting": st, "events": events, "words": rows, "summary": summ}

    def summary_table(self) -> None:
        self.out.banner("p42 的表格（三列）" + ("  [FAKE]" if self.fake else ""), C.AMBER if self.fake else C.BLUE)
        self.out(f"{'feed chunk':>24} │ {'D_alg mean':>10} │ {'emit−end med':>12} {'fed−end med':>11} {'wall−end med':>12} │ worst word (wall)")
        for st in self.settings:
            r = self.log["runs"].get(st["key"])
            if not r:
                self.out(f"{st['name']:>24} │ {'(not run)':>10}")
                continue
            s = r["summary"]
            self.out(f"{st['name']:>24} │ {s['d_alg']['mean_ms']:10.0f} │ {s.get('median_gap_frame_ms', float('nan')):+12.0f} "
                     f"{s.get('median_gap_fed_ms', float('nan')):+11.0f} {s.get('median_gap_wall_ms', float('nan')):+12.0f} │ "
                     f"{s['worst_wall']['word']} {s['worst_wall']['gap_ms']:+.0f}" if "worst_wall" in s else "—")
        self.out()
        self.out("讀法：emit−end 是模型自己學會的延遲（不隨 feed chunk 變）；fed−end 多出來的是架構那一項 c + F/2（隨 F 線性、斜率 1/2）；wall−end 再多的是計算。")
        self.tl.draw_summary()

    def menu(self) -> None:
        names = {s["key"]: s["name"] for s in self.settings}
        print(f"\n{C.BOLD}鍵{C.RESET}：{C.BOLD}1{C.RESET} feed {names['1']}   {C.BOLD}2{C.RESET} {names['2']}   {C.BOLD}3{C.RESET} {names['3']}   "
              f"{C.BOLD}a{C.RESET} 摘要表＋第四格   {C.BOLD}p{C.RESET} 播音   {C.BOLD}m{C.RESET} 靜音   {C.BOLD}h{C.RESET} 選單   {C.BOLD}q{C.RESET} 離開")

    def dispatch(self, k: str) -> bool:
        if k in ("1", "2", "3"):
            self.run_setting(k)
        elif k == "a":
            self.summary_table()
        elif k == "p":
            C.play(self.wav)
        elif k == "m":
            self.mute = not self.mute
            print("靜音" if self.mute else "播音")
        elif k == "h":
            self.menu()
        elif k == "q":
            return False
        return True

    def run_all(self) -> None:
        for st in self.settings:
            self.run_setting(st["key"], pace=True)
        self.summary_table()


RESET_ = C.RESET


def interactive(d: Demo) -> None:
    d.menu()
    if d.tl.interactive:
        # 鍵在視窗裡按（與 W3 Demo 3 相同；在終端機 block 讀鍵會讓 macOS 的視窗凍住）
        alive = {"ok": True}

        def on_key(ev):
            if ev.key is None:
                return
            if not d.dispatch(ev.key):
                alive["ok"] = False
                d.tl.plt.close(d.tl.fig)

        d.tl.fig.canvas.mpl_connect("key_press_event", on_key)
        print("（視窗要有焦點才收得到鍵）")
        d.tl.plt.show(block=True)
    else:
        print("（沒有互動式的 matplotlib 後端：鍵改在終端機按）")
        while d.dispatch(C.getkey()):
            pass


def save(d: Demo, folder: Path) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "rehearsal.json").write_text(json.dumps(d.log, ensure_ascii=False, indent=1), encoding="utf-8")
    (folder / "rehearsal.txt").write_text(d.out.text(), encoding="utf-8")
    d.tl.save(folder / "timeline.png")
    print(f"→ {folder}/  (rehearsal.json, rehearsal.txt, timeline.png)")


# ── replay：從彩排紀錄重畫，不碰模型 ──────────────────────────────
class Replay:
    def __init__(self, out: Out, log: dict):
        self.out, self.log = out, log
        al = log["alignment"]
        al.setdefault("text", " ".join(w["word"] for w in al["words"]))
        self.al = al
        self.settings = log["config"]["settings"]["items"]
        self.fake = log["backend"]["backend"] == "FAKE"
        self.tl = Timeline(al, self.settings, fake=self.fake)
        self.wav = _p(log["config"]["audio"]["wav"])
        self.mute = False
        out.banner(f"REPLAY（{log['time']} 的彩排，不是現場算的；{log['backend']['backend']} {log['backend'].get('version')}）", C.AMBER)

    def run_setting(self, key: str) -> None:
        r = self.log["runs"].get(key)
        if not r:
            print("彩排沒有這一種餵法")
            return
        st, summ = r["setting"], r["summary"]
        self.out.banner(f"feed {st['name']}  —  replay", C.AMBER)
        for e in r["events"]:
            mark = C.GREEN if e["tok"].startswith(WORD_START) else ""
            self.out(f"{e['t_fed']:7.3f} {e['t_wall']:7.3f} {e['t_frame']:8.3f}   {mark}{e['tok']}{C.RESET}")
        self.tl.draw_row(key, r["words"], summ)
        self.out(f"hyp: {summ['hyp_text']}")
        for w in r["words"]:
            self.out(f"{w['ref']:>9} {w['end']:12.3f} │ {ms(w['gap_frame']):>9} {ms(w['gap_fed']):>8} {ms(w['gap_wall']):>9}")
        self.out(f"median gap: emit {summ.get('median_gap_frame_ms', float('nan')):+.0f}  fed {summ.get('median_gap_fed_ms', float('nan')):+.0f}  "
                 f"wall {summ.get('median_gap_wall_ms', float('nan')):+.0f} ms  ·  D_alg mean {summ['d_alg']['mean_ms']:.0f} ms")

    def dispatch(self, k: str) -> bool:
        if k in ("1", "2", "3"):
            self.run_setting(k)
        elif k == "a":
            Demo.summary_table(self)  # 同一張表、同一個第四格
        elif k == "p":
            C.play(self.wav)
        elif k == "h":
            Demo.menu(self)
        elif k == "q":
            return False
        return True

    menu = Demo.menu


# ── check／fetch／export／selftest ─────────────────────────────
def load_audio() -> np.ndarray:
    wav = _p(CFG["audio"]["wav"])
    if not wav.exists():
        sys.exit(f"找不到 {wav}——W3 Demo 3 的彩排紀錄（runs/rehearsal/utterance.wav）要先在。")
    return C.load_wav(wav, int(CFG["audio"]["fs"])).astype(np.float32)


def check() -> None:
    mc = CFG["model"]
    d = Path(os.path.expanduser(mc["cache_dir"])) / mc["name"]
    rep: dict = {"time": C.now(), "versions": C.versions(), "model_dir": str(d), "files": {}}
    rep["versions"]["sherpa_onnx"] = _version("sherpa-onnx")
    rep["versions"]["onnx"] = _version("onnx")
    C.banner("check：版本、模型檔、metadata、整句離線解一次")
    print(json.dumps(rep["versions"], ensure_ascii=False, indent=1))
    for k in ("encoder", "decoder", "joiner", "tokens"):
        p = d / mc[k]
        rep["files"][k] = {"path": str(p), "exists": p.exists(), "bytes": p.stat().st_size if p.exists() else None}
        print(f"{k:8} {'OK ' if p.exists() else 'MISSING'} {p}")
    if not all(v["exists"] for v in rep["files"].values()):
        print("模型檔不齊，先跑 ./present.sh fetch")
        (HERE / "runs").mkdir(exist_ok=True)
        (HERE / "runs" / "check.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1), encoding="utf-8")
        return
    meta = read_metadata(d / mc["encoder"])
    rep["meta"] = meta
    print("encoder metadata:", json.dumps(meta, ensure_ascii=False))
    pf, cf = pad_frames(meta), chunk_frames(meta)
    rep["derived"] = {"decode_chunk_len_frames": cf, "chunk_ms": cf * mc["input_frame_ms"], "pad_frames": pf, "pad_ms": pf * mc["input_frame_ms"],
                      "from_metadata": "T" in meta and "decode_chunk_len" in meta}
    print(f"→ model chunk {cf} frames = {cf * mc['input_frame_ms']:.0f} ms；pad m = {pf} frames = {pf * mc['input_frame_ms']:.0f} ms"
          f"（{'metadata' if rep['derived']['from_metadata'] else '沒讀到 metadata，用 config 預設'}）")
    for st in CFG["settings"]["items"]:
        if st["feed_ms"] % (cf * mc["input_frame_ms"]) != 0:
            print(f"{C.RED}settings {st['key']} 的 feed_ms {st['feed_ms']} 不是 chunk {cf * mc['input_frame_ms']:.0f} ms 的整數倍{C.RESET}")
    al = load_alignment(_p(CFG["audio"]["alignment"]))
    x = load_audio()
    be = Sherpa()
    rep["load_s"] = be.load_s
    t0 = time.perf_counter()
    events = be.run(np.concatenate([x, np.zeros(int(CFG["audio"]["tail_silence_s"] * CFG["audio"]["fs"]), dtype=np.float32)]),
                    feed_ms=1000 * (len(x) / CFG["audio"]["fs"] + CFG["audio"]["tail_silence_s"]), pace=False)
    rep["offline"] = {"seconds": round(time.perf_counter() - t0, 3), "tokens": [e["tok"] for e in events], "timestamps": [e["t_frame"] for e in events]}
    hyp = group_words(events)
    rows = match_words(al["words"], hyp)
    rep["offline"]["hyp_text"] = " ".join(w["word"] for w in hyp)
    rep["offline"]["ref_text"] = al["text"]
    rep["offline"]["n_matched"] = sum(r["gap_wall"] is not None for r in rows)
    rep["offline"]["n_ref"] = len(rows)
    grid = sorted({round(e["t_frame"] % (mc["output_frame_ms"] / 1000), 3) for e in events})
    rep["offline"]["timestamps_on_40ms_grid"] = all(g in (0.0, round(mc["output_frame_ms"] / 1000, 3)) for g in grid)
    print(f"offline decode {rep['offline']['seconds']:.2f} s（整句一次餵、不計步調）")
    print(f"hyp: {rep['offline']['hyp_text']}")
    print(f"ref: {al['text']}   ({rep['offline']['n_matched']}/{rep['offline']['n_ref']} 字對上)")
    print("tokens:", " ".join(e["tok"] for e in events))
    print("timestamps:", " ".join(f"{e['t_frame']:.2f}" for e in events), "  — 都在 40 ms 格上" if rep["offline"]["timestamps_on_40ms_grid"] else "  — 不在 40 ms 格上，output_frame_ms 要改")
    print(f"emit − acoustic end（離線、無 chunk 等待）：", " ".join(f"{r['ref']}{ms(r['gap_frame']).strip()}" for r in rows if r["gap_frame"] is not None))
    (HERE / "runs").mkdir(exist_ok=True)
    (HERE / "runs" / "check.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1), encoding="utf-8")
    print("→ runs/check.json")


def fetch() -> None:
    import urllib.request
    mc = CFG["model"]
    cache = Path(os.path.expanduser(mc["cache_dir"]))
    cache.mkdir(parents=True, exist_ok=True)
    d = cache / mc["name"]
    if all((d / mc[k]).exists() for k in ("encoder", "decoder", "joiner", "tokens")):
        print(f"已經有了：{d}")
        return
    tb = cache / (mc["name"] + ".tar.bz2")
    if not tb.exists():
        print(f"下載 {mc['tarball']} → {tb}（約 250 MB 的 fp32 encoder 加 int8 版，共數百 MB）")
        urllib.request.urlretrieve(mc["tarball"], tb)
    print("解壓……")
    with tarfile.open(tb, "r:bz2") as t:
        t.extractall(cache, filter="data")
    missing = [mc[k] for k in ("encoder", "decoder", "joiner", "tokens") if not (d / mc[k]).exists()]
    if missing:
        sys.exit(f"解壓後缺：{missing}。看看 {d} 底下實際的檔名，改 demo_config.toml 的 [model]。")
    print(f"→ {d}")
    print("下一步：./present.sh check")


def export() -> None:
    src = HERE / "runs" / "rehearsal" / "rehearsal.json"
    if not src.exists():
        sys.exit("沒有 runs/rehearsal/rehearsal.json，先跑 ./present.sh rehearse")
    log = json.loads(src.read_text(encoding="utf-8"))
    if log["backend"]["backend"] == "FAKE":
        sys.exit("彩排紀錄是 FAKE，不匯出。")
    dst = HERE.parent.parent / "slides" / "assets" / "w04" / "w04-metrics.json"
    data = json.loads(dst.read_text(encoding="utf-8")) if dst.exists() else {
        "_note": "量測數字（進版控的出處，speech_ai/CLAUDE.md 4.8）。w04-data.json 是 make_figs_w04.py 算出來的、每次重產；量測的放這裡。"}
    rows = {}
    for key, r in log["runs"].items():
        s = r["summary"]
        rows[key] = {"feed_ms": s["feed_ms"], "d_alg_mean_ms": s["d_alg"]["mean_ms"], "d_alg_worst_ms": s["d_alg"]["worst_ms"],
                     "d_alg_formula": s["d_alg"].get("formula"), "model_chunk_ms": s["d_alg"].get("c_ms"), "pad_ms": s["d_alg"]["m_ms"],
                     "naive_mean_ms": s["d_alg"].get("naive_mean_ms"), "median_gap_emit_ms": s.get("median_gap_frame_ms"), "median_gap_fed_ms": s.get("median_gap_fed_ms"),
                     "median_gap_wall_ms": s.get("median_gap_wall_ms"), "worst_wall": s.get("worst_wall"),
                     "median_fed_minus_emit_ms": s.get("median_fed_minus_frame_ms"), "median_wall_minus_fed_ms": s.get("median_wall_minus_fed_ms"),
                     "n_matched": s["n_matched"], "n_ref": s["n_ref"], "hyp_text": s["hyp_text"],
                     "per_word": [{"word": w["ref"], "end": w["end"], "gap_emit_ms": None if w["gap_frame"] is None else round(1000 * w["gap_frame"]),
                                   "gap_fed_ms": None if w["gap_fed"] is None else round(1000 * w["gap_fed"]),
                                   "gap_wall_ms": None if w["gap_wall"] is None else round(1000 * w["gap_wall"])} for w in r["words"]]}
    data["demo2"] = {"source": "demos/w04_d2_stream_chunks/runs/rehearsal/rehearsal.json", "time": log["time"], "backend": log["backend"],
                     "sentence": log["alignment"].get("text") or " ".join(w["word"] for w in log["alignment"]["words"]),
                     "acoustic_end_source": log["alignment"]["source"], "settings": rows}
    dst.write_text(json.dumps(data, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"→ {dst}  (demo2)")


def recompute() -> None:
    """從已存的事件重做後處理（併字、比對、統計、圖、txt）。時鐘一個都不動。"""
    import matplotlib
    matplotlib.use("Agg")
    mc = CFG["model"]
    src = HERE / "runs" / "rehearsal" / "rehearsal.json"
    if not src.exists():
        sys.exit("沒有 runs/rehearsal/rehearsal.json")
    log = json.loads(src.read_text(encoding="utf-8"))
    al = load_alignment(_p(CFG["audio"]["alignment"]))
    meta = log["backend"]["meta"]
    pf, cf = pad_frames(meta), chunk_frames(meta)
    out = Out()
    out.banner(f"recompute — {log['backend']['backend']} {log['backend'].get('version')}，彩排 {log['time']}；model chunk {cf} frames, pad {pf} frames")
    settings = log["config"]["settings"]["items"]
    tl = Timeline(al, settings, fake=log["backend"]["backend"] == "FAKE")
    for key, r in log["runs"].items():
        feed = float(r["setting"]["feed_ms"])
        events = r["events"]
        rows = match_words(al["words"], group_words(events))
        summ = summarize(rows, events, feed, pf, mc["input_frame_ms"], cf)
        r["words"], r["summary"] = rows, summ
        tl.draw_row(key, rows, summ)
        out.banner(f"feed {r['setting']['name']}")
        out(f"{'t_fed':>7} {'t_wall':>7} {'t_frame':>8}   token")
        for e in events:
            out(f"{e['t_fed']:7.3f} {e['t_wall']:7.3f} {e['t_frame']:8.3f}   {e['tok']!r}")
        out()
        out(f"hyp: {summ['hyp_text']}")
        out(f"ref: {al['text']}   ({summ['n_matched']}/{summ['n_ref']} words matched)")
        out()
        out(f"{'word':>9} {'acoustic end':>12} │ {'emit−end':>9} {'fed−end':>8} {'wall−end':>9}")
        for w in rows:
            out(f"{w['ref']:>9} {w['end']:12.3f} │ {ms(w['gap_frame']):>9} {ms(w['gap_fed']):>8} {ms(w['gap_wall']):>9}" if w["gap_wall"] is not None
                else f"{w['ref']:>9} {w['end']:12.3f} │ {'—':>9} {'—':>8} {'—':>9}   {w['hyp'] or 'missing'}")
        da = summ["d_alg"]
        out()
        out(f"median gap: emit {summ.get('median_gap_frame_ms', float('nan')):+.0f} ms · fed {summ.get('median_gap_fed_ms', float('nan')):+.0f} ms · wall {summ.get('median_gap_wall_ms', float('nan')):+.0f} ms")
        out(f"worst word (wall): {summ['worst_wall']['word']} {summ['worst_wall']['gap_ms']:+.0f} ms" if "worst_wall" in summ else "worst word: —")
        out(f"computed D_alg: c = {da['c_ms']:.0f}, pad = {da['m_ms']:.0f}, F = {da['C_ms']:.0f} → mean c + F/2 = {da['mean_ms']:.0f} ms, worst c + F = {da['worst_ms']:.0f} ms (naive pad + F/2 = {da['naive_mean_ms']:.0f})")
        out(f"measured  fed − emit: median {summ.get('median_fed_minus_frame_ms', float('nan')):+.0f} ms · wall − fed: median {summ.get('median_wall_minus_fed_ms', float('nan')):+.0f} ms")
    # 摘要表（同 Demo.summary_table，但不靠 Demo 物件）
    out.banner("p42 的表格（三列）")
    out(f"{'feed chunk':>24} │ {'D_alg mean':>10} │ {'emit−end med':>12} {'fed−end med':>11} {'wall−end med':>12} │ worst word (wall)")
    for st in settings:
        s_ = log["runs"][st["key"]]["summary"]
        out(f"{st['name']:>24} │ {s_['d_alg']['mean_ms']:10.0f} │ {s_.get('median_gap_frame_ms', float('nan')):+12.0f} "
            f"{s_.get('median_gap_fed_ms', float('nan')):+11.0f} {s_.get('median_gap_wall_ms', float('nan')):+12.0f} │ "
            f"{s_['worst_wall']['word']} {s_['worst_wall']['gap_ms']:+.0f}")
    tl.draw_summary()
    log.setdefault("recomputed", []).append({"time": C.now(), "note": "post-processing redone from saved events (grouping, matching, D_alg); clocks untouched"})
    src.write_text(json.dumps(log, ensure_ascii=False, indent=1), encoding="utf-8")
    (src.parent / "rehearsal.txt").write_text(out.text(), encoding="utf-8")
    tl.save(src.parent / "timeline.png")
    print(f"→ {src.parent}/  (rehearsal.json, rehearsal.txt, timeline.png 重寫)")
    chk = HERE / "runs" / "check.json"
    if chk.exists():
        rep_ = json.loads(chk.read_text(encoding="utf-8"))
        o = rep_.get("offline")
        if o and "tokens" in o:
            evs = [{"tok": t, "t_frame": ts, "t_fed": 0.0, "t_wall": 0.0} for t, ts in zip(o["tokens"], o["timestamps"])]
            hyp = group_words(evs)
            rows = match_words(al["words"], hyp)
            o["hyp_text"] = " ".join(w["word"] for w in hyp)
            o["n_matched"] = sum(r["gap_frame"] is not None for r in rows)
            o["emit_minus_end_ms"] = [[r["ref"], None if r["gap_frame"] is None else round(1000 * r["gap_frame"])] for r in rows]
            g = [r["gap_frame"] for r in rows if r["gap_frame"] is not None]
            o["emit_minus_end_median_ms"] = round(1000 * statistics.median(g)) if g else None
            rep_.setdefault("recomputed", []).append(C.now())
            chk.write_text(json.dumps(rep_, ensure_ascii=False, indent=1), encoding="utf-8")
            print(f"→ {chk}  (offline.hyp_text / n_matched / emit_minus_end_ms 重寫)")


def selftest() -> None:
    fails = 0

    def ok(name: str, cond: bool, detail: str = "") -> None:
        nonlocal fails
        fails += not cond
        print(f"  {'OK  ' if cond else 'FAIL'} {name}{('  ' + detail) if detail else ''}")

    C.banner("selftest（不需要模型）")
    al = load_alignment(_p(CFG["audio"]["alignment"]))
    ok("對齊：17 個字、T = 292", len(al["words"]) == 17 and al["T"] == 292, f"{len(al['words'])} 字, T = {al['T']}")
    ok("第一個字 MISTER 0.56–0.80 s", al["words"][0]["word"] == "MISTER" and al["words"][0]["start"] == 0.56 and al["words"][0]["end"] == 0.80,
       f"{al['words'][0]}")
    ok("最後一個字 GOSPEL 結尾 5.42 s", al["words"][-1]["word"] == "GOSPEL" and al["words"][-1]["end"] == 5.42, f"{al['words'][-1]}")
    ok("字的起訖單調", all(al["words"][i]["end"] <= al["words"][i + 1]["start"] for i in range(16)))

    ev = [{"tok": "▁MIS", "t_frame": 0.8, "t_fed": 1.28, "t_wall": 1.3}, {"tok": "TER", "t_frame": 0.84, "t_fed": 1.28, "t_wall": 1.3},
          {"tok": "▁QU", "t_frame": 1.2, "t_fed": 1.28, "t_wall": 1.31}, {"tok": "IL", "t_frame": 1.24, "t_fed": 2.56, "t_wall": 2.6},
          {"tok": "TER", "t_frame": 1.28, "t_fed": 2.56, "t_wall": 2.6}, {"tok": "▁IS", "t_frame": 1.4, "t_fed": 2.56, "t_wall": 2.6}]
    w = group_words(ev)
    ok("BPE 併字：▁ 起新字", [x["word"] for x in w] == ["MISTER", "QUILTER", "IS"], str([x["word"] for x in w]))
    ok("字的時鐘取最後一個片段", w[1]["last"]["t_fed"] == 2.56 and w[1]["first"]["t_frame"] == 1.2)
    rows = match_words(al["words"][:4], w)
    ok("比對：3 對上、THE 缺", [r["hyp"] for r in rows] == ["MISTER", "QUILTER", "IS", None])
    ok("差 = 時鐘 − 聲學結尾", rows[0]["gap_frame"] == round(0.84 - 0.80, 3) and rows[0]["gap_fed"] == round(1.28 - 0.80, 3))
    sub = match_words(al["words"][:3], group_words([{"tok": "▁MISTER", "t_frame": 0.8, "t_fed": 1, "t_wall": 1}, {"tok": "▁QUILT", "t_frame": 1, "t_fed": 1, "t_wall": 1},
                                                      {"tok": "▁IS", "t_frame": 1.4, "t_fed": 2, "t_wall": 2}]))
    ok("替換的字標 ≠、不算差", sub[1]["hyp"] == "≠QUILT" and sub[1]["gap_wall"] is None and sub[2]["gap_wall"] is not None)

    da = d_alg_ms(320, 13, 10.0, 32)
    ok("D_alg：feed 320 → c 320、pad 130、mean c + F/2 = 480、worst 640", (da["c_ms"], da["m_ms"], da["mean_ms"], da["worst_ms"]) == (320.0, 130.0, 480.0, 640.0), str(da))
    ok("D_alg：feed 640 → 640、1280 → 960", d_alg_ms(640, 13, 10.0, 32)["mean_ms"] == 640.0 and d_alg_ms(1280, 13, 10.0, 32)["mean_ms"] == 960.0)
    ok("D_alg：naive pad + F/2 留著對照 = 290", da["naive_mean_ms"] == 290.0)
    sp = group_words([{"tok": " MISTER", "t_frame": 0.8, "t_fed": 1, "t_wall": 1}, {"tok": " ", "t_frame": 1.08, "t_fed": 1, "t_wall": 1},
                      {"tok": "QUI", "t_frame": 1.12, "t_fed": 1, "t_wall": 1}, {"tok": "L", "t_frame": 1.24, "t_fed": 1, "t_wall": 1}, {"tok": "TER", "t_frame": 1.36, "t_fed": 1, "t_wall": 1}, {"tok": " IS", "t_frame": 1.6, "t_fed": 2, "t_wall": 2}])
    ok("sherpa-onnx 的空白詞首（含單獨的 \" \"）", [x["word"] for x in sp] == ["MISTER", "QUILTER", "IS"] and sp[1]["last"]["t_frame"] == 1.36, str([x["word"] for x in sp]))
    meta = {"decode_chunk_len": "32", "T": "45"}
    ok("metadata → chunk 32、pad 13", chunk_frames(meta) == 32 and pad_frames(meta) == 13)
    ok("沒有 metadata → config 預設", chunk_frames({}) == 32 and pad_frames({}) == 13)

    fk = Fake(al)
    x = np.zeros(int(6.6 * CFG["audio"]["fs"]), dtype=np.float32)
    for feed in (320, 640, 1280):
        evs = fk.run(x, feed, pace=False)
        rows = match_words(al["words"], group_words(evs))
        s = summarize(rows, evs, feed, 13, 10.0, 32)
        ok(f"FAKE feed {feed}：17 字全對上", s["n_matched"] == 17, s["hyp_text"])
        fm = [e["t_fed"] - e["t_frame"] for e in evs]
        ok(f"FAKE feed {feed}：fed − frame ∈ [m + 40 ms, m + 40 ms + C]", all(0.17 - 1e-9 <= v <= 0.17 + feed / 1000 + 1e-9 for v in fm),
           f"min {min(fm):.3f} max {max(fm):.3f}")
        ok(f"FAKE feed {feed}：t_frame 在 40 ms 格上", all(abs(e["t_frame"] / 0.04 - round(e["t_frame"] / 0.04)) < 1e-6 for e in evs))
        ok(f"FAKE feed {feed}：median wall − fed = 20 ms", s["median_wall_minus_fed_ms"] == 20.0, str(s["median_wall_minus_fed_ms"]))
    print(f"\n{'全部通過' if not fails else f'{fails} 項失敗'}")
    if fails:
        sys.exit(1)


# ── main ───────────────────────────────────────────────────────
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["show", "rehearse", "replay", "check", "fetch", "fake", "selftest", "export", "recompute"])
    ap.add_argument("--mute", action="store_true", help="不播音（rehearse 一律不播）")
    args = ap.parse_args()

    if args.cmd == "selftest":
        selftest()
        return
    if args.cmd == "fetch":
        fetch()
        return
    if args.cmd == "check":
        check()
        return
    if args.cmd == "export":
        export()
        return
    if args.cmd == "recompute":
        recompute()
        return

    rehearsal = HERE / "runs" / "rehearsal"
    out = Out()
    if args.cmd == "replay":
        src = rehearsal / "rehearsal.json"
        if not src.exists():
            sys.exit(f"沒有彩排紀錄 {src}。先跑 ./present.sh rehearse。")
        r = Replay(out, json.loads(src.read_text(encoding="utf-8")))
        interactive(r)  # type: ignore[arg-type]
        return

    al = load_alignment(_p(CFG["audio"]["alignment"]))
    x = load_audio()
    if args.cmd == "fake":
        d = Demo(out, Fake(al), al, x, tag="fake", mute=True)
        try:
            interactive(d)
        finally:
            if d.log["runs"]:   # 存 runs/live/fake-…（不進版控），看圖用；export 會拒絕 FAKE
                save(d, HERE / "runs" / "live" / ("fake-" + C.now().replace(":", "")))
        return

    be = Sherpa()
    if args.cmd == "rehearse":
        import matplotlib
        matplotlib.use("Agg")
        d = Demo(out, be, al, x, tag="rehearsal", mute=True)
        d.run_all()
        save(d, rehearsal)
        print("\n下一步：./present.sh export（寫 slides/assets/w04/w04-metrics.json）；把三列的中位數、最差字與日期填進 p42 的表格與【提示】。")
        return

    d = Demo(out, be, al, x, mute=args.mute)
    try:
        interactive(d)
    finally:
        if d.log["runs"]:
            save(d, HERE / "runs" / "live" / C.now().replace(":", ""))


if __name__ == "__main__":
    main()

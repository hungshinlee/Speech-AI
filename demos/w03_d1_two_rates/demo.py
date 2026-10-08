# -*- coding: utf-8 -*-
# 此檔由 bin/publish_demos.py 從 private repo 的 <course>/demos/ 複製而來，請勿在公開 repo 直接編輯。
"""W3 demo 1：同一句話、兩種語速——frame 數變了，符號數沒變，對應關係沒人知道。

  show       課堂：錄兩段（正常語速、快一倍）→ 先問學生「哪些數字變了」→ 按 s 兩張 spectrogram 上下並排、起點對齊
  rehearse   課前：同 show，但 s 會把兩段音、圖與數字存進 runs/rehearsal/（進版控 = 現場的退路）
  replay     現場收不到音時：載入 runs/rehearsal/ 的兩段，直接進並排畫面（畫面標明非現場）
  fake       沒有麥克風也能演練流程（合成的「句子」，第二段各音段縮短；不是給學生聽的）
  devices    列出音訊輸入裝置

畫面上的鍵：
  r        錄下一段（第 1 段 → 第 2 段；兩段都有時從第 1 段重來）。倒數之後錄 record_seconds 秒
  1 / 2    重錄指定的那一段
  s        並排（兩段都錄好之後才有作用）。rehearse 存 runs/rehearsal/，show 存 runs/live/<時間>/
  w / n    25 ms 窗（每欄 = 一個 ASR frame）／ 30 ms narrowband 窗；hop 固定 10 ms，欄數不變
  p        有聲 frame 的 F0 點開關      z   頻率上限 5 kHz ↔ 8 kHz
  q        離開

按 s 之前畫面上不出現任何時間軸——「哪些數字變了、哪些沒變」要先問再揭曉。
數字的定義：T = 裁掉前後靜音之後的長度 ÷ 10 ms（與 ASR 前端 25 ms／10 ms 的 frame 數只差邊界的一兩個）；
U = 句子去掉空白與標點的字元數。兩者都印在終端機與 metrics.json 裡。

與 W2 Demo 1 的關係：同一套 dsp.spectrogram_db、同一個 Mic 類別，差別是這裡是「錄完再看」不是即時捲動——
對齊起點只有錄完才做得到。
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import warnings
import threading
import time
import tomllib
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import common as C  # noqa: E402
import dsp          # noqa: E402

CFG = tomllib.loads((HERE / "demo_config.toml").read_text(encoding="utf-8"))
FS = CFG["audio"]["fs"]
HOP = int(round(FS * CFG["frames"]["hop_ms"] / 1000))          # 一個 frame = 這麼多 sample
LABELS = ("normal rate", "twice as fast")
FILES = ("normal", "fast")


def symbol_count(text: str) -> int:
    """U：去掉空白與標點。華語一字一符號；英文就是字母數。"""
    return len(re.sub(r"[\s\W_]", "", text, flags=re.UNICODE))


CJK_FONT_FILES = (
    # macOS：Fonts/ 裡的 PingFang.ttc 是空殼，真的字形在 FontServices.framework；STHeiti、Hiragino、Arial Unicode 是備援
    "/System/Library/PrivateFrameworks/FontServices.framework/Versions/A/Resources/Fonts/ApplePingFang.ttc",
    "/System/Library/Fonts/PingFang.ttc",
    "/System/Library/Fonts/STHeiti Medium.ttc",
    "/System/Library/Fonts/STHeiti Light.ttc",
    "/System/Library/Fonts/Hiragino Sans GB.ttc",
    "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
    "/Library/Fonts/Arial Unicode.ttf",
    # Linux：Noto CJK
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/noto-cjk/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
)


def cjk_font_for(text: str):
    """回傳畫得出 `text` 每一個字的 FontProperties；找不到回 None（畫面上就是方塊，不影響數字）。

    不能用字型「名稱」交給 rcParams 去找：macOS 的 PingFang.ttc 在 /System/Library/Fonts 只是空殼，
    .ttc 又只會登錄第一個 face，名稱對不上時 matplotlib 整行退回 DejaVu、只在 log 裡抱怨一句。
    這裡改成直接開檔、逐字檢查 glyph index，第一個全部有字形的就用。
    """
    from matplotlib import font_manager as fm
    from matplotlib.ft2font import FT2Font

    need = [c for c in text if not c.isascii()] + ["「", "」"]
    seen: set[str] = set()
    known = [f.fname for f in fm.fontManager.ttflist
             if any(k in f.name for k in ("PingFang", "Heiti", "Hiragino", "Unicode", "CJK", "Song", "Kai", "Ming"))]
    for path in (*CJK_FONT_FILES, *known):
        if path in seen or not Path(path).is_file():
            continue
        seen.add(path)
        try:
            face = FT2Font(path)
            if all(face.get_char_index(ord(c)) for c in need):
                return fm.FontProperties(fname=path)
        except Exception:  # 空殼、壞檔、不是 FreeType 認得的格式
            continue
    return None


# ── 裁靜音、數 frame ────────────────────────────────────────────
def frame_db(x: np.ndarray) -> np.ndarray:
    n = len(x) // HOP
    fr = x[: n * HOP].reshape(n, HOP)
    return 20 * np.log10(np.sqrt(np.mean(fr * fr, axis=1)) + 1e-9)


def trim(x: np.ndarray) -> tuple[np.ndarray, bool]:
    """回傳 (裁掉前後靜音的訊號, 有沒有真的偵測到語音)。

    門檻取兩者之高：「峰值以下 threshold_db」與「底噪以上 floor_margin_db」（底噪 = frame 能量的第 10 百分位）。
    只看峰值時，Mac 內建麥克風的底噪離語音峰值只有 30–38 dB，35 dB 的門檻等於沒裁（2026-09-29 六次都是 T = 400）。
    起訖點另外要求連續 min_run_ms 都在門檻之上，單一 frame 的呼吸聲或雜音不算（同一批錄音裡句尾一個 −35 dB 的
    frame 曾讓 fast 那段從 90 變 330 個 frame）。
    """
    t = CFG["trim"]
    e = frame_db(x)
    if len(e) == 0:
        return x, False
    floor = float(np.percentile(e, 10))
    th = max(e.max() - t["threshold_db"], floor + t.get("floor_margin_db", 15))
    above = e > th
    min_run = max(1, int(round(t.get("min_run_ms", 50) / (1000 * HOP / FS))))

    def first_run(mask: np.ndarray) -> int | None:
        run = 0
        for i, b in enumerate(mask):
            run = run + 1 if b else 0
            if run >= min_run:
                return i - min_run + 1
        return None

    a0 = first_run(above)
    b0 = first_run(above[::-1])
    if a0 is None or b0 is None:
        return x, False
    on0, on1 = a0, len(e) - 1 - b0
    if (on1 - on0 + 1) * HOP < FS * t["min_speech_ms"] / 1000:
        return x, False
    pad = int(FS * t["pad_ms"] / 1000)
    a = max(0, on0 * HOP - pad)
    b = min(len(x), (on1 + 1) * HOP + pad)
    return x[a:b], True


def voiced_frames(x: np.ndarray) -> tuple[int, int]:
    """(有聲 frame 數, 無聲 frame 數)，以 10 ms hop 走 autocorr F0。粗分類，量級可信。"""
    pc = CFG["pitch"]
    flen = int(FS * pc["frame_ms"] / 1000)
    v = u = 0
    for a in range(0, max(0, len(x) - flen) + 1, HOP):
        f0, _ = dsp.f0_autocorr(x[a:a + flen], FS, pc["floor_hz"], pc["ceiling_hz"], pc["voicing"])
        if np.isnan(f0):
            u += 1
        else:
            v += 1
    return v, u


@dataclass
class Take:
    raw: np.ndarray
    x: np.ndarray                    # 裁掉前後靜音之後
    speech_found: bool
    f0: list[tuple[float, float]] = field(default_factory=list)   # (秒, Hz) 只放有聲的

    @property
    def T(self) -> int:              # frame 數：長度 ÷ 10 ms
        return int(round(len(self.x) / HOP))

    @property
    def seconds(self) -> float:
        return len(self.x) / FS

    def analyse(self) -> None:
        pc = CFG["pitch"]
        flen = int(FS * pc["frame_ms"] / 1000)
        self.f0 = []
        for a in range(0, max(0, len(self.x) - flen) + 1, HOP):
            f0, _ = dsp.f0_autocorr(self.x[a:a + flen], FS, pc["floor_hz"], pc["ceiling_hz"], pc["voicing"])
            if not np.isnan(f0):
                self.f0.append(((a + flen / 2) / FS, float(f0)))

    @property
    def voiced(self) -> int:
        return len(self.f0)

    @property
    def unvoiced(self) -> int:
        pc = CFG["pitch"]
        flen = int(FS * pc["frame_ms"] / 1000)
        n = len(range(0, max(0, len(self.x) - flen) + 1, HOP))
        return n - self.voiced


def make_take(raw: np.ndarray) -> Take:
    x, ok = trim(raw)
    t = Take(raw, x, ok)
    t.analyse()
    return t


# ── 音源 ────────────────────────────────────────────────────────
class Mic:
    """與 W2 Demo 1 相同：一直開著的 InputStream，tick 時把累積的 chunk 拿走。"""
    def __init__(self):
        import sounddevice as sd
        self.q: deque[np.ndarray] = deque()
        self.lock = threading.Lock()
        dev = CFG["audio"].get("device") or None
        self.stream = sd.InputStream(samplerate=FS, channels=1, dtype="float32", device=dev,
                                     blocksize=int(FS * 0.02), callback=self._cb)
        self.stream.start()

    def _cb(self, indata, frames, time_, status):
        with self.lock:
            self.q.append(indata[:, 0].astype(np.float64).copy())

    def read(self) -> np.ndarray:
        with self.lock:
            chunks, self.q = list(self.q), deque()
        return np.concatenate(chunks) if chunks else np.zeros(0)

    def close(self):
        self.stream.stop()
        self.stream.close()


class NoMic:
    def read(self) -> np.ndarray:
        return np.zeros(0)

    def close(self):
        pass


def fake_takes() -> tuple[Take, Take]:
    """合成的「句子」：/a/ /i/ /u/ /a/ /i/ 五個音段夾著短靜音。第二段母音縮到 0.45 倍、靜音縮到 0.7 倍——
    快講時母音縮得多、停頓縮得少，所以 T 不會剛好一半。只是流程演練，不是給學生聽的。"""
    def build(vs: float, gs: float) -> np.ndarray:
        segs = [np.zeros(int(FS * 0.35))]
        for k, v in enumerate("aiuai"):
            f0 = np.linspace(130, 105, 50) if k % 2 == 0 else np.linspace(115, 140, 50)
            segs.append(dsp.synth_vowel(FS, 0.22 * vs, f0, dsp.VOWELS[v]))
            segs.append(np.zeros(int(FS * (0.06 if k != 2 else 0.18) * gs)))
        segs.append(np.zeros(int(FS * 0.4)))
        x = np.concatenate(segs)
        rng = np.random.default_rng(0)
        return x + 1e-3 * rng.standard_normal(len(x))
    return make_take(build(1.0, 1.0)), make_take(build(0.45, 0.7))


# ── 畫面 ────────────────────────────────────────────────────────
class Viewer:
    def __init__(self, source, mode: str, takes: list[Take | None] | None = None, tag: str = ""):
        import matplotlib.pyplot as plt
        for k in [k for k in plt.rcParams if k.startswith("keymap.")]:
            plt.rcParams[k] = []
        # 句子是中文，只有左上角那一行用到。不能靠 rcParams 填字型名稱：macOS 的 PingFang.ttc 是
        # 空殼（本體在 FontServices.framework 裡）、STHeiti 是 .ttc 而 matplotlib 只登錄第一個 face，
        # 名稱對不上就整行退回 DejaVu → 八個豆腐。改成直接開字型檔、確認每個字都有字形才採用。
        self.cjk_font = cjk_font_for(CFG["sentence"]["text"])
        warnings.filterwarnings("ignore", message="Glyph .* missing from font")
        self.plt = plt
        self.src, self.mode, self.tag = source, mode, tag
        self.takes: list[Take | None] = takes if takes else [None, None]
        self.state = "idle"                # idle | countdown | recording
        self.slot = 0
        self.t_deadline = 0.0
        self.rec: list[np.ndarray] = []
        self.rec_n = 0
        self.compared = False
        self.wide, self.show_pitch, self.fmax = True, True, CFG["display"]["fmax_hz"][0]
        self.msg = ""
        self.level_db = -80.0

        d = CFG["display"]
        self.fig, self.axes = plt.subplots(2, 1, figsize=(d["fig_w"], d["fig_h"]), sharex=True)
        self.fig.subplots_adjust(left=0.07, right=0.95, top=0.83, bottom=0.09, hspace=0.34)
        self.ims = []
        self.pitches = []
        self.shades = []
        for ax, lab in zip(self.axes, LABELS):
            im = ax.imshow(np.zeros((2, 2)), origin="lower", aspect="auto", cmap="magma",
                           extent=[0, 1, 0, FS / 2000], vmin=-d["dynamic_range_db"], vmax=0)
            im.set_visible(False)
            self.ims.append(im)
            ax.set_ylabel("frequency (kHz)")
            axp = ax.twinx()                       # F0 疊在右軸，與 W2 Demo 1 相同
            (p,) = axp.plot([], [], ".", color="#5ee0ff", ms=4)
            axp.set_ylim(CFG["pitch"]["floor_hz"], CFG["pitch"]["ceiling_hz"])
            axp.set_ylabel("F0 (Hz)", color="#1899b8")
            self.pitches.append(p)
            ax.set_title(lab, loc="left", fontsize=13, fontweight="bold")
            ax.set_xticks([])
        self.axes[1].set_xlabel("time from the first voiced frame (s)")
        self.suptitle = self.fig.suptitle("", fontsize=14, x=0.07, y=0.975, ha="left",
                                          **({"fontproperties": self.cjk_font} if self.cjk_font else {}))
        if self.cjk_font:
            self.suptitle.set_fontsize(14)      # fontproperties 會蓋掉 fontsize，補回來
        else:
            print("  [warn] 找不到能畫中文的字型，左上角句子會是方塊（不影響數字）。", file=sys.stderr)
        self.summary = self.fig.text(0.07, 0.905, "", fontsize=15, ha="left", va="center", color="#b3261e", fontweight="bold")
        self.notes = [ax.text(0.5, 0.5, "", transform=ax.transAxes, ha="center", va="center", fontsize=22, color="0.35")
                      for ax in self.axes]
        from matplotlib.patches import Rectangle
        self.bars = [ax.add_patch(Rectangle((0.0, 0.08), 0.0, 0.12, transform=ax.transAxes, color="#d1345b", alpha=0.0))
                     for ax in self.axes]
        self.fig.canvas.mpl_connect("key_press_event", self.on_key)
        self.draw_idle_state()

    # -- 狀態文字 --
    def draw_idle_state(self) -> None:
        for i, (ax, note) in enumerate(zip(self.axes, self.notes)):
            if self.compared:
                note.set_text("")
                continue
            if self.state in ("countdown", "recording") and i == self.slot:
                continue
            note.set_text("recorded ✓  (press s when both are in)" if self.takes[i] else f"press r to record take {i + 1}")
        U = symbol_count(CFG["sentence"]["text"])
        head = f"「{CFG['sentence']['text']}」  U = {U} symbols   ·   " + self.hint()
        if self.tag:
            head += f"      {self.tag}"
        self.suptitle.set_text(head)
        if not self.compared:
            self.summary.set_text(self.msg)

    def hint(self) -> str:
        if self.state == "countdown":
            return f"take {self.slot + 1}: get ready…"
        if self.state == "recording":
            return f"take {self.slot + 1}: RECORDING"
        return "r = record   s = compare   q = quit"

    # -- 錄音狀態機 --
    def start_take(self, slot: int) -> None:
        if self.state != "idle" or isinstance(self.src, NoMic):
            return
        self.slot, self.state = slot, "countdown"
        self.compared = False
        for im, p in zip(self.ims, self.pitches):
            im.set_visible(False)
            p.set_data([], [])
        for ax in self.axes:
            ax.set_xticks([])
        self.t_deadline = time.monotonic() + CFG["audio"]["countdown_s"]
        self.msg = ""
        self.draw_idle_state()

    def tick(self, _=None):
        x = self.src.read()
        if self.state == "countdown":
            left = self.t_deadline - time.monotonic()
            self.notes[self.slot].set_text(f"take {self.slot + 1} — {LABELS[self.slot]}\n\n{int(np.ceil(max(left, 0)))}")
            if left <= 0:
                self.state, self.rec, self.rec_n = "recording", [], 0
                self.notes[self.slot].set_text(f"take {self.slot + 1} — {LABELS[self.slot]}\n\n● speak now")
        elif self.state == "recording":
            if len(x):
                self.rec.append(x)
                self.rec_n += len(x)
                self.level_db = 20 * np.log10(np.sqrt(np.mean(x * x)) + 1e-9)
            bar = self.bars[self.slot]
            bar.set_alpha(0.9)
            bar.set_width(float(np.clip((self.level_db + 60) / 60, 0.02, 1.0)))   # -60 dBFS → 0，0 dBFS → 滿
            if self.rec_n >= CFG["audio"]["record_seconds"] * FS:
                raw = np.concatenate(self.rec)[: int(CFG["audio"]["record_seconds"] * FS)]
                self.takes[self.slot] = make_take(raw)
                bar.set_alpha(0.0)
                self.state = "idle"
                if not self.takes[self.slot].speech_found:
                    self.msg = f"take {self.slot + 1}: no speech detected — check ./present.sh devices, then press {self.slot + 1} to redo"
                    self.takes[self.slot] = None
                elif self.slot == 0 and self.takes[1] is None:
                    self.msg = "take 1 in.  Now press r for the fast one."
                else:
                    self.msg = "both takes in.  Ask the question, then press s."
        self.draw_idle_state()
        return self.ims + self.pitches

    # -- 並排 --
    def compare(self) -> None:
        if not all(self.takes):
            self.msg = "need both takes first (r)"
            self.draw_idle_state()
            return
        d, f = CFG["display"], CFG["frames"]
        win = f["win_ms"] if self.wide else d["narrow_ms"]
        tmax = max(t.seconds for t in self.takes)
        for ax, im, p, note, t in zip(self.axes, self.ims, self.pitches, self.notes, self.takes):
            S = dsp.spectrogram_db(t.x, FS, win, f["hop_ms"], f["n_fft"])
            ref = max(S.max(), d["floor_db"])
            im.set_data(S - ref)
            im.set_extent([0, S.shape[1] * f["hop_ms"] / 1000, 0, FS / 2000])
            im.set_visible(True)
            ax.set_xlim(0, tmax)
            ax.set_ylim(0, self.fmax / 1000)
            ax.set_xticks(np.arange(0, tmax + 1e-9, 0.25))
            if self.show_pitch and t.f0:
                tt, ff = zip(*t.f0)
                p.set_data(tt, ff)
            else:
                p.set_data([], [])
            note.set_text("")
        for s in self.shades:
            s.remove()
        self.shades = []
        for ax, t in zip(self.axes, self.takes):
            if t.seconds < tmax - 1e-6:
                self.shades.append(ax.axvspan(t.seconds, tmax, color="0.85", alpha=0.6, lw=0))
            self.shades.append(ax.axvline(t.seconds, color="#d1345b", lw=1.5, ls="--"))
        t1, t2 = self.takes
        for ax, lab, t in zip(self.axes, LABELS, self.takes):
            ax.set_title(f"{lab}   —   T = {t.T} frames   ({t.seconds:.2f} s ÷ 10 ms;  window {win:g} ms, each column one frame)",
                         loc="left", fontsize=13, fontweight="bold")
        U = symbol_count(CFG["sentence"]["text"])
        self.compared = True
        self.msg = ""
        self.summary.set_text(f"T = {t1.T} → {t2.T} frames (×{t2.T / t1.T:.2f})    U = {U} both times    "
                              f"T/U = {t1.T / U:.1f} → {t2.T / U:.1f}    which frame ↔ which symbol?  nobody says.")
        self.draw_idle_state()
        self.fig.canvas.draw_idle()

    def metrics(self) -> dict:
        U = symbol_count(CFG["sentence"]["text"])
        out = {"time": C.now(), "mode": self.mode, "sentence": CFG["sentence"]["text"], "U": U,
               "frame_hop_ms": CFG["frames"]["hop_ms"], "config": CFG, "versions": C.versions(), "takes": {}}
        for name, lab, t in zip(FILES, LABELS, self.takes):
            out["takes"][name] = {"label": lab, "raw_seconds": round(len(t.raw) / FS, 3),
                                  "trimmed_seconds": round(t.seconds, 3), "T_frames": t.T,
                                  "frames_per_symbol": round(t.T / U, 2),
                                  "voiced_frames": t.voiced, "unvoiced_frames": t.unvoiced}
        t1, t2 = self.takes
        out["ratio_T"] = round(t2.T / t1.T, 3)
        out["ratio_voiced"] = round(t2.voiced / max(t1.voiced, 1), 3)
        out["ratio_unvoiced"] = round(t2.unvoiced / max(t1.unvoiced, 1), 3)
        return out

    def report(self) -> None:
        m = self.metrics()
        C.banner("同一句話、兩種語速", C.AMBER)
        print(f"  句子：「{m['sentence']}」   U = {m['U']} symbols（兩段相同）")
        print(f"  {'':14s}{'trimmed':>9s}{'T (10 ms)':>11s}{'T / U':>8s}{'voiced':>8s}{'unvoiced':>10s}")
        for name in FILES:
            t = m["takes"][name]
            print(f"  {t['label']:14s}{t['trimmed_seconds']:>8.2f}s{t['T_frames']:>11d}{t['frames_per_symbol']:>8.1f}"
                  f"{t['voiced_frames']:>8d}{t['unvoiced_frames']:>10d}")
        print(f"  T 比例 {m['ratio_T']:.2f}    有聲 frame 比例 {m['ratio_voiced']:.2f}    無聲 frame 比例 {m['ratio_unvoiced']:.2f}"
              f"   （有聲／無聲是 autocorr 粗分類，量級可信、逐 frame 不可信）")
        C.note("  T = 裁掉前後靜音的長度 ÷ 10 ms；與 ASR 前端 25/10 ms 的 frame 數只差邊界的一兩個。")

    def save(self, folder: Path) -> None:
        folder.mkdir(parents=True, exist_ok=True)
        peak = max(float(np.max(np.abs(t.x))) for t in self.takes) + 1e-9
        for name, t in zip(FILES, self.takes):
            C.save_wav(folder / f"{name}.wav", t.x, FS, peak)
            C.save_wav(folder / f"{name}_raw.wav", t.raw, FS, peak)
        self.fig.savefig(folder / "compare.png", dpi=130)
        (folder / "metrics.json").write_text(json.dumps(self.metrics(), ensure_ascii=False, indent=1), encoding="utf-8")
        C.note(f"  存到 {folder}/（normal.wav、fast.wav、*_raw.wav、compare.png、metrics.json）")

    # -- 鍵盤 --
    def on_key(self, ev) -> None:
        k = ev.key
        if k == "r":
            if self.takes[0] is None:
                self.start_take(0)
            elif self.takes[1] is None:
                self.start_take(1)
            else:
                self.takes = [None, None]
                self.start_take(0)
        elif k in ("1", "2"):
            self.start_take(int(k) - 1)
        elif k == "s":
            if self.state != "idle":
                return
            self.compare()
            if self.compared:
                self.report()
                if self.mode == "rehearse":
                    self.save(HERE / "runs" / "rehearsal")
                elif self.mode == "show":
                    self.save(HERE / "runs" / "live" / C.now().replace(":", ""))
        elif k in ("w", "n", "p", "z"):
            if k == "w":
                self.wide = True
            elif k == "n":
                self.wide = False
            elif k == "p":
                self.show_pitch = not self.show_pitch
            else:
                a, b = CFG["display"]["fmax_hz"]
                self.fmax = b if self.fmax == a else a
            if self.compared:
                self.compare()
        elif k == "q":
            self.plt.close(self.fig)
        self.fig.canvas.draw_idle()


def load_pair(folder: Path) -> list[Take]:
    takes = []
    for name in FILES:
        p = folder / f"{name}_raw.wav"
        p = p if p.exists() else folder / f"{name}.wav"
        if not p.exists():
            sys.exit(f"找不到 {p}。先 ./present.sh rehearse、錄兩段、按 s。")
        takes.append(make_take(C.load_wav(p, FS)))
    return takes


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["show", "rehearse", "replay", "fake", "devices"])
    ap.add_argument("folder", nargs="?", help="replay 用：含 normal.wav 與 fast.wav 的資料夾（預設 runs/rehearsal）")
    ap.add_argument("--selftest", metavar="PNG", help="不開視窗：fake 兩段 → 並排 → 存圖、印數字（測試用）")
    args = ap.parse_args()

    if args.cmd == "devices":
        import sounddevice as sd
        print(sd.query_devices())
        return
    if args.selftest:
        import matplotlib
        matplotlib.use("Agg")

    takes, tag, src = None, "", NoMic()
    if args.cmd == "fake":
        takes, tag = list(fake_takes()), "FAKE (synthetic)"
    elif args.cmd == "replay":
        folder = Path(args.folder) if args.folder else HERE / "runs" / "rehearsal"
        takes, tag = load_pair(folder), "REPLAY (not live)"
    else:
        try:
            src = Mic()
        except Exception as e:
            sys.exit(f"開不了麥克風：{e}\n看裝置：./present.sh devices；退路：./present.sh replay")

    v = Viewer(src, args.cmd, takes, tag)
    if args.cmd == "replay":
        v.compare()
        v.report()
    if args.selftest:
        v.on_key(type("E", (), {"key": "s"})())
        v.on_key(type("E", (), {"key": "n"})())
        v.fig.savefig(args.selftest, dpi=110)
        t1, t2 = v.takes
        print(f"selftest ok: T={t1.T},{t2.T} ratio={t2.T / t1.T:.2f} voiced={t1.voiced},{t2.voiced}")
        return

    from matplotlib.animation import FuncAnimation
    anim = FuncAnimation(v.fig, v.tick, interval=50, blit=False, cache_frame_data=False)
    try:
        v.plt.show()
    finally:
        src.close()
    del anim


if __name__ == "__main__":
    main()

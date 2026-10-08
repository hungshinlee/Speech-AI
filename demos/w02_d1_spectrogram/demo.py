# -*- coding: utf-8 -*-
# 此檔由 bin/publish_demos.py 從 private repo 的 <course>/demos/ 複製而來，請勿在公開 repo 直接編輯。
"""W2 demo 1：即時 spectrogram、pitch track 與 F1–F2 母音圖。

  show       課堂：麥克風即時收音
  rehearse   課前：同 show，另外可以按 1／2／3 把畫面上這段音存成退路（runs/rehearsal/）
  replay X   現場收不到音時：載入 runs/rehearsal/X.wav，畫面凍結，w／n／p／z 照樣能切
  fake       沒有麥克風也能演練流程（合成的 /a/ /i/ /u/，F0 一路往上滑）

畫面上的鍵：
  w / n    wideband（5 ms 窗）／ narrowband（30 ms 窗）——同一段音，只換窗長
  space    凍結／繼續。**先凍結再切 w／n**，學生才看得出是同一段音
  p        pitch track 開關          z   頻率上限 5 kHz ↔ 8 kHz
  c        清掉母音圖上的軌跡        s   存目前這段音與畫面（runs/live/）
  1 2 3    （只在 rehearse）存成 vowels／tones／question 三個退路
  q        離開

這支程式負責「嘴型一變、圖立刻跟著動」的第一印象。要精確讀數（共振峰頻率、F0 值）用 Praat，
見同資料夾的 w02_d1.praat 與 README。
"""
from __future__ import annotations

import argparse
import sys
import threading
import tomllib
from collections import deque
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import common as C  # noqa: E402
import dsp          # noqa: E402

CFG = tomllib.loads((HERE / "demo_config.toml").read_text(encoding="utf-8"))
FS = CFG["audio"]["fs"]
BUF_S = CFG["display"]["seconds"]
SLOTS = {"1": "vowels", "2": "tones", "3": "question"}


# ── 音源 ────────────────────────────────────────────────────────
class Mic:
    def __init__(self):
        import sounddevice as sd
        self.q: deque[np.ndarray] = deque()
        self.lock = threading.Lock()
        dev = CFG["audio"].get("device") or None
        self.stream = sd.InputStream(samplerate=FS, channels=1, dtype="float32", device=dev,
                                     blocksize=int(FS * 0.02), callback=self._cb)
        self.stream.start()

    def _cb(self, indata, frames, time, status):
        with self.lock:
            self.q.append(indata[:, 0].astype(np.float64).copy())

    def read(self, n_hint: int) -> np.ndarray:
        with self.lock:
            chunks, self.q = list(self.q), deque()
        return np.concatenate(chunks) if chunks else np.zeros(0)

    def close(self):
        self.stream.stop()
        self.stream.close()


class Fake:
    """/a/ → /i/ → /u/ 各 1.2 秒循環，F0 在每個母音裡由 110 Hz 滑到 180 Hz。"""
    def __init__(self):
        seg = [dsp.synth_vowel(FS, 1.2, np.linspace(110, 180, 100), dsp.VOWELS[v]) for v in "aiu"]
        gap = np.zeros(int(FS * 0.2))
        self.loop = np.concatenate([np.concatenate([s, gap]) for s in seg])
        self.pos = 0

    def read(self, n_hint: int) -> np.ndarray:
        idx = (self.pos + np.arange(n_hint)) % len(self.loop)
        self.pos = (self.pos + n_hint) % len(self.loop)
        return self.loop[idx]

    def close(self):
        pass


class Still:
    """replay 用：一次把整個檔案交出去，之後不再有新的音。"""
    def __init__(self, x: np.ndarray):
        self.x = x

    def read(self, n_hint: int) -> np.ndarray:
        x, self.x = self.x, np.zeros(0)
        return x

    def close(self):
        pass


# ── 畫面 ────────────────────────────────────────────────────────
class Viewer:
    def __init__(self, source, mode: str, frozen: bool = False):
        import matplotlib.pyplot as plt
        for k in [k for k in plt.rcParams if k.startswith("keymap.")]:
            plt.rcParams[k] = []                      # 把 s／f／p／q 等預設快捷鍵讓出來
        self.plt = plt
        self.src, self.mode, self.frozen = source, mode, frozen
        self.buf = np.zeros(int(FS * BUF_S))
        self.total = 0                                # 到目前為止收進來的 sample 數
        self.next_hop = 0                             # 下一個要分析的 frame 起點（絕對位置）
        self.track: deque[tuple[int, float, float, float]] = deque()   # (位置, f0, F1, F2)
        self.wide, self.show_pitch, self.fmax = True, True, CFG["display"]["fmax_hz"][0]
        self.msg = ""

        d = CFG["display"]
        self.fig = plt.figure(figsize=(d["fig_w"], d["fig_h"]))
        gs = self.fig.add_gridspec(1, 2, width_ratios=[3.2, 1], wspace=0.34)
        self.ax = self.fig.add_subplot(gs[0])
        self.axp = self.ax.twinx()
        self.axv = self.fig.add_subplot(gs[1])
        self.im = self.ax.imshow(np.zeros((2, 2)), origin="lower", aspect="auto", cmap="magma",
                                 extent=[-BUF_S, 0, 0, FS / 2000], vmin=-d["dynamic_range_db"], vmax=0)
        self.ax.set_xlabel("time (s)")
        self.ax.set_ylabel("frequency (kHz)")
        (self.pitch,) = self.axp.plot([], [], ".", color="#5ee0ff", ms=5)
        self.axp.set_ylim(CFG["pitch"]["floor_hz"], CFG["pitch"]["ceiling_hz"])
        self.axp.set_ylabel("F0 (Hz)", color="#1899b8")
        # 母音圖：照語音學慣例兩個軸都反過來（左上 = 高前母音 /i/）
        self.axv.set_xlim(2800, 500)
        self.axv.set_ylim(1000, 150)
        self.axv.set_xlabel("F2 (Hz)")
        self.axv.set_ylabel("F1 (Hz)")
        self.axv.set_title("vowel space", fontsize=11)
        for v, (f1, f2, _) in dsp.VOWELS.items():
            self.axv.text(f2, f1, f"/{v}/", color="0.6", fontsize=15, ha="center", va="center")
        (self.trail,) = self.axv.plot([], [], ".", color="0.55", ms=4)
        (self.dot,) = self.axv.plot([], [], "o", color="#d1345b", ms=13)
        self.fig.canvas.mpl_connect("key_press_event", self.on_key)
        self.ticks = 0

    # -- 資料 --
    def ingest(self, x: np.ndarray) -> None:
        if len(x) == 0:
            return
        x = x[-len(self.buf):]
        self.buf = np.concatenate([self.buf[len(x):], x])
        self.total += len(x)
        pc = CFG["pitch"]
        flen, hop = int(FS * pc["frame_ms"] / 1000), int(FS * pc["hop_ms"] / 1000)
        while self.next_hop + flen <= self.total:
            a = len(self.buf) - (self.total - self.next_hop)
            if a >= 0:
                fr = self.buf[a:a + flen]
                f0, _ = dsp.f0_autocorr(fr, FS, pc["floor_hz"], pc["ceiling_hz"], pc["voicing"])
                f1, f2 = dsp.formants(fr[:int(FS * 0.03)], FS) if not np.isnan(f0) else (np.nan, np.nan)
                self.track.append((self.next_hop + flen // 2, f0, f1, f2))
            self.next_hop += hop
        while self.track and self.track[0][0] < self.total - len(self.buf):
            self.track.popleft()

    # -- 繪圖 --
    def redraw(self) -> None:
        d = CFG["display"]
        win = d["wide_ms"] if self.wide else d["narrow_ms"]
        S = dsp.spectrogram_db(self.buf, FS, win, d["hop_ms"], d["n_fft"])
        ref = max(S.max(), d["floor_db"])             # 安靜時不要把底噪拉到全亮
        self.im.set_data(S - ref)
        self.ax.set_ylim(0, self.fmax / 1000)
        kind = "WIDEBAND" if self.wide else "NARROWBAND"
        see = "glottal pulses + formants" if self.wide else "harmonics"
        state = "  ❚❚ FROZEN" if self.frozen else ""
        self.ax.set_title(f"{kind}  window = {win:g} ms  →  {see}{state}    {self.msg}", fontsize=12, loc="left")
        if self.track:
            pos = np.array([t[0] for t in self.track], dtype=float)
            t = (pos - self.total) / FS
            f0 = np.array([t_[1] for t_ in self.track])
            self.pitch.set_data(t, f0) if self.show_pitch else self.pitch.set_data([], [])
            n_trail = CFG["vowel_space"]["trail_frames"]
            ff = np.array([(t_[3], t_[2]) for t_ in list(self.track)[-n_trail:]])
            ff = ff[~np.isnan(ff).any(axis=1)]
            self.trail.set_data(ff[:, 0], ff[:, 1]) if len(ff) else self.trail.set_data([], [])
            recent = ff[-CFG["vowel_space"]["smooth_frames"]:]
            # 只在最近這幾個 frame 都有聲時才畫紅點，取中位數壓掉 LPC 偶發的跳點
            if len(recent) == CFG["vowel_space"]["smooth_frames"] and not self.frozen_silent():
                self.dot.set_data([np.median(recent[:, 0])], [np.median(recent[:, 1])])
            else:
                self.dot.set_data([], [])
        else:
            self.pitch.set_data([], [])

    def frozen_silent(self) -> bool:
        last = list(self.track)[-1]
        return (self.total - last[0]) / FS > 0.15

    def tick(self, _=None):
        self.ticks += 1
        if not self.frozen:
            self.ingest(self.src.read(int(FS * CFG["display"]["tick_ms"] / 1000)))
        self.redraw()
        return self.im, self.pitch, self.trail, self.dot

    # -- 鍵盤 --
    def save(self, folder: Path, name: str) -> None:
        folder.mkdir(parents=True, exist_ok=True)
        C.save_wav(folder / f"{name}.wav", self.buf, FS, float(np.max(np.abs(self.buf))) + 1e-9)
        self.fig.savefig(folder / f"{name}.png", dpi=130)
        self.msg = f"saved {folder.name}/{name}"

    def on_key(self, ev) -> None:
        k = ev.key
        if k == "w":
            self.wide = True
        elif k == "n":
            self.wide = False
        elif k == " ":
            self.frozen = not self.frozen
        elif k == "p":
            self.show_pitch = not self.show_pitch
        elif k == "z":
            a, b = CFG["display"]["fmax_hz"]
            self.fmax = b if self.fmax == a else a
        elif k == "c":
            self.track.clear()
        elif k == "s":
            self.save(HERE / "runs" / "live", C.now().replace(":", ""))
        elif k in SLOTS and self.mode == "rehearse":
            self.save(HERE / "runs" / "rehearsal", SLOTS[k])
        elif k == "q":
            self.plt.close(self.fig)
        if self.frozen:
            self.redraw()
            self.fig.canvas.draw_idle()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["show", "rehearse", "replay", "fake", "devices"])
    ap.add_argument("name", nargs="?", default="vowels", help="replay 用：vowels／tones／question，或一個 WAV 路徑")
    ap.add_argument("--selftest", metavar="PNG", help="不開視窗，跑幾十個 tick 後存圖（測試用）")
    args = ap.parse_args()

    if args.cmd == "devices":
        import sounddevice as sd
        print(sd.query_devices())
        return
    if args.selftest:
        import matplotlib
        matplotlib.use("Agg")

    frozen = False
    if args.cmd == "fake":
        src = Fake()
    elif args.cmd == "replay":
        p = Path(args.name)
        p = p if p.suffix == ".wav" else HERE / "runs" / "rehearsal" / f"{args.name}.wav"
        if not p.exists():
            sys.exit(f"找不到 {p}。先跑 ./present.sh rehearse 並按 1／2／3 存退路。")
        src = Still(C.load_wav(p, FS))
    else:
        try:
            src = Mic()
        except Exception as e:
            sys.exit(f"開不了麥克風：{e}\n看裝置：./present.sh devices；退路：./present.sh replay vowels")

    v = Viewer(src, args.cmd)
    if args.cmd == "replay":
        v.tick()
        v.frozen = True
        v.msg = "REPLAY (not live)"
    if args.selftest:
        for i in range(60):
            v.tick()
            if i == 40:
                v.on_key(type("E", (), {"key": "n"})())
        v.fig.savefig(args.selftest, dpi=110)
        f0 = [t[1] for t in v.track if not np.isnan(t[1])]
        print(f"ticks={v.ticks} frames={len(v.track)} voiced={len(f0)} f0 range={min(f0):.0f}–{max(f0):.0f} Hz")
        return

    from matplotlib.animation import FuncAnimation
    anim = FuncAnimation(v.fig, v.tick, interval=CFG["display"]["tick_ms"], blit=False, cache_frame_data=False)
    try:
        v.plt.show()
    finally:
        src.close()
    del anim


if __name__ == "__main__":
    main()

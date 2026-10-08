# -*- coding: utf-8 -*-
# 此檔由 bin/publish_demos.py 從 private repo 的 <course>/demos/ 複製而來，請勿在公開 repo 直接編輯。
"""speech_ai/demos 底下所有 demo 共用的小工具（終端機樣式、鎖定檔、WAV 讀寫、錄音、播放、單鍵輸入）。

只有 numpy 與 scipy 在模組頂層 import。sounddevice 需要 PortAudio，只在真的要收音時才 import，
所以沒有音訊裝置的機器（或 --fake 模式）也能跑。
"""
from __future__ import annotations

import datetime as dt
import platform
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
from scipy.io import wavfile

DEMOS = Path(__file__).resolve().parent
LOCK = DEMOS / "versions.lock"
LOCK_HEADER = "# 由 demos/setup.sh 產生。要換版本就刪掉對應那一行再重跑。"

BOLD, DIM, RESET = "\033[1m", "\033[2m", "\033[0m"
BLUE, AMBER, RED, GREEN, GREY = "\033[34m", "\033[33m", "\033[31m", "\033[32m", "\033[90m"


def banner(text: str, color: str = BLUE) -> None:
    line = "─" * 72
    print(f"\n{color}{line}\n{BOLD}{text}{RESET}\n{color}{line}{RESET}")


def note(text: str) -> None:
    print(f"{GREY}{text}{RESET}")


def now() -> str:
    return dt.datetime.now().strftime("%Y-%m-%dT%H:%M:%S")


def versions() -> dict:
    from importlib import metadata
    v = {"python": platform.python_version(), "platform": platform.platform()}
    for pkg in ("numpy", "scipy", "matplotlib", "sounddevice"):
        try:
            v[pkg] = metadata.version(pkg)
        except metadata.PackageNotFoundError:
            v[pkg] = None
    return v


# ── WAV ─────────────────────────────────────────────────────────
def load_wav(path: Path, fs: int) -> np.ndarray:
    """讀 WAV → mono float64、重取樣到 fs。只吃 WAV（scipy）；其他格式先用 ffmpeg 或 afconvert 轉。"""
    import dsp
    fs_in, x = wavfile.read(str(path))
    return dsp.resample(dsp.to_mono_float(x), fs_in, fs)


def save_wav(path: Path, x: np.ndarray, fs: int, peak: float | None = None) -> None:
    """存 16-bit PCM。peak 給定時，所有版本用同一個增益（不要各自正規化，否則響度差會蓋過音質差）。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    g = 0.89 / peak if peak else 1.0
    wavfile.write(str(path), fs, np.int16(np.clip(x * g, -1, 1) * 32767))


def record(seconds: float, fs: int) -> np.ndarray:
    import sounddevice as sd
    note(f"錄音 {seconds:g} 秒……（macOS 第一次會跳麥克風權限，要允許「終端機」）")
    x = sd.rec(int(seconds * fs), samplerate=fs, channels=1, dtype="float32")
    sd.wait()
    return x[:, 0].astype(np.float64)


def play(path: Path) -> None:
    """macOS 用內建的 afplay（不需要 PortAudio）；其他平台退到 sounddevice。Ctrl-C 可中斷。"""
    try:
        if shutil.which("afplay"):
            subprocess.run(["afplay", str(path)], check=False)
        else:
            import sounddevice as sd
            fs, x = wavfile.read(str(path))
            sd.play(x, fs)
            sd.wait()
    except KeyboardInterrupt:
        pass
    except Exception as e:                               # 沒有音訊裝置時不要讓整個 demo 掛掉
        print(f"{RED}播放失敗：{e}{RESET}")


# ── 單鍵輸入 ────────────────────────────────────────────────────
def getkey() -> str:
    """讀一個鍵，不必按 Enter。不是 tty（例如被 pipe）時退回整行輸入的第一個字元。"""
    if not sys.stdin.isatty():
        line = sys.stdin.readline()
        return line[:1] if line else "q"
    import termios
    import tty
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setcbreak(fd)
        return sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)

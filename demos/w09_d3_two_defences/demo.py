# -*- coding: utf-8 -*-
# 此檔由 bin/publish_demos.py 從 private repo 的 <course>/demos/ 複製而來，請勿在公開 repo 直接編輯。
"""W9 demo 3：兩道防線各自在哪裡破——偵測器（AASIST，2019 年的資料訓的）與 watermark（AudioSeal）。

  show       課堂：載偵測器、watermark、codec 與 Vocos，啟動時全部算好，開一個視窗；鍵在視窗裡按。離開時存 runs/live/<時間>/
  rehearse   課前：不互動，全部算一遍，.wav、圖、數字存 runs/rehearsal/（進版控 = 現場的退路），寫 slides/assets/w09/w09-metrics.json → demo3
  replay     現場出狀況時：從 runs/rehearsal/ 讀回，同一個視窗、同樣的鍵（除了空白鍵），不碰 torch
  record     錄授課者自己的一句（→ ../_media/w09_d3/own.wav，不進版控），彩排時當「現場對麥克風」那一格的預演
  check      課前體檢：版本；偵測器與 watermark 載不載得起、cpu／mps 差多少、分數隨音量變多少；codec、Vocos、ffmpeg → runs/check.json
  fetch      課前一週：clone AASIST 的 repo（對 commit、權重對 sha256）；下載 AudioSeal 的兩個權重（對 sha256，commit 寫進 model.lock）；
             確認 W6 Demo 1 的 codec、W8 Demo 3 的 Vocos 與前幾週的 .wav 都在（要網路）
  fake       沒有模型也能演練流程（偵測器、watermark、codec 都是替身，畫面標明 FAKE）
  selftest   不開視窗、不需要模型：取窗、調音量、接合、SNR、統計量、fake 流程；有彩排紀錄時對 w09-metrics.json
  export     只重寫 w09-metrics.json → demo3（從 runs/rehearsal/rehearsal.json）

畫面上的鍵（順序照投影片 p89、p90 講稿的【操作】）。
前半——偵測器，一條分數軸、一條門檻：
  r        真人錄音（W3 那句）與參考組（dev-clean 每位說話人一句，小點）
  空白鍵   對麥克風說一句，現場給分（replay 時沒有；彩排時用 record 錄的那一句）
  1–6      Griffin-Lim、SpeechT5、F5-TTS、Kokoro、Qwen3-TTS、MMS-TTS：播一段，點出現在軸上
  c        真人錄音過 Mimi、過 EnCodec（各 8 層）   x  加映：別的層數、Vocos、BigVGAN
後半——watermark，一條逐取樣點的機率曲線（色線 = 加了 watermark 的，灰線 = 沒加的同一段走同一條路）：
  w        A／B：F5 的輸出、加了 watermark 的     d  detector 的曲線     s  接進真人錄音中間
  a n p    重取樣、加噪音、MP3（每按一次換下一格）
  e        EnCodec：n_q 由高到低，每按一次降一格     m  Mimi     v  Vocos
  b        後半的長條圖（每一格：加了的 vs. 沒加的）
  t  投影片上那兩張表要的摘要     y  再播一次     ?  選單     q  離開

兩個自己定的判準（都寫在 demo_config.toml，看結果之前定的）：
  偵測器   s(x) = bona fide 的 logit − spoof 的 logit，各窗平均；s > 0 叫 bona fide。0 是模型自己的分界，不是 ASVspoof 上的 EER 門檻。
  watermark  逐取樣點機率 > 0.5 的比例過半叫「偵測到」。沒加 watermark 的同一段走同一條路也算一次——那是這一格的假警報。
這個 demo 不訓練任何東西，也不對偵測器或 watermark 做對抗式攻擊。
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import io
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import tomllib
import warnings
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import common as C  # noqa: E402
import dsp          # noqa: E402

CFG = tomllib.loads((HERE / "demo_config.toml").read_text(encoding="utf-8"))
FS = int(CFG["audio"]["fs"])
FS_CODEC = 24000
DET = CFG["detector"]
WM = CFG["watermark"]
ATT = CFG["attacks"]
MEDIA = (HERE / DET["dir"]).resolve()
AASIST_DIR = MEDIA / "aasist"
OWN_WAV = (HERE / CFG["own"]["path"]).resolve()
REF_DIR = (HERE / CFG["reference"]["dir"]).resolve()
W6_DIR = (HERE / CFG["codecs"]["prev"]).resolve()
MODEL_LOCK = HERE / "model.lock"
METRICS_JSON = HERE.parents[1] / "slides" / "assets" / "w09" / "w09-metrics.json"     # parents[1] = speech_ai/
REHEARSAL = HERE / "runs" / "rehearsal"
THRESHOLD = float(DET["threshold"])
MESSAGE = [int(b) for b in WM["message"]]
ITEMS = CFG["items"]
KIND_COLOR = {"real": "#2a9d4b", "synth": "#5b3fa0", "resynth": "#c8742a", "ref": "#9aa0a6"}
KIND_LABEL = {"real": "real", "synth": "synthetic", "resynth": "real, resynthesised"}
WM_COLOR, CTRL_COLOR = "#2a7fc8", "#9aa0a6"
GROUP_LABEL = {"none": "nothing done", "splice": "spliced into a real recording", "resample": "resampling", "noise": "added noise",
               "mp3": "MP3", "encodec": "EnCodec", "mimi": "Mimi", "vocos": "Vocos (mel → waveform)"}
GROUP_KEY = {"a": "resample", "n": "noise", "p": "mp3", "e": "encodec", "m": "mimi", "v": "vocos"}


# ── 小工具 ───────────────────────────────────────────────────
def rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(x)) + 1e-20))


def level(x: np.ndarray, target: float) -> np.ndarray:
    """調到固定的 RMS（偵測器的分數會隨音量變）。"""
    return x * (target / rms(x))


def window_starts(n: int, L: int, hop: int) -> list[int]:
    """n 個取樣點取長 L 的窗：每 hop 一窗，最後一窗貼齊句尾。n < L 時只有一窗（那一窗是重複接長的）。"""
    if n <= L:
        return [0]
    s = list(range(0, n - L + 1, hop))
    if s[-1] != n - L:
        s.append(n - L)
    return s


def windows(x: np.ndarray, L: int, hop: int) -> np.ndarray:
    if len(x) < L:
        return np.tile(x, L // len(x) + 1)[:L][None]            # AASIST 的 data_utils.pad：短的重複接長
    return np.stack([x[s:s + L] for s in window_starts(len(x), L, hop)])


def add_noise(x: np.ndarray, snr_db: float, rng: np.random.Generator) -> np.ndarray:
    n = rng.standard_normal(len(x))
    return x + n * (rms(x) / rms(n)) * 10 ** (-snr_db / 20)


def round_trip(x: np.ndarray, hz: int) -> np.ndarray:
    y = dsp.resample(dsp.resample(x, FS, hz), hz, FS)
    return fit(y, len(x))


def fit(y: np.ndarray, n: int) -> np.ndarray:
    return y[:n] if len(y) >= n else np.pad(y, (0, n - len(y)))


def mp3_round_trip(x: np.ndarray, kbps: int) -> np.ndarray:
    """16 kHz wav → MP3（ffmpeg 的 libmp3lame）→ 16 kHz wav。ffmpeg 會補上編碼器的延遲，長度照原來的裁。"""
    with tempfile.TemporaryDirectory() as td:
        a, m, b = Path(td) / "a.wav", Path(td) / "a.mp3", Path(td) / "b.wav"
        C.save_wav(a, x, FS)
        subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", str(a), "-b:a", f"{kbps}k", str(m)], check=True)
        subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", str(m), "-ar", str(FS), "-ac", "1", str(b)], check=True)
        return fit(C.load_wav(b, FS), len(x))


def splice(host: np.ndarray, seg: np.ndarray, at: int) -> np.ndarray:
    """把 seg 插進 host 的第 at 個取樣點（不是蓋掉）。"""
    return np.concatenate([host[:at], seg, host[at:]])


def curve100(p: np.ndarray) -> list[float]:
    """逐取樣點的機率 → 每 10 ms 一個平均（畫圖與存檔用）。"""
    k = FS // 100
    n = len(p) // k
    return [round(float(v), 3) for v in p[:n * k].reshape(n, k).mean(axis=1)]


def det_stats(p: np.ndarray, bits: np.ndarray) -> dict:
    """p：逐取樣點「有 watermark」的機率；bits：解出來的 16 bit 各自是 1 的機率。"""
    frac = float(np.mean(p > WM["sample_threshold"]))
    got = (np.asarray(bits) > 0.5).astype(int)
    return {"mean": float(np.mean(p)), "frac": frac, "detected": bool(frac > WM["detect_fraction"]),
            "bit_acc": float(np.mean(got == np.asarray(MESSAGE))), "curve": curve100(p)}


def crossings(p: np.ndarray, thr: float = 0.5, smooth_ms: float = 20.0) -> tuple[int | None, int | None]:
    """平滑後高於 thr 的最長一段的頭尾取樣點（定位用；零星的短突起不算）。"""
    k = max(1, int(FS * smooth_ms / 1000))
    q = np.convolve(p, np.ones(k) / k, mode="same") > thr
    if not q.any():
        return None, None
    edges = np.flatnonzero(np.diff(np.concatenate([[0], q.astype(int), [0]])))
    starts, ends = edges[0::2], edges[1::2]
    i = int(np.argmax(ends - starts))
    return int(starts[i]), int(ends[i] - 1)


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def read_lock(path: Path = MODEL_LOCK) -> dict[str, str]:
    out = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip() and not line.startswith("#"):
                repo, sha = line.split()[:2]
                out[repo] = sha
    return out


def resolve_device(d: str) -> str:
    import torch
    if d == "auto":
        return "mps" if torch.backends.mps.is_available() else "cpu"
    return d


def _load_w6():
    p = W6_DIR / "demo.py"
    if not p.exists():
        sys.exit(f"找不到 W6 Demo 1 的程式 {p}（demo_config.toml [codecs].prev）")
    spec = importlib.util.spec_from_file_location("w06_d1_demo", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def fmt(v, nd: int = 2) -> str:
    return "—" if v is None else f"{v:+.{nd}f}"


class Out:
    def __init__(self, echo: bool = True):
        self.buf, self.echo = io.StringIO(), echo

    def __call__(self, s: str = "") -> None:
        if self.echo:
            print(s, flush=True)
        self.buf.write(re.sub(r"\033\[[0-9;]*m", "", s) + "\n")

    def banner(self, s: str, color: str = C.BLUE) -> None:
        if self.echo:
            C.banner(s, color)
        self.buf.write(f"\n{'─' * 72}\n{s}\n{'─' * 72}\n")

    def text(self) -> str:
        return self.buf.getvalue()


# ── 偵測器 ───────────────────────────────────────────────────
class AASIST:
    """clovaai/aasist 的 Model 與它附的 AASIST.pth。score(x16) → 各窗的 log-odds 與平均。"""

    def __init__(self, device: str = "cpu"):
        import torch
        self.torch = torch
        src = AASIST_DIR / "models" / "AASIST.py"
        w = AASIST_DIR / DET["weights"]
        if not src.exists() or not w.exists():
            sys.exit(f"沒有 {os.path.relpath(AASIST_DIR, HERE)}：先跑 ./present.sh fetch")
        spec = importlib.util.spec_from_file_location("aasist_model", src)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        cfg = json.loads((AASIST_DIR / DET["config"]).read_text(encoding="utf-8"))["model_config"]
        self.model = mod.Model(cfg)
        self.model.load_state_dict(torch.load(str(w), map_location="cpu", weights_only=True))
        self.device = resolve_device(device)
        self.model.eval().to(self.device)
        self.n_params = sum(p.numel() for p in self.model.parameters())
        self.nb_samp = int(cfg["nb_samp"])

    def score(self, x16: np.ndarray, do_level: bool = True) -> dict:
        torch = self.torch
        x = level(x16, DET["input_rms"]) if do_level else x16
        w = windows(x, int(DET["window"]), int(DET["hop"]))
        with torch.no_grad():
            _, out = self.model(torch.tensor(w, dtype=torch.float32, device=self.device))
        out = out.detach().float().cpu().numpy()
        lo = out[:, 1] - out[:, 0]
        return {"s": float(lo.mean()), "windows": [float(v) for v in lo], "bonafide_logit_first": float(out[0, 1])}

    def info(self) -> dict:
        return {"repo": DET["repo"], "commit": DET["commit"], "weights": DET["weights"], "weights_sha256": DET["weights_sha256"],
                "trained_on": "ASVspoof 2019 LA (train partition)", "device": self.device, "n_params": self.n_params,
                "window": int(DET["window"]), "hop": int(DET["hop"]), "input_rms": DET["input_rms"], "threshold": THRESHOLD}


class FakeDetector:
    """替身：分數只看 4 kHz 以上的能量比與一個由內容決定的擾動——沒有任何意義，只為了演練流程。"""
    fake = True

    def score(self, x16: np.ndarray, do_level: bool = True) -> dict:
        w = windows(x16, int(DET["window"]), int(DET["hop"]))
        out = []
        for seg in w:
            S = np.abs(np.fft.rfft(seg)) ** 2
            hf = 10 * math.log10(S[len(S) // 2:].sum() / S.sum() + 1e-12)
            out.append(float(np.clip(6.0 + 0.35 * hf + 2.0 * math.sin(float(np.abs(seg).sum())), -9, 9)))
        return {"s": float(np.mean(out)), "windows": out, "bonafide_logit_first": out[0] / 2}

    def info(self) -> dict:
        return {"repo": "FAKE", "device": "cpu", "threshold": THRESHOLD}


# ── watermark ────────────────────────────────────────────────
class Seal:
    """audioseal 的 generator 與 detector（16 kHz、16 bit）。權重從 fetch 抓下來的檔案路徑載入。"""

    def __init__(self, device: str = "cpu"):
        import torch
        from huggingface_hub import hf_hub_download
        self.torch = torch
        self.commit = read_lock().get(WM["repo"])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            from audioseal import AudioSeal
            g = hf_hub_download(WM["repo"], WM["generator"], revision=self.commit, local_files_only=True)
            d = hf_hub_download(WM["repo"], WM["detector"], revision=self.commit, local_files_only=True)
            for p, want in ((g, WM["generator_sha256"]), (d, WM["detector_sha256"])):
                if sha256(Path(p)) != want:          # audioseal 用 pickle 載入（weights_only=False），所以每次載入前都對一次
                    sys.exit(f"{C.RED}{Path(p).name} 的 sha256 與設定檔不符——不要載入，先查原因。{C.RESET}")
            self.gen = AudioSeal.load_generator(g, nbits=len(MESSAGE))
            self.det = AudioSeal.load_detector(d, nbits=len(MESSAGE))
        self.device = resolve_device(device)
        self.gen.eval().to(self.device)
        self.det.eval().to(self.device)
        self.msg = torch.tensor([MESSAGE], device=self.device)

    def _t(self, x: np.ndarray):
        return self.torch.tensor(x, dtype=self.torch.float32, device=self.device)[None, None]

    def mark(self, x16: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        with self.torch.no_grad(), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            wm = self.gen.get_watermark(self._t(x16), FS, message=self.msg)[0, 0].detach().cpu().numpy().astype(np.float64)
        return x16 + wm, wm

    def detect(self, x16: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        with self.torch.no_grad(), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            r, m = self.det(self._t(x16), FS)
        return r[0, 1].detach().cpu().numpy().astype(np.float64), m[0].detach().cpu().numpy().astype(np.float64)

    def info(self) -> dict:
        from importlib.metadata import version
        return {"package": f"audioseal {version('audioseal')}", "repo": WM["repo"], "commit": self.commit, "device": self.device,
                "generator": WM["generator"], "detector": WM["detector"], "nbits": len(MESSAGE), "message": MESSAGE,
                "sample_threshold": WM["sample_threshold"], "detect_fraction": WM["detect_fraction"],
                "n_params": {"generator": sum(p.numel() for p in self.gen.parameters()), "detector": sum(p.numel() for p in self.det.parameters())}}


class FakeSeal:
    """替身：watermark 是一個固定音量的 7 kHz 正弦波（聽得見），detector 看每 10 ms 那個頻率是不是一根窄峰。只為了演練流程。"""
    fake = True
    F = 7000.0

    def mark(self, x16: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        t = np.arange(len(x16)) / FS
        wm = 0.03 * np.sin(2 * np.pi * self.F * t)
        return x16 + wm, wm

    def detect(self, x16: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        k = 160
        n = len(x16) // k
        fr = x16[:n * k].reshape(n, k) * np.hanning(k)
        S = np.abs(np.fft.rfft(fr, axis=1)) ** 2
        b = int(round(self.F / (FS / k)))
        ratio = S[:, b - 1:b + 2].sum(axis=1) / (S[:, b - 8:b + 9].sum(axis=1) + 1e-12)     # 窄峰對它旁邊：噪音約 3/17，純音接近 1
        ratio = np.array([np.median(ratio[max(0, i - 15):i + 16]) for i in range(n)])                # 單一 frame 的雜訊峰不算
        p = np.clip((ratio - 0.30) / 0.25, 0.0, 1.0)
        p = fit(np.repeat(p, k), len(x16))
        hit = float(np.mean(p > 0.5)) > 0.3
        bits = np.asarray(MESSAGE, dtype=float) if hit else np.asarray([(i * 7 % 3 == 0) for i in range(len(MESSAGE))], dtype=float)
        return p, bits

    def info(self) -> dict:
        return {"package": "FAKE", "device": "cpu", "nbits": len(MESSAGE), "message": MESSAGE,
                "sample_threshold": WM["sample_threshold"], "detect_fraction": WM["detect_fraction"]}


# ── 重新生成：codec 與 Vocos ─────────────────────────────────
class Regen:
    """16 kHz → 24 kHz → codec（encode 一次，只留前 n_q 層 decode）→ 16 kHz。codec 是 W6 Demo 1 的 Codec。"""

    def __init__(self, codecs: dict, vocos=None):
        self.codecs, self.vocos = codecs, vocos

    def codec_many(self, key: str, x16: np.ndarray, nqs: list[int]) -> dict[int, np.ndarray]:
        c = self.codecs[key]
        x24 = dsp.resample(x16, FS, FS_CODEC)
        codes = c.encode_all(x24)                                 # (N_q, T)；decode 用這一次 encode 留下的 scale／mask
        return {int(q): fit(dsp.resample(c.decode(codes[:int(q)], len(x24)), FS_CODEC, FS), len(x16)) for q in nqs}

    def voc(self, x16: np.ndarray) -> np.ndarray:
        return fit(dsp.resample(self.vocos.resynth(dsp.resample(x16, FS, FS_CODEC)), FS_CODEC, FS), len(x16))

    def info(self) -> dict:
        d = {k: c.info() for k, c in self.codecs.items()}
        if self.vocos is not None:
            d["vocos"] = self.vocos.info()
        return d


class VocosResynth:
    """charactr/vocos-mel-24khz：波形 → 它自己的 mel 前端 → 波形（W8 Demo 3 的那一份權重）。"""

    def __init__(self, device: str = "auto"):
        import torch
        from huggingface_hub import snapshot_download
        from vocos import Vocos
        self.torch = torch
        self.repo = CFG["vocos"]["repo"]
        self.commit = read_lock((HERE / CFG["vocos"]["lock"]).resolve()).get(self.repo)
        d = snapshot_download(self.repo, revision=self.commit, local_files_only=True, allow_patterns=["config.yaml", "pytorch_model.bin", "README.md"])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.model = Vocos.from_hparams(str(Path(d) / "config.yaml"))
            self.model.load_state_dict(torch.load(str(Path(d) / "pytorch_model.bin"), map_location="cpu", weights_only=True))
        self.device = resolve_device(device)
        self.model.eval().to(self.device)

    def resynth(self, x24: np.ndarray) -> np.ndarray:
        with self.torch.no_grad():
            mel = self.model.feature_extractor(self.torch.tensor(x24, dtype=self.torch.float32, device=self.device)[None])
            return self.model.decode(mel)[0].detach().cpu().numpy().astype(np.float64)

    def info(self) -> dict:
        return {"repo": self.repo, "commit": self.commit, "device": self.device}


class FakeVocos:
    fake = True

    def resynth(self, x24: np.ndarray) -> np.ndarray:
        X = np.fft.rfft(x24)
        X[int(len(X) * 0.5):] = 0                         # 替身：把 6 kHz 以上砍掉（FakeSeal 的 7 kHz 會不見）
        return np.fft.irfft(X, len(x24))

    def info(self) -> dict:
        return {"repo": "FAKE"}


def versions() -> dict:
    from importlib import metadata
    v = C.versions()
    for pkg in ("torch", "torchaudio", "transformers", "audioseal", "vocos", "huggingface_hub"):
        try:
            v[pkg] = metadata.version(pkg)
        except metadata.PackageNotFoundError:
            v[pkg] = None
    v["ffmpeg"] = shutil.which("ffmpeg") is not None
    return v


# ── 主體 ─────────────────────────────────────────────────────
class Demo:
    def __init__(self, out: Out, detector, seal, regen: Regen | None, folder: Path, tag: str = ""):
        self.out, self.detector, self.seal, self.regen, self.folder, self.tag = out, detector, seal, regen, folder, tag
        self.items: dict[str, dict] = {}        # 前半：name → {label, kind, key, s, windows, called, wav(Path)}
        self.order: list[str] = []
        self.ref: list[dict] = []               # 參考組：每句 {id, s}
        self.conds: dict[str, dict] = {}        # 後半：name → {label, group, marked{…}, control{…}, wav(Path)}
        self.cond_order: list[str] = []
        self.splice_info: dict = {}
        self.wm_info: dict = {}
        self.shown: list[str] = []              # 分數軸上已經出現的
        self.view = ("axis", None)
        self.step = {g: -1 for g in GROUP_KEY.values()}
        self.last: Path | None = None
        self.fig = None
        self.quiet = False
        self.live_ok = True
        self.skipped: list[str] = []
        self.t_compute = None

    # ── 前半：給分 ──
    def _score(self, name: str, label: str, kind: str, key: str, x16: np.ndarray, wav: Path | None, **extra) -> dict:
        r = self.detector.score(x16)
        it = {"label": label, "kind": kind, "key": key, "s": r["s"], "windows": r["windows"], "bonafide_logit_first": r["bonafide_logit_first"],
              "called": "bona fide" if r["s"] > THRESHOLD else "spoof", "dur_s": len(x16) / FS, "wav": wav, **extra}
        self.items[name] = it
        if name not in self.order:
            self.order.append(name)
        return it

    def _own(self, name: str, x16: np.ndarray) -> Path:
        p = self.folder / f"{name}.wav"
        C.save_wav(p, x16, FS)
        return p

    def compute(self) -> None:
        out = self.out
        t0 = time.perf_counter()
        real_p = (HERE / CFG["audio"]["real"]).resolve()
        if not real_p.exists():
            sys.exit(f"沒有 {real_p}（W3 Demo 3 的那一句）")
        self.real = C.load_wav(real_p, FS)
        self._score("real", CFG["audio"]["real_label"], "real", "r", self.real, real_p)
        # 參考組
        if REF_DIR.exists():
            for p in sorted(REF_DIR.glob("*.wav")):
                self.ref.append({"id": p.stem, "s": self.detector.score(C.load_wav(p, FS))["s"]})
        else:
            out(f"  {C.AMBER}沒有 {os.path.relpath(REF_DIR, HERE)}（W2 Demo 3 的 fetch 解出來的 40 句）：參考組略過{C.RESET}")
        # 授課者錄好的那一句（彩排用；現場是空白鍵）
        if OWN_WAV.exists() and not getattr(self.detector, "fake", False):
            self._score("own", "the lecturer (recorded for the rehearsal)", "real", " ", C.load_wav(OWN_WAV, FS), OWN_WAV, private=True)
        # 前幾週的輸出
        srcs = {}
        for it in ITEMS:
            p = (HERE / it["path"]).resolve()
            if not p.exists():
                self.skipped.append(f"{it['name']}（{it['path']} 不在）")
                continue
            x = C.load_wav(p, FS)
            srcs[it["name"]] = x
            self._score(it["name"], it["label"], it["kind"], it["key"], x, p, source=it["path"])
        # 真人錄音過 codec
        if self.regen is not None:
            want = [(k_, c_, int(q)) for k_, lst in (("c", CFG["codecs"]["real_main"]), ("x", CFG["codecs"]["real_extra"])) for c_, q in lst]
            for codec in dict.fromkeys(c_ for _, c_, _ in want):
                if codec not in self.regen.codecs:
                    continue
                ys = self.regen.codec_many(codec, self.real, [q for _, c_, q in want if c_ == codec])
                for key_, c_, nq in want:
                    if c_ != codec:
                        continue
                    name = f"real_{codec}{nq:02d}"
                    lab = f"real → {'Mimi' if codec == 'mimi' else 'EnCodec'}, n_q = {nq}"
                    self._score(name, lab, "resynth", key_, ys[nq], self._own(f"det_{name}", ys[nq]), codec=codec, n_q=nq)
            # 鍵 c 的排在鍵 x 的前面
            self.order.sort(key=lambda n: 1 if self.items[n]["key"] == "x" else 0)
        # 後半
        car = WM["carrier"]
        if car not in srcs:
            sys.exit(f"watermark 要用的那一段（items 裡的 {car}）不在")
        self._watermark(srcs[car], self.items[car]["label"])
        self.t_compute = time.perf_counter() - t0
        for s in self.skipped:
            out(f"  {C.AMBER}略過：{s}{C.RESET}")
        out(f"  全部算好：{self.t_compute:.1f} s（{len(self.items)} 段給分、參考組 {len(self.ref)} 句、watermark {len(self.conds)} 格）")

    def _cond(self, name: str, label: str, group: str, marked: np.ndarray, control: np.ndarray, **extra) -> None:
        pm, bm = self.seal.detect(marked)
        pc, bc = self.seal.detect(control)
        wav = self.folder / f"wm_{name}.wav"
        C.save_wav(wav, marked, FS)
        self.conds[name] = {"label": label, "group": group, "marked": det_stats(pm, bm), "control": det_stats(pc, bc), "wav": wav, **extra}
        self.cond_order.append(name)

    def _watermark(self, x: np.ndarray, label: str) -> None:
        y, wm = self.seal.mark(x)
        self.carrier, self.marked = x, y
        self.carrier_wav = self.folder / "wm_carrier.wav"
        C.save_wav(self.carrier_wav, x, FS)
        self.wm_info = {"carrier": label, "dur_s": len(x) / FS, "snr_db": 20 * math.log10(rms(x) / rms(wm)),
                        "peak_wm": float(np.abs(wm).max()), "peak_carrier": float(np.abs(x).max())}
        self._cond("none", "watermarked, nothing done", "none", y, x)
        # 接進真人錄音中間
        a, n = int(WM["splice_from"] * FS), int(WM["splice_len"] * FS)
        at = int(WM["splice_at"] * FS)
        seg_m, seg_c = y[a:a + n], x[a:a + n]
        sp_m, sp_c = splice(self.real, seg_m, at), splice(self.real, seg_c, at)
        self._cond("splice", f"spliced into a real recording ({at / FS:g}–{(at + len(seg_m)) / FS:g} s)", "splice", sp_m, sp_c)
        pm, _ = self.seal.detect(sp_m)
        lo, hi = crossings(pm, WM["sample_threshold"])
        inside = pm[at:at + len(seg_m)]
        outside = np.concatenate([pm[:at], pm[at + len(seg_m):]])
        self.splice_info = {"true_start_s": at / FS, "true_end_s": (at + len(seg_m)) / FS,
                            "found_start_s": None if lo is None else lo / FS, "found_end_s": None if hi is None else (hi + 1) / FS,
                            "mean_inside": float(inside.mean()), "mean_outside": float(outside.mean()),
                            "frac_inside": float(np.mean(inside > WM["sample_threshold"])), "frac_outside": float(np.mean(outside > WM["sample_threshold"]))}
        for hz in ATT["resample_hz"]:
            self._cond(f"resample{hz}", f"16 → {hz / 1000:g} → 16 kHz", "resample", round_trip(y, hz), round_trip(x, hz), hz=hz)
        rng = np.random.default_rng(int(ATT["noise_seed"]))
        for snr in ATT["noise_snr_db"]:
            n_ = rng.standard_normal(len(x))
            n_ = n_ * (rms(x) / rms(n_)) * 10 ** (-snr / 20)          # 同一段噪音加在兩邊
            self._cond(f"noise{snr}", f"white noise, SNR {snr} dB", "noise", y + n_, x + n_, snr_db=snr)
        if shutil.which("ffmpeg"):
            for kb in ATT["mp3_kbps"]:
                try:
                    self._cond(f"mp3_{kb}", f"MP3, {kb} kbps", "mp3", mp3_round_trip(y, kb), mp3_round_trip(x, kb), kbps=kb)
                except Exception as e:
                    self.skipped.append(f"MP3 {kb} kbps（ffmpeg：{str(e)[:80]}）")
        else:
            self.skipped.append("MP3（這台機器上沒有 ffmpeg）")
        if self.regen is not None:
            for codec, nqs in (("encodec", ATT["encodec_nq"]), ("mimi", ATT["mimi_nq"])):
                if codec not in self.regen.codecs:
                    continue
                ym, yc = self.regen.codec_many(codec, y, nqs), self.regen.codec_many(codec, x, nqs)
                for nq in nqs:
                    lab = f"EnCodec, n_q = {nq} ({nq * 0.75:g} kbps)" if codec == "encodec" else f"Mimi, n_q = {nq}"
                    self._cond(f"{codec}{int(nq):02d}", lab, codec, ym[int(nq)], yc[int(nq)], n_q=int(nq))
            if ATT["vocos"] and self.regen.vocos is not None:
                self._cond("vocos", "Vocos: mel, then back to a waveform", "vocos", self.regen.voc(y), self.regen.voc(x))

    # ── 畫面 ──
    def make_fig(self):
        import matplotlib.pyplot as plt
        plt.rcParams["keymap.save"] = []
        plt.rcParams["keymap.quit"] = []
        plt.rcParams["keymap.fullscreen"] = []
        plt.rcParams["keymap.pan"] = []
        plt.rcParams["keymap.yscale"] = []
        plt.rcParams["keymap.xscale"] = []
        plt.rcParams["keymap.grid"] = []
        plt.rcParams["keymap.back"] = []
        plt.rcParams["keymap.forward"] = []
        plt.rcParams["keymap.home"] = []
        self.plt = plt
        self.fig = plt.figure(figsize=(CFG["display"]["fig_w"], CFG["display"]["fig_h"]))
        self.draw()
        return self.fig

    def redraw(self):
        if self.fig is not None:
            self.draw()
            self.fig.canvas.draw_idle()

    def _suptitle(self, s: str):
        self.fig.suptitle(("[FAKE — not a real model]  " if self.tag == "FAKE" else "") + s, fontsize=15, x=0.02, ha="left", fontweight="bold")

    def draw(self):
        self.fig.clear()
        mode, arg = self.view
        if mode == "axis":
            self._draw_axis()
        elif mode == "curve":
            self._draw_curves(arg)
        else:
            self._draw_bars()

    def _draw_axis(self):
        ax = self.fig.add_axes([0.33, 0.10, 0.64, 0.78])
        self._suptitle("Demo 3 · the detector: AASIST, trained on ASVspoof 2019 LA")
        allv = [w for it in self.items.values() for w in it["windows"]] + [r["s"] for r in self.ref] + [THRESHOLD]
        lo, hi = min(allv) - 1.0, max(allv) + 1.0
        ax.axvspan(lo, THRESHOLD, color="#d62728", alpha=0.06)
        ax.axvspan(THRESHOLD, hi, color="#2a9d4b", alpha=0.06)
        ax.axvline(THRESHOLD, color="black", lw=1.6)
        ax.text(THRESHOLD, 1.005, "threshold (fixed in advance)", transform=ax.get_xaxis_transform(), ha="center", va="bottom", fontsize=11)
        ax.text(lo + 0.15, -0.085, "← called spoof", transform=ax.get_xaxis_transform(), ha="left", va="top", fontsize=12, color="#d62728")
        ax.text(hi - 0.15, -0.085, "called bona fide →", transform=ax.get_xaxis_transform(), ha="right", va="top", fontsize=12, color="#2a9d4b")
        rows = list(self.shown)
        n_rows = max(len(rows) + (1 if self.ref and "real" in self.shown else 0), 8)
        y = 0
        ticks, labels, colors = [], [], []
        for name in rows:
            it = self.items[name]
            col = KIND_COLOR[it["kind"]]
            if len(it["windows"]) > 1:
                ax.plot(it["windows"], [y] * len(it["windows"]), "o", ms=5, color=col, alpha=0.35, zorder=2)
                ax.plot([min(it["windows"]), max(it["windows"])], [y, y], "-", lw=1, color=col, alpha=0.35, zorder=1)
            ax.plot([it["s"]], [y], "o", ms=13, color=col, zorder=3, markeredgecolor="white")
            ax.text(it["s"], y - 0.33, f"{it['s']:+.1f}", ha="center", va="bottom", fontsize=11, color=col)
            ticks.append(y)
            labels.append(it["label"])
            colors.append(col)
            y += 1
            if name == "real" and self.ref:
                v = [r["s"] for r in self.ref]
                ax.plot(v, [y] * len(v), "|", ms=14, color=KIND_COLOR["ref"], zorder=2)
                k = sum(1 for s in v if s > THRESHOLD)
                ticks.append(y)
                labels.append(f"real · dev-clean, {len(v)} speakers: {k} bona fide")
                colors.append("#555555")
                y += 1
        ax.set_ylim(n_rows - 0.5, -0.7)
        ax.set_xlim(lo, hi)
        ax.set_yticks(ticks)
        ax.set_yticklabels(labels, fontsize=12)
        for t, c in zip(ax.get_yticklabels(), colors):
            t.set_color(c)
        ax.set_xlabel(r"score $s(\mathbf{x})$ = bona fide logit − spoof logit (mean over ≈ 4 s windows; small dots = each window)", fontsize=11)
        ax.tick_params(axis="x", labelsize=11)
        for s in ("top", "right", "left"):
            ax.spines[s].set_visible(False)
        ax.tick_params(axis="y", length=0)
        if not rows:
            ax.text(0.5, 0.5, "r  real      space  live      1–6  generators      c  real speech through a codec", transform=ax.transAxes,
                    ha="center", va="center", fontsize=13, color="#888888")

    def _draw_curves(self, names: list[str]):
        names = [n for n in names if n in self.conds]
        n = len(names)
        self._suptitle("Demo 3 · the watermark: AudioSeal's detector, per sample")
        for i, name in enumerate(names):
            c = self.conds[name]
            h = 0.76 / n
            ax = self.fig.add_axes([0.07, 0.10 + (n - 1 - i) * h, 0.90, h * (0.72 if n > 1 else 0.86)])
            for who, col, lw in (("control", CTRL_COLOR, 1.4), ("marked", WM_COLOR, 2.0)):
                v = c[who]["curve"]
                ax.plot(np.arange(len(v)) / 100, v, color=col, lw=lw)
            ax.axhline(WM["sample_threshold"], color="black", lw=0.8, ls=":")
            if name == "splice" and self.splice_info:
                ax.axvspan(self.splice_info["true_start_s"], self.splice_info["true_end_s"], color=WM_COLOR, alpha=0.10)
            ax.set_ylim(-0.04, 1.04)
            ax.set_xlim(0, max(len(c["marked"]["curve"]), 1) / 100)
            ax.set_yticks([0, 0.5, 1])
            ax.tick_params(labelsize=10)
            m, k = c["marked"], c["control"]
            verdict = "detected" if m["detected"] else "NOT detected"
            warn = "   · the unmarked one is ALSO flagged" if k["detected"] else ""
            good = m["detected"] and not k["detected"]
            if name == "splice" and self.splice_info:
                si = self.splice_info
                verdict = f"inside the splice mean {si['mean_inside']:.2f}, outside {si['mean_outside']:.2f}"
                good = si["mean_inside"] > 0.5 > si["mean_outside"]
            ax.set_title(f"{c['label']}     marked: mean {m['mean']:.2f}, {m['frac'] * 100:.0f}% of samples > 0.5, bits {m['bit_acc'] * 16:.0f}/16 → {verdict}"
                         f"     ·     unmarked: mean {k['mean']:.2f}, {k['frac'] * 100:.0f}%{warn}", fontsize=11.5, loc="left",
                         color="black" if good else "#d62728")
            for s in ("top", "right"):
                ax.spines[s].set_visible(False)
            if i == n - 1:
                ax.set_xlabel("time (s)      blue = watermarked · grey = the same audio without the watermark, through the same processing", fontsize=11)
            else:
                ax.set_xticklabels([])

    def _draw_bars(self):
        self._suptitle("Demo 3 · the watermark: fraction of samples the detector flags")
        ax = self.fig.add_axes([0.34, 0.09, 0.62, 0.79])
        names = [n for n in self.cond_order]
        ys = np.arange(len(names))
        ax.barh(ys - 0.2, [self.conds[n]["marked"]["frac"] for n in names], height=0.38, color=WM_COLOR, label="watermarked")
        ax.barh(ys + 0.2, [self.conds[n]["control"]["frac"] for n in names], height=0.38, color=CTRL_COLOR, label="not watermarked, same processing")
        ax.axvline(WM["detect_fraction"], color="black", lw=1.2)
        ax.set_yticks(ys)
        ax.set_yticklabels([self.conds[n]["label"] if len(self.conds[n]["label"]) < 44 else self.conds[n]["label"][:42] + "…" for n in names], fontsize=11)
        ax.set_ylim(len(names) - 0.4, -0.6)
        ax.set_xlim(0, 1.0)
        ax.set_xlabel("fraction of samples with Det > 0.5   (line: the decision at one half)", fontsize=11)
        ax.legend(loc="lower right", fontsize=11, frameon=False)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)

    # ── 動作 ──
    def _play(self, path: Path | None, what: str) -> None:
        if path is None or not Path(path).exists():
            self.out(f"  （沒有音檔：{what}）")
            return
        self.last = Path(path)
        if self.quiet:
            return
        self.out(f"  {C.DIM}▶ {what}{C.RESET}")
        C.play(Path(path))

    def _line(self, name: str) -> str:
        it = self.items[name]
        col = C.GREEN if it["called"] == "bona fide" else C.RED
        w = "" if len(it["windows"]) < 2 else f"   windows {min(it['windows']):+.1f} … {max(it['windows']):+.1f}"
        return f"  {it['label']:<46s} s = {it['s']:+6.2f}   {col}{it['called']:<9s}{C.RESET}{w}"

    def reveal(self, names: list[str], play: bool = True) -> None:
        self.view = ("axis", None)
        for name in names:
            if name not in self.items:
                continue
            if play:
                self._play(self.items[name]["wav"], self.items[name]["label"])
            if name not in self.shown:
                self.shown.append(name)
            self.out(self._line(name))
            self.redraw()
            if self.fig is not None and not self.quiet:      # 一次按鍵揭露兩段時（鍵 c），中間那一次也要畫出來
                self.fig.canvas.draw()
                self.fig.canvas.flush_events()
        if "real" in names and self.ref:
            v = [r["s"] for r in self.ref]
            k = sum(1 for s in v if s > THRESHOLD)
            self.out(f"  參考組 dev-clean {len(v)} 位說話人各一句：{k} 句叫 bona fide、{len(v) - k} 句叫 spoof；中位數 {np.median(v):+.2f}，最低 {min(v):+.2f}，最高 {max(v):+.2f}")

    def live(self) -> None:
        if not self.live_ok:
            self.out(f"  {C.AMBER}replay 不收音；彩排時錄的那一句{'在表上' if 'own' in self.items else '沒有'}。{C.RESET}")
            if "own" in self.items:
                self.reveal(["own"], play=False)
            return
        if getattr(self.detector, "fake", False):
            rng = np.random.default_rng(7)
            x = 0.05 * rng.standard_normal(int(CFG["own"]["seconds"] * FS))
        else:
            x = C.record(float(CFG["own"]["seconds"]), FS)
        if rms(x) < 1e-4:
            self.out(f"  {C.RED}收到的幾乎是全零——麥克風權限（終端機 app）或輸入裝置（系統設定 → 聲音）。{C.RESET}")
            return
        p = self.folder / "live.wav"
        C.save_wav(p, x, FS)
        self._score("live", "the lecturer, live (this room, this microphone)", "real", " ", x, p, private=True)
        self.reveal(["live"], play=False)

    def by_key(self, key: str) -> list[str]:
        return [n for n in self.order if self.items[n]["key"] == key and n not in ("live", "own")]

    def ab(self) -> None:
        self.out.banner("A／B：F5 的輸出，與加了 watermark 的")
        i = self.wm_info
        self.out(f"  watermark 的能量比訊號低 {i['snr_db']:.1f} dB（SNR）；最大振幅 {i['peak_wm']:.4f} 對 {i['peak_carrier']:.3f}")
        self._play(self.carrier_wav, "A：F5 的輸出（沒加）")
        self._play(self.conds["none"]["wav"], "B：加了 watermark")

    def cond_line(self, name: str) -> str:
        c = self.conds[name]
        m, k = c["marked"], c["control"]
        v = f"{C.GREEN}偵測到{C.RESET}" if m["detected"] else f"{C.RED}沒偵測到{C.RESET}"
        if c["group"] == "splice":              # 只有中間一截有 watermark：整段的比例不是判準，看裡面與外面
            v = f"{C.DIM}（看定位，不看整段的比例）{C.RESET}"
        fa = f"   {C.RED}沒加的也被判成有{C.RESET}" if k["detected"] else ""
        return (f"  {c['label']:<44s} 加了的：平均 {m['mean']:.3f}、{m['frac'] * 100:5.1f}% > 0.5、bit {m['bit_acc'] * 16:2.0f}/16 → {v}"
                f"   ｜ 沒加的：平均 {k['mean']:.3f}、{k['frac'] * 100:5.1f}%{fa}")

    def show_conds(self, names: list[str], play: str | None = None) -> None:
        names = [n for n in names if n in self.conds]
        if not names:
            self.out(f"  {C.AMBER}這一格沒有算（見啟動時的「略過」）。{C.RESET}")
            return
        self.view = ("curve", names)
        self.redraw()
        for n in names:
            self.out(self.cond_line(n))
        if play:
            self._play(self.conds[play]["wav"], self.conds[play]["label"])

    def show_splice(self) -> None:
        self.out.banner("接進真人錄音中間：曲線只在中間升起來嗎")
        self.show_conds(["splice"], play="splice")
        s = self.splice_info
        if s:
            f0 = "—" if s["found_start_s"] is None else f"{s['found_start_s']:.2f}"
            f1 = "—" if s["found_end_s"] is None else f"{s['found_end_s']:.2f}"
            self.out(f"  接進去的是 {s['true_start_s']:.2f}–{s['true_end_s']:.2f} s；曲線高於 0.5 的最長一段是 {f0}–{f1} s。"
                     f"裡面平均 {s['mean_inside']:.3f}、外面平均 {s['mean_outside']:.3f}")

    def step_group(self, g: str) -> None:
        names = [n for n in self.cond_order if self.conds[n]["group"] == g]
        if not names:
            self.out(f"  {C.AMBER}{GROUP_LABEL[g]}：沒有算（見啟動時的「略過」）。{C.RESET}")
            return
        self.step[g] = (self.step[g] + 1) % (len(names) + (1 if len(names) > 1 else 0))
        i = self.step[g]
        if i == 0:
            self.out.banner(GROUP_LABEL[g])
        if i < len(names):
            self.show_conds(["none", names[i]] if g != "none" else names, play=names[i])
        else:                                   # 走完一輪：全部並排
            self.view = ("curve", names[:5])
            self.redraw()
            self.out(f"  {GROUP_LABEL[g]}：全部並排")

    def bars(self) -> None:
        self.view = ("bars", None)
        self.redraw()
        self.wm_table()

    def det_table(self) -> None:
        self.out.banner("前半 · 偵測器：每一段的分數與判定（門檻 0，事先定的）")
        for name in self.order:
            self.out(self._line(name))
        if self.ref:
            v = [r["s"] for r in self.ref]
            k = sum(1 for s in v if s > THRESHOLD)
            self.out(f"  {'real · dev-clean，' + str(len(v)) + ' 位說話人各一句':<40s} 中位數 {np.median(v):+.2f}（{min(v):+.2f} … {max(v):+.2f}）   {k}/{len(v)} 叫 bona fide")
        syn = [n for n in self.order if self.items[n]["kind"] == "synth"]
        res = [n for n in self.order if self.items[n]["kind"] == "resynth"]
        self.out(f"  合成的 {len(syn)} 段裡被叫成 spoof 的：{sum(self.items[n]['called'] == 'spoof' for n in syn)}；"
                 f"真人錄音重新合成的 {len(res)} 段裡被叫成 spoof 的：{sum(self.items[n]['called'] == 'spoof' for n in res)}")

    def wm_table(self) -> None:
        self.out.banner("後半 · watermark：每一格（加了的 ｜ 沒加的同一段走同一條路）")
        for n in self.cond_order:
            self.out(self.cond_line(n))

    def summary(self) -> None:
        self.det_table()
        self.wm_table()

    def menu(self) -> None:
        self.out(f"\n  {C.BOLD}前半{C.RESET}  r 真人  空白鍵 現場  1–6 生成器  c 真人過 codec  x 加映   "
                 f"{C.BOLD}後半{C.RESET}  w A／B  d 曲線  s 接合  a 重取樣  n 噪音  p MP3  e EnCodec  m Mimi  v Vocos  b 長條圖\n"
                 f"  t 摘要  y 再播  ? 選單  q 離開")

    def dispatch(self, k: str) -> bool:
        if k in ("q", "\x03", "\x04"):
            return False
        if k == "r":
            self.out.banner("真人錄音")
            self.reveal(["real"])
        elif k in (" ", "space"):
            self.out.banner("現場對麥克風說一句")
            self.live()
        elif k in "123456" and len(k) == 1:
            self.reveal(self.by_key(k))
        elif k == "c":
            self.out.banner("真人錄音，過一次 codec")
            self.reveal(self.by_key("c"))
        elif k == "x":
            self.out.banner("加映：別的層數與 vocoder")
            self.reveal(self.by_key("x"), play=False)
        elif k == "w":
            self.ab()
        elif k == "d":
            self.out.banner("detector 逐取樣點的機率")
            self.show_conds(["none"])
        elif k == "s":
            self.show_splice()
        elif k in GROUP_KEY:
            self.step_group(GROUP_KEY[k])
        elif k == "b":
            self.bars()
        elif k == "t":
            self.summary()
        elif k == "y":
            self._play(self.last, "再播一次") if self.last else self.out("  還沒播過")
        elif k == "?":
            self.menu()
        return True

    def on_key(self, ev) -> None:
        if ev.key is None:
            return
        if not self.dispatch(ev.key):
            self.plt.close(self.fig)

    # ── 存檔 ──
    def rehearse_figs(self) -> None:
        import matplotlib
        matplotlib.use("Agg", force=True)
        self.make_fig()
        self.shown = [n for n in self.order if n != "live"]
        self.view = ("axis", None)
        self.draw()
        self.fig.savefig(self.folder / "axis.png", dpi=110)
        for g in ["none", "splice", *GROUP_KEY.values()]:
            names = [n for n in self.cond_order if self.conds[n]["group"] == g][:5]
            if names:
                self.view = ("curve", names)
                self.draw()
                self.fig.savefig(self.folder / f"curve_{g}.png", dpi=110)
        self.view = ("bars", None)
        self.draw()
        self.fig.savefig(self.folder / "bars.png", dpi=110)

    def record_dict(self) -> dict:
        def clean(d: dict) -> dict:
            o = {k: v for k, v in d.items() if k != "wav"}
            w = d.get("wav")
            if w is not None and not d.get("private"):
                try:
                    o["wav"] = str(Path(w).resolve().relative_to(self.folder.resolve()))
                except ValueError:
                    o["wav"] = os.path.relpath(Path(w).resolve(), HERE)
            return o
        v = [r["s"] for r in self.ref]
        return {"time": C.now(), "tag": self.tag, "versions": versions() if self.tag != "REPLAY" else None,
                "models": {"detector": self.detector.info() if self.detector else None, "watermark": self.seal.info() if self.seal else None,
                           "regen": self.regen.info() if self.regen else None},
                "threshold": THRESHOLD, "order": self.order, "items": {n: clean(self.items[n]) for n in self.order},
                "reference": {"dir": CFG["reference"]["dir"], "scores": self.ref,
                              "n": len(v), "n_bonafide": sum(1 for s in v if s > THRESHOLD), "median": float(np.median(v)) if v else None,
                              "min": min(v) if v else None, "max": max(v) if v else None},
                "watermark": self.wm_info, "splice": self.splice_info, "cond_order": self.cond_order,
                "conds": {n: clean(self.conds[n]) for n in self.cond_order}, "skipped": self.skipped, "seconds": self.t_compute}

    def save(self) -> None:
        self.folder.mkdir(parents=True, exist_ok=True)
        (self.folder / "rehearsal.json").write_text(json.dumps(self.record_dict(), ensure_ascii=False, indent=1, default=float), encoding="utf-8")
        (self.folder / "rehearsal.txt").write_text(self.out.text(), encoding="utf-8")


# ── 組裝 ─────────────────────────────────────────────────────
def build(cmd: str, out: Out, device: str | None = None) -> Demo:
    folder = REHEARSAL if cmd == "rehearse" else HERE / "runs" / ("_selftest" if cmd == "selftest" else "live") / time.strftime("%Y%m%d-%H%M%S")
    if cmd == "selftest":
        folder = HERE / "runs" / "_selftest"
    folder.mkdir(parents=True, exist_ok=True)
    W6 = _load_w6()
    if cmd in ("fake", "selftest"):
        codecs = {k: W6.FakeCodec(k) for k in ("mimi", "encodec")}
        return Demo(out, FakeDetector(), FakeSeal(), Regen(codecs, FakeVocos()), folder, tag="FAKE")
    t0 = time.perf_counter()
    detector = AASIST(device or DET["device"])
    seal = Seal(device or WM["device"])
    codecs = {k: W6.Codec(k, device or CFG["codecs"]["device"]) for k in ("mimi", "encodec")}
    vocos = VocosResynth(device or CFG["vocos"]["device"]) if ATT["vocos"] else None
    out(f"  模型載好：{time.perf_counter() - t0:.1f} s（AASIST 在 {detector.device}、AudioSeal 在 {seal.device}、"
        f"codec 在 {codecs['mimi'].device.type}{'、Vocos 在 ' + vocos.device if vocos else ''}）")
    return Demo(out, detector, seal, Regen(codecs, vocos), folder)


def build_replay(out: Out) -> Demo:
    p = REHEARSAL / "rehearsal.json"
    if not p.exists():
        sys.exit("沒有 runs/rehearsal/rehearsal.json：先 ./present.sh rehearse")
    rec = json.loads(p.read_text(encoding="utf-8"))
    d = Demo(out, None, None, None, REHEARSAL, tag="REPLAY")
    d.live_ok = False

    def wav(v):
        if not v:
            return None
        q = (REHEARSAL / v)
        return q if q.exists() else (HERE / v).resolve()
    d.order = rec["order"]
    d.items = {n: {**it, "wav": wav(it.get("wav"))} for n, it in rec["items"].items()}
    d.ref = rec["reference"]["scores"]
    d.cond_order = rec["cond_order"]
    d.conds = {n: {**c, "wav": wav(c.get("wav"))} for n, c in rec["conds"].items()}
    d.wm_info, d.splice_info, d.skipped = rec["watermark"], rec["splice"], rec.get("skipped", [])
    d.carrier_wav = REHEARSAL / "wm_carrier.wav"
    out(f"  讀回 {rec['time']} 的彩排（{len(d.items)} 段給分、watermark {len(d.conds)} 格）")
    return d


def run(d: Demo, plot: bool) -> None:
    d.menu()
    if plot:
        import matplotlib.pyplot as plt
        d.make_fig()
        d.fig.canvas.mpl_connect("key_press_event", d.on_key)
        plt.show()
    else:
        while d.dispatch(C.getkey()):
            pass


# ── record ───────────────────────────────────────────────────
def record() -> None:
    C.banner("錄授課者自己的一句（彩排用；不進版控、不上站）", C.AMBER)
    sec = float(CFG["own"]["seconds"])
    print(f"  按 Enter 之後說 {sec:g} 秒（隨便一句話，用上課那支麥克風、那個房間最好）。")
    input()
    x = C.record(sec, FS)
    if rms(x) < 1e-4:
        sys.exit(f"  {C.RED}收到的幾乎是全零：麥克風權限或輸入裝置。{C.RESET}")
    OWN_WAV.parent.mkdir(parents=True, exist_ok=True)
    C.save_wav(OWN_WAV, x, FS)
    print(f"  → {os.path.relpath(OWN_WAV, HERE)}（RMS {rms(x):.3f}）")
    C.play(OWN_WAV)


# ── fetch ────────────────────────────────────────────────────
def fetch() -> None:
    C.banner("下載偵測器（AASIST）與 watermark（AudioSeal）；確認前幾週的東西都在", C.AMBER)
    MEDIA.mkdir(parents=True, exist_ok=True)
    if not (AASIST_DIR / "models" / "AASIST.py").exists():
        if shutil.which("git") is None:
            sys.exit("  需要 git（xcode-select --install）")
        subprocess.run(["git", "clone", DET["repo"], str(AASIST_DIR)], check=True)
    subprocess.run(["git", "-C", str(AASIST_DIR), "checkout", "--quiet", DET["commit"]], check=True)
    sha = subprocess.run(["git", "-C", str(AASIST_DIR), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    if sha != DET["commit"]:
        sys.exit(f"  {C.RED}{DET['repo']} 現在是 {sha}，設定檔鎖的是 {DET['commit']}。{C.RESET}")
    h = sha256(AASIST_DIR / DET["weights"])
    if h != DET["weights_sha256"]:
        sys.exit(f"  {C.RED}{DET['weights']} 的 sha256 是 {h}，設定檔鎖的是 {DET['weights_sha256']}——不要載入，先查原因。{C.RESET}")
    print(f"  {DET['repo']} @ {sha[:12]} → {os.path.relpath(AASIST_DIR, HERE)}/；{DET['weights']} sha256 對")

    from huggingface_hub import HfApi, hf_hub_download, snapshot_download
    lock = read_lock()
    sha = lock.get(WM["repo"]) or HfApi().model_info(WM["repo"]).sha
    for f, want in ((WM["generator"], WM["generator_sha256"]), (WM["detector"], WM["detector_sha256"])):
        p = hf_hub_download(WM["repo"], f, revision=sha)
        h = sha256(Path(p))
        if h != want:
            sys.exit(f"  {C.RED}{f} 的 sha256 是 {h}，設定檔鎖的是 {want}——不要載入（audioseal 用 pickle 載權重），先查原因。{C.RESET}")
        print(f"  {WM['repo']}/{f} @ {sha[:12]}  {Path(p).stat().st_size / 1e6:.0f} MB  sha256 對")
    MODEL_LOCK.write_text(f"{WM['repo']} {sha}\n", encoding="utf-8")
    print(f"  → {MODEL_LOCK.name}")

    W6 = _load_w6()
    for key in ("mimi", "encodec"):
        try:
            W6.model_dir(key, offline=True)
            print(f"  W6 Demo 1 · {W6.CFG['codecs'][key]['repo']}：在")
        except Exception:
            print(f"  {C.AMBER}W6 Demo 1 · {W6.CFG['codecs'][key]['repo']} 不在快取：先跑 ../w06_d1_layer_listen/present.sh fetch{C.RESET}")
    try:
        vl = read_lock((HERE / CFG["vocos"]["lock"]).resolve())
        snapshot_download(CFG["vocos"]["repo"], revision=vl.get(CFG["vocos"]["repo"]), local_files_only=True, allow_patterns=["config.yaml", "pytorch_model.bin", "README.md"])
        print(f"  W8 Demo 3 · {CFG['vocos']['repo']}：在")
    except Exception:
        print(f"  {C.AMBER}W8 Demo 3 · {CFG['vocos']['repo']} 不在快取：先跑 ../w08_d3_vocoder_three_things/present.sh fetch{C.RESET}")
    for it in [{"name": "real", "path": CFG["audio"]["real"]}, *ITEMS]:
        ok = (HERE / it["path"]).resolve().exists()
        print(f"  {it['name']:<13s} {it['path']}：{'在' if ok else C.AMBER + '不在（那個 demo 的 rehearse 還沒跑）' + C.RESET}")
    n = len(list(REF_DIR.glob("*.wav"))) if REF_DIR.exists() else 0
    print(f"  參考組 {CFG['reference']['dir']}：{n} 句" + ("" if n else f"{C.AMBER}（沒有：../w02_d3_frontend_mismatch/present.sh fetch 會解出來；沒有也能跑）{C.RESET}"))
    print(f"  ffmpeg：{'在' if shutil.which('ffmpeg') else C.AMBER + '不在（MP3 那一格會跳過；brew install ffmpeg）' + C.RESET}")
    print(f"  授課者自己的一句：{'在' if OWN_WAV.exists() else '還沒錄（./present.sh record；沒有也能跑）'}")
    print("下一步：./present.sh check")


# ── check ────────────────────────────────────────────────────
def check(device: str | None = None) -> None:
    import torch
    C.banner("課前體檢", C.AMBER)
    rec = {"time": C.now(), "versions": versions(), "mps_available": bool(torch.backends.mps.is_available()),
           "mps_fallback_env": os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK")}
    for k, v in rec["versions"].items():
        print(f"  {k:<16s} {v}")
    print(f"  MPS：{rec['mps_available']}   PYTORCH_ENABLE_MPS_FALLBACK={rec['mps_fallback_env']}（0 = 嚴格：不支援的運算直接報錯）")
    real = C.load_wav((HERE / CFG["audio"]["real"]).resolve(), FS)
    devs = ["cpu"] + (["mps"] if rec["mps_available"] else [])

    C.banner("偵測器 AASIST")
    rec["detector"] = {}
    base = None
    for dv in devs:
        try:
            t0 = time.perf_counter()
            m = AASIST(dv)
            t1 = time.perf_counter()
            r = m.score(real)
            t2 = time.perf_counter()
            r = m.score(real)
            t3 = time.perf_counter()
            rec["detector"][dv] = {"ok": True, "load_s": t1 - t0, "first_s": t2 - t1, "score_s": t3 - t2, "s": r["s"], "windows": r["windows"], "n_params": m.n_params}
            print(f"  {dv}：載 {t1 - t0:.2f} s、一段 {t3 - t2:.3f} s；W3 那句 s = {r['s']:+.4f}（各窗 {', '.join(f'{w:+.2f}' for w in r['windows'])}）")
            if dv == "cpu":
                base = m
        except Exception as e:
            rec["detector"][dv] = {"ok": False, "error": f"{type(e).__name__}: {str(e)[:300]}"}
            print(f"  {C.RED}{dv}：{type(e).__name__}: {str(e)[:200]}{C.RESET}")
    if "mps" in rec["detector"] and rec["detector"]["mps"].get("ok") and rec["detector"]["cpu"].get("ok"):
        diff = abs(rec["detector"]["mps"]["s"] - rec["detector"]["cpu"]["s"])
        rec["detector"]["mps_minus_cpu"] = diff
        print(f"  mps 對 cpu 的分數差 {diff:.4f}" + ("" if diff < 0.05 else f"  {C.AMBER}（差得多：demo_config.toml [detector].device 留在 cpu）{C.RESET}"))
    if base is not None:
        g = {}
        for target in (0.01, 0.02, 0.05, 0.1, 0.2):
            g[str(target)] = base.score(level(real, target), do_level=False)["s"]
        rec["detector"]["gain"] = g
        print("  分數隨音量（RMS → s）：" + "   ".join(f"{k} → {v:+.2f}" for k, v in g.items()) + f"   ← 所以每一段先調到 RMS {DET['input_rms']}")
        off = base.score(real, do_level=False)
        rec["detector"]["no_level"] = off["s"]
        print(f"  不調音量、原始的 W3 那句（RMS {rms(real):.3f}）：s = {off['s']:+.2f}；官方流程的 bona fide logit（第一窗）= {off['bonafide_logit_first']:+.2f}")

    C.banner("watermark AudioSeal")
    rec["watermark"] = {}
    ref_p = None
    for dv in devs:
        try:
            t0 = time.perf_counter()
            s = Seal(dv)
            t1 = time.perf_counter()
            y, wm = s.mark(real)
            t2 = time.perf_counter()
            p, b = s.detect(y)
            t3 = time.perf_counter()
            q, _ = s.detect(real)
            st = det_stats(p, b)
            rec["watermark"][dv] = {"ok": True, "load_s": t1 - t0, "mark_s": t2 - t1, "detect_s": t3 - t2, "mean_marked": st["mean"], "frac_marked": st["frac"],
                                    "bit_acc": st["bit_acc"], "mean_unmarked": float(q.mean()), "snr_db": 20 * math.log10(rms(real) / rms(wm))}
            print(f"  {dv}：載 {t1 - t0:.2f} s、加 {t2 - t1:.2f} s、偵測 {t3 - t2:.2f} s；真人錄音加了之後平均 {st['mean']:.3f}、bit {st['bit_acc'] * 16:.0f}/16；沒加的平均 {q.mean():.3f}；"
                  f"watermark 比訊號低 {rec['watermark'][dv]['snr_db']:.1f} dB")
            if dv == "cpu":
                ref_p = (y, p)
            elif ref_p is not None:
                p2, _ = s.detect(ref_p[0])
                d = float(np.max(np.abs(p2 - ref_p[1])))
                rec["watermark"]["mps_minus_cpu_max"] = d
                print(f"  同一段加了 watermark 的聲音，detector 在 mps 對 cpu 的逐點最大差 {d:.4f}")
        except Exception as e:
            rec["watermark"][dv] = {"ok": False, "error": f"{type(e).__name__}: {str(e)[:300]}"}
            print(f"  {C.RED}{dv}：{type(e).__name__}: {str(e)[:200]}{C.RESET}")

    C.banner("codec（W6 Demo 1）、Vocos（W8 Demo 3）、ffmpeg")
    rec["regen"] = {}
    W6 = _load_w6()
    for key in ("mimi", "encodec"):
        try:
            t0 = time.perf_counter()
            c = W6.Codec(key, device or CFG["codecs"]["device"])
            y = Regen({key: c}).codec_many(key, real, [8])[8]
            rec["regen"][key] = {"ok": True, "device": c.device.type, "n_q_cfg": c.n_q_cfg, "seconds": time.perf_counter() - t0, "rms_ratio": rms(y) / rms(real)}
            print(f"  {key}：{c.device.type}，載＋一次來回 {time.perf_counter() - t0:.1f} s；8 層重建的 RMS 是原音的 {rms(y) / rms(real):.2f} 倍")
        except Exception as e:
            rec["regen"][key] = {"ok": False, "error": f"{type(e).__name__}: {str(e)[:300]}"}
            print(f"  {C.RED}{key}：{type(e).__name__}: {str(e)[:200]}{C.RESET}")
    try:
        t0 = time.perf_counter()
        v = VocosResynth(device or CFG["vocos"]["device"])
        y = Regen({}, v).voc(real)
        rec["regen"]["vocos"] = {"ok": True, "device": v.device, "seconds": time.perf_counter() - t0, "rms_ratio": rms(y) / rms(real)}
        print(f"  Vocos：{v.device}，載＋一次來回 {time.perf_counter() - t0:.1f} s；RMS 是原音的 {rms(y) / rms(real):.2f} 倍")
    except Exception as e:
        rec["regen"]["vocos"] = {"ok": False, "error": f"{type(e).__name__}: {str(e)[:300]}"}
        print(f"  {C.RED}Vocos：{type(e).__name__}: {str(e)[:200]}{C.RESET}")
    if shutil.which("ffmpeg"):
        try:
            y = mp3_round_trip(real, 64)
            lag = int(np.argmax(np.correlate(y[:8000], real[:8000], mode="full"))) - 7999
            rec["ffmpeg"] = {"ok": True, "lag_samples": lag}
            print(f"  ffmpeg：MP3 來回可以；對原音的延遲 {lag} 個取樣點" + ("" if abs(lag) < 80 else f"  {C.AMBER}（沒有對齊：曲線會整條平移，數字不受影響）{C.RESET}"))
        except Exception as e:
            rec["ffmpeg"] = {"ok": False, "error": str(e)[:200]}
            print(f"  {C.RED}ffmpeg：{str(e)[:200]}{C.RESET}")
    else:
        rec["ffmpeg"] = {"ok": False, "error": "not installed"}
        print(f"  {C.AMBER}沒有 ffmpeg：MP3 那一格會跳過（brew install ffmpeg）{C.RESET}")
    (HERE / "runs").mkdir(exist_ok=True)
    (HERE / "runs" / "check.json").write_text(json.dumps(rec, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    print("\n  → runs/check.json\n下一步：./present.sh record（選配）→ ./present.sh rehearse")


# ── export ───────────────────────────────────────────────────
def export(rec_path: Path) -> None:
    rec = json.loads(rec_path.read_text(encoding="utf-8"))
    d = json.loads(METRICS_JSON.read_text(encoding="utf-8")) if METRICS_JSON.exists() else {
        "_note": "量測數字（進版控的出處，speech_ai/CLAUDE.md 4.8）。w09-data.json 是 make_figs_w09.py 算出來的、每次重產；量測的放這裡。"}
    conds = {n: {k: ({kk: vv for kk, vv in v.items() if kk != "curve"} if k in ("marked", "control") else v) for k, v in c.items() if k != "wav"}
             for n, c in rec["conds"].items()}
    ref = {k: v for k, v in rec["reference"].items() if k != "scores"}
    d["demo3"] = {"source": "demos/w09_d3_two_defences/runs/rehearsal/rehearsal.json", "time": rec["time"], "tag": rec.get("tag", ""),
                  "models": rec["models"], "versions": rec.get("versions"),
                  "detector": {"threshold": rec["threshold"], "order": rec["order"],
                               "items": {n: {k: v for k, v in it.items() if k != "wav"} for n, it in rec["items"].items()}, "reference": ref},
                  "watermark": {"carrier": rec["watermark"], "splice": rec["splice"], "order": rec["cond_order"], "conds": conds},
                  "skipped": rec.get("skipped", []),
                  "definitions": {
                      "s": "AASIST's bona fide logit minus its spoof logit (log-odds), averaged over windows of 64600 samples (hop 32300, last window flush with the end; shorter audio is tiled). Input scaled to a fixed RMS first.",
                      "threshold": "0 = the model's own two-class decision boundary, fixed before looking at any result. Not the EER threshold on ASVspoof 2019 LA (that needs the dataset).",
                      "called": "'bona fide' if s > threshold, else 'spoof'",
                      "reference": "one utterance per LibriSpeech dev-clean speaker (the 40 sentences of W2 Demo 3), scored the same way",
                      "marked / control": "the watermarked audio, and the same audio without the watermark, sent through the same processing",
                      "mean": "mean over samples of the AudioSeal detector's per-sample probability",
                      "frac": "fraction of samples whose probability exceeds 0.5 (what audioseal's detect_watermark() returns)",
                      "detected": "frac > 0.5",
                      "bit_acc": "fraction of the 16 message bits decoded correctly (0.5 = chance)",
                      "regeneration": "16 kHz → 24 kHz → codec (encode once, decode the first n_q codebooks) → 16 kHz; Vocos: 16 → 24 kHz → its own mel front end → waveform → 16 kHz",
                      "what_we_heard": "filled by hand after the rehearsal (w09.qmd p89, p90)"}}
    METRICS_JSON.parent.mkdir(parents=True, exist_ok=True)
    METRICS_JSON.write_text(json.dumps(d, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    print(f"  → {os.path.relpath(METRICS_JSON, HERE)}（demo3）")


# ── selftest ─────────────────────────────────────────────────
def selftest() -> None:
    C.banner("selftest（不需要模型）", C.AMBER)
    n_ok = [0, 0]

    def ok(cond: bool, msg: str):
        n_ok[0 if cond else 1] += 1
        print(f"  {C.GREEN + '✓' if cond else C.RED + '✗'}{C.RESET} {msg}")

    L, hop = int(DET["window"]), int(DET["hop"])
    ok(window_starts(L, L, hop) == [0] and window_starts(L - 1, L, hop) == [0], "剛好一窗或不到一窗：只有一窗")
    s = window_starts(93760, L, hop)
    ok(s == [0, 93760 - L], f"5.86 s（93760 點）：兩窗，第二窗貼齊句尾 {s}")
    s = window_starts(L + 2 * hop + 5, L, hop)
    ok(s == [0, hop, 2 * hop, 2 * hop + 5], f"最後一窗貼齊句尾、不重複 {s}")
    x = np.arange(1000, dtype=float)
    w = windows(x, L, hop)
    ok(w.shape == (1, L) and w[0, 1000] == 0 and w[0, 999] == 999, "短的重複接長（AASIST 的 pad）")
    rng = np.random.default_rng(0)
    x = rng.standard_normal(FS)
    ok(abs(rms(level(x, 0.05)) - 0.05) < 1e-9, "level：RMS 調到 0.05")
    y = add_noise(x, 20.0, np.random.default_rng(1))
    snr = 20 * math.log10(rms(x) / rms(y - x))
    ok(abs(snr - 20.0) < 1e-6, f"add_noise：SNR {snr:.3f} dB")
    host, seg = np.zeros(100), np.ones(30)
    sp = splice(host, seg, 40)
    ok(len(sp) == 130 and sp[39] == 0 and sp[40] == 1 and sp[69] == 1 and sp[70] == 0, "splice：插入、不是蓋掉")
    p = np.zeros(FS)
    p[4000:12000] = 1.0
    lo, hi = crossings(p)
    ok(abs(lo - 4000) < 200 and abs(hi - 11999) < 200, f"crossings：{lo}–{hi}（真值 4000–11999）")
    st = det_stats(p, np.asarray(MESSAGE, dtype=float))
    ok(abs(st["frac"] - 0.5) < 1e-9 and st["detected"] is False and st["bit_acc"] == 1.0 and len(st["curve"]) == 100, "det_stats：一半的取樣點 → 不算偵測到（要過半）；100 點的曲線")
    st = det_stats(np.ones(FS), 1 - np.asarray(MESSAGE, dtype=float))
    ok(st["detected"] and st["bit_acc"] == 0.0, "det_stats：全 1 → 偵測到；bit 全反 → 0")
    t = np.arange(FS) / FS
    tone = np.sin(2 * np.pi * 440 * t)
    rt = round_trip(tone, 8000)
    ok(len(rt) == len(tone) and rms(rt - tone) / rms(tone) < 0.05, f"round_trip 8 kHz：440 Hz 的正弦波留著（相對誤差 {rms(rt - tone) / rms(tone):.4f}）")
    hi_tone = np.sin(2 * np.pi * 6000 * t)
    ok(rms(round_trip(hi_tone, 8000)) / rms(hi_tone) < 0.05, "round_trip 8 kHz：6 kHz 的正弦波不見了")
    ok(len(MESSAGE) == 16 and set(MESSAGE) <= {0, 1}, "訊息是 16 個 bit")
    keys = [it["key"] for it in ITEMS]
    ok(all(k in "123456x" for k in keys) and len({it["name"] for it in ITEMS}) == len(ITEMS), "items 的鍵與名字")
    ok(WM["carrier"] in {it["name"] for it in ITEMS}, "watermark 的 carrier 在 items 裡")

    # fake 流程（要 W3 那句與 W6 Demo 1 的程式；前幾週的 .wav 不在就略過那幾格）
    if (HERE / CFG["audio"]["real"]).resolve().exists():
        car = next(it for it in ITEMS if it["name"] == WM["carrier"])
        if (HERE / car["path"]).resolve().exists():
            out = Out(echo=False)
            d = build("selftest", out)
            d.quiet = True
            d.compute()
            ok("real" in d.items and all(n in d.conds for n in ("none", "splice", "encodec02", "mimi08", "vocos")), f"fake 流程：{len(d.items)} 段給分、{len(d.conds)} 格")
            ok(d.conds["none"]["marked"]["detected"] and not d.conds["none"]["control"]["detected"], "fake：加了的偵測到、沒加的沒有")
            si = d.splice_info
            ok(si["found_start_s"] is not None and abs(si["found_start_s"] - si["true_start_s"]) < 0.2 and abs(si["found_end_s"] - si["true_end_s"]) < 0.2,
               f"fake：接合的位置找得回來（{si['found_start_s']:.2f}–{si['found_end_s']:.2f} 對 {si['true_start_s']:.2f}–{si['true_end_s']:.2f}）")
            ok(not d.conds["vocos"]["marked"]["detected"], "fake：替身 Vocos 把替身 watermark 洗掉")
            for k in "r123456cxwdsanpeeeeeemvbt?":
                d.dispatch(k)
            d.dispatch(" ")
            ok("live" in d.items, "fake：按鍵走一遍、空白鍵加一格")
            rd = d.record_dict()
            json.dumps(rd, default=float)
            ok(all("wav" not in it or isinstance(it["wav"], str) for it in rd["items"].values()), "record_dict 可以存成 json")
            shutil.rmtree(d.folder, ignore_errors=True)
        else:
            print(f"  {C.AMBER}（{car['path']} 不在：fake 流程略過）{C.RESET}")
    else:
        print(f"  {C.AMBER}（{CFG['audio']['real']} 不在：fake 流程略過）{C.RESET}")
    # 彩排紀錄與 w09-metrics.json 一致
    rp = REHEARSAL / "rehearsal.json"
    if rp.exists() and METRICS_JSON.exists():
        rec = json.loads(rp.read_text(encoding="utf-8"))
        m = json.loads(METRICS_JSON.read_text(encoding="utf-8")).get("demo3")
        if m:
            same = m["time"] == rec["time"] and all(abs(m["detector"]["items"][n]["s"] - rec["items"][n]["s"]) < 1e-9 for n in rec["order"]) \
                and all(abs(m["watermark"]["conds"][n]["marked"]["frac"] - rec["conds"][n]["marked"]["frac"]) < 1e-9 for n in rec["cond_order"])
            ok(same, "w09-metrics.json → demo3 與 runs/rehearsal/rehearsal.json 一致")
        else:
            print(f"  {C.AMBER}（w09-metrics.json 還沒有 demo3：./present.sh export）{C.RESET}")
    print(f"\n  {n_ok[0]} 項通過、{n_ok[1]} 項失敗")
    if n_ok[1]:
        sys.exit(1)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", nargs="?", default="show", choices=["show", "rehearse", "replay", "record", "check", "fetch", "fake", "selftest", "export"])
    ap.add_argument("--no-plot", action="store_true", help="只在終端機，不開圖")
    ap.add_argument("--device", default=None, help="覆蓋 demo_config.toml 的裝置（cpu／mps／auto）")
    args = ap.parse_args()
    if args.cmd == "fetch":
        return fetch()
    if args.cmd == "check":
        return check(args.device)
    if args.cmd == "selftest":
        return selftest()
    if args.cmd == "record":
        return record()
    if args.cmd == "export":
        return export(REHEARSAL / "rehearsal.json")
    out = Out()
    if args.cmd == "replay":
        out.banner("REPLAY：讀回彩排（不碰 torch）", C.AMBER)
        d = build_replay(out)
        return run(d, not args.no_plot)
    out.banner({"show": "W9 Demo 3 · 兩道防線各自在哪裡破", "rehearse": "彩排：全部算一遍，存 runs/rehearsal/", "fake": "FAKE：替身，沒有任何模型"}[args.cmd],
               C.AMBER if args.cmd != "show" else C.BLUE)
    d = build(args.cmd, out, args.device)
    if args.cmd == "rehearse":
        d.quiet = True
        for old in REHEARSAL.glob("*"):
            if old.is_file() and old.suffix in (".wav", ".png"):
                old.unlink()
    d.compute()
    if args.cmd == "rehearse":
        d.shown = list(d.order)
        d.summary()
        s = d.splice_info
        out(f"\n  接合：接進去 {s['true_start_s']:.2f}–{s['true_end_s']:.2f} s；曲線高於 0.5 的最長一段是 {s['found_start_s']}–{s['found_end_s']} s；裡面平均 {s['mean_inside']:.3f}、外面 {s['mean_outside']:.3f}")
        i = d.wm_info
        out(f"  watermark 比訊號低 {i['snr_db']:.1f} dB")
        d.rehearse_figs()
        d.save()
        export(REHEARSAL / "rehearsal.json")
        print(f"  → {os.path.relpath(REHEARSAL, HERE)}/")
        return
    try:
        run(d, not args.no_plot)
    finally:
        d.save()


if __name__ == "__main__":
    main()

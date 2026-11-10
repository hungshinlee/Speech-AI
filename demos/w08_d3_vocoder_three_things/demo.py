# -*- coding: utf-8 -*-
# 此檔由 bin/publish_demos.py 從 private repo 的 <course>/demos/ 複製而來，請勿在公開 repo 直接編輯。
"""W8 demo 3：vocoder 的三件事——同一句真人語音、三個 vocoder；再換兩份預測出來的 mel（Demo 1 的 SpeechT5 與 F5）；
最後一段不經 mel 的台語與客語（MMS-TTS，VITS 直接出波形）。

  show       課堂：載三個網路、把全部格子算好（M5 Max 上幾十秒），開一個視窗；鍵在視窗裡按（下面的鍵）。離開時存 runs/live/<時間>/
  rehearse   課前：不互動，全部算一遍，.wav、mel（.npy）、圖、數字存 runs/rehearsal/（進版控 = 現場的退路），寫 slides/assets/w08/w08-metrics.json → demo3
  replay     現場出狀況時：從 runs/rehearsal/ 讀回，同一個視窗、同樣的鍵，不碰 torch
  check      課前體檢：版本、MPS；Vocos、BigVGAN、VITS 在 cpu／mps 上載不載得起、同一個輸入的數值差與耗時（兩個裝置的 .wav 存 runs/check/，用耳朵比）；
             換算在真人錄音上的誤差；兩個「100-band log-mel」差多少 → runs/check.json
  fetch      課前一週：下載 Vocos、BigVGAN（含程式碼）、MMS-TTS nan／hak／eng、SpeechT5 的前端設定（鎖 commit → model.lock）、從 Demo 1 複製真人錄音（要網路）
  fake       沒有模型也能演練流程：Vocos 與 BigVGAN 用 Griffin-Lim 冒充（畫面標明 FAKE）
  selftest   不開視窗、不需要模型：filterbank、換算（恆等；真人錄音上 SpeechT5 → Vocos 的誤差）、Griffin-Lim 單調性、fake 流程、白話字的字元；有彩排紀錄時對 w08-metrics.json
  export     只重寫 w08-metrics.json → demo3（從 runs/rehearsal/rehearsal.json）

畫面上的鍵（順序照投影片 p71／p72 講稿的【操作】）：
  0  播原音（真人錄音，24 kHz）
  1  目前這一列的 mel → Griffin-Lim（W2 Demo 2 的演算法，64 輪），播；印 fit、LSD、耗時
  2  → Vocos，播          3  → BigVGAN，播
  g  選真人錄音那一列（預設）
  4  選 SpeechT5 的 mel（Demo 1 回歸出來的）那一列，依序播 Griffin-Lim → Vocos → BigVGAN
  5  選 F5 的 mel（Demo 1 抽出來的）那一列，依序播三個
  m  不換算、硬餵：這一列 Vocos 定義的 mel 原樣給 BigVGAN（同樣叫 100-band log-mel 24 kHz，矩陣不同——W2 Demo 3）
  n  Demo 1 當時的聲音（SpeechT5 → 它自己的 HiFi-GAN；F5 → Vocos）當對照；真人那一列就是原音
  6  MMS-TTS nan：一句白話字的台語，播；印 tokenizer 留下幾個字元、丟掉哪些
  7  MMS-TTS hak：一句客語白話字
  c  同一句台語改餵漢字給 nan：tokenizer 一個字都不留
  e  MMS-TTS eng：W3 那句（退路）
  t  數字表（三列 × 三欄的 fit／LSD／耗時、RTF；MMS 三句）
  r  再播一次     h  選單     q  離開

三個 vocoder 各用自己的前端（W2 Demo 3 的教訓）：Vocos 與 BigVGAN 都是 100 bins、24 kHz、hop 256，但 mel 尺度（HTK／Slaney）、正規化（無／面積）、
補邊、floor 都不同；Griffin-Lim 吃 Vocos 定義的 mel（F5 原生的那一種）經 pseudo-inverse。預測的 mel 要給別的前端時走對數域的內插換算（demo_config.toml [convert]）。
量：fit = vocoder 的輸出用它自己的前端重新分析、對它拿到的 mel 的 RMS 差（dB；三件事裡的第一件做得多準）；LSD = 對原音（只有真人那一列有原音）。
兩個都是參考，不是判準——判準是耳朵。
"""
from __future__ import annotations

import argparse
import json
import math
import os
import platform
import shutil
import sys
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
MODEL_LOCK = HERE / "model.lock"
METRICS_JSON = HERE.parents[1] / "slides" / "assets" / "w08" / "w08-metrics.json"     # parents[1] = speech_ai/
REHEARSAL = HERE / "runs" / "rehearsal"
CHECK_DIR = HERE / "runs" / "check"
FS = CFG["audio"]["fs"]
N_FFT, HOP = 1024, 256
ROWS = ("real", "speecht5", "f5")
VOCS = ("gl", "vocos", "bigvgan")
ROW_LABEL = {"real": "real recording", "speecht5": "SpeechT5 mel (Demo 1, regressed)", "f5": "F5 mel (Demo 1, sampled)"}
VOC_LABEL = {"gl": f"Griffin-Lim ({CFG['griffin_lim']['iters']} it)", "vocos": "Vocos", "bigvgan": "BigVGAN v2"}
FIT_FLOOR = CFG["convert"]["fit_floor_ln"]
MMS_LANGS = ("nan", "hak", "eng")
LN10 = math.log(10.0)


# ── 前端 ─────────────────────────────────────────────────────
class FrontEnd:
    """一個 vocoder（或 TTS 模型）的 mel 前端：取樣率、hop、filterbank 矩阵 M (bins, n_fft/2+1)、log 的底、floor。
    陣列一律 (bins, frames)、模型自己的單位。to_ln 把它換成自然對數（SpeechT5 是 log10）。"""

    def __init__(self, key: str, fs: int, M: np.ndarray, base: float, floor_ln: float, label: str, fmin: float, fmax: float):
        self.key, self.fs, self.M, self.base, self.floor_ln, self.label, self.fmin, self.fmax = key, fs, np.asarray(M, dtype=np.float64), base, floor_ln, label, fmin, fmax
        self.freqs = np.linspace(0, fs / 2, self.M.shape[1])
        self.centers = np.array([self.freqs[np.argmax(self.M[k])] for k in range(self.M.shape[0])])
        self.gain_ln = np.log(np.maximum(self.M.sum(axis=1), 1e-12))     # 每個 bin 的權重和：一個 bin 裡的頻譜是平的時候，mel = gain × 頻譜
        self._mel_fn = None          # 真的前端（torch）由模型那邊掛上來；沒有就用 numpy 近似

    @property
    def fps(self) -> float:
        return self.fs / HOP

    @property
    def n_mels(self) -> int:
        return self.M.shape[0]

    def to_ln(self, logmel: np.ndarray) -> np.ndarray:
        return np.asarray(logmel, dtype=np.float64) * self.base

    def mel(self, x: np.ndarray, fs_in: int) -> np.ndarray:
        """波形 → 這個前端的 log-mel (bins, frames)，模型自己的單位（Vocos／BigVGAN 是 ln）。"""
        x = dsp.resample(np.asarray(x, dtype=np.float64), fs_in, self.fs)
        if self._mel_fn is not None:
            return self._mel_fn(x)
        return self.numpy_mel(x)

    def numpy_mel(self, x: np.ndarray) -> np.ndarray:
        """fake／selftest／replay 用的近似（dsp 的 STFT，center）：不是模型的程式碼，但同一個 filterbank 尺度。"""
        S = np.abs(dsp.stft(x, N_FFT, HOP, N_FFT))
        out = np.log(np.maximum(self.M @ S, math.exp(self.floor_ln)))
        return out / self.base

    def to_db(self, logmel: np.ndarray) -> np.ndarray:
        return 20.0 / LN10 * self.to_ln(logmel)


def numpy_front_ends() -> dict[str, FrontEnd]:
    """三個前端的 numpy 版 filterbank（真的模型載進來之後，Vocos／BigVGAN／SpeechT5 的 M 會被模型自己的矩陣換掉）。"""
    fe = {}
    Mv = dsp.mel_filterbank(FS, N_FFT, 100, "htk", area_norm=False, fmin=0.0, fmax=FS / 2)
    Mb = dsp.mel_filterbank(FS, N_FFT, 100, "slaney", area_norm=True, fmin=0.0, fmax=FS / 2)
    M5 = dsp.mel_filterbank(16000, N_FFT, 80, "slaney", area_norm=True, fmin=80.0, fmax=7600.0)
    fe["vocos"] = FrontEnd("vocos", FS, Mv, 1.0, math.log(1e-7), "Vocos front end · 100 bins · HTK · no norm · ln(clamp 1e-7)", 0.0, FS / 2)
    fe["bigvgan"] = FrontEnd("bigvgan", FS, Mb, 1.0, math.log(1e-5), "BigVGAN front end · 100 bins · Slaney · area norm · ln(clamp 1e-5)", 0.0, FS / 2)
    fe["speecht5"] = FrontEnd("speecht5", 16000, M5, LN10, math.log(1e-10), "SpeechT5 front end · 80 bins · 16 kHz · Slaney · log10", 80.0, 7600.0)
    return fe


# ── 換算與量 ──────────────────────────────────────────────────
def convert(logmel: np.ndarray, src: FrontEnd, dst: FrontEnd) -> np.ndarray:
    """一個前端的 log-mel → 另一個前端的 log-mel（對數域內插；demo_config.toml [convert]）。回傳 dst 自己的單位。"""
    lnS = src.to_ln(logmel) - src.gain_ln[:, None]                      # 每個 bin 裡「平均的線性頻譜」的 ln
    T = lnS.shape[1]
    ratio = dst.fps / src.fps
    T2 = int(round(T * ratio))
    if T2 != T:
        t_src, t_dst = np.arange(T) / src.fps, np.arange(T2) / dst.fps
        lnS = np.stack([np.interp(t_dst, t_src, lnS[k]) for k in range(lnS.shape[0])])
    inband = (src.centers >= src.fmin) & (src.centers <= src.fmax)
    cs = src.centers[inband]
    out = np.empty((dst.n_mels, T2))
    for j in range(T2):
        col = lnS[inband, j]
        out[:, j] = np.interp(dst.centers, cs, col, left=col[0], right=col.min())    # 來源沒有的頻帶：那一格最安靜的值
    out += dst.gain_ln[:, None]
    return np.maximum(out, dst.floor_ln) / dst.base


def mel_fit_db(a: np.ndarray, b: np.ndarray, fe: FrontEnd, floor_ln: float = FIT_FLOOR) -> float:
    """兩份同一個前端的 log-mel 的 RMS 差（dB = 20·Δln／ln10），兩邊都先夾在 floor（沒有能量的格子不算帳）。"""
    a, b = fe.to_ln(a), fe.to_ln(b)
    n = min(a.shape[1], b.shape[1])
    a, b = np.maximum(a[:, :n], floor_ln), np.maximum(b[:, :n], floor_ln)
    return float(20.0 / LN10 * np.sqrt(np.mean((a - b) ** 2)))


def energy_lag(a: np.ndarray, b: np.ndarray, fe: FrontEnd, max_lag: int = 6) -> int:
    """兩份 log-mel 的逐 frame 能量最相關的 lag（frames）；換算對了應該是 0。"""
    n = min(a.shape[1], b.shape[1])
    ea, eb = np.exp(fe.to_ln(a[:, :n])).sum(axis=0), np.exp(fe.to_ln(b[:, :n])).sum(axis=0)
    best = max(range(-max_lag, max_lag + 1),
               key=lambda l: np.corrcoef(ea[max(0, l):n + min(0, l)], eb[max(0, -l):n - max(0, l)])[0, 1])
    return int(best)


def lsd_db(ref: np.ndarray, est: np.ndarray) -> float:
    c = CFG["lsd"]
    L, H = int(FS * c["win_ms"] / 1000), int(FS * c["hop_ms"] / 1000)
    return dsp.log_spectral_distance(ref, est, L, H, c["n_fft"])


def rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(x)))) if len(x) else 0.0


def pinv_magnitude(logmel: np.ndarray, fe: FrontEnd) -> np.ndarray:
    """W2 Demo 2 的做法：log-mel → mel magnitude → pseudo-inverse → 線性 magnitude (n_fft/2+1, T)，負值截 0。"""
    return np.maximum(np.linalg.pinv(fe.M) @ np.exp(fe.to_ln(logmel)), 0.0)


# ── 模型 ─────────────────────────────────────────────────────
def read_lock() -> dict[str, str]:
    if not MODEL_LOCK.exists():
        return {}
    out = {}
    for line in MODEL_LOCK.read_text(encoding="utf-8").splitlines():
        if line.strip() and not line.startswith("#"):
            repo, sha = line.split()[:2]
            out[repo] = sha
    return out


REPOS = {"vocos": CFG["vocos"]["repo"], "bigvgan": CFG["bigvgan"]["repo"], "speecht5": CFG["speecht5_frontend"]["repo"],
         **{f"mms_{l}": CFG["mms"][l] for l in MMS_LANGS}}
PATTERNS = {"vocos": ["config.yaml", "pytorch_model.bin", "README.md"],
            "bigvgan": ["*.py", "alias_free_activation/torch/*", "config.json", CFG["bigvgan"]["weights"], "LICENSE", "README.md"],
            "speecht5": ["preprocessor_config.json"],
            **{f"mms_{l}": ["*.json", "model.safetensors", "README.md"] for l in MMS_LANGS}}


def model_dir(key: str, offline: bool = True) -> str:
    from huggingface_hub import snapshot_download
    repo = REPOS[key]
    return snapshot_download(repo, revision=read_lock().get(repo), local_files_only=offline, allow_patterns=PATTERNS[key])


def quiet_hf() -> None:
    try:
        from transformers.utils import logging as hf_logging
        hf_logging.set_verbosity_error()
        hf_logging.disable_progress_bar()
    except ImportError:
        pass


def resolve_device(d: str) -> str:
    import torch
    if d == "auto":
        return "mps" if torch.backends.mps.is_available() else "cpu"
    return d


def sync(dev: str) -> None:
    import torch
    if dev == "mps":
        torch.mps.synchronize()


class _quiet:
    """把 stdout 關掉一下（HF 的 Loading weights 進度條、BigVGAN 的 print）。"""

    def __enter__(self):
        self._o, sys.stdout = sys.stdout, open(os.devnull, "w")
        return self

    def __exit__(self, *a):
        sys.stdout.close()
        sys.stdout = self._o


def speecht5_filterbank() -> np.ndarray:
    """SpeechT5 的 mel filterbank（80, 513），從它的 preprocessor_config.json 建——只為了換算，不載模型。"""
    from transformers import SpeechT5FeatureExtractor
    fe = SpeechT5FeatureExtractor.from_pretrained(model_dir("speecht5"))
    return np.asarray(fe.mel_filters).T


class GriffinLim:
    """W2 Demo 2 的 Griffin-Lim（../dsp.py，原始版本、無 momentum）當第一個「vocoder」：吃 Vocos 定義的 mel，pseudo-inverse 之後迭代猜相位。"""
    key, label, device = "gl", VOC_LABEL["gl"], "numpy"

    def __init__(self, fe: FrontEnd, iters: int | None = None, seed: int | None = None):
        self.fe = fe
        self.iters = CFG["griffin_lim"]["iters"] if iters is None else iters
        self.seed = CFG["griffin_lim"]["seed"] if seed is None else seed
        self.fake = False

    def decode(self, logmel: np.ndarray) -> tuple[np.ndarray, float]:
        t0 = time.perf_counter()
        mag = pinv_magnitude(logmel, self.fe)
        length = (logmel.shape[1] - 1) * HOP
        y, self.log, _ = dsp.griffin_lim(mag, N_FFT, HOP, N_FFT, length, self.iters, seed=self.seed)
        return y, time.perf_counter() - t0

    def info(self) -> dict:
        return {"algorithm": "Griffin & Lim 1984 (dsp.griffin_lim, no momentum)", "iters": self.iters, "seed": self.seed, "mel_inverse": "pinv",
                "input": "Vocos-defined mel", "n_fft": N_FFT, "hop": HOP}


class VocosVocoder:
    key, label = "vocos", VOC_LABEL["vocos"]

    def __init__(self, device: str):
        import torch
        from vocos import Vocos
        self.torch = torch
        self.device = resolve_device(device)
        d = model_dir("vocos")
        with _quiet(), warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)      # vocos 裡的 torch.nn.utils.weight_norm 已 deprecated
            self.model = Vocos.from_hparams(str(Path(d) / "config.yaml"))
            self.model.load_state_dict(torch.load(str(Path(d) / "pytorch_model.bin"), map_location="cpu"))
        self.model.eval().to(self.device)
        self.n_params = sum(p.numel() for p in self.model.parameters())
        fb = self.model.feature_extractor.mel_spec.mel_scale.fb.detach().cpu().numpy().T          # (100, 513)
        self.fe = FrontEnd("vocos", FS, fb, 1.0, math.log(1e-7), "Vocos front end · 100 bins · HTK · no norm · ln(clamp 1e-7)", 0.0, FS / 2)
        self.fe._mel_fn = self._mel
        self.fake = False

    def _mel(self, x: np.ndarray) -> np.ndarray:
        with self.torch.no_grad():
            return self.model.feature_extractor(self.torch.tensor(x, dtype=self.torch.float32)[None].to(self.device))[0].cpu().numpy()

    def decode(self, logmel: np.ndarray) -> tuple[np.ndarray, float]:
        mel = self.torch.tensor(np.asarray(logmel), dtype=self.torch.float32)[None].to(self.device)
        sync(self.device)
        t0 = time.perf_counter()
        with self.torch.no_grad():
            y = self.model.decode(mel)
        sync(self.device)
        return y[0].cpu().numpy().astype(np.float64), time.perf_counter() - t0

    def info(self) -> dict:
        return {"repo": REPOS["vocos"], "commit": read_lock().get(REPOS["vocos"]), "device": self.device, "n_params": self.n_params,
                "head": "ISTFT (n_fft 1024, hop 256)", "front_end": self.fe.label}


_BIGVGAN = {}


def import_bigvgan(d: str) -> dict:
    """BigVGAN 的程式碼住在 HF repo 裡（bigvgan.py 等，平的 import：utils、env、activations、meldataset）。
    utils.py 在 import 時把 matplotlib 換成 Agg——import 完換回來，課堂的視窗才開得起來。
    不用 BigVGAN.from_pretrained：它的 _from_pretrained 簽名是給 huggingface_hub 0.x 的（要 proxies／resume_download），1.x 不給，會 TypeError。"""
    if _BIGVGAN:
        return _BIGVGAN
    import matplotlib
    backend = matplotlib.get_backend()
    sys.path.insert(0, d)
    try:
        with warnings.catch_warnings(), _quiet():
            warnings.simplefilter("ignore")
            import bigvgan      # noqa: F401
            import meldataset   # noqa: F401
    finally:
        sys.path.remove(d)
    if matplotlib.get_backend() != backend:
        matplotlib.use(backend, force=True)
    _BIGVGAN.update(bigvgan=bigvgan, meldataset=meldataset)
    return _BIGVGAN


class BigVGANVocoder:
    key, label = "bigvgan", VOC_LABEL["bigvgan"]

    def __init__(self, device: str):
        import torch
        self.torch = torch
        self.device = resolve_device(device)
        d = model_dir("bigvgan")
        m = import_bigvgan(d)
        self.h = m["bigvgan"].load_hparams_from_json(str(Path(d) / "config.json"))
        with _quiet(), warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)      # weight_norm、torch.jit.script 的 deprecation 來自 BigVGAN 的程式碼
            self.model = m["bigvgan"].BigVGAN(self.h, use_cuda_kernel=False)
            sd = torch.load(str(Path(d) / CFG["bigvgan"]["weights"]), map_location="cpu")
            try:
                self.model.load_state_dict(sd["generator"])
                self.model.remove_weight_norm()
            except RuntimeError:                       # 權重已經拿掉 weight norm
                self.model.remove_weight_norm()
                self.model.load_state_dict(sd["generator"])
        self.model.eval().to(self.device)
        self.n_params = sum(p.numel() for p in self.model.parameters())
        import librosa
        fb = librosa.filters.mel(sr=self.h.sampling_rate, n_fft=self.h.n_fft, n_mels=self.h.num_mels, fmin=self.h.fmin, fmax=self.h.fmax)
        self.fe = FrontEnd("bigvgan", self.h.sampling_rate, fb, 1.0, math.log(1e-5),
                           "BigVGAN front end · 100 bins · Slaney · area norm · ln(clamp 1e-5)", 0.0, self.h.sampling_rate / 2)
        self.fe._mel_fn = self._mel
        self.mel_spectrogram = m["meldataset"].mel_spectrogram
        warnings.filterwarnings("ignore", category=FutureWarning, module=r"torch\.jit")   # 第一次前向時 alias-free 的 torch.jit.script deprecation
        self.fake = False

    def _mel(self, x: np.ndarray) -> np.ndarray:
        h = self.h
        with self.torch.no_grad():
            y = self.torch.tensor(x, dtype=self.torch.float32)[None]
            return self.mel_spectrogram(y, h.n_fft, h.num_mels, h.sampling_rate, h.hop_size, h.win_size, h.fmin, h.fmax)[0].numpy()

    def decode(self, logmel: np.ndarray) -> tuple[np.ndarray, float]:
        mel = self.torch.tensor(np.asarray(logmel), dtype=self.torch.float32)[None].to(self.device)
        sync(self.device)
        t0 = time.perf_counter()
        with self.torch.no_grad():
            y = self.model(mel)
        sync(self.device)
        return y[0, 0].cpu().numpy().astype(np.float64), time.perf_counter() - t0

    def info(self) -> dict:
        return {"repo": REPOS["bigvgan"], "commit": read_lock().get(REPOS["bigvgan"]), "device": self.device, "n_params": self.n_params,
                "upsample_rates": list(self.h.upsample_rates), "activation": self.h.activation, "cuda_kernel": False, "front_end": self.fe.label}


class FakeVocoder:
    """沒有模型時冒充 Vocos／BigVGAN：256 輪、換 seed 的 Griffin-Lim，吃同一個（numpy 近似的）前端。畫面標明 FAKE。"""

    def __init__(self, key: str, fe: FrontEnd, seed: int):
        self.key, self.label, self.fe, self.device, self.fake = key, f"FAKE {VOC_LABEL[key]}", fe, "numpy", True
        self.gl = GriffinLim(fe, iters=CFG["fake"]["gl_iters"], seed=seed)

    def decode(self, logmel):
        return self.gl.decode(logmel)

    def info(self):
        return {"fake": True, "stand_in": self.gl.info()}


class Mms:
    """MMS-TTS 的一個語言：VITS（transformers 的 VitsModel），字元輸入、16 kHz 波形輸出。vocab 以外的字元 tokenizer 直接丟掉。"""

    def __init__(self, lang: str, device: str):
        import torch
        from transformers import VitsModel, VitsTokenizer
        self.torch, self.lang = torch, lang
        self.device = resolve_device(device)
        d = model_dir(f"mms_{lang}")
        quiet_hf()
        with _quiet():
            self.tok = VitsTokenizer.from_pretrained(d)
            self.model = VitsModel.from_pretrained(d).eval().to(self.device)
        self.vocab = set(self.tok.get_vocab())
        self.fs = int(self.model.config.sampling_rate)
        self.n_params = sum(p.numel() for p in self.model.parameters())
        self.fake = False

    def dropped(self, text: str) -> list[str]:
        return sorted({ch for ch in text.lower() if ch not in self.vocab and not ch.isspace()})

    def say(self, text: str, seed: int) -> dict:
        inp = self.tok(text, return_tensors="pt")
        n_ids = int(inp["input_ids"].shape[1])
        rec = {"text": text, "n_ids": n_ids, "kept": self.tok.decode(inp["input_ids"][0]), "dropped": self.dropped(text), "fs": self.fs, "seed": seed}
        if n_ids == 0:
            rec.update(wav=np.zeros(0), seconds=0.0, error="tokenizer kept nothing")
            return rec
        self.torch.manual_seed(seed)
        inp = {k: v.to(self.device) for k, v in inp.items()}
        sync(self.device)
        t0 = time.perf_counter()
        try:
            with self.torch.no_grad():
                out = self.model(**inp)
            sync(self.device)
            wav = out.waveform[0].cpu().numpy().astype(np.float64)
            rec.update(wav=wav, seconds=time.perf_counter() - t0, audio_s=len(wav) / self.fs)
        except RuntimeError as e:
            rec.update(wav=np.zeros(0), seconds=0.0, error=f"{type(e).__name__}: {str(e)[:120]}")
        return rec

    def info(self) -> dict:
        return {"repo": REPOS[f"mms_{self.lang}"], "commit": read_lock().get(REPOS[f"mms_{self.lang}"]), "device": self.device,
                "n_params": self.n_params, "fs": self.fs, "vocab_size": len(self.vocab),
                "noise_scale": float(self.model.noise_scale), "noise_scale_duration": float(self.model.noise_scale_duration), "speaking_rate": float(self.model.speaking_rate)}


class FakeMms:
    """沒有模型時：合成母音串冒充（長度照字元數），tokenizer 的「丟字」用 vocab.json（抓過的話）或一張硬寫的白話字字母表判。"""

    def __init__(self, lang: str):
        self.lang, self.fs, self.fake, self.device = lang, 16000, True, "numpy"
        self.vocab = None
        try:
            self.vocab = set(json.loads((Path(model_dir(f"mms_{lang}")) / "vocab.json").read_text(encoding="utf-8")))
        except Exception:
            pass

    def dropped(self, text: str) -> list[str]:
        if self.vocab is None:
            return sorted({ch for ch in text.lower() if ord(ch) > 0x2FFF})      # 沒有 vocab：只能判漢字一類
        return sorted({ch for ch in text.lower() if ch not in self.vocab and not ch.isspace()})

    def say(self, text: str, seed: int) -> dict:
        kept = "".join(ch for ch in text.lower() if ch.isspace() or ch not in self.dropped(text))
        letters = [ch for ch in kept if not ch.isspace()]
        rec = {"text": text, "n_ids": 2 * len(letters) + 1 if letters else 0, "kept": kept.strip(), "dropped": self.dropped(text), "fs": self.fs, "seed": seed, "fake": True}
        if not letters:
            rec.update(wav=np.zeros(0), seconds=0.0, error="tokenizer kept nothing")
            return rec
        rng = np.random.default_rng(seed)
        parts = [dsp.synth_vowel(self.fs, 0.12, 110.0 + 20 * rng.standard_normal(), rng.choice([(730, 1090, 2440), (270, 2290, 3010), (300, 870, 2240)]))
                 for _ in letters]
        wav = np.concatenate(parts) * 0.3
        rec.update(wav=wav, seconds=0.0, audio_s=len(wav) / self.fs)
        return rec

    def info(self) -> dict:
        return {"fake": True, "lang": self.lang}


# ── 紀錄 ─────────────────────────────────────────────────────
class Out:
    def __init__(self):
        self.lines: list[str] = []

    def __call__(self, s: str = "") -> None:
        print(s)
        self.lines.append(s)

    def banner(self, s: str, color: str = C.BLUE) -> None:
        C.banner(s, color)
        self.lines.append(f"\n== {s} ==")

    def text(self) -> str:
        import re
        return re.sub(r"\x1b\[[0-9;]*m", "", "\n".join(self.lines)) + "\n"


# ── 主體 ─────────────────────────────────────────────────────
class Demo:
    """三列（真人／SpeechT5／F5）× 三欄（GL／Vocos／BigVGAN）的格子，加一欄「硬餵」與 MMS 的三句。全部在啟動時算好。"""

    def __init__(self, out: Out, fe: dict[str, FrontEnd], vocoders: dict, mms: dict, folder: Path, tag: str = ""):
        self.out, self.fe, self.voc, self.mms, self.folder, self.tag = out, fe, vocoders, mms, folder, tag
        self.folder.mkdir(parents=True, exist_ok=True)
        self.inputs: dict[str, dict[str, np.ndarray]] = {}      # inputs[row][fe_key] = 給那個前端的 log-mel；inputs[row]["own"] = 原本的（畫面左格）
        self.cells: dict[tuple[str, str], dict] = {}            # cells[(row, voc)] = {"wav", "seconds", "fit_db", "lsd_db", "rtf", "T", "remel"}
        self.mms_out: dict[str, dict] = {}
        self.row = "real"
        self.last: tuple | None = None
        self.fig = None
        self.mode = ("grid", None)
        self.ref_rms = 0.05
        self.real: np.ndarray | None = None
        self.demo1: dict[str, Path] = {}
        self.cold: dict[str, float] = {}

    # ── 輸入 ──
    def load_inputs(self) -> None:
        a, p = CFG["audio"], CFG["predicted"]
        real_path = HERE / a["real_path"]
        if not real_path.exists():
            src = (HERE / a["real_source"]).resolve()
            if not src.exists():
                sys.exit(f"沒有 {real_path} 也沒有 Demo 1 的 {src}：先跑 ./present.sh fetch（或 Demo 1 的 fetch）")
            real_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, real_path)
        x16 = C.load_wav(real_path, 16000)
        self.real = dsp.resample(x16, 16000, FS)
        self.ref_rms = rms(self.real)
        fv, fb = self.fe["vocos"], self.fe["bigvgan"]
        self.inputs["real"] = {"own": fv.mel(self.real, FS), "vocos": None, "bigvgan": fb.mel(self.real, FS), "own_fe": "vocos"}
        self.inputs["real"]["vocos"] = self.inputs["real"]["own"]
        pdir = (HERE / p["dir"]).resolve()
        for row, key in (("speecht5", "speecht5_mel"), ("f5", "f5_mel")):
            f = pdir / p[key]
            if not f.exists():
                self.out(f"  {C.AMBER}沒有 Demo 1 的 {f.name}（{pdir}）：{row} 那一列跳過。Demo 1 先 rehearse。{C.RESET}")
                continue
            mel = np.load(f).astype(np.float64)
            src = self.fe[row] if row == "speecht5" else fv
            self.inputs[row] = {"own": mel, "own_fe": src.key,
                                "vocos": mel if row == "f5" else convert(mel, src, fv),
                                "bigvgan": convert(mel, src, fb)}
            wav = pdir / p[f"{row}_wav"]
            if wav.exists():
                self.demo1[row] = wav

    # ── 算 ──
    def compute(self) -> None:
        if self.real is None:
            self.load_inputs()
        # 暖機：torch 的第一次呼叫含 kernel 編譯／快取，不算進耗時（另記 cold）
        warm = np.full((100, 40), -8.0)
        self.cold = {}
        for k in ("vocos", "bigvgan"):
            v = self.voc[k]
            if not getattr(v, "fake", False):
                _, self.cold[k] = v.decode(warm)
        for row in ROWS:
            if row not in self.inputs:
                continue
            for vk in VOCS:
                self._cell(row, vk)
            # 硬餵：這一列 Vocos 定義的 mel 原樣給 BigVGAN
            self._cell(row, "bigvgan_mismatch")
        self.write_wavs()

    def _cell(self, row: str, vk: str) -> dict:
        inp = self.inputs[row]
        if vk == "bigvgan_mismatch":
            v, mel = self.voc["bigvgan"], inp["vocos"]
        else:
            v, mel = self.voc[vk], inp["vocos" if vk == "gl" else vk]
        y, sec = v.decode(mel)
        T = mel.shape[1]
        remel = v.fe.mel(y, FS)
        rec = {"wav": y, "seconds": sec, "T": T, "audio_s": len(y) / FS, "rtf": sec / (len(y) / FS) if len(y) else None,
               "fit_db": mel_fit_db(remel, mel, v.fe),
               "lsd_db": lsd_db(self.real, y) if row == "real" else None, "remel": remel, "device": v.device, "fake": getattr(v, "fake", False)}
        if vk == "bigvgan_mismatch":
            # 硬餵的 mel 比 BigVGAN 自己的前端會算出來的高多少（平均 dB）：Slaney 面積正規化對 HTK 不正規化，差的是一個常數加一個斜率
            proper = convert(mel, self.fe["vocos"], v.fe)
            rec["given_above_own_scale_db"] = float(20 / LN10 * np.mean(np.maximum(mel[:, :T], FIT_FLOOR) - np.maximum(proper[:, :T], FIT_FLOOR)))
            rec["clip_frac"] = float(np.mean(np.abs(y) > 0.99)) if len(y) else 0.0
        self.cells[(row, vk)] = rec
        return rec

    def compute_mms(self) -> None:
        t = CFG["mms"]["text"]
        plan = [("nan", t["nan_poj"]), ("nan_han", t["nan_han"]), ("hak", t["hak_pfs"]), ("eng", t["eng"])]
        for key, text in plan:
            lang = key.split("_")[0]
            if lang not in self.mms:
                continue
            rec = self.mms[lang].say(text, CFG["mms"]["seed"])
            rec["lang"] = lang
            self.mms_out[key] = rec
        self.write_wavs()

    # ── 檔 ──
    def wav_path(self, name: str) -> Path:
        return self.folder / f"{name}.wav"

    def _playable(self, y: np.ndarray) -> tuple[np.ndarray, float]:
        g = self.ref_rms / rms(y) if len(y) and rms(y) > 0 else 1.0
        z = y * g
        return z, max(float(np.abs(z).max()), 1e-9) / 0.89

    def write_wavs(self) -> None:
        if self.real is not None and not self.wav_path("real").exists():
            z, pk = self._playable(self.real)
            C.save_wav(self.wav_path("real"), z, FS, peak=pk)
        for (row, vk), rec in self.cells.items():
            p = self.wav_path(f"{row}_{vk}")
            if not p.exists() and len(rec["wav"]):
                z, pk = self._playable(rec["wav"])
                C.save_wav(p, z, FS, peak=pk)
        for key, rec in self.mms_out.items():
            p = self.wav_path(f"mms_{key}")
            if not p.exists() and len(rec["wav"]):
                z, pk = self._playable(rec["wav"])
                C.save_wav(p, z, rec["fs"], peak=pk)

    # ── 視窗 ──
    def make_fig(self):
        import matplotlib.pyplot as plt
        setup_fonts()
        self.plt = plt
        d = CFG["display"]
        self.fig = plt.figure(figsize=(d["fig_w"], d["fig_h"]))
        self.fig.canvas.mpl_connect("key_press_event", self.on_key)
        self.draw()
        return self.fig

    def redraw(self):
        if self.fig is not None:
            self.fig.canvas.draw_idle()
            if self.plt.get_backend().lower() not in ("agg", "pdf", "svg"):     # rehearse 用 Agg，沒有視窗可以 pause
                self.plt.pause(0.05)

    def _mel_panel(self, ax, logmel: np.ndarray, fe: FrontEnd, title: str, color: str = "black"):
        d = CFG["display"]
        S = fe.to_db(logmel)
        top = float(np.percentile(S, 99.5))
        ax.imshow(S, origin="lower", aspect="auto", cmap="magma", vmin=top - d["db_range"], vmax=top,
                  extent=[0, logmel.shape[1] / fe.fps, 0, fe.n_mels])
        ticks = [1000, 2000, 4000, 8000] if fe.fs > 16000 else [1000, 2000, 4000]
        ax.set_yticks([float(np.interp(h, fe.centers, np.arange(fe.n_mels))) for h in ticks])
        ax.set_yticklabels([f"{h // 1000}k" for h in ticks], fontsize=8)
        ax.set_xlabel("s", fontsize=8)
        ax.tick_params(labelsize=8)
        kw = {"fontproperties": cjk_font()} if has_cjk(title) and cjk_font() is not None else {}
        ax.set_title(title, fontsize=9.5, color=color, loc="left", **kw)

    def _cell_title(self, row: str, vk: str) -> str:
        rec = self.cells.get((row, vk))
        if rec is None:
            return f"{VOC_LABEL.get(vk, vk)} — not computed"
        parts = [f"fit {rec['fit_db']:.1f} dB"]
        if rec["lsd_db"] is not None:
            parts.append(f"LSD {rec['lsd_db']:.1f} dB")
        parts.append(f"{rec['seconds']:.2f} s (RTF {rec['rtf']:.3f})" if rec["rtf"] else f"{rec['seconds']:.2f} s")
        lab = VOC_LABEL["bigvgan"] + " ← Vocos mel, unconverted" if vk == "bigvgan_mismatch" else VOC_LABEL[vk]
        return f"{'FAKE ' if rec.get('fake') else ''}{lab}\n" + " · ".join(parts)

    def draw(self):
        if self.fig is None:
            return
        self.fig.clf()
        kind, arg = self.mode
        if kind == "mms":
            rec = self.mms_out.get(arg)
            ax = self.fig.add_subplot(1, 1, 1)
            if rec is None or not len(rec["wav"]):
                ax.axis("off")
                msg = rec["error"] if rec else "not computed"
                ax.text(0.5, 0.6, f"MMS-TTS {arg.split('_')[0]}: {msg}", ha="center", fontsize=16, color="firebrick", transform=ax.transAxes)
                body = f"input: {rec['text'] if rec else ''}\nkept: {rec['kept']!r}   dropped: {' '.join(rec['dropped']) if rec else ''}"
                kw = {"fontproperties": cjk_font()} if has_cjk(body) and cjk_font() is not None else {}
                ax.text(0.5, 0.4, body, ha="center", fontsize=15, transform=ax.transAxes, **kw)
            else:
                fe16 = FrontEnd("mms16", rec["fs"], dsp.mel_filterbank(rec["fs"], N_FFT, 80, "slaney", area_norm=True), 1.0, math.log(1e-7), "", 0.0, rec["fs"] / 2)
                self._mel_panel(ax, fe16.numpy_mel(rec["wav"]), fe16,
                                f"MMS-TTS {rec['lang']} (VITS, {rec['fs'] // 1000} kHz, straight to waveform) · {rec['audio_s']:.2f} s of audio in {rec['seconds']:.2f} s"
                                f"\ninput: {rec['text']}\ntokenizer kept: {rec['kept']}   dropped: {' '.join(rec['dropped']) or '—'}")
            self.fig.tight_layout()
            self.redraw()
            return
        row = self.row
        inp = self.inputs.get(row)
        axes = [self.fig.add_subplot(1, 4, i + 1) for i in range(4)]
        if inp is None:
            axes[0].text(0.5, 0.5, f"{row}: no input", ha="center", transform=axes[0].transAxes)
        else:
            own_fe = self.fe[inp["own_fe"]]
            self._mel_panel(axes[0], inp["own"], own_fe, f"input: {ROW_LABEL[row]}\n{own_fe.label.split(' · ')[0]}, {own_fe.n_mels} bins, {own_fe.fs // 1000} kHz")
            for ax, vk in zip(axes[1:], VOCS):
                rec = self.cells.get((row, vk))
                if rec is None:
                    ax.axis("off")
                    continue
                v = self.voc[vk]
                hl = (self.last == (row, vk))
                self._mel_panel(ax, rec["remel"], v.fe, self._cell_title(row, vk), color="firebrick" if hl else "black")
        self.fig.suptitle(f"W8 Demo 3 · one mel, three vocoders · row: {ROW_LABEL.get(row, row)}   —   outputs re-analysed with each vocoder's own front end; "
                          f"fit = RMS log-mel error vs the mel given" + (f"   [{self.tag}]" if self.tag else ""), fontsize=9.5)
        self.fig.tight_layout()
        self.redraw()

    # ── 鍵 ──
    def _play(self, path: Path, what: str) -> None:
        self.out(f"  ▶ {what}")
        C.play(path)

    def play_cell(self, vk: str) -> None:
        row = self.row
        rec = self.cells.get((row, vk))
        if rec is None:
            self.out(f"  {C.AMBER}{row} × {vk}：沒有算（Demo 1 的 mel 不在？）{C.RESET}")
            return
        self.last = (row, vk)
        self.mode = ("grid", None)
        self.draw()
        lab = "BigVGAN ← Vocos-defined mel, unconverted" if vk == "bigvgan_mismatch" else VOC_LABEL[vk]
        s = f"{ROW_LABEL[row]} → {lab}: fit {rec['fit_db']:.2f} dB"
        if rec["lsd_db"] is not None:
            s += f", LSD vs original {rec['lsd_db']:.2f} dB"
        s += f", {rec['seconds']:.3f} s for {rec['audio_s']:.2f} s of audio (RTF {rec['rtf']:.3f}, {rec['device']})"
        if vk == "bigvgan_mismatch":
            s += f"; the mel it was handed sits {rec['given_above_own_scale_db']:+.1f} dB above its own scale; {100 * rec['clip_frac']:.1f}% of samples clipped"
        self._play(self.wav_path(f"{row}_{vk}"), s)

    def select_row(self, row: str, play_all: bool) -> None:
        if row not in self.inputs:
            self.out(f"  {C.AMBER}{row} 那一列沒有輸入{C.RESET}")
            return
        self.row = row
        self.mode = ("grid", None)
        self.last = None
        self.draw()
        self.out(f"  列：{ROW_LABEL[row]}")
        if play_all:
            for vk in VOCS:
                self.play_cell(vk)

    def play_original(self) -> None:
        self.last = None
        self._play(self.wav_path("real"), f"original recording ({CFG['audio']['real_utt']}, resampled to {FS // 1000} kHz)")

    def play_demo1(self) -> None:
        if self.row == "real":
            self.play_original()
            return
        p = self.demo1.get(self.row)
        if p is None:
            self.out(f"  {C.AMBER}沒有 Demo 1 的 {self.row} .wav{C.RESET}")
            return
        self._play(p, f"Demo 1's own output for {ROW_LABEL[self.row]} ({'SpeechT5 HiFi-GAN, 16 kHz' if self.row == 'speecht5' else 'Vocos'}): {p.name}")

    def play_mms(self, key: str) -> None:
        rec = self.mms_out.get(key)
        if rec is None:
            lang = key.split("_")[0]
            self.out(f"  {C.AMBER}MMS {lang} 沒有載（fetch 過了嗎？）{C.RESET}")
            return
        self.mode = ("mms", key)
        self.last = ("mms", key)
        self.draw()
        self.out(f"  MMS-TTS {rec['lang']}  input: {rec['text']}")
        self.out(f"    tokenizer: {rec['n_ids']} ids (incl. blanks) → kept {rec['kept']!r}; dropped: {' '.join(rec['dropped']) or '—'}")
        if rec.get("error"):
            self.out(f"    {C.RED}{rec['error']}{C.RESET} — nothing to play")
            return
        self._play(self.wav_path(f"mms_{key}"), f"{rec['audio_s']:.2f} s of audio in {rec['seconds']:.2f} s ({self.mms[rec['lang']].device})")

    def again(self) -> None:
        if self.last is None:
            self.play_original()
        elif self.last[0] == "mms":
            self.play_mms(self.last[1])
        else:
            self.play_cell(self.last[1])

    def table(self) -> None:
        o = self.out
        o.banner("Demo 3 · numbers (fit = RMS log-mel error of the output vs the mel given, dB; LSD vs the original, dB; wall-clock s / RTF)")
        head = f"  {'input mel':<34}" + "".join(f"{VOC_LABEL[v]:>30}" for v in VOCS)
        o(head)
        for row in ROWS:
            if row not in self.inputs:
                continue
            cells = []
            for vk in VOCS:
                r = self.cells.get((row, vk))
                if r is None:
                    cells.append(f"{'—':>30}")
                    continue
                s = f"fit {r['fit_db']:4.1f}"
                if r["lsd_db"] is not None:
                    s += f" LSD {r['lsd_db']:4.1f}"
                s += f" {r['seconds']:6.2f}s"
                cells.append(f"{s:>30}")
            o(f"  {ROW_LABEL[row]:<34}" + "".join(cells))
        o("")
        for row in ROWS:
            r = self.cells.get((row, "bigvgan_mismatch"))
            if r:
                o(f"  {ROW_LABEL[row]:<34} Vocos-defined mel fed to BigVGAN unconverted: fit {r['fit_db']:.1f} dB; handed a mel {r['given_above_own_scale_db']:+.1f} dB above its own scale; {100 * r['clip_frac']:.1f}% clipped")
        o("")
        for vk in ("vocos", "bigvgan"):
            v = self.voc[vk]
            r = self.cells.get(("real", vk))
            if r:
                o(f"  {VOC_LABEL[vk]:<14} {getattr(v, 'n_params', 0) / 1e6:6.1f} M params · {v.device:>5} · real row {r['seconds']:.3f} s = RTF {r['rtf']:.4f}"
                  + (f" · first call {self.cold[vk]:.2f} s" if vk in self.cold else ""))
        g = self.voc["gl"]
        o(f"  {VOC_LABEL['gl']:<14} numpy · pinv + {g.iters} iterations · last inconsistency {g.log[-1]['inconsistency']:.4f}" if hasattr(g, "log") else "")
        if self.mms_out:
            o("")
            for key, r in self.mms_out.items():
                s = f"  MMS {r['lang']:<4} {key:<8} {r['n_ids']:4d} ids  dropped {' '.join(r['dropped']) or '—':<12}"
                s += f" {r['audio_s']:.2f} s audio / {r['seconds']:.2f} s" if not r.get("error") else f" {r['error']}"
                o(s)
        o("")
        o(f"  {C.GREY}fit and LSD are references, not the judge — the ear is. Row conversions: SpeechT5 (80 bins, 16 kHz, log10) and the F5 → BigVGAN cell go through [convert].{C.RESET}")

    def menu(self) -> None:
        print(f"\n{C.BOLD}鍵{C.RESET}  0 原音 ｜ g 真人列  4 SpeechT5 列（播三個）  5 F5 列（播三個） ｜ 1 GL  2 Vocos  3 BigVGAN  m 硬餵  n Demo 1 的聲音 ｜ "
              f"6 nan  7 hak  c 漢字給 nan  e eng ｜ t 表  r 再播  h 選單  q 離開")

    def dispatch(self, k: str) -> bool:
        if k == "0":
            self.play_original()
        elif k == "g":
            self.select_row("real", False)
        elif k == "4":
            self.select_row("speecht5", True)
        elif k == "5":
            self.select_row("f5", True)
        elif k in ("1", "2", "3"):
            self.play_cell(VOCS[int(k) - 1])
        elif k == "m":
            self.play_cell("bigvgan_mismatch")
        elif k == "n":
            self.play_demo1()
        elif k == "6":
            self.play_mms("nan")
        elif k == "7":
            self.play_mms("hak")
        elif k == "c":
            self.play_mms("nan_han")
        elif k == "e":
            self.play_mms("eng")
        elif k == "t":
            self.table()
        elif k == "r":
            self.again()
        elif k == "h":
            self.menu()
        elif k == "q":
            return False
        return True

    def on_key(self, ev) -> None:
        if not self.dispatch(ev.key):
            self.plt.close(self.fig)

    # ── 彩排 ──
    def rehearse_figs(self) -> None:
        import matplotlib
        matplotlib.use("Agg", force=True)
        self.make_fig()
        for row in ROWS:
            if row in self.inputs:
                self.row, self.mode, self.last = row, ("grid", None), None
                self.draw()
                self.fig.savefig(self.folder / f"grid_{row}.png", dpi=110)
        for key in self.mms_out:
            self.mode = ("mms", key)
            self.draw()
            self.fig.savefig(self.folder / f"mms_{key}.png", dpi=110)
        self.plt.close(self.fig)
        self.fig = None

    def save(self) -> None:
        self.write_wavs()
        for row, inp in self.inputs.items():
            np.save(self.folder / f"{row}_input_{inp['own_fe']}_mel.npy", inp["own"].astype(np.float32))
            for k in ("vocos", "bigvgan"):
                if row == "real" or (row == "f5" and k == "vocos"):
                    continue
                np.save(self.folder / f"{row}_to_{k}_mel.npy", inp[k].astype(np.float32))
        for (row, vk), rec in self.cells.items():
            np.save(self.folder / f"{row}_{vk}_remel.npy", rec["remel"].astype(np.float32))
        rec = {"time": C.now(), "tag": self.tag, "versions": versions(), "config": CFG, "ref_rms": self.ref_rms,
               "real": {"utt": CFG["audio"]["real_utt"], "seconds": len(self.real) / FS, "fs": FS, "source_fs": 16000},
               "front_ends": {k: {"label": f.label, "fs": f.fs, "n_mels": f.n_mels, "fmin": f.fmin, "fmax": f.fmax, "floor_ln": f.floor_ln} for k, f in self.fe.items()},
               "vocoders": {k: v.info() for k, v in self.voc.items()},
               "cold_first_call_s": self.cold,
               "cells": {f"{row}/{vk}": {k: (round(v, 6) if isinstance(v, float) else v) for k, v in r.items() if k not in ("wav", "remel")}
                         for (row, vk), r in self.cells.items()},
               "inputs": {row: {"own_fe": inp["own_fe"], "T_own": int(inp["own"].shape[1]), "T_vocos": int(inp["vocos"].shape[1]), "T_bigvgan": int(inp["bigvgan"].shape[1])}
                          for row, inp in self.inputs.items()},
               "mms": {k: {**{kk: vv for kk, vv in r.items() if kk != "wav"}, "model": self.mms[r["lang"]].info()} for k, r in self.mms_out.items()},
               "demo1_wavs": {k: str(p.relative_to(HERE.parent)) for k, p in self.demo1.items()}}
        (self.folder / f"{self.tag or 'session'}.json").write_text(json.dumps(rec, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
        (self.folder / f"{self.tag or 'session'}.txt").write_text(self.out.text(), encoding="utf-8")
        C.note(f"  → {self.folder}/")


CJK_FONT_FILES = (
    # macOS（PingFang 是 .ttc，matplotlib 用名稱找不一定對得上，所以用路徑）
    "/System/Library/Fonts/PingFang.ttc", "/System/Library/Fonts/STHeiti Medium.ttc", "/System/Library/Fonts/Hiragino Sans GB.ttc",
    "/System/Library/Fonts/Supplemental/Songti.ttc", "/Library/Fonts/Arial Unicode.ttf",
    # Linux
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc", "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc", "/usr/share/fonts/opentype/ipafont-gothic/ipag.ttf",
)
CJK_FONT_NAMES = ("PingFang TC", "PingFang SC", "Heiti TC", "Hiragino Sans GB", "Noto Sans CJK TC", "Noto Sans CJK JP", "WenQuanYi Zen Hei", "IPAGothic")
_CJK = {}


def cjk_font():
    """系統裡一個有漢字的字型（FontProperties），先用路徑、再用名稱找；沒有就 None（漢字會是方塊，聲音不受影響）。"""
    if "fp" in _CJK:
        return _CJK["fp"]
    from matplotlib import font_manager
    fp = None
    for f in CJK_FONT_FILES:
        if Path(f).exists():
            try:
                font_manager.fontManager.addfont(f)
                fp = font_manager.FontProperties(fname=f)
                fp.get_name()
                break
            except Exception:
                fp = None
    if fp is None:
        have = {f.name for f in font_manager.fontManager.ttflist}
        name = next((n for n in CJK_FONT_NAMES if n in have), None)
        if name:
            fp = font_manager.FontProperties(family=name)
    _CJK["fp"] = fp
    return fp


def has_cjk(text: str) -> bool:
    return any(ord(ch) > 0x2E7F for ch in text)


def setup_fonts() -> None:
    """圖上有漢字（MMS 那句、漢字輸入）：找一個 CJK 字型；含漢字的文字由 _mel_panel／draw 明確指定它（不靠 fallback——
    2026-11-08 Mac 上用名稱設 font.sans-serif 備援沒生效，整句豆腐）。"""
    fp = cjk_font()
    import matplotlib
    if fp is not None and fp.get_name():
        matplotlib.rcParams["font.sans-serif"] = ["DejaVu Sans", fp.get_name()] + list(matplotlib.rcParams["font.sans-serif"])
    warnings.filterwarnings("ignore", message=r"Glyph .* missing from font")
    warnings.filterwarnings("ignore", message=r"FigureCanvasAgg is non-interactive")


def versions() -> dict:
    v = C.versions()
    from importlib import metadata
    for pkg in ("torch", "torchaudio", "transformers", "vocos", "librosa", "huggingface_hub"):
        try:
            v[pkg] = metadata.version(pkg)
        except metadata.PackageNotFoundError:
            v[pkg] = None
    v["PYTORCH_ENABLE_MPS_FALLBACK"] = os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK")
    return v


# ── replay ───────────────────────────────────────────────────
class ReplayVoc:
    def __init__(self, key: str, fe: FrontEnd, info: dict, device: str):
        self.key, self.fe, self._info, self.device, self.fake = key, fe, info, device, info.get("fake", False)
        self.label = VOC_LABEL[key]
        self.n_params = info.get("n_params", 0)
        self.iters = info.get("iters", CFG["griffin_lim"]["iters"])

    def info(self):
        return self._info


class ReplayMms:
    def __init__(self, lang: str, info: dict):
        self.lang, self._info, self.device, self.fake = lang, info, info.get("device", "replay"), info.get("fake", False)

    def info(self):
        return self._info


def build_replay(out: Out) -> Demo:
    rec_path = REHEARSAL / "rehearsal.json"
    if not rec_path.exists():
        sys.exit(f"沒有 {rec_path}：先在有模型的機器上 ./present.sh rehearse")
    rec = json.loads(rec_path.read_text(encoding="utf-8"))
    fe = numpy_front_ends()
    voc = {k: ReplayVoc(k, fe["vocos" if k == "gl" else k], rec["vocoders"][k], rec["vocoders"][k].get("device", "replay")) for k in VOCS}
    mms = {r["lang"]: ReplayMms(r["lang"], r["model"]) for r in rec["mms"].values()}
    d = Demo(out, fe, voc, mms, REHEARSAL, tag="replay")
    d.real = C.load_wav(REHEARSAL / "real.wav", FS)
    d.ref_rms = rec["ref_rms"]
    d.cold = rec.get("cold_first_call_s", {})
    for row, meta in rec["inputs"].items():
        own = np.load(REHEARSAL / f"{row}_input_{meta['own_fe']}_mel.npy").astype(np.float64)
        inp = {"own": own, "own_fe": meta["own_fe"]}
        for k in ("vocos", "bigvgan"):
            f = REHEARSAL / f"{row}_to_{k}_mel.npy"
            inp[k] = np.load(f).astype(np.float64) if f.exists() else (own if (row == "f5" and k == "vocos") else None)
        d.inputs[row] = inp
    for key, c in rec["cells"].items():
        row, vk = key.split("/")
        remel = REHEARSAL / f"{row}_{vk}_remel.npy"
        d.cells[(row, vk)] = {**c, "wav": np.zeros(1), "remel": np.load(remel).astype(np.float64) if remel.exists() else np.zeros((100, 1))}
    for key, m in rec["mms"].items():
        d.mms_out[key] = {**{k: v for k, v in m.items() if k != "model"}, "wav": np.zeros(1) if not m.get("error") else np.zeros(0)}
    for row, p in rec.get("demo1_wavs", {}).items():
        d.demo1[row] = (HERE.parent / p).resolve()
    out(f"  replay：{rec['time']} 的彩排（{rec['versions'].get('platform', '?')}）")
    return d


def build(cmd: str, out: Out, folder: Path | None = None) -> Demo:
    fe = numpy_front_ends()
    if cmd == "fake":
        voc = {"gl": GriffinLim(fe["vocos"]), "vocos": FakeVocoder("vocos", fe["vocos"], seed=1), "bigvgan": FakeVocoder("bigvgan", fe["bigvgan"], seed=2)}
        mms = {l: FakeMms(l) for l in MMS_LANGS}
        return Demo(out, fe, voc, mms, folder or HERE / "runs" / "live" / (C.now().replace(":", "") + "-fake"), tag="FAKE")
    quiet_hf()
    inf = CFG["infer"]
    t0 = time.perf_counter()
    vocos = VocosVocoder(inf["vocos_device"])
    out(f"  Vocos 載好：{vocos.n_params / 1e6:.1f} M 參數，{vocos.device}（{time.perf_counter() - t0:.1f} s）")
    t0 = time.perf_counter()
    bigvgan = BigVGANVocoder(inf["bigvgan_device"])
    out(f"  BigVGAN 載好：{bigvgan.n_params / 1e6:.1f} M 參數，{bigvgan.device}（{time.perf_counter() - t0:.1f} s）")
    fe["vocos"], fe["bigvgan"] = vocos.fe, bigvgan.fe
    try:
        M5 = speecht5_filterbank()
        fe["speecht5"] = FrontEnd("speecht5", 16000, M5, LN10, math.log(1e-10), fe["speecht5"].label, 80.0, 7600.0)
    except Exception as e:
        out(f"  {C.AMBER}SpeechT5 的 preprocessor_config.json 不在（{type(e).__name__}）：用 numpy 近似的 filterbank 換算。fetch 一次就有。{C.RESET}")
    voc = {"gl": GriffinLim(vocos.fe), "vocos": vocos, "bigvgan": bigvgan}
    mms = {}
    for l in MMS_LANGS:
        try:
            t0 = time.perf_counter()
            mms[l] = Mms(l, inf["mms_device"])
            out(f"  MMS-TTS {l} 載好：{mms[l].n_params / 1e6:.1f} M，vocab {len(mms[l].vocab)}，{mms[l].device}（{time.perf_counter() - t0:.1f} s）")
        except Exception as e:
            out(f"  {C.AMBER}MMS-TTS {l} 載不起來：{type(e).__name__}: {str(e)[:100]}{C.RESET}")
    folder = REHEARSAL if cmd == "rehearse" else HERE / "runs" / "live" / C.now().replace(":", "")
    return Demo(out, fe, voc, mms, folder, tag="rehearsal" if cmd == "rehearse" else "")


# ── check ────────────────────────────────────────────────────
def check() -> None:
    import torch
    C.banner("check：版本、裝置、三個網路在 cpu／mps 上的數值差與耗時、換算的誤差")
    rec = {"time": C.now(), "versions": versions(), "mps_available": bool(torch.backends.mps.is_available()),
           "model_lock": read_lock(), "fallback_warnings": []}
    for k, v in rec["versions"].items():
        print(f"  {k:<28} {v}")
    print(f"  MPS available: {rec['mps_available']}；PYTORCH_ENABLE_MPS_FALLBACK={rec['versions']['PYTORCH_ENABLE_MPS_FALLBACK']}（present.sh check 固定給 0）")
    quiet_hf()
    devices = ["cpu"] + (["mps"] if rec["mps_available"] else [])
    CHECK_DIR.mkdir(parents=True, exist_ok=True)
    # 真人錄音
    real_path = HERE / CFG["audio"]["real_path"]
    if not real_path.exists():
        shutil.copy2((HERE / CFG["audio"]["real_source"]).resolve(), real_path)
    x = dsp.resample(C.load_wav(real_path, 16000), 16000, FS)
    out_wavs = {}
    real_fe = numpy_front_ends()
    with warnings.catch_warnings(record=True) as ws:
        warnings.simplefilter("always")
        for name, cls in (("vocos", VocosVocoder), ("bigvgan", BigVGANVocoder)):
            r = {}
            mel = None
            for dev in devices:
                try:
                    t0 = time.perf_counter()
                    v = cls(dev)
                    load_s = time.perf_counter() - t0
                    if mel is None:
                        mel = v.fe.mel(x, FS)
                        real_fe[name] = v.fe
                    _, cold = v.decode(mel)
                    y, warm = v.decode(mel)
                    r[dev] = {"load_s": load_s, "first_call_s": cold, "second_call_s": warm, "rtf": warm / (len(y) / FS), "n_params": v.n_params}
                    out_wavs[(name, dev)] = y
                    C.save_wav(CHECK_DIR / f"{name}_{dev}.wav", y, FS, peak=max(float(np.abs(y).max()), 1e-9) / 0.89)
                    print(f"  {name:<8} {dev:<4} 載 {load_s:5.1f} s · 第一次 {cold:6.3f} s · 第二次 {warm:6.3f} s（RTF {r[dev]['rtf']:.4f}）")
                except Exception as e:
                    r[dev] = {"error": f"{type(e).__name__}: {str(e)[:160]}"}
                    print(f"  {C.RED}{name:<8} {dev:<4} {r[dev]['error']}{C.RESET}")
            if all(d in out_wavs for d in [(name, d) for d in devices]) and len(devices) == 2:
                a, b = out_wavs[(name, "cpu")], out_wavs[(name, "mps")]
                n = min(len(a), len(b))
                r["cpu_vs_mps_max_abs_diff"] = float(np.abs(a[:n] - b[:n]).max())
                r["cpu_vs_mps_lsd_db"] = lsd_db(a[:n], b[:n])
                print(f"           cpu 對 mps：最大差 {r['cpu_vs_mps_max_abs_diff']:.2e}，LSD {r['cpu_vs_mps_lsd_db']:.3f} dB → runs/check/{name}_cpu.wav、{name}_mps.wav 用耳朵比")
            rec[name] = r
        # MMS：RNG 在 cpu 與 mps 上不同，只量跑不跑、多久、多長
        mms_rec = {}
        text = CFG["mms"]["text"]["nan_poj"]
        for dev in devices:
            try:
                m = Mms("nan", dev)
                r1 = m.say(text, CFG["mms"]["seed"])
                r2 = m.say(text, CFG["mms"]["seed"])
                same = bool(len(r1["wav"]) == len(r2["wav"]) and np.allclose(r1["wav"], r2["wav"]))
                mms_rec[dev] = {"seconds": r1["seconds"], "audio_s": r1.get("audio_s"), "n_ids": r1["n_ids"], "dropped": r1["dropped"], "same_seed_same_output": same}
                print(f"  MMS nan  {dev:<4} {r1['n_ids']} ids，{r1.get('audio_s', 0):.2f} s 音訊 / {r1['seconds']:.2f} s；同 seed 兩次相同：{same}；丟掉：{' '.join(r1['dropped']) or '—'}")
                han = m.say(CFG["mms"]["text"]["nan_han"], 1)
                mms_rec[dev]["han_n_ids"] = han["n_ids"]
                mms_rec[dev]["han_error"] = han.get("error")
                print(f"           漢字那句：{han['n_ids']} ids → {han.get('error')}")
            except Exception as e:
                mms_rec[dev] = {"error": f"{type(e).__name__}: {str(e)[:160]}"}
                print(f"  {C.RED}MMS nan  {dev:<4} {mms_rec[dev]['error']}{C.RESET}")
        rec["mms_nan"] = mms_rec
        rec["fallback_warnings"] = sorted({str(w.message)[:200] for w in ws if "fall back" in str(w.message).lower() or "fallback" in str(w.message).lower()})
    if rec["fallback_warnings"]:
        print(f"  {C.AMBER}MPS fallback 警告：{rec['fallback_warnings']}{C.RESET}")
    # 換算在真人錄音上的誤差（Demo 1 算的兩份 real mel 對本 demo 的換算）
    try:
        real_fe["speecht5"] = FrontEnd("speecht5", 16000, speecht5_filterbank(), LN10, math.log(1e-10), real_fe["speecht5"].label, 80.0, 7600.0)
    except Exception as e:
        print(f"  {C.AMBER}SpeechT5 的 preprocessor_config.json 不在（{type(e).__name__}）：換算檢查用 numpy 近似的 filterbank{C.RESET}")
    rec["conversion"] = conversion_check(print_it=True, fe=real_fe)
    (HERE / "runs").mkdir(exist_ok=True)
    (HERE / "runs" / "check.json").write_text(json.dumps(rec, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    C.note(f"  → runs/check.json；runs/check/*.wav 兩個裝置各一份，用耳朵比（§6）")


def conversion_check(print_it: bool = False, fe: dict[str, FrontEnd] | None = None) -> dict:
    """同一段真人錄音在不同前端各有一份 mel（Demo 1 算的 SpeechT5／F5 兩份；本 demo 的 BigVGAN 前端）：換算之後對得多準。"""
    fe = fe or numpy_front_ends()
    p = CFG["predicted"]
    pdir = (HERE / p["dir"]).resolve()
    out = {}
    try:
        r5 = np.load(pdir / p["real_speecht5_mel"]).astype(np.float64)
        rv = np.load(pdir / p["real_f5_mel"]).astype(np.float64)
    except FileNotFoundError as e:
        out["skipped"] = f"Demo 1 的 real mel 不在：{e.filename}"
        if print_it:
            print(f"  {C.AMBER}{out['skipped']}{C.RESET}")
        return out
    fv, fb, f5 = fe["vocos"], fe["bigvgan"], fe["speecht5"]
    c = convert(r5, f5, fv)
    inband = fv.centers <= 7600
    out["speecht5_to_vocos"] = {"fit_db": mel_fit_db(c, rv, fv), "fit_db_below_7600": mel_fit_db(c[inband], rv[inband], fv), "lag_frames": energy_lag(c, rv, fv),
                                "T": [int(r5.shape[1]), int(c.shape[1]), int(rv.shape[1])]}
    x = dsp.resample(C.load_wav(HERE / CFG["audio"]["real_path"], 16000), 16000, FS) if (HERE / CFG["audio"]["real_path"]).exists() else None
    if x is not None:
        rb = fb.mel(x, FS)
        cb = convert(rv, fv, fb)
        out["vocos_to_bigvgan"] = {"fit_db": mel_fit_db(cb, rb, fb), "lag_frames": energy_lag(cb, rb, fb)}
        n = min(rv.shape[1], rb.shape[1])
        out["vocos_mel_vs_bigvgan_mel_unconverted_db"] = mel_fit_db(rv[:, :n], rb[:, :n], fb)
        out["vocos_mel_minus_bigvgan_mel_mean_db"] = float(20 / LN10 * np.mean(np.maximum(rv[:, :n], FIT_FLOOR) - np.maximum(rb[:, :n], FIT_FLOOR)))
    ident = convert(rv, fv, fv)
    out["vocos_to_vocos_identity_db"] = mel_fit_db(ident, rv, fv)
    if print_it:
        s = out["speecht5_to_vocos"]
        print(f"  換算 SpeechT5 mel → Vocos mel（真人錄音，Demo 1 的兩份）：fit {s['fit_db']:.1f} dB，7.6 kHz 以下 {s['fit_db_below_7600']:.1f} dB，lag {s['lag_frames']} frames（要 0）；frames {s['T']}")
        if "vocos_to_bigvgan" in out:
            v = out["vocos_to_bigvgan"]
            print(f"  換算 Vocos mel → BigVGAN mel：fit {v['fit_db']:.1f} dB，lag {v['lag_frames']}；**不換算**直接比兩份「100-band log-mel」：差 {out['vocos_mel_vs_bigvgan_mel_unconverted_db']:.1f} dB（平均 {out['vocos_mel_minus_bigvgan_mel_mean_db']:+.1f} dB）")
        print(f"  Vocos → Vocos 恆等：{out['vocos_to_vocos_identity_db']:.3f} dB（要 ≈ 0）")
    return out


# ── fetch ────────────────────────────────────────────────────
def fetch() -> None:
    from huggingface_hub import HfApi, snapshot_download
    C.banner("fetch：下載 Vocos、BigVGAN（含程式碼）、MMS-TTS nan／hak／eng、SpeechT5 的前端設定；鎖 commit → model.lock；複製 Demo 1 的真人錄音")
    api = HfApi()
    lock = read_lock()
    lines = []
    for key, repo in REPOS.items():
        sha = lock.get(repo) or api.model_info(repo).sha
        t0 = time.perf_counter()
        d = snapshot_download(repo, revision=sha, allow_patterns=PATTERNS[key])
        size = sum(f.stat().st_size for f in Path(d).rglob("*") if f.is_file()) / 1e6
        lic = "?"
        rd = Path(d) / "README.md"
        if rd.exists():
            for line in rd.read_text(encoding="utf-8", errors="replace").splitlines()[:40]:
                if line.lower().startswith("license"):
                    lic = line.split(":", 1)[-1].strip()
                    break
        print(f"  {repo:<44} {sha[:10]}  {size:7.1f} MB  license: {lic}  ({time.perf_counter() - t0:.1f} s)")
        lines.append(f"{repo} {sha}")
    MODEL_LOCK.write_text("\n".join(lines) + "\n", encoding="utf-8")
    real_path = HERE / CFG["audio"]["real_path"]
    src = (HERE / CFG["audio"]["real_source"]).resolve()
    if not src.exists():
        sys.exit(f"沒有 Demo 1 的 {src}：先跑 ../w08_d1_mean_vs_sample/present.sh fetch")
    real_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, real_path)
    (real_path.parent / "real.txt").write_text(f"{CFG['audio']['real_utt']}\n{CFG['audio']['real_text']}\nLibriSpeech dev-clean, CC BY 4.0; copied from {CFG['audio']['real_source']}\n", encoding="utf-8")
    print(f"  {real_path.relative_to(HERE)} ← {src.relative_to(HERE.parent)}")
    pdir = (HERE / CFG["predicted"]["dir"]).resolve()
    for k in ("speecht5_mel", "f5_mel", "speecht5_wav", "f5_wav", "real_speecht5_mel", "real_f5_mel"):
        f = pdir / CFG["predicted"][k]
        print(f"  Demo 1 {k:<18} {'✓' if f.exists() else C.RED + '✗ 沒有（Demo 1 先 rehearse）' + C.RESET} {f.relative_to(HERE.parent)}")
    C.note(f"  → {MODEL_LOCK.relative_to(HERE)}；下一步：./present.sh check")


# ── export ───────────────────────────────────────────────────
def export(rec_path: Path) -> None:
    rec = json.loads(rec_path.read_text(encoding="utf-8"))
    cells = {}
    for key, c in rec["cells"].items():
        cells[key] = {k: c.get(k) for k in ("fit_db", "lsd_db", "seconds", "rtf", "audio_s", "T", "device", "fake", "given_above_own_scale_db", "clip_frac") if c.get(k) is not None}
    demo3 = {"source": str(rec_path.relative_to(HERE.parents[1])), "time": rec["time"], "tag": rec["tag"],
             "real": rec["real"], "front_ends": rec["front_ends"], "vocoders": rec["vocoders"], "cold_first_call_s": rec.get("cold_first_call_s", {}),
             "cells": cells, "inputs": rec["inputs"],
             "mms": {k: {kk: m.get(kk) for kk in ("lang", "text", "n_ids", "kept", "dropped", "audio_s", "seconds", "error")} | {"model": m["model"]} for k, m in rec["mms"].items()},
             "versions": {k: rec["versions"].get(k) for k in ("platform", "python", "torch", "torchaudio", "transformers", "vocos", "librosa", "numpy")},
             "definitions": {"fit_db": "RMS of (log-mel of the vocoder output, re-analysed with that vocoder's own front end) minus (the log-mel it was given), both floored at ln 1e-4, in dB = 20·Δln/ln 10",
                             "lsd_db": "log-spectral distance to the original recording at 24 kHz (25 ms / 10 ms / n_fft 1024), real row only",
                             "seconds": "wall-clock of one decode after a warm-up call; Griffin-Lim includes the pseudo-inverse",
                             "conversion": CFG["convert"]}}
    if METRICS_JSON.exists():
        data = json.loads(METRICS_JSON.read_text(encoding="utf-8"))
    else:
        METRICS_JSON.parent.mkdir(parents=True, exist_ok=True)
        data = {"_note": "由各 demo 的 export 寫；手改會被覆蓋"}
    data["demo3"] = demo3
    METRICS_JSON.write_text(json.dumps(data, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    C.note(f"  → {METRICS_JSON.relative_to(HERE.parents[1])} → demo3")


# ── selftest ─────────────────────────────────────────────────
def selftest() -> None:
    C.banner("selftest（不開視窗、不需要模型）")
    fails = []

    def ok(cond: bool, msg: str):
        print(f"  {'✓' if cond else C.RED + '✗'} {msg}{C.RESET}")
        if not cond:
            fails.append(msg)

    fe = numpy_front_ends()
    ok(fe["vocos"].M.shape == (100, 513) and fe["bigvgan"].M.shape == (100, 513) and fe["speecht5"].M.shape == (80, 513), "三個 filterbank 的形狀 (100,513)(100,513)(80,513)")
    ok(all(np.all(np.diff(f.centers) > 0) for f in fe.values()), "三個前端的 bin 中心頻率單調遞增")
    ok(fe["vocos"].M.max() > 0.5 and fe["bigvgan"].M.max() < 0.1, f"Vocos 的 filterbank 峰值 ≈ 1（不正規化，{fe['vocos'].M.max():.2f}）、BigVGAN 的面積正規化（峰值 {fe['bigvgan'].M.max():.3f} ≪ 1）")
    # 換算：恆等、以及 HTK ↔ Slaney 來回
    rng = np.random.default_rng(0)
    smooth = np.cumsum(rng.standard_normal((100, 50)) * 0.05, axis=1) - 2.0
    ident = convert(smooth, fe["vocos"], fe["vocos"])
    ok(mel_fit_db(ident, smooth, fe["vocos"]) < 1e-6, f"Vocos → Vocos 恆等：{mel_fit_db(ident, smooth, fe['vocos']):.2e} dB")
    back = convert(convert(smooth, fe["vocos"], fe["bigvgan"]), fe["bigvgan"], fe["vocos"])
    ok(mel_fit_db(back, smooth, fe["vocos"]) < 1.5, f"Vocos → BigVGAN → Vocos 來回：{mel_fit_db(back, smooth, fe['vocos']):.2f} dB（< 1.5）")
    sm5 = convert(smooth, fe["vocos"], fe["speecht5"])
    ok(sm5.shape == (80, int(round(50 * 62.5 / 93.75))), f"Vocos → SpeechT5 的形狀 {sm5.shape}（80, 33）：frame rate 93.75 → 62.5")
    # 真人錄音上的換算（Demo 1 的兩份 real mel 在的話）
    cc = conversion_check(print_it=False, fe=fe)
    if "skipped" in cc:
        print(f"  {C.GREY}– {cc['skipped']}（換算對真人錄音的檢查跳過）{C.RESET}")
    else:
        s = cc["speecht5_to_vocos"]
        ok(s["lag_frames"] == 0, f"SpeechT5 mel → Vocos mel（真人錄音）時間對齊 lag {s['lag_frames']} frames")
        ok(s["fit_db_below_7600"] < 10.0, f"SpeechT5 mel → Vocos mel（真人錄音）7.6 kHz 以下 fit {s['fit_db_below_7600']:.1f} dB（< 10；window 64 ms 對 43 ms、尺度不同，不會是 0）")
        if "vocos_to_bigvgan" in cc:
            ok(cc["vocos_to_bigvgan"]["fit_db"] < 5.0, f"Vocos mel → BigVGAN mel（真人錄音）fit {cc['vocos_to_bigvgan']['fit_db']:.1f} dB（< 5）")
            ok(cc["vocos_mel_vs_bigvgan_mel_unconverted_db"] > 10.0, f"不換算、兩份「100-band log-mel」直接比：差 {cc['vocos_mel_vs_bigvgan_mel_unconverted_db']:.1f} dB（> 10——同名不同矩陣）")
    # Griffin-Lim：合成母音，inconsistency 單調不增、fit 合理
    fs = FS
    vow = dsp.synth_vowel(fs, 1.0, 120.0, (730, 1090, 2440))
    mel = fe["vocos"].numpy_mel(vow)
    gl = GriffinLim(fe["vocos"], iters=32)
    y, sec = gl.decode(mel)
    inc = [l["inconsistency"] for l in gl.log]
    ok(all(b <= a + 1e-9 for a, b in zip(inc, inc[1:])), f"Griffin-Lim 的 inconsistency 單調不增（{inc[0]:.3f} → {inc[-1]:.4f}）")
    ok(mel_fit_db(fe["vocos"].numpy_mel(y), mel, fe["vocos"]) < 4.5, f"Griffin-Lim 32 輪的 fit {mel_fit_db(fe['vocos'].numpy_mel(y), mel, fe['vocos']):.2f} dB（< 4.5；合成母音、pinv 的 100 → 513 本來就有損）")
    ok(len(y) == (mel.shape[1] - 1) * HOP, f"Griffin-Lim 的輸出長度 {len(y)} = (T − 1)·hop")
    # 白話字
    t = CFG["mms"]["text"]
    ok("̍" in t["nan_poj"] and "̍" in t["hak_pfs"], "台語與客語的句子都有第八聲的組合符號 U+030D")
    ok(all(ord(ch) > 0x2FFF or ch in "，。" for ch in t["nan_han"].replace(" ", "")), "漢字那句全是漢字與全形標點（tokenizer 應該一個都不留）")
    try:
        vocab = set(json.loads((Path(model_dir("mms_nan")) / "vocab.json").read_text(encoding="utf-8")))
        letters = {ch for ch in t["nan_poj"].lower() if not ch.isspace() and ch not in ",.!?"}
        missing = sorted(letters - vocab)
        ok(not missing, f"白話字句子的字元都在 mms-tts-nan 的 vocab 裡" + (f"（缺 {missing}）" if missing else f"（{len(letters)} 種）"))
        ok(not ({ch for ch in t['nan_han']} & vocab), "漢字那句沒有任何字元在 vocab 裡")
    except Exception:
        print(f"  {C.GREY}– mms-tts-nan 的 vocab.json 不在（fetch 之後才有），字元檢查跳過{C.RESET}")
    # fake 流程
    out = Out()
    d = build("fake", out, folder=HERE / "runs" / "_selftest")
    C.play = lambda *a, **k: None
    d.compute()
    d.compute_mms()
    ok(("real", "gl") in d.cells and ("real", "vocos") in d.cells and ("real", "bigvgan") in d.cells, "fake 流程：真人那一列三格都算了")
    ok(all(d.cells[("real", v)]["lsd_db"] is not None for v in VOCS), "fake 流程：真人那一列有 LSD")
    pred_rows = [r for r in ("speecht5", "f5") if r in d.inputs]
    ok(all((r, "bigvgan") in d.cells for r in pred_rows), f"fake 流程：預測的列 {pred_rows} 都過了 BigVGAN（換算）" if pred_rows else "fake 流程：Demo 1 的 mel 不在，預測的列跳過")
    ok("nan_han" in d.mms_out and d.mms_out["nan_han"]["n_ids"] == 0, "fake 流程：漢字給 nan → 0 個 token")
    for k in ("0", "g", "1", "2", "3", "m", "n", "4", "5", "6", "7", "c", "e", "t", "r", "h"):
        d.dispatch(k)
    ok(True, "fake 流程：16 個鍵都走過（沒有視窗）")
    shutil.rmtree(d.folder, ignore_errors=True)
    # 彩排紀錄對 metrics
    rec_path = REHEARSAL / "rehearsal.json"
    if rec_path.exists() and METRICS_JSON.exists():
        rec = json.loads(rec_path.read_text(encoding="utf-8"))
        m = json.loads(METRICS_JSON.read_text(encoding="utf-8")).get("demo3", {})
        ok(m.get("time") == rec["time"], f"w08-metrics.json → demo3 與 rehearsal.json 同一次（{rec['time']}）")
        same = all(abs(m["cells"][k]["fit_db"] - rec["cells"][k]["fit_db"]) < 1e-6 for k in rec["cells"] if k in m.get("cells", {}))
        ok(same, "metrics 的 fit 與 rehearsal.json 逐格相同")
    else:
        print(f"  {C.GREY}– 還沒有彩排紀錄或 w08-metrics.json，對帳跳過{C.RESET}")
    print()
    if fails:
        sys.exit(f"{C.RED}{len(fails)} 項沒過{C.RESET}")
    print(f"{C.GREEN}selftest 全過{C.RESET}")


# ── main ─────────────────────────────────────────────────────
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", nargs="?", default="show", choices=["show", "rehearse", "replay", "check", "fetch", "fake", "selftest", "export"])
    ap.add_argument("--no-plot", action="store_true", help="只在終端機，不開圖")
    args = ap.parse_args()
    if args.cmd == "selftest":
        return selftest()
    if args.cmd == "fetch":
        return fetch()
    if args.cmd == "check":
        return check()
    if args.cmd == "export":
        return export(REHEARSAL / "rehearsal.json")
    out = Out()
    out.banner(f"W8 Demo 3 · the same mel, three vocoders ({args.cmd})")
    if args.cmd == "replay":
        d = build_replay(out)
    else:
        d = build(args.cmd, out)
        t0 = time.perf_counter()
        d.compute()
        d.compute_mms()
        out(f"  全部格子算好（{time.perf_counter() - t0:.1f} s）：{len(d.cells)} 格 vocoder、{len(d.mms_out)} 句 MMS")
    if args.cmd == "rehearse":
        d.table()
        d.rehearse_figs()
        d.save()
        export(d.folder / "rehearsal.json")
        print("\n下一步：./present.sh 自己按一遍 0 1 2 3 4 5 m n 6 7 c e t；聽到的寫進 p71 的表（what we heard）與 p72【提示】（hak 的腔、nan 的變調），runs/rehearsal/*.png 看過。")
        return
    d.make_fig()
    d.menu()
    try:
        if args.no_plot:
            C.note("  --no-plot：鍵在終端機按。")
            while d.dispatch(C.getkey()):
                pass
        else:
            C.note("  鍵在視窗裡按（視窗要有焦點）。終端機這裡是紀錄。")
            d.plt.show()
    finally:
        if args.cmd in ("show", "fake"):
            d.save()


if __name__ == "__main__":
    main()

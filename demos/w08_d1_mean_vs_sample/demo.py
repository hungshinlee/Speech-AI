# -*- coding: utf-8 -*-
# 此檔由 bin/publish_demos.py 從 private repo 的 <course>/demos/ 複製而來，請勿在公開 repo 直接編輯。
"""W8 demo 1：回歸的平均聲與抽樣的變異——同一句話，SpeechT5（回歸 mel）與 F5-TTS（從噪音抽 mel），F5 抽三次。

  show       課堂：載兩個模型、把全部版本算好（M5 Max 上約半分鐘），開一個視窗；鍵在視窗裡按（下面的鍵）。離開時存 runs/live/<時間>/
  rehearse   課前：不互動，全部版本算一遍，.wav、mel（.npy）、圖、數字存 runs/rehearsal/（進版控 = 現場的退路），寫 slides/assets/w08/w08-metrics.json → demo1
  replay     現場出狀況時：從 runs/rehearsal/ 讀回，同一個視窗、同樣的鍵，不碰 torch
  check      課前體檢：版本、MPS、兩個模型在 cpu／mps 上載不載得起、同一個輸入的數值差、耗時；F5 在兩個裝置上各抽一次存成 .wav，用耳朵比（§6 第 7 項）→ runs/check.json
  fetch      課前一週：下載模型與 x-vector（鎖 commit → model.lock）、複製 target 錄音、從 dev-clean.tar.gz 解出 prompt（要網路）
  fake       沒有模型也能演練流程：用真人錄音的 mel 冒充兩個模型（畫面標明 FAKE）
  selftest   不開視窗、不需要模型：頻率軸、細紋理指標、F5 的長度公式、fake 流程；有彩排紀錄時對 w08-metrics.json
  export     只重寫 w08-metrics.json → demo1（從 runs/rehearsal/rehearsal.json）

畫面上的鍵（順序照投影片 p12 講稿的【操作】）：
  1  SpeechT5：回歸出來的 mel（左上），播它的 HiFi-GAN 出的聲音
  2  F5-TTS seed 1：抽出來的 mel（右上），播 Vocos 出的聲音
  g  真人錄音：經兩個模型各自的前端（下排），播原音——對照組
  x  只看 2–6 kHz ↔ 全頻帶
  3  F5 再抽兩次（seed 2、3）：換成三張 mel 並排，依序播，印三個長度
  z  三個 seed 的 F0 曲線疊在同一個時間軸（真人錄音虛線）
  s  SpeechT5 換一個 seed（它的 decoder pre-net 在推論時也開著 dropout；問答用）
  t  數字表（長度與它是怎麼決定的、細紋理對真人錄音的比例、seed 之間差多少、耗時）
  r  再播一次     h  選單     q  離開

兩個模型都用自己的前端：SpeechT5 是 80 bins log10-mel（16 kHz、hop 16 ms、80–7600 Hz，Slaney 尺度、Slaney 正規化），
F5 是 100 bins ln-mel（24 kHz、hop 256、0–12 kHz，HTK 尺度、不正規化，Vocos 的定義）。真人錄音也各算一份，比較只在同一個前端裡做。
細紋理（fine detail）：log-mel（dB）減掉自己 3 × 3 的區域平均之後在一個頻帶裡的 RMS，只算有聲音的 frame；報「模型 ÷ 真人錄音」。
它是參考，不是判準——判準是耳朵。
"""
from __future__ import annotations

import argparse
import io
import json
import math
import os
import re
import shutil
import sys
import tarfile
import time
import tomllib
import warnings
import zipfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import common as C  # noqa: E402
import dsp          # noqa: E402

CFG = tomllib.loads((HERE / "demo_config.toml").read_text(encoding="utf-8"))
TXT = CFG["text"]
MODEL_LOCK = HERE / "model.lock"
METRICS_JSON = HERE.parents[1] / "slides" / "assets" / "w08" / "w08-metrics.json"     # parents[1] = speech_ai/
REHEARSAL = HERE / "runs" / "rehearsal"
CHECK_DIR = HERE / "runs" / "check"
ST5_SEEDS = CFG["speecht5"]["seeds"]
F5_SEEDS = CFG["f5"]["seeds"]
MAIN_BAND = tuple(CFG["detail"]["main_band"])
BANDS = [tuple(b) for b in CFG["detail"]["bands"]]
SILENCE_DBFS, SILENCE_CHUNK_MS, PROMPT_PAD_MS = -42.0, 10, 50     # f5_tts.infer.utils_infer：remove_silence_edges(-42) ＋ 50 ms


def band_name(b) -> str:
    return f"{b[0] / 1000:g}–{b[1] / 1000:g} kHz"


# ── 前端（頻率軸）─────────────────────────────────────────────
class FrontEnd:
    """一個模型的 mel 前端：取樣率、hop、每個 bin 的中心頻率與 filter 支撐寬度（從實際的 filterbank 矩陣讀），log 的底數。
    mel 的陣列一律是 (bins, frames)、模型自己的 log 單位；to_db 把它換成 dB（10·log10 power：log10-mel 乘 10、ln-mel 乘 10/ln10——
    兩個前端都是 magnitude 經 filterbank 再取 log，這裡的「dB」只是同一個縮放，比較只在同一個前端裡做）。"""

    def __init__(self, key: str, fs: int, hop: int, fb: np.ndarray, to_db: float, label: str, real: bool = True):
        self.key, self.fs, self.hop, self.to_db, self.label, self.real = key, fs, hop, to_db, label, real
        n_freqs = fb.shape[0]
        freqs = np.linspace(0, fs / 2, n_freqs)
        self.centers = np.array([freqs[np.argmax(fb[:, k])] for k in range(fb.shape[1])])
        sup = [freqs[fb[:, k] > 0] for k in range(fb.shape[1])]
        self.widths = np.array([(s.max() - s.min()) if len(s) else 0.0 for s in sup])
        self._st5_fe = self._f5_mel = None

    @property
    def fps(self) -> float:
        return self.fs / self.hop

    def mel(self, x: np.ndarray, fs_in: int) -> np.ndarray:
        """真人錄音（或任何波形）→ 這個前端的 log-mel (bins, frames)，模型自己的定義。"""
        x = dsp.resample(x, fs_in, self.fs)
        if not self.real:
            return self._numpy_mel(x)
        if self.key == "speecht5":
            if self._st5_fe is None:
                from transformers import SpeechT5FeatureExtractor
                self._st5_fe = SpeechT5FeatureExtractor.from_pretrained(model_dir("speecht5"))
            out = self._st5_fe(audio_target=x.astype(np.float32), sampling_rate=self.fs, return_tensors="np")
            return np.asarray(out["input_values"][0]).T
        import torch
        if self._f5_mel is None:
            f5_modules = import_f5()["modules"]
            self._f5_mel = f5_modules.MelSpec(n_fft=1024, hop_length=256, win_length=1024, n_mel_channels=100,
                                              target_sample_rate=24000, mel_spec_type="vocos")
        with torch.no_grad():
            return self._f5_mel(torch.tensor(x, dtype=torch.float32)[None])[0].numpy()

    def _numpy_mel(self, x: np.ndarray) -> np.ndarray:
        """fake／selftest 用的近似（不是模型的前端）：dsp 的 STFT 加同樣尺度的 filterbank。"""
        S = np.abs(dsp.stft(x, 1024, self.hop, 1024))
        M = self._fb_numpy @ S
        return np.log10(np.maximum(M, 1e-10)) if self.key == "speecht5" else np.log(np.maximum(M, 1e-5))

    def bin_of(self, hz: float) -> float:
        return float(np.interp(hz, self.centers, np.arange(len(self.centers))))

    def resolvable_below(self, f0: float) -> float:
        """相鄰 bin 的間距超過 F0/2 的第一個頻率：在那之上，這個前端分不出這個聲音的個別諧波
        （沿頻率軸取樣一個週期 F0 的起伏，每個週期至少要兩個 bin）。"""
        spacing = np.diff(self.centers)
        idx = np.flatnonzero(spacing > f0 / 2)
        return float(self.centers[idx[0]]) if len(idx) else float(self.centers[-1])


def front_ends(real: bool = True) -> dict[str, FrontEnd]:
    fe = {}
    fb_st5 = dsp.mel_filterbank(16000, 1024, 80, "slaney", area_norm=True, fmin=80.0, fmax=7600.0).T
    fb_f5 = dsp.mel_filterbank(24000, 1024, 100, "htk", area_norm=False, fmin=0.0, fmax=12000.0).T
    if real:
        from transformers import SpeechT5FeatureExtractor
        fb_st5 = SpeechT5FeatureExtractor.from_pretrained(model_dir("speecht5")).mel_filters
        import torchaudio
        fb_f5 = torchaudio.functional.melscale_fbanks(n_freqs=513, f_min=0.0, f_max=12000.0, n_mels=100, sample_rate=24000,
                                                      norm=None, mel_scale="htk").numpy()
    fe["speecht5"] = FrontEnd("speecht5", 16000, 256, np.asarray(fb_st5), 10.0, "SpeechT5's front end · 80 bins · 16 kHz · hop 16 ms", real)
    fe["f5"] = FrontEnd("f5", 24000, 256, np.asarray(fb_f5), 10.0 / math.log(10), "F5's front end · 100 bins · 24 kHz · hop 10.7 ms", real)
    for k, f in fe.items():
        f._fb_numpy = (fb_st5 if k == "speecht5" else fb_f5).T
    return fe


# ── 指標 ─────────────────────────────────────────────────────
def active_frames(Sdb: np.ndarray, active_db: float) -> np.ndarray:
    m = Sdb.mean(axis=0)
    return m > m.max() - active_db


def fine_detail(Sdb: np.ndarray, fe: FrontEnd, band, size=None, active_db=None) -> float:
    """log-mel（dB）減掉 size 的區域平均，在 band 內、有聲音的 frame 上取 RMS（dB）。"""
    from scipy.ndimage import uniform_filter
    size = tuple(size or CFG["detail"]["size"])
    act = active_frames(Sdb, active_db if active_db is not None else CFG["detail"]["active_db"])
    res = Sdb - uniform_filter(Sdb, size=size, mode="nearest")
    b = (fe.centers >= band[0]) & (fe.centers <= band[1])
    if not b.any() or not act.any():
        return float("nan")
    return float(np.sqrt(np.mean(res[b][:, act] ** 2)))


def f0_track(x: np.ndarray, fs: int) -> tuple[np.ndarray, np.ndarray]:
    c = CFG["f0"]
    flen = int(round(c["frame_ms"] * fs / 1000)); hop = int(round(c["hop_ms"] * fs / 1000))
    ts, f0 = [], []
    for a in range(0, max(1, len(x) - flen), hop):
        v, _ = dsp.f0_autocorr(x[a:a + flen], fs, c["floor_hz"], c["ceiling_hz"], c["voicing"])
        ts.append((a + flen / 2) / fs); f0.append(v)
    return np.array(ts), np.array(f0, dtype=float)


def f0_pair(a: np.ndarray, b: np.ndarray) -> dict:
    """同一個時間軸上兩條 F0：兩邊都有聲的 frame 上的差（cents），以及 voicing 一致率。"""
    k = min(len(a), len(b)); a, b = a[:k], b[:k]
    va, vb = ~np.isnan(a), ~np.isnan(b)
    both = va & vb
    d = 1200 * np.log2(b[both] / a[both]) if both.any() else np.array([])
    return {"frames_voiced_both": int(both.sum()), "voicing_agree": float(np.mean(va == vb)) if k else float("nan"),
            "median_abs_cents": float(np.median(np.abs(d))) if len(d) else float("nan"),
            "rms_cents": float(np.sqrt(np.mean(d ** 2))) if len(d) else float("nan")}


def pauses(x: np.ndarray, fs: int, min_ms: float = 100.0) -> dict:
    """句子內部的停頓：10 ms 一格的能量，門檻 max(峰值 − 35 dB, 底噪 + 15 dB)（W3 Demo 1 裁靜音的規則），
    第一個與最後一個有聲音的格子之間、連續 ≥ min_ms 的安靜段。回最長那一段的起點與長度、總數、語音的起訖。"""
    hop = int(fs * 0.010)
    n = len(x) // hop
    e = 20 * np.log10(np.sqrt(np.mean(x[:n * hop].reshape(n, hop) ** 2, axis=1)) + 1e-9)
    thr = max(e.max() - 35.0, np.percentile(e, 10) + 15.0)
    on = np.flatnonzero(e > thr)
    if len(on) == 0:
        return {"speech_start_s": None, "speech_end_s": None, "n": 0, "longest_start_s": None, "longest_s": 0.0}
    a, b = on[0], on[-1]
    quiet = ~(e[a:b + 1] > thr)
    runs, i = [], 0
    while i < len(quiet):
        if quiet[i]:
            j = i
            while j < len(quiet) and quiet[j]:
                j += 1
            if (j - i) * 10 >= min_ms:
                runs.append((a + i, j - i))
            i = j
        else:
            i += 1
    lo = max(runs, key=lambda r: r[1]) if runs else None
    return {"speech_start_s": round(a * 0.01, 2), "speech_end_s": round((b + 1) * 0.01, 2), "n": len(runs),
            "longest_start_s": round(lo[0] * 0.01, 2) if lo else None, "longest_s": round(lo[1] * 0.01, 2) if lo else 0.0}


def mel_pair(A: np.ndarray, B: np.ndarray, fe: FrontEnd) -> float:
    """同長度的兩張 mel（dB）在有聲 frame、7.6 kHz 以下的平均絕對差。"""
    k = min(A.shape[1], B.shape[1]); A, B = A[:, :k], B[:, :k]
    act = active_frames(A, CFG["detail"]["active_db"]) & active_frames(B, CFG["detail"]["active_db"])
    b = fe.centers <= CFG["display"]["max_hz"]
    return float(np.mean(np.abs(A[b][:, act] - B[b][:, act])))


def rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(x ** 2)) + 1e-12)


# ── F5 的 prompt 與長度（照 f5_tts.infer.utils_infer 的做法，用 numpy 寫，不需要 FFmpeg）──
def trim_edges(x: np.ndarray, fs: int, thr_dbfs: float = SILENCE_DBFS, chunk_ms: int = SILENCE_CHUNK_MS) -> np.ndarray:
    """前後 10 ms 一格、dBFS 低於 −42 的格子去掉（pydub detect_leading_silence 的規則；F5 的 remove_silence_edges）。"""
    n = int(fs * chunk_ms / 1000)
    k = len(x) // n
    if k == 0:
        return x
    db = 20 * np.log10(np.sqrt(np.mean(x[:k * n].reshape(k, n) ** 2, axis=1)) + 1e-12)
    loud = np.flatnonzero(db >= thr_dbfs)
    if len(loud) == 0:
        return x
    return x[loud[0] * n:(loud[-1] + 1) * n]


def f5_ref_text_pre(t: str) -> str:
    """preprocess_ref_audio_text 的最後一步：句尾補成 ". "（F5TTS.infer 把這個結果交給 infer_batch_process）。"""
    if not (t.endswith(". ") or t.endswith("。")):
        t = t + " " if t.endswith(".") else t + ". "
    return t


def f5_ref_text(t: str) -> str:
    """infer_batch_process 實際拿去算長度的 prompt 文字：上一步之後，最後一個字元是 1 byte 時再補一個空格。"""
    t = f5_ref_text_pre(t)
    if len(t[-1].encode("utf-8")) == 1:
        t = t + " "
    return t


def f5_length(ref_samples: int, ref_text: str, gen_text: str, speed: float = 1.0, hop: int = 256) -> dict:
    """infer_batch_process 的長度：T_gen = ⌊T^p · |ℓ| / |ℓ^p| / speed⌋，|·| 是 UTF-8 bytes（中文字一個算 3）；
    生成的文字不到 10 bytes 時 speed 換成 0.3。T^p 是 prompt（含 50 ms 墊）的 frame 數。"""
    rt = f5_ref_text(ref_text)
    ref_frames = ref_samples // hop
    ref_bytes, gen_bytes = len(rt.encode("utf-8")), len(gen_text.encode("utf-8"))
    sp = 0.3 if gen_bytes < 10 else speed
    return {"ref_frames": ref_frames, "ref_bytes": ref_bytes, "gen_bytes": gen_bytes, "speed": sp,
            "gen_frames": int(ref_frames / ref_bytes * gen_bytes / sp), "ref_text_used": rt}


def f5_max_chars(ref_text: str, ref_seconds: float, speed: float = 1.0) -> int:
    """infer_process 切句的上限（F5TTS.infer 先走這一步）；句子短於它就是一整塊，與本 demo 直接呼叫 infer_batch_process 相同。"""
    return int(len(ref_text.encode("utf-8")) / ref_seconds * (22 - ref_seconds) * speed)


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


PATTERNS = {"speecht5": ["*.json", "*.model", "*.bin"], "speecht5_vocoder": ["*.json", "*.bin"],
            "f5": [CFG["f5"]["ckpt"]], "f5_vocoder": ["config.yaml", "pytorch_model.bin"], "xvectors": [CFG["speecht5"]["xvector_zip"]]}
REPOS = {"speecht5": CFG["speecht5"]["repo"], "speecht5_vocoder": CFG["speecht5"]["vocoder"], "f5": CFG["f5"]["repo"],
         "f5_vocoder": CFG["f5"]["vocoder"], "xvectors": CFG["speecht5"]["xvectors"]}


def model_dir(key: str, offline: bool = True) -> str:
    from huggingface_hub import snapshot_download
    repo = REPOS[key]
    return snapshot_download(repo, revision=read_lock().get(repo), local_files_only=offline, allow_patterns=PATTERNS[key],
                             repo_type="dataset" if key == "xvectors" else None)


def quiet_hf() -> None:
    """transformers 的載入報告與進度條在課堂上是雜訊。"""
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


_F5 = {}


def import_f5() -> dict:
    """f5_tts.infer.utils_infer 在 import 時做兩件事：設 PYTORCH_ENABLE_MPS_FALLBACK=1、把 matplotlib 換成 Agg。
    torch 先 import（這時環境變數還沒設；MPS 的 fallback 在 torch 載入時註冊——就我們的理解），MPS 不支援的運算應該直接報錯，
    不會悄悄改在 CPU 上跑；check 另外記下變數值與「fall back」的警告。matplotlib 的 backend import 完換回來，課堂的視窗才開得起來。"""
    if _F5:
        return _F5
    import torch  # noqa: F401
    import matplotlib
    backend = matplotlib.get_backend()
    fallback_before = os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        from f5_tts.infer import utils_infer
        from f5_tts.model import DiT, modules
        from f5_tts.model.utils import seed_everything
    if matplotlib.get_backend() != backend:
        matplotlib.use(backend, force=True)
    _F5.update(utils_infer=utils_infer, DiT=DiT, modules=modules, seed_everything=seed_everything,
               mps_fallback_env_before=fallback_before, mps_fallback_env_after=os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK"))
    return _F5


def arctic_xvector() -> tuple[np.ndarray, int]:
    """CMU ARCTIC 某位說話人全部句子的 x-vector 平均，再正規化成單位長度（每一條本來就是單位長度）。"""
    spk = CFG["speecht5"]["arctic_speaker"]
    z = zipfile.ZipFile(Path(model_dir("xvectors")) / CFG["speecht5"]["xvector_zip"])
    vs = [np.load(io.BytesIO(z.read(n))) for n in z.namelist() if f"_{spk}_arctic-" in n and n.endswith(".npy")]
    if not vs:
        raise RuntimeError(f"x-vector zip 裡沒有 {spk}")
    v = np.mean(vs, axis=0)
    return v / np.linalg.norm(v), len(vs)


class SpeechT5:
    """回歸 mel：SpeechT5ForTextToSpeech.generate_speech（stop token 決定長度）→ SpeechT5HifiGan。"""
    key, label, fs = "speecht5", "SpeechT5", 16000

    def __init__(self, device: str, vocoder_device: str):
        import torch
        from transformers import SpeechT5ForTextToSpeech, SpeechT5HifiGan, SpeechT5Processor
        self.torch = torch
        self.device, self.vdev = resolve_device(device), resolve_device(vocoder_device)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.proc = SpeechT5Processor.from_pretrained(model_dir("speecht5"))
            self.model = SpeechT5ForTextToSpeech.from_pretrained(model_dir("speecht5")).eval().to(self.device)
            self.vocoder = SpeechT5HifiGan.from_pretrained(model_dir("speecht5_vocoder")).eval().to(self.vdev)
        xv, self.n_xvectors = arctic_xvector()
        self.xvector = torch.tensor(xv, dtype=torch.float32)[None]
        self.n_params = sum(p.numel() for p in self.model.parameters())
        self.reduction_factor = int(self.model.config.reduction_factor)
        self.prenet_dropout = float(self.model.config.speech_decoder_prenet_dropout)

    def synth(self, text: str, seed: int) -> dict:
        torch, c = self.torch, CFG["speecht5"]
        ids = self.proc(text=text, return_tensors="pt")["input_ids"].to(self.device)
        torch.manual_seed(seed)
        t0 = time.perf_counter()
        with torch.no_grad():
            mel = self.model.generate_speech(ids, self.xvector.to(self.device), threshold=c["threshold"],
                                             minlenratio=c["minlenratio"], maxlenratio=c["maxlenratio"])
        sync(self.device)
        t1 = time.perf_counter()
        with torch.no_grad():
            wav = self.vocoder(mel.to(self.vdev))
        sync(self.vdev)
        t2 = time.perf_counter()
        mel = mel.detach().float().cpu().numpy().T                        # (80, frames)，log10
        T = mel.shape[1]
        return {"mel": mel, "wav": wav.detach().float().cpu().numpy().reshape(-1).astype(np.float64), "fs": self.fs,
                "T": T, "seconds": T * 256 / 16000, "acoustic_s": t1 - t0, "vocoder_s": t2 - t1, "n_text_tokens": int(ids.shape[1]),
                "how_T": f"stop token: generation ends at the first step whose stop probability > {c['threshold']} "
                         f"({self.reduction_factor} frames per step)"}

    def info(self) -> dict:
        return {"repo": REPOS["speecht5"], "commit": read_lock().get(REPOS["speecht5"]), "vocoder": REPOS["speecht5_vocoder"],
                "vocoder_commit": read_lock().get(REPOS["speecht5_vocoder"]), "device": self.device, "vocoder_device": self.vdev,
                "n_params": self.n_params, "reduction_factor": self.reduction_factor, "prenet_dropout_at_inference": self.prenet_dropout,
                "xvector": {"repo": REPOS["xvectors"], "commit": read_lock().get(REPOS["xvectors"]),
                            "speaker": CFG["speecht5"]["arctic_speaker"], "n_utterances_averaged": self.n_xvectors}}


class _TimedVocoder:
    """把 Vocos 包起來：mel 搬到 vocoder 的裝置、量 decode 的時間（F5 的 infer_batch_process 呼叫 vocoder.decode）。"""

    def __init__(self, voc, dev: str):
        self.voc, self.dev, self.seconds = voc, dev, 0.0

    def decode(self, mel):
        t0 = time.perf_counter()
        y = self.voc.decode(mel.to(self.dev))
        sync(self.dev)
        self.seconds += time.perf_counter() - t0
        return y


class F5:
    """從噪音抽 mel：F5-TTS 的 infer_batch_process（CLI 與 F5TTS.infer 最後都走這一個函式）→ Vocos。
    prompt 的前處理（裁前後靜音、補 50 ms）在這支程式裡用 numpy 做，與 preprocess_ref_audio_text 同規則；
    直接傳波形進去，所以不需要 torchaudio.load／FFmpeg。"""
    key, label, fs = "f5", "F5-TTS", 24000

    def __init__(self, device: str, vocoder_device: str):
        import torch
        import yaml
        from importlib.resources import files
        f = import_f5()
        self.torch, self.f = torch, f
        self.device, self.vdev = resolve_device(device), resolve_device(vocoder_device)
        c = CFG["f5"]
        cfg = yaml.safe_load(files("f5_tts").joinpath(f"configs/{c['model']}.yaml").read_text())
        self.arch = cfg["model"]["arch"]
        ckpt = str(Path(model_dir("f5")) / c["ckpt"])
        with warnings.catch_warnings(), _quiet():
            warnings.simplefilter("ignore")
            voc = f["utils_infer"].load_vocoder("vocos", True, model_dir("f5_vocoder"), self.vdev)
            self.model = f["utils_infer"].load_model(f["DiT"], self.arch, ckpt, mel_spec_type="vocos", vocab_file="",
                                                     ode_method=c["ode_method"], use_ema=True, device=self.device)
        # load_model 不呼叫 eval()（CFM.sample 開頭才呼叫）；check 直接呼叫一次速度場，不先切成 eval 的話 DiT 的 dropout（0.1）還開著
        self.model.eval()
        voc.eval()
        self.vocoder = _TimedVocoder(voc, self.vdev)
        self.n_params = sum(p.numel() for p in self.model.parameters())

    def prompt(self, x16: np.ndarray) -> np.ndarray:
        x = dsp.resample(x16, 16000, self.fs)
        x = trim_edges(x, self.fs)
        return np.concatenate([x, np.zeros(int(self.fs * PROMPT_PAD_MS / 1000))])

    def synth(self, prompt24: np.ndarray, ref_text: str, text: str, seed: int, nfe: int | None = None,
              cfg_strength: float | None = None, sway: float | None = None) -> dict:
        torch, c = self.torch, CFG["f5"]
        nfe = nfe or c["nfe"]
        cfg_strength = c["cfg_strength"] if cfg_strength is None else cfg_strength
        sway = c["sway_sampling_coef"] if sway is None else sway
        L = f5_length(len(prompt24), ref_text, text, c["speed"])
        audio = torch.tensor(prompt24, dtype=torch.float32)[None]
        self.f["seed_everything"](seed)
        self.vocoder.seconds = 0.0
        t0 = time.perf_counter()
        with _quiet():
            gen = self.f["utils_infer"].infer_batch_process((audio, self.fs), f5_ref_text_pre(ref_text), [text], self.model, self.vocoder,
                                                            mel_spec_type="vocos", progress=None, target_rms=0.1, cross_fade_duration=0.15,
                                                            nfe_step=nfe, cfg_strength=cfg_strength, sway_sampling_coef=sway,
                                                            speed=c["speed"], fix_duration=None, device=self.device)
            wav, sr, spec = next(gen)
        sync(self.device)
        total = time.perf_counter() - t0
        T = spec.shape[1]
        if T != L["gen_frames"]:
            print(f"  {C.AMBER}F5 的長度 {T} ≠ 公式算的 {L['gen_frames']}（f5-tts 改了算法？看 infer_batch_process）{C.RESET}")
        return {"mel": spec, "wav": np.asarray(wav, dtype=np.float64), "fs": sr, "T": T, "seconds": T * 256 / sr,
                "acoustic_s": total - self.vocoder.seconds, "vocoder_s": self.vocoder.seconds, "nfe": nfe, "cfg_strength": cfg_strength,
                "w_slide": cfg_strength + 1, "sway": sway, "length": L,
                "how_T": f"from the character ratio: T = ⌊{L['ref_frames']} · {L['gen_bytes']} / {L['ref_bytes']}⌋ = {L['gen_frames']} "
                         f"(prompt frames × UTF-8 bytes of the text / bytes of the prompt's text)"}

    def info(self) -> dict:
        c = CFG["f5"]
        return {"repo": REPOS["f5"], "commit": read_lock().get(REPOS["f5"]), "ckpt": c["ckpt"], "model": c["model"],
                "vocoder": REPOS["f5_vocoder"], "vocoder_commit": read_lock().get(REPOS["f5_vocoder"]), "device": self.device,
                "vocoder_device": self.vdev, "n_params": self.n_params, "nfe": c["nfe"], "cfg_strength_alpha": c["cfg_strength"],
                "w_slide": c["cfg_strength"] + 1, "sway_sampling_coef": c["sway_sampling_coef"], "ode_method": c["ode_method"],
                "arch": self.arch, "mps_fallback_env": {"before_import": self.f["mps_fallback_env_before"], "after_import": self.f["mps_fallback_env_after"]}}


class _quiet:
    """F5 的函式會 print（vocab、model 路徑）；課堂上不要那些行。"""

    def __enter__(self):
        self._o = sys.stdout
        sys.stdout = io.StringIO()

    def __exit__(self, *a):
        sys.stdout = self._o


class FakeModel:
    """沒有模型時的替身：真人錄音的 mel，「回歸」= 區域平均、「抽樣」= 加上 seed 決定的紋理。畫面標 FAKE。"""

    def __init__(self, key: str, fe: FrontEnd, real_mel: np.ndarray, real_wav: np.ndarray):
        self.key, self.fe, self.real_mel, self.real_wav = key, fe, real_mel, real_wav
        self.label = ("SpeechT5" if key == "speecht5" else "F5-TTS") + " [FAKE]"
        self.fs = fe.fs
        self.device = self.vdev = "fake"

    def synth(self, *a, seed: int = 1, **k) -> dict:
        from scipy.ndimage import uniform_filter
        rng = np.random.default_rng(1000 + seed)
        M = self.real_mel.copy()
        if self.key == "speecht5":
            M = uniform_filter(M, size=tuple(CFG["fake"]["smooth"]), mode="nearest")
            M = M[:, :max(10, int(M.shape[1] * (0.9 + 0.01 * seed)))]
        else:
            M = M + rng.standard_normal(M.shape) * CFG["fake"]["texture_db"] / self.fe.to_db
        T = M.shape[1]
        x = dsp.resample(self.real_wav, 16000, self.fs)
        return {"mel": M, "wav": x * (1 + 0.05 * seed), "fs": self.fs, "T": T, "seconds": T * self.fe.hop / self.fe.fs,
                "acoustic_s": 0.0, "vocoder_s": 0.0, "how_T": "FAKE", "nfe": None}

    def info(self) -> dict:
        return {"repo": "fake", "device": "fake"}


# ── 紀錄 ─────────────────────────────────────────────────────
class Out:
    def __init__(self):
        self.buf = io.StringIO()

    def __call__(self, s: str = "") -> None:
        print(s, flush=True)
        self.buf.write(re.sub(r"\033\[[0-9;]*m", "", s) + "\n")

    def banner(self, s: str, color: str = C.BLUE) -> None:
        C.banner(s, color)
        self.buf.write(f"\n{'─' * 72}\n{s}\n{'─' * 72}\n")

    def text(self) -> str:
        return self.buf.getvalue()


def item_name(model: str, seed: int) -> str:
    return f"{model}_seed{seed}"


# ── 主體 ─────────────────────────────────────────────────────
class Demo:
    """一個 session：真人錄音（兩個前端各一份 mel）＋ SpeechT5 × 它的 seeds ＋ F5 × 它的 seeds。
    items[name] = {"model", "seed", "mel", "Sdb", "wav", "fs", "T", "seconds", "how_T", "detail", "wav_path", …}"""

    def __init__(self, out: Out, fe: dict, models: dict, folder: Path, tag: str = ""):
        self.out, self.fe, self.models, self.folder, self.tag = out, fe, models, folder, tag
        self.items: dict[str, dict] = {}
        self.real: dict = {}
        self.f0: dict[str, tuple] = {}
        self.log: dict = {"time": C.now(), "versions": versions(), "tag": tag, "text": dict(TXT),
                          "env": {"PYTORCH_ENABLE_MPS_FALLBACK_at_start": os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK")},
                          "models": {k: m.info() for k, m in models.items()}, "front_ends": {}, "real": {}, "items": {}, "pairs": {}, "played": []}
        self.fig = self.plt = None
        self.mode = "pair"
        self.shown: set[str] = set()
        self.cur_f5 = F5_SEEDS[0]
        self.st5_i = 0
        self.zoom = False
        self.show_f0 = False
        self.last = None

    # ── 準備 ──
    def load_real(self) -> None:
        p = HERE / CFG["audio"]["target_path"]
        if not p.exists():
            sys.exit(f"沒有 {p.relative_to(HERE)}：先跑 ./present.sh fetch")
        x = C.load_wav(p, 16000)
        self.real = {"wav": x, "fs": 16000, "mel": {}, "Sdb": {}}
        for k, fe in self.fe.items():
            M = fe.mel(x, 16000)
            self.real["mel"][k] = M
            self.real["Sdb"][k] = M * fe.to_db
        ts, f0 = f0_track(x, 16000)
        self.f0["real"] = (ts, f0)
        v = f0[~np.isnan(f0)]
        med = float(np.median(v)) if len(v) else float("nan")
        self.real["f0_median"] = med
        self.log["real"] = {"utt": TXT["target_utt"], "seconds": round(len(x) / 16000, 3), "f0_median_hz": round(med, 1),
                            "pauses": pauses(x, 16000),
                            "frames": {k: int(M.shape[1]) for k, M in self.real["mel"].items()},
                            "detail_db": {k: {band_name(b): fine_detail(self.real["Sdb"][k], self.fe[k], b) for b in BANDS} for k in self.fe}}
        for k, fe in self.fe.items():
            self.log["front_ends"][k] = {"label": fe.label, "fs": fe.fs, "hop": fe.hop, "fps": round(fe.fps, 3), "n_bins": len(fe.centers),
                                         "max_center_hz": round(float(fe.centers[-1]), 1),
                                         "bin_spacing_hz": {f"{f}": round(float(np.interp(f, fe.centers[1:], np.diff(fe.centers))), 1) for f in (500, 1000, 2000, 4000, 6000)},
                                         "harmonics_resolvable_below_hz": round(fe.resolvable_below(med), 0) if np.isfinite(med) else None}

    def compute(self) -> None:
        o = self.out
        o(f"\n{C.BOLD}{TXT['target']}{C.RESET}")
        pz = self.log["real"]["pauses"]
        o(f"  真人錄音 {TXT['target_utt']}：{self.log['real']['seconds']} s   F0 中位數 {self.log['real']['f0_median_hz']} Hz   最長停頓 {pz['longest_s']:.2f} s @ {pz['longest_start_s']} s")
        for k, fe in self.fe.items():
            r = self.log["front_ends"][k]
            o(f"    {fe.label}：{self.real['mel'][k].shape[1]} frames；bin 間距 2 kHz {r['bin_spacing_hz']['2000']} Hz、4 kHz {r['bin_spacing_hz']['4000']} Hz"
              f" → 這個聲音的個別諧波在 {r['harmonics_resolvable_below_hz'] / 1000:.1f} kHz 以上分不出來")
        if "speecht5" in self.models:
            for s in ST5_SEEDS:
                self.synth("speecht5", s)
        if "f5" in self.models:
            m = self.models["f5"]
            px = C.load_wav(HERE / CFG["audio"]["prompt_path"], 16000) if isinstance(m, F5) else None
            self.prompt24 = m.prompt(px) if px is not None else None
            if self.prompt24 is not None:
                L = f5_length(len(self.prompt24), TXT["prompt_text"], TXT["target"], CFG["f5"]["speed"])
                mc = f5_max_chars(f5_ref_text_pre(TXT["prompt_text"]), len(self.prompt24) / 24000)
                self.log["f5_prompt"] = {"utt": TXT["prompt_utt"], "text": TXT["prompt_text"], "seconds_after_trim_and_pad": round(len(self.prompt24) / 24000, 3),
                                         "length": L, "max_chars": mc, "one_chunk": len(TXT["target"].encode("utf-8")) <= mc}
                o(f"  F5 prompt {TXT['prompt_utt']}：{len(self.prompt24) / 24000:.2f} s（裁前後靜音、補 50 ms）；長度 T = ⌊{L['ref_frames']} · {L['gen_bytes']} / {L['ref_bytes']}⌋ = {L['gen_frames']} frames = {L['gen_frames'] * 256 / 24000:.2f} s")
            for s in F5_SEEDS:
                self.synth("f5", s)
        self.pairs()
        self.write_wavs()

    def synth(self, model: str, seed: int) -> None:
        m = self.models[model]
        if isinstance(m, F5):
            r = m.synth(self.prompt24, TXT["prompt_text"], TXT["target"], seed)
        elif isinstance(m, SpeechT5):
            r = m.synth(TXT["target"], seed)
        else:
            r = m.synth(seed=seed)
        fe = self.fe[model]
        r.update(model=model, seed=seed, Sdb=r["mel"] * fe.to_db)
        r["detail_db"] = {band_name(b): fine_detail(r["Sdb"], fe, b) for b in BANDS}
        r["detail_pct"] = {band_name(b): 100 * r["detail_db"][band_name(b)] / self.log["real"]["detail_db"][model][band_name(b)] for b in BANDS}
        self.f0[item_name(model, seed)] = f0_track(r["wav"], r["fs"])
        name = item_name(model, seed)
        self.items[name] = r
        rec = {k: v for k, v in r.items() if k not in ("mel", "Sdb", "wav")}
        rec["detail_db"] = {k: round(v, 3) for k, v in r["detail_db"].items()}
        rec["detail_pct"] = {k: round(v, 1) for k, v in r["detail_pct"].items()}
        for k in ("seconds", "acoustic_s", "vocoder_s"):
            rec[k] = round(r[k], 3)
        v = self.f0[name][1]; v = v[~np.isnan(v)]
        rec["f0_median_hz"] = round(float(np.median(v)), 1) if len(v) else None
        rec["f0_p10_p90_hz"] = [round(float(np.percentile(v, 10)), 1), round(float(np.percentile(v, 90)), 1)] if len(v) else None
        rec["pauses"] = pauses(r["wav"], r["fs"])
        r["pauses"], r["f0_median_hz"] = rec["pauses"], rec["f0_median_hz"]
        self.log["items"][name] = rec
        pz = rec["pauses"]
        self.out(f"  {self.models[model].label:<9s} seed {seed}   T = {r['T']:4d} frames = {r['seconds']:5.2f} s   細紋理 {band_name(MAIN_BAND)} = "
                 f"{r['detail_pct'][band_name(MAIN_BAND)]:5.1f}% of the real recording   F0 中位數 {rec['f0_median_hz']} Hz   最長停頓 {pz['longest_s']:.2f} s @ {pz['longest_start_s']} s"
                 f"   acoustic {r['acoustic_s']:6.2f} s · vocoder {r['vocoder_s']:5.2f} s")

    def pairs(self) -> None:
        """F5 三個 seed 兩兩比：mel 的平均絕對差、F0 差（cents）；SpeechT5 的長度。"""
        fe = self.fe["f5"]
        f5n = [item_name("f5", s) for s in F5_SEEDS if item_name("f5", s) in self.items]
        for i in range(len(f5n)):
            for j in range(i + 1, len(f5n)):
                a, b = self.items[f5n[i]], self.items[f5n[j]]
                self.log["pairs"][f"{f5n[i]}|{f5n[j]}"] = {"same_T": a["T"] == b["T"], "mel_mean_abs_db": round(mel_pair(a["Sdb"], b["Sdb"], fe), 3),
                                                           **{k: (round(v, 3) if isinstance(v, float) else v) for k, v in f0_pair(self.f0[f5n[i]][1], self.f0[f5n[j]][1]).items()}}
        if f5n:
            a = self.items[f5n[0]]
            k = min(a["Sdb"].shape[1], self.real["Sdb"]["f5"].shape[1])
            self.log["pairs"]["real|" + f5n[0]] = {"mel_mean_abs_db_first_frames": round(mel_pair(self.real["Sdb"]["f5"][:, :k], a["Sdb"][:, :k], fe), 3)}
        st = [self.items[item_name("speecht5", s)]["T"] for s in ST5_SEEDS if item_name("speecht5", s) in self.items]
        self.log["pairs"]["speecht5_T_by_seed"] = st

    def wav_path(self, name: str) -> Path:
        return self.folder / f"{name}.wav"

    def write_wavs(self) -> None:
        """播放檔：每個版本調到真人錄音的 RMS（量測用原始輸出，不受影響）。"""
        target = rms(self.real["wav"])
        self.folder.mkdir(parents=True, exist_ok=True)
        C.save_wav(self.folder / "real.wav", self.real["wav"], 16000)
        for name, r in self.items.items():
            g = target / rms(r["wav"])
            y = r["wav"] * g
            pk = float(np.max(np.abs(y)))
            if pk > 0.99:
                y = y * 0.99 / pk
            C.save_wav(self.wav_path(name), y, r["fs"])
            r["wav_path"] = self.wav_path(name)
            self.log["items"][name]["wav"] = os.path.relpath(self.wav_path(name), HERE)
            self.log["items"][name]["play_gain"] = round(float(g), 4)

    # ── 視窗 ──
    def make_fig(self):
        import matplotlib.pyplot as plt
        self.plt = plt
        d = CFG["display"]
        self.fig = plt.figure(figsize=(d["fig_w"], d["fig_h"]))
        self.fig.canvas.mpl_connect("key_press_event", self.on_key)
        self.draw()
        return self.fig

    def redraw(self):
        if self.fig is not None:
            self.fig.canvas.draw_idle()
            self.plt.pause(0.05)

    def _mel(self, ax, Sdb: np.ndarray, fe: FrontEnd, title: str, color: str = "black"):
        d = CFG["display"]
        top = fe.bin_of(d["max_hz"]) + 0.5
        S = Sdb[: int(math.ceil(top))]
        vmax = float(S.max())
        ax.imshow(S, origin="lower", aspect="auto", cmap="magma", vmin=vmax - d["db_range"], vmax=vmax,
                  extent=[0, S.shape[1] / fe.fps, -0.5, S.shape[0] - 0.5], interpolation="nearest")
        ticks = [f for f in (250, 500, 1000, 2000, 3000, 4000, 6000, 7000) if f <= d["max_hz"]]
        ax.set_yticks([fe.bin_of(f) for f in ticks]); ax.set_yticklabels([f"{f / 1000:g}k" for f in ticks], fontsize=9)
        lo, hi = fe.bin_of(d["zoom_hz"][0]), fe.bin_of(d["zoom_hz"][1])
        if self.zoom:
            ax.set_ylim(lo - 0.5, hi + 0.5)
        else:
            ax.set_ylim(-0.5, top)
            for yb in (lo, hi):
                ax.axhline(yb, color="white", lw=0.6, ls=(0, (4, 4)), alpha=0.55)
        ax.set_title(title, fontsize=10, loc="left", color=color)
        ax.set_xlabel("time (s)", fontsize=9); ax.set_ylabel("Hz (mel bins)", fontsize=9)
        ax.tick_params(labelsize=9)

    def _hint(self, ax, text: str):
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_color("#cccccc")
        ax.text(0.5, 0.5, text, ha="center", va="center", fontsize=11, color="#777777", transform=ax.transAxes)

    def _title(self, name: str) -> str:
        r = self.items[name]
        m = self.models[r["model"]]
        what = "regressed mel" if r["model"] == "speecht5" else "drawn mel"
        how = "stop token" if r["model"] == "speecht5" else "character ratio"
        return (f"{m.label} seed {r['seed']} — {what} · T = {r['T']} frames = {r['seconds']:.2f} s ({how})\n"
                f"fine detail {band_name(MAIN_BAND)}: {r['detail_pct'][band_name(MAIN_BAND)]:.0f}% of the real recording")

    def draw(self):
        if self.fig is None:
            return
        self.fig.clf()
        sup = "W8 Demo 1 — a regressed mel, a drawn mel, three draws" + (f"   [{self.tag}]" if self.tag else "") + ("   · zoom 2–6 kHz" if self.zoom else "")
        self.fig.suptitle(sup, fontsize=12)
        if self.mode == "pair":
            gs = self.fig.add_gridspec(2, 2)
            ax = [[self.fig.add_subplot(gs[i, j]) for j in range(2)] for i in range(2)]
            st = item_name("speecht5", ST5_SEEDS[self.st5_i])
            if "speecht5" in self.shown and st in self.items:
                self._mel(ax[0][0], self.items[st]["Sdb"], self.fe["speecht5"], self._title(st))
            else:
                self._hint(ax[0][0], "1 — SpeechT5 (regression)")
            f5 = item_name("f5", self.cur_f5)
            if "f5" in self.shown and f5 in self.items:
                self._mel(ax[0][1], self.items[f5]["Sdb"], self.fe["f5"], self._title(f5))
            else:
                self._hint(ax[0][1], "2 — F5-TTS (flow matching), seed 1")
            for j, k in enumerate(("speecht5", "f5")):
                if "real" in self.shown:
                    short = "SpeechT5's front end (80 bins, 16 kHz)" if k == "speecht5" else "F5's front end (100 bins, 24 kHz)"
                    self._mel(ax[1][j], self.real["Sdb"][k], self.fe[k],
                              f"real recording {TXT['target_utt']} · {short}\nT = {self.real['mel'][k].shape[1]} frames = {self.log['real']['seconds']:.2f} s",
                              color="#555555")
                else:
                    self._hint(ax[1][j], "g — the real recording, through this model's own front end")
        else:
            gs = self.fig.add_gridspec(2, 3, height_ratios=[1.25, 1])
            for j, s in enumerate(F5_SEEDS):
                a = self.fig.add_subplot(gs[0, j])
                n = item_name("f5", s)
                if n in self.items:
                    r = self.items[n]
                    self._mel(a, r["Sdb"], self.fe["f5"], f"F5-TTS seed {s} · T = {r['T']} = {r['seconds']:.2f} s")
                else:
                    self._hint(a, f"seed {s}")
            a = self.fig.add_subplot(gs[1, :])
            if self.show_f0:
                cols = ["#5b3fa0", "#c8742a", "#2a7fc8"]
                ts, f0 = self.f0["real"]
                a.plot(ts, f0, "o", ms=2.5, mfc="none", color="#999999", label=f"real recording ({self.log['real']['seconds']:.2f} s)")
                for s, col in zip(F5_SEEDS, cols):
                    n = item_name("f5", s)
                    if n in self.f0:
                        t2, f2 = self.f0[n]
                        pz = self.items[n]["pauses"]
                        a.plot(t2, f2, ".", ms=4, color=col, label=f"F5 seed {s} · median {self.items[n]['f0_median_hz']} Hz · longest pause {pz['longest_s']:.2f} s")
                a.set_ylim(CFG["f0"]["floor_hz"] - 5, 240); a.set_ylabel("$F_0$ (Hz)"); a.set_xlabel("time (s)")
                a.grid(alpha=0.3); a.legend(fontsize=9, loc="upper right", ncol=2)
                a.set_title("$F_0$ (autocorrelation, W2 dsp.py) — same length, three different contours and pauses", fontsize=10, loc="left")
            else:
                self._hint(a, "z — the three $F_0$ tracks on one time axis")
        self.fig.tight_layout()
        self.redraw()

    # ── 鍵 ──
    def _play(self, path: Path, what: str) -> None:
        self.out(f"  ▶ {what}")
        self.last = (path, what)
        self.log["played"].append(what)
        C.play(path)

    def show_st5(self, next_seed: bool = False) -> None:
        if "speecht5" not in self.models and not any(k.startswith("speecht5") for k in self.items):
            self.out(f"  {C.AMBER}沒有 SpeechT5{C.RESET}"); return
        if next_seed:
            self.st5_i = (self.st5_i + 1) % len(ST5_SEEDS)
        n = item_name("speecht5", ST5_SEEDS[self.st5_i])
        if n not in self.items:
            self.out(f"  {C.AMBER}沒有 {n}{C.RESET}"); return
        self.mode = "pair"; self.shown.add("speecht5")
        self.draw()
        r = self.items[n]
        self.out(f"  SpeechT5 seed {r['seed']}：T = {r['T']} frames = {r['seconds']:.2f} s（stop token）   細紋理 {band_name(MAIN_BAND)} {r['detail_pct'][band_name(MAIN_BAND)]:.0f}%")
        if next_seed:
            self.out(f"    SpeechT5 三個 seed 的長度：{self.log['pairs'].get('speecht5_T_by_seed')} frames——decoder pre-net 的 dropout 在推論時也開著")
        self._play(r["wav_path"], f"SpeechT5 seed {r['seed']}")

    def show_f5(self, seed: int) -> None:
        n = item_name("f5", seed)
        if n not in self.items:
            self.out(f"  {C.AMBER}沒有 {n}{C.RESET}"); return
        self.mode = "pair"; self.shown.add("f5"); self.cur_f5 = seed
        self.draw()
        r = self.items[n]
        self.out(f"  F5 seed {seed}：T = {r['T']} frames = {r['seconds']:.2f} s（{r['how_T']}）   細紋理 {band_name(MAIN_BAND)} {r['detail_pct'][band_name(MAIN_BAND)]:.0f}%")
        self._play(r["wav_path"], f"F5 seed {seed}")

    def show_real(self) -> None:
        self.mode = "pair"; self.shown.add("real")
        self.draw()
        self._play(self.folder / "real.wav", f"real recording {TXT['target_utt']}")

    def show_seeds(self) -> None:
        self.mode = "seeds"
        self.draw()
        Ts = [self.items[item_name("f5", s)]["T"] for s in F5_SEEDS if item_name("f5", s) in self.items]
        self.out(f"  F5 三個 seed 的長度：{Ts} frames——長度不是抽的，是 prompt 與字數的比例算出來的")
        for s in F5_SEEDS[1:]:
            n = item_name("f5", s)
            if n in self.items:
                self._play(self.items[n]["wav_path"], f"F5 seed {s}")

    def toggle_f0(self) -> None:
        self.mode = "seeds"; self.show_f0 = not self.show_f0
        self.draw()
        if self.show_f0:
            pz = self.log["real"]["pauses"]
            self.out(f"    真人錄音：F0 中位數 {self.log['real']['f0_median_hz']} Hz，最長停頓 {pz['longest_s']:.2f} s（{pz['longest_start_s']} s 起）")
            for s in F5_SEEDS:
                n = item_name("f5", s)
                if n in self.items:
                    r = self.items[n]; pz = r["pauses"]
                    self.out(f"    F5 seed {s}：F0 中位數 {r['f0_median_hz']} Hz，最長停頓 {pz['longest_s']:.2f} s（{pz['longest_start_s']} s 起）")
            for k, v in self.log["pairs"].items():
                if k.startswith("f5_seed") and "|" in k:
                    a, b = k.split("|")
                    self.out(f"    {a} vs {b}：同一格有聲／無聲一致 {v['voicing_agree']:.2f}、mel 平均差 {v['mel_mean_abs_db']:.1f} dB")

    def toggle_zoom(self) -> None:
        self.zoom = not self.zoom
        self.draw()

    def again(self) -> None:
        if self.last:
            self._play(*self.last)

    def table(self) -> None:
        o = self.out
        o(f"\n{C.BOLD}{TXT['target']}{C.RESET}")
        bl = [band_name(b) for b in BANDS]
        o(f"  {'':<20s}{'T':>6s}{'sec':>7s}   " + "".join(f"{b:>13s}" for b in bl) + f"{'F0 med':>8s}{'pause':>7s}{'acoustic':>10s}{'vocoder':>9s}   length decided by")
        for k in ("speecht5", "f5"):
            fe = self.fe[k]
            rr = self.log["real"]
            o(f"  {C.GREY}real · {('SpeechT5 FE' if k == 'speecht5' else 'F5 FE'):<13s}{self.real['mel'][k].shape[1]:>6d}{rr['seconds']:>7.2f}   "
              + "".join(f"{rr['detail_db'][k][b]:>9.2f} dB " for b in bl) + f"{rr['f0_median_hz']:>8.0f}{rr['pauses']['longest_s']:>7.2f}{C.RESET}")
            for s in (ST5_SEEDS if k == "speecht5" else F5_SEEDS):
                n = item_name(k, s)
                if n not in self.items:
                    continue
                r = self.items[n]
                how = "stop token" if k == "speecht5" else f"⌊{r['length']['ref_frames']}·{r['length']['gen_bytes']}/{r['length']['ref_bytes']}⌋" if "length" in r else r["how_T"]
                o(f"  {self.models[k].label + ' seed ' + str(s):<20s}{r['T']:>6d}{r['seconds']:>7.2f}   "
                  + "".join(f"{r['detail_pct'][b]:>12.0f}%" for b in bl) + f"{(r.get('f0_median_hz') or float('nan')):>8.0f}{r['pauses']['longest_s']:>7.2f}"
                  + f"{r['acoustic_s']:>9.2f}s{r['vocoder_s']:>8.2f}s   {how}")
        o("  細紋理：log-mel（dB）減掉自己 3 × 3 的區域平均，在頻帶內、有聲 frame 上的 RMS；模型那幾列是「÷ 同一個前端的真人錄音」。參考，不是判準。")
        o("  F0 med：自相關 F0 的中位數（Hz）；pause：句子內部最長的停頓（s，10 ms 能量、門檻 max(峰值 − 35, 底噪 + 15) dB）。")
        f = self.log["front_ends"]
        o(f"  個別諧波分得出來的上限（bin 間距 < F0/2，F0 = 真人錄音的中位數 {self.log['real']['f0_median_hz']} Hz）："
          f"SpeechT5 前端 {f['speecht5']['harmonics_resolvable_below_hz'] / 1000:.1f} kHz、F5 前端 {f['f5']['harmonics_resolvable_below_hz'] / 1000:.1f} kHz")
        for k, v in self.log["pairs"].items():
            if k.startswith("f5_seed") and "|" in k:
                o(f"  {k}: same T {v['same_T']} · mel |Δ| {v['mel_mean_abs_db']:.2f} dB · voicing agree {v['voicing_agree']:.2f} · frame-wise F0 |Δ| median {v['median_abs_cents']:.0f} cents (timing and pitch mixed)")
        if self.log["pairs"].get("speecht5_T_by_seed"):
            o(f"  SpeechT5 seeds {ST5_SEEDS}: T = {self.log['pairs']['speecht5_T_by_seed']} frames（pre-net dropout）")
        m = self.log["models"].get("f5", {})
        if m.get("nfe"):
            o(f"  F5：NFE {m['nfe']}、cfg_strength α = {m['cfg_strength_alpha']:g}（投影片的 w = α + 1 = {m['w_slide']:g}）、sway {m['sway_sampling_coef']:g}；每一步兩次前向（有條件、無條件）")

    def menu(self) -> None:
        self.out(f"\n{C.BOLD}p12{C.RESET}  1 SpeechT5   2 F5 seed 1   g 真人錄音   x 2–6 kHz   3 再抽兩次   z F0   s SpeechT5 換 seed   t 數字表   r 再播   h 選單   q 離開")

    def dispatch(self, k: str) -> bool:
        if k == "1":
            self.show_st5()
        elif k == "2":
            self.show_f5(F5_SEEDS[0])
        elif k == "g":
            self.show_real()
        elif k == "x":
            self.toggle_zoom()
        elif k == "3":
            self.show_seeds()
        elif k == "z":
            self.toggle_f0()
        elif k == "s":
            self.show_st5(next_seed=True)
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
        import matplotlib.pyplot as plt
        self.plt = plt
        d = CFG["display"]
        for zoom, nm in ((False, "pair.png"), (True, "pair_zoom.png")):
            self.fig = plt.figure(figsize=(d["fig_w"], d["fig_h"]))
            self.mode, self.zoom, self.shown = "pair", zoom, {"speecht5", "f5", "real"}
            self.st5_i, self.cur_f5 = 0, F5_SEEDS[0]
            self.draw(); self.fig.savefig(self.folder / nm, dpi=100); plt.close(self.fig)
        self.fig = plt.figure(figsize=(d["fig_w"], d["fig_h"]))
        self.mode, self.zoom, self.show_f0 = "seeds", False, True
        self.draw(); self.fig.savefig(self.folder / "seeds.png", dpi=100); plt.close(self.fig)
        st = [item_name("speecht5", s) for s in ST5_SEEDS if item_name("speecht5", s) in self.items]
        if st:
            fig, axes = plt.subplots(1, len(st), figsize=(d["fig_w"], 3.6), squeeze=False)
            for a, n in zip(axes[0], st):
                r = self.items[n]
                self._mel(a, r["Sdb"], self.fe["speecht5"], f"SpeechT5 seed {r['seed']} · T = {r['T']} = {r['seconds']:.2f} s")
            fig.suptitle("SpeechT5 with three seeds (decoder pre-net dropout stays on at inference)", fontsize=11)
            fig.tight_layout(); fig.savefig(self.folder / "speecht5_seeds.png", dpi=100); plt.close(fig)
        self.fig = None

    def save(self) -> None:
        stem = "rehearsal" if self.folder.name == "rehearsal" else "session"
        self.folder.mkdir(parents=True, exist_ok=True)
        np.save(self.folder / "real_speecht5_mel.npy", self.real["mel"]["speecht5"].astype(np.float32))
        np.save(self.folder / "real_f5_mel.npy", self.real["mel"]["f5"].astype(np.float32))
        for name, r in self.items.items():
            np.save(self.folder / f"{name}_mel.npy", r["mel"].astype(np.float32))
        (self.folder / f"{stem}.json").write_text(json.dumps(self.log, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
        (self.folder / f"{stem}.txt").write_text(self.out.text(), encoding="utf-8")
        C.note(f"  存到 {self.folder}/（{stem}.json、{stem}.txt、.wav、*_mel.npy）")


def versions() -> dict:
    v = C.versions()
    from importlib import metadata
    for pkg in ("torch", "torchaudio", "transformers", "huggingface_hub", "f5-tts", "vocos", "x-transformers", "torchdiffeq", "sentencepiece", "soundfile"):
        try:
            v[pkg] = metadata.version(pkg)
        except metadata.PackageNotFoundError:
            v[pkg] = None
    return v


# ── replay ───────────────────────────────────────────────────
class ReplayModel:
    def __init__(self, key: str, info: dict):
        self.key, self.label = key, ("SpeechT5" if key == "speecht5" else "F5-TTS")
        self._info = info

    def info(self):
        return self._info


def build_replay(out: Out) -> Demo:
    rec_path = REHEARSAL / "rehearsal.json"
    if not rec_path.exists():
        sys.exit("沒有 runs/rehearsal/rehearsal.json：先 ./present.sh rehearse")
    rec = json.loads(rec_path.read_text(encoding="utf-8"))
    fe = front_ends(real=False)
    for k, f in fe.items():                                   # 頻率軸與彩排時一樣（filterbank 的定義相同，數值見 selftest）
        f.real = False
    models = {k: ReplayModel(k, v) for k, v in rec["models"].items()}
    d = Demo(out, fe, models, REHEARSAL, tag="replay " + rec["time"])
    d.log = rec
    d.log["played"] = []
    x = C.load_wav(REHEARSAL / "real.wav", 16000)
    d.real = {"wav": x, "fs": 16000, "mel": {}, "Sdb": {}, "f0_median": rec["real"]["f0_median_hz"]}
    for k in fe:
        M = np.load(REHEARSAL / f"real_{k}_mel.npy")
        d.real["mel"][k] = M; d.real["Sdb"][k] = M * fe[k].to_db
    d.f0["real"] = f0_track(x, 16000)
    d.log["real"].setdefault("pauses", pauses(x, 16000))                     # 舊版的彩排紀錄沒有這兩個欄位
    for name, r in rec["items"].items():
        M = np.load(REHEARSAL / f"{name}_mel.npy")
        wav = (HERE / r["wav"]).resolve()
        y = C.load_wav(wav, r["fs"])
        item = dict(r); item.update(mel=M, Sdb=M * fe[r["model"]].to_db, wav=y, wav_path=wav)
        d.f0[name] = f0_track(y, r["fs"])
        item.setdefault("pauses", pauses(y, r["fs"]))
        if item.get("f0_median_hz") is None:
            v = d.f0[name][1]; v = v[~np.isnan(v)]
            item["f0_median_hz"] = round(float(np.median(v)), 1) if len(v) else None
        d.items[name] = item
    out(f"  replay：{rec['time']} 的彩排，{len(d.items)} 個版本")
    return d


def build(cmd: str, out: Out, device: str | None = None) -> Demo:
    if cmd == "fake":
        folder = HERE / "runs" / "live" / (C.now().replace(":", "") + ".fake")
        fe = front_ends(real=False)
        x = C.load_wav(HERE / CFG["audio"]["target_path"], 16000)
        models = {k: FakeModel(k, fe[k], fe[k].mel(x, 16000), x) for k in fe}
        d = Demo(out, fe, models, folder, tag="FAKE")
    else:
        folder = REHEARSAL if cmd == "rehearse" else HERE / "runs" / "live" / C.now().replace(":", "")
        try:
            import torch  # noqa: F401
        except ImportError:
            sys.exit("沒有 torch：先跑 ./setup.sh（建 ../.venv-tts）；或 ./present.sh fake 演練流程")
        quiet_hf()
        fe = front_ends(real=True)
        c = CFG["infer"]
        models = {}
        for key, cls, dev in (("speecht5", SpeechT5, c["speecht5_device"]), ("f5", F5, c["f5_device"])):
            t0 = time.perf_counter()
            try:
                models[key] = cls(device or dev, device or c["vocoder_device"])
                m = models[key]
                out(f"  載入 {m.label:<9s} {m.n_params / 1e6:6.1f} M 參數   {m.device} / vocoder {m.vdev}   {time.perf_counter() - t0:5.1f} s")
            except Exception as e:
                out(f"  {C.RED}{key} 載入失敗：{e!r}{C.RESET}（跳過；先 ./present.sh check）")
        if not models:
            sys.exit("兩個模型都載不起來")
        d = Demo(out, fe, models, folder, tag="rehearsal" if cmd == "rehearse" else "")
    d.load_real()
    d.compute()
    return d


# ── check ────────────────────────────────────────────────────
def check(device: str | None = None) -> None:
    C.banner("課前體檢", C.AMBER)
    res = {"time": C.now(), "versions": versions(), "env": {"PYTORCH_ENABLE_MPS_FALLBACK": os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK")}}
    if os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") not in (None, "0"):
        print(f"  {C.AMBER}PYTORCH_ENABLE_MPS_FALLBACK = {os.environ['PYTORCH_ENABLE_MPS_FALLBACK']}：MPS 不支援的運算會悄悄改在 CPU 上跑；"
              f"check 要嚴格就用 ./present.sh check（它設成 0）{C.RESET}")
    for k, v in res["versions"].items():
        print(f"  {k:<16s} {v}")
    try:
        import torch
    except ImportError:
        sys.exit("  沒有 torch：先跑 ./setup.sh")
    quiet_hf()
    mps = bool(torch.backends.mps.is_available())
    print(f"  {'mps':<16s} built={torch.backends.mps.is_built()} available={mps}")
    res["mps_available"] = mps
    fa, fb = front_ends(real=False), front_ends(real=True)
    res["front_end_axes_numpy_vs_model"] = {k: float(np.max(np.abs(fa[k].centers - fb[k].centers))) for k in fa}
    print(f"  頻率軸（replay 用的 numpy filterbank 對模型的）最大差：{res['front_end_axes_numpy_vs_model']} Hz")
    devs = [device] if device else (["cpu"] + (["mps"] if mps else []))
    lock = read_lock()
    for k, repo in REPOS.items():
        print(f"  {repo:<32s} @ {str(lock.get(repo))[:12]}")
    x16 = C.load_wav(HERE / CFG["audio"]["target_path"], 16000)
    p16 = C.load_wav(HERE / CFG["audio"]["prompt_path"], 16000)
    CHECK_DIR.mkdir(parents=True, exist_ok=True)
    caught: list[str] = []

    def record_warnings(ws):
        for w in ws:
            msg = str(w.message)
            if "fall back" in msg or "not currently supported on the MPS" in msg:
                caught.append(msg[:200])

    # 1. SpeechT5：兩個裝置各生成一次（seed 1），同一份 mel 經 HiFi-GAN 在兩個裝置上的差
    res["speecht5"] = {}
    mel_ref = None
    for dev in devs:
        r = {}
        try:
            with warnings.catch_warnings(record=True) as ws:
                warnings.simplefilter("always")
                t0 = time.perf_counter(); m = SpeechT5(dev, dev); r["load_s"] = round(time.perf_counter() - t0, 2)
                m.synth("Hello.", 0)                                                                # 暖機
                out = m.synth(TXT["target"], ST5_SEEDS[0])
            record_warnings(ws)
            r.update({"T": out["T"], "seconds": round(out["seconds"], 3), "acoustic_s": round(out["acoustic_s"], 3), "vocoder_s": round(out["vocoder_s"], 3),
                      "prenet_dropout_at_inference": m.prenet_dropout, "reduction_factor": m.reduction_factor})
            if mel_ref is None:
                mel_ref = out["mel"]
            with torch.no_grad():
                y = m.vocoder(torch.tensor(mel_ref.T, dtype=torch.float32).to(m.vdev)).float().cpu().numpy().reshape(-1)
            r["_vocoder_out"] = y
            C.save_wav(CHECK_DIR / f"speecht5_{dev}.wav", out["wav"] / max(1e-9, np.max(np.abs(out["wav"]))) * 0.9, 16000)
            print(f"  SpeechT5 {dev}: 載入 {r['load_s']} s · T = {out['T']} frames（{out['seconds']:.2f} s）· acoustic {out['acoustic_s']:.2f} s · HiFi-GAN {out['vocoder_s']:.3f} s"
                  f" · pre-net dropout {m.prenet_dropout} 推論時開著")
        except Exception as e:
            r["error"] = repr(e)
            print(f"  SpeechT5 {dev}: {C.RED}失敗{C.RESET} {e!r}")
        res["speecht5"][dev] = r
    if all("_vocoder_out" in res["speecht5"].get(d, {}) for d in ("cpu", "mps")):
        a, b = res["speecht5"]["cpu"].pop("_vocoder_out"), res["speecht5"]["mps"].pop("_vocoder_out")
        n = min(len(a), len(b))
        res["speecht5"]["hifigan_mps_vs_cpu"] = {"max_abs": float(np.max(np.abs(a[:n] - b[:n]))), "lsd_db": dsp.log_spectral_distance(a[:n], b[:n], 400, 160, 512)}
        print(f"  HiFi-GAN 同一份 mel：mps 對 cpu 最大差 {res['speecht5']['hifigan_mps_vs_cpu']['max_abs']:.2e}、LSD {res['speecht5']['hifigan_mps_vs_cpu']['lsd_db']:.3f} dB")
    for dv in res["speecht5"].values():
        if isinstance(dv, dict):
            dv.pop("_vocoder_out", None)

    # 2. F5：同一個輸入（cpu 上抽的固定噪音、prompt mel、文字）過一次速度場；同一份 mel 經 Vocos；各抽一次 NFE 32 存 .wav 給耳朵
    res["f5"] = {}
    fields, vocs = {}, {}
    for dev in devs:
        r = {}
        try:
            with warnings.catch_warnings(record=True) as ws:
                warnings.simplefilter("always")
                t0 = time.perf_counter(); m = F5(dev, dev); r["load_s"] = round(time.perf_counter() - t0, 2)
                prompt24 = m.prompt(p16)
                v, t_fwd = f5_field_once(m, prompt24)
                fields[dev] = v
                r["field_forward_cfg_s"] = round(t_fwd, 4)
                mel_probe = m.f["modules"].MelSpec(mel_spec_type="vocos")(torch.tensor(dsp.resample(x16, 16000, 24000), dtype=torch.float32)[None])
                with torch.no_grad():
                    vocs[dev] = m.vocoder.voc.decode(mel_probe.to(m.vdev)).float().cpu().numpy().reshape(-1)
                out = m.synth(prompt24, TXT["prompt_text"], TXT["target"], F5_SEEDS[0])
            record_warnings(ws)
            r.update({"T": out["T"], "seconds": round(out["seconds"], 3), "acoustic_s": round(out["acoustic_s"], 3), "vocoder_s": round(out["vocoder_s"], 3),
                      "nfe": out["nfe"], "cfg_strength_alpha": out["cfg_strength"], "length": out["length"],
                      "mps_fallback_env_after_import": m.f["mps_fallback_env_after"]})
            C.save_wav(CHECK_DIR / f"f5_{dev}.wav", out["wav"] / max(1e-9, np.max(np.abs(out["wav"]))) * 0.9, 24000)
            print(f"  F5 {dev}: 載入 {r['load_s']} s · 速度場一次前向（含 CFG 的兩份）{t_fwd * 1000:.0f} ms · NFE {out['nfe']} 抽一次 acoustic {out['acoustic_s']:.1f} s"
                  f" · Vocos {out['vocoder_s']:.3f} s · T = {out['T']}（{out['seconds']:.2f} s）")
        except Exception as e:
            r["error"] = repr(e)
            print(f"  F5 {dev}: {C.RED}失敗{C.RESET} {e!r}")
        res["f5"][dev] = r
    if "cpu" in fields and "mps" in fields:
        a, b = fields["cpu"], fields["mps"]
        res["f5"]["field_mps_vs_cpu"] = {"max_abs": float(np.max(np.abs(a - b))), "rel_l2": float(np.linalg.norm(a - b) / (np.linalg.norm(a) + 1e-12))}
        print(f"  F5 速度場同一個輸入：mps 對 cpu 最大差 {res['f5']['field_mps_vs_cpu']['max_abs']:.2e}、相對 L2 {res['f5']['field_mps_vs_cpu']['rel_l2']:.2e}")
    if "cpu" in vocs and "mps" in vocs:
        a, b = vocs["cpu"], vocs["mps"]; n = min(len(a), len(b))
        res["f5"]["vocos_mps_vs_cpu"] = {"max_abs": float(np.max(np.abs(a[:n] - b[:n]))), "lsd_db": dsp.log_spectral_distance(a[:n], b[:n], 600, 240, 1024)}
        print(f"  Vocos 同一份 mel：mps 對 cpu 最大差 {res['f5']['vocos_mps_vs_cpu']['max_abs']:.2e}、LSD {res['f5']['vocos_mps_vs_cpu']['lsd_db']:.3f} dB")
    res["mps_fallback_warnings"] = caught
    if caught:
        print(f"  {C.AMBER}有運算退回 CPU：{caught[:3]}{C.RESET}")
    (HERE / "runs").mkdir(exist_ok=True)
    (HERE / "runs" / "check.json").write_text(json.dumps(res, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    print(f"\n  → runs/check.json；用耳朵比：afplay runs/check/f5_cpu.wav；afplay runs/check/f5_mps.wav（§6 第 7 項）")
    print("  兩個裝置的 F5 用同一個 seed 也抽不到同一個樣本（亂數產生器不同），比的是品質不是逐點相同。下一步：./present.sh rehearse")


def f5_field_once(m: F5, prompt24: np.ndarray) -> tuple[np.ndarray, float]:
    """速度場在 τ = 0.5、同一個輸入上的一次前向（有條件與無條件兩份，就是 CFG 每一步的成本）。噪音在 cpu 上抽，兩個裝置吃同一份。"""
    torch = m.torch
    f = m.f
    L = f5_length(len(prompt24), TXT["prompt_text"], TXT["target"])
    total = L["ref_frames"] + L["gen_frames"]
    with torch.no_grad():
        cond = m.model.mel_spec(torch.tensor(prompt24, dtype=torch.float32)[None].to(m.device)).permute(0, 2, 1)
        cond = torch.nn.functional.pad(cond, (0, 0, 0, total - cond.shape[1]))
        mask = torch.zeros(1, total, 1, dtype=torch.bool, device=m.device); mask[:, :L["ref_frames"]] = True
        step_cond = torch.where(mask, cond, torch.zeros_like(cond))
        text = f["utils_infer"].convert_char_to_pinyin([f5_ref_text(TXT["prompt_text"]) + TXT["target"]])
        from f5_tts.model.utils import list_str_to_idx
        ids = list_str_to_idx(text, m.model.vocab_char_map).to(m.device)
        g = torch.Generator().manual_seed(0)
        x = torch.randn(1, total, 100, generator=g).to(m.device)
        t = torch.tensor([0.5], device=m.device)
        for _ in range(2):                                                              # 第一次是暖機
            t0 = time.perf_counter()
            pred = m.model.transformer(x=x, cond=step_cond, text=ids, time=t, mask=None, cfg_infer=True, cache=False)
            sync(m.device)
            dt = time.perf_counter() - t0
    return pred.float().cpu().numpy(), dt


# ── fetch ────────────────────────────────────────────────────
def fetch() -> None:
    from huggingface_hub import HfApi, snapshot_download
    C.banner("下載模型與 x-vector、準備兩段錄音", C.AMBER)
    lock = read_lock()
    api = HfApi()
    for key, repo in REPOS.items():
        is_ds = key == "xvectors"
        info = api.dataset_info(repo) if is_ds else api.model_info(repo)
        sha = lock.get(repo) or info.sha
        path = snapshot_download(repo, revision=sha, allow_patterns=PATTERNS[key], repo_type="dataset" if is_ds else None)
        lock[repo] = sha
        card = getattr(info, "card_data", None)
        lic = card.get("license") if card else None
        size = sum(f.stat().st_size for f in Path(path).rglob("*") if f.is_file()) / 1e6
        print(f"  {repo} @ {sha[:12]}   {size:.0f} MB   license (card): {lic}")
    MODEL_LOCK.write_text("".join(f"{r} {s}\n" for r, s in lock.items()), encoding="utf-8")
    src = (HERE / CFG["audio"]["target_source"]).resolve()
    dst = HERE / CFG["audio"]["target_path"]
    dst.parent.mkdir(parents=True, exist_ok=True)
    if src.exists():
        shutil.copyfile(src, dst)
        if src.with_suffix(".txt").exists():
            shutil.copyfile(src.with_suffix(".txt"), dst.with_suffix(".txt"))
        print(f"  {src.name} → {dst.relative_to(HERE)}（{TXT['target_utt']}）")
    else:
        print(f"  {C.AMBER}找不到 {src}：先跑 ../w03_d3_blank_heatmap/present.sh fetch{C.RESET}")
    tar = (HERE / CFG["audio"]["dev_clean_tar"]).resolve()
    if not tar.exists():
        print(f"  {C.AMBER}找不到 {tar}：先跑 ../w03_d3_blank_heatmap/present.sh fetch（會下載 {CFG['audio']['librispeech_url']}）{C.RESET}")
    else:
        utt = TXT["prompt_utt"]
        spk, ch, _ = utt.split("-")
        member = f"LibriSpeech/dev-clean/{spk}/{ch}/{utt}.flac"
        import soundfile as sf
        with tarfile.open(tar, "r:gz") as tf:
            data = tf.extractfile(member).read()
            trans = tf.extractfile(f"LibriSpeech/dev-clean/{spk}/{ch}/{spk}-{ch}.trans.txt").read().decode()
        y, fs = sf.read(io.BytesIO(data))
        p = HERE / CFG["audio"]["prompt_path"]
        C.save_wav(p, dsp.resample(np.asarray(y, dtype=np.float64), fs, 16000), 16000)
        line = next(l for l in trans.splitlines() if l.startswith(utt + " "))
        p.with_suffix(".txt").write_text(line.split(" ", 1)[1] + "\n", encoding="utf-8")
        print(f"  {member} → {p.relative_to(HERE)}（{len(y) / fs:.2f} s）：{line.split(' ', 1)[1]}")
    print("  → model.lock\n下一步：./present.sh check")


# ── export ───────────────────────────────────────────────────
def export(rec_path: Path) -> None:
    rec = json.loads(rec_path.read_text(encoding="utf-8"))
    d = json.loads(METRICS_JSON.read_text(encoding="utf-8")) if METRICS_JSON.exists() else {
        "_note": "量測數字（進版控的出處，speech_ai/CLAUDE.md 4.8）。w08-data.json 是 make_figs_w08.py 算出來的、每次重產；量測的放這裡。"}
    items = {}
    for name, r in rec["items"].items():
        items[name] = {k: r[k] for k in ("model", "seed", "T", "seconds", "detail_db", "detail_pct", "acoustic_s", "vocoder_s",
                                         "f0_median_hz", "f0_p10_p90_hz", "pauses") if k in r}
        if "length" in r:
            items[name]["length"] = {k: v for k, v in r["length"].items() if k != "ref_text_used"}
            items[name].update(nfe=r.get("nfe"), cfg_strength_alpha=r.get("cfg_strength"), w_slide=r.get("w_slide"), sway=r.get("sway"))
    d["demo1"] = {"source": "demos/w08_d1_mean_vs_sample/runs/rehearsal/rehearsal.json", "time": rec["time"], "tag": rec.get("tag", ""),
                  "text": rec["text"], "models": rec["models"], "front_ends": rec["front_ends"], "real": rec["real"],
                  "f5_prompt": {k: v for k, v in rec.get("f5_prompt", {}).items() if k != "length"}, "items": items, "pairs": rec["pairs"],
                  "definitions": {"detail_db": "RMS over a band and over active frames of (log-mel in dB minus its own 3x3 local mean); active = frame mean within 25 dB of the loudest frame",
                                  "detail_pct": "100 x model detail / real-recording detail, both in that model's own front end",
                                  "harmonics_resolvable_below_hz": "first mel-bin centre where the spacing to the next bin exceeds F0/2 (F0 = median of the real recording)",
                                  "w_slide": "the slides' CFG weight w = F5 cfg_strength (alpha, F5 Eq. 4) + 1",
                                  "pauses": "10 ms frame energy, threshold max(peak - 35, 10th percentile + 15) dB, quiet runs >= 100 ms between the first and last active frame",
                                  "f0_median_hz": "median of the autocorrelation F0 (W2 dsp.py, 40 ms / 10 ms, 60-300 Hz) over voiced frames",
                                  "what_we_heard": "filled by hand after the rehearsal (w08.qmd p12)"}}
    METRICS_JSON.parent.mkdir(parents=True, exist_ok=True)
    METRICS_JSON.write_text(json.dumps(d, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    print(f"  → {METRICS_JSON}（demo1）")


# ── selftest ─────────────────────────────────────────────────
def selftest() -> None:
    C.banner("selftest", C.AMBER)
    fails = []

    def ok(cond: bool, msg: str):
        print(f"  {'OK ' if cond else C.RED + 'FAIL' + C.RESET}  {msg}")
        if not cond:
            fails.append(msg)

    fe = front_ends(real=False)
    st, f5 = fe["speecht5"], fe["f5"]
    # 1. 頻率軸：SpeechT5 前端 80 個 bin、中心在 80–7600 Hz 內、Slaney 尺度 1 kHz 以下等距；F5 前端 100 個、0–12 kHz、HTK
    ok(len(st.centers) == 80 and 80 <= st.centers[0] < 200 and st.centers[-1] < 7600, f"SpeechT5 前端：80 bins，{st.centers[0]:.0f}–{st.centers[-1]:.0f} Hz")
    lin = np.diff(st.centers[st.centers < 900])
    ok(np.ptp(lin) <= 2 * (8000 / 512) + 1e-9, f"SpeechT5 前端 1 kHz 以下等距（間距 {lin.min():.1f}–{lin.max():.1f} Hz，FFT bin 15.6 Hz 的量化）")
    ok(len(f5.centers) == 100 and f5.centers[-1] < 12000, f"F5 前端：100 bins，最高中心 {f5.centers[-1]:.0f} Hz")
    ok(st.resolvable_below(110) < st.resolvable_below(220), "F0 越高，分得出諧波的上限越高")
    # 2. 細紋理：平滑的 mel → 0；加雜訊 → > 0；自己對自己的比 = 1
    T = 300
    t = np.linspace(0, 1, T)
    smooth = 40 * np.outer(np.exp(-np.arange(80) / 30), 1 + 0.2 * np.sin(2 * np.pi * t))
    ok(fine_detail(smooth, st, MAIN_BAND) < 0.05, f"平滑的 mel：細紋理 {fine_detail(smooth, st, MAIN_BAND):.3f} dB")
    noisy = smooth + np.random.default_rng(0).standard_normal(smooth.shape) * 2
    ok(fine_detail(noisy, st, MAIN_BAND) > 1.0, f"加 2 dB 雜訊：細紋理 {fine_detail(noisy, st, MAIN_BAND):.2f} dB")
    # 3. F5 的長度公式：對一個手算的例子（prompt 480 frames、"Hi." → "Hi.  " 5 bytes、生成 20 bytes → ⌊480·20/5⌋ = 1920）
    L = f5_length(480 * 256, "Hi.", "x" * 20)
    ok(L["ref_bytes"] == 5 and L["gen_frames"] == 1920, f"長度公式：ref {L['ref_bytes']} bytes、T_gen {L['gen_frames']}")
    L2 = f5_length(480 * 256, "Hi.", "短")
    ok(L2["speed"] == 0.3 and L2["gen_bytes"] == 3, "生成的文字 < 10 bytes 時 speed 換成 0.3；中文字一個 3 bytes")
    ok(f5_ref_text("Near the board") == "Near the board.  " and f5_ref_text("ok.") == "ok.  ", "ref_text 的句尾：補 '. ' 再補 ' '")
    # 4. 裁前後靜音
    x = np.concatenate([np.zeros(2400), 0.3 * np.sin(2 * np.pi * 200 * np.arange(24000) / 24000), np.zeros(4800)])
    y = trim_edges(x, 24000)
    ok(abs(len(y) - 24000) <= 240, f"trim_edges：{len(x)} → {len(y)} samples")
    tone = 0.3 * np.sin(2 * np.pi * 150 * np.arange(16000) / 16000)
    pz = pauses(np.concatenate([np.zeros(3200), tone, np.zeros(6400), tone, np.zeros(3200)]), 16000)
    ok(pz["n"] == 1 and abs(pz["longest_s"] - 0.40) <= 0.02 and abs(pz["longest_start_s"] - 1.20) <= 0.02,
       f"pauses：中間 0.40 s 的停頓量到 {pz['longest_s']} s @ {pz['longest_start_s']} s")
    # 5. fake 流程：兩個模型的替身、視窗的全部鍵、表、彩排的圖、replay 讀回
    import matplotlib
    matplotlib.use("Agg")
    tgt = HERE / CFG["audio"]["target_path"]
    if not tgt.exists():
        print("  --   沒有 audio/target.wav，跳過 fake 流程（先 ./present.sh fetch）")
    else:
        out = Out()
        folder = HERE / "runs" / "_selftest"
        xr = C.load_wav(tgt, 16000)
        models = {k: FakeModel(k, fe[k], fe[k].mel(xr, 16000), xr) for k in fe}
        d = Demo(out, fe, models, folder, tag="selftest FAKE")
        d.load_real(); d.compute()
        ok(len(d.items) == len(ST5_SEEDS) + len(F5_SEEDS), f"fake：{len(d.items)} 個版本")
        r_st, r_f5 = d.items[item_name("speecht5", 1)], d.items[item_name("f5", 1)]
        ok(r_st["detail_pct"][band_name(MAIN_BAND)] < 80 < r_f5["detail_pct"][band_name(MAIN_BAND)],
           f"fake：區域平均的細紋理低（{r_st['detail_pct'][band_name(MAIN_BAND)]:.0f}%）、加紋理的高（{r_f5['detail_pct'][band_name(MAIN_BAND)]:.0f}%）")
        C.play = lambda *a, **k: None                                                   # selftest 不出聲
        d.make_fig()
        for k in "12gx3zst" + "x":
            d.dispatch(k)
        d.table()
        d.rehearse_figs(); d.save()
        ok(all((folder / n).exists() for n in ("pair.png", "pair_zoom.png", "seeds.png", "speecht5_seeds.png", "session.json")), "fake：圖與紀錄存了")
        shutil.rmtree(folder, ignore_errors=True)
    # 6. 有彩排紀錄：對 w08-metrics.json
    rec_path = REHEARSAL / "rehearsal.json"
    if rec_path.exists() and METRICS_JSON.exists():
        rec = json.loads(rec_path.read_text(encoding="utf-8"))
        mj = json.loads(METRICS_JSON.read_text(encoding="utf-8")).get("demo1", {})
        bad = [n for n, r in rec["items"].items() if mj.get("items", {}).get(n, {}).get("T") != r["T"]
               or abs(mj["items"][n]["detail_pct"][band_name(MAIN_BAND)] - r["detail_pct"][band_name(MAIN_BAND)]) > 1e-6]
        ok(not bad, f"彩排紀錄對 w08-metrics.json → demo1：{len(bad)} 處不一致")
        ok(all((REHEARSAL / f"{n}_mel.npy").exists() and (HERE / r["wav"]).exists() for n, r in rec["items"].items()), "彩排的 mel 與 .wav 都在（replay 用）")
        import matplotlib
        matplotlib.use("Agg")
        C.play = lambda *a, **k: None
        d = build_replay(Out())
        d.make_fig()
        for k in "12gx3zst":
            d.dispatch(k)
        ok(len(d.items) == len(rec["items"]), f"replay：讀回 {len(d.items)} 個版本、鍵都走得過")
    else:
        print("  --   沒有彩排紀錄或 w08-metrics.json，跳過比對")
    print(f"\n  {'全部通過' if not fails else str(len(fails)) + ' 項失敗'}")
    sys.exit(1 if fails else 0)


# ── main ─────────────────────────────────────────────────────
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["show", "rehearse", "replay", "check", "fetch", "fake", "selftest", "export"])
    ap.add_argument("--device", help="show／rehearse／check：蓋過 demo_config.toml 的 device（cpu／mps／auto），三個模型一起")
    ap.add_argument("--no-plot", action="store_true", help="show／replay／fake：只在終端機，不開圖")
    args = ap.parse_args()

    if args.cmd == "check":
        check(args.device); return
    if args.cmd == "fetch":
        fetch(); return
    if args.cmd == "selftest":
        selftest(); return
    if args.cmd == "export":
        export(REHEARSAL / "rehearsal.json"); return

    if args.cmd == "rehearse" or args.no_plot:
        import matplotlib
        matplotlib.use("Agg")

    out = Out()
    out.banner("W8 Demo 1：a regressed mel, a drawn mel, three draws")
    d = build_replay(out) if args.cmd == "replay" else build(args.cmd, out, args.device)

    if args.cmd == "rehearse":
        d.table()
        d.rehearse_figs()
        d.save()
        export(d.folder / "rehearsal.json")
        print("\n下一步：./present.sh 自己按一遍 1 2 g x x 3 z s t；聽到的寫進 p12 的表（what we heard），runs/rehearsal/*.png 看過。")
        return

    d.make_fig()
    d.menu()
    if args.no_plot:
        C.note("  --no-plot：鍵在終端機按。")
        while d.dispatch(C.getkey()):
            pass
    else:
        C.note("  鍵在視窗裡按（視窗要有焦點）。終端機這裡是紀錄。")
        d.plt.show()
    if args.cmd in ("show", "fake"):
        d.save()


if __name__ == "__main__":
    main()

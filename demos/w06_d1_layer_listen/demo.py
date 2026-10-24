# -*- coding: utf-8 -*-
# 此檔由 bin/publish_demos.py 從 private repo 的 <course>/demos/ 複製而來，請勿在公開 repo 直接編輯。
"""W6 demo 1：逐層重建聆聽——同一句話經三個 neural codec（Mimi／EnCodec／DAC），一層一層加回去；只留第一層；華語句子聽聲調。

  show       課堂：載三個 codec、對兩段輸入算好全部設定的重建（幾秒到幾十秒），開一個三格的視窗；鍵在視窗裡按（下面的鍵）。離開時存 runs/live/<時間>/
  rehearse   課前：不互動，全部設定算一遍，重建的 .wav、指標、log-mel 圖存 runs/rehearsal/（進版控 = 現場的退路），寫 slides/assets/w06/w06-metrics.json → demo1
  replay     現場出狀況時：從 runs/rehearsal/ 讀回 .wav 與指標，同一個視窗、同樣的鍵，不碰 torch
  record     課前：錄華語句子（或 --from 匯入一個 WAV）→ ../_media/w06_d1/mandarin.wav（不進版控、不上站）
  check      課前體檢：版本、三個 codec 載不載得起、config 裡的 frame rate／codebook 數／大小對 encode 出來的矩陣、切列 decode 對層數參數、mps 對 cpu → runs/check.json
  fetch      課前一週：下載三個 codec（鎖 commit → model.lock）、把 W3 那句複製到 audio/（要網路）
  fake       沒有模型也能演練流程：低通＋雜訊模擬層數（畫面標明 FAKE，不是給學生看的）
  selftest   不開視窗：bitrate 公式、F0 估計、指標的恆等、fake 流程；有彩排紀錄時對 w06-metrics.json
  export     只重寫 w06-metrics.json → demo1（從 runs/rehearsal/rehearsal.json）

畫面上的鍵（順序照投影片 p33／p34 講稿的【操作】）：
  o  播原句                       1 2 4 8  Mimi 只留前 1／2／4／8 層（12.5 Hz；1 層 = 蒸餾 WavLM 的 semantic 層）
  e  EnCodec 24 kHz，每按一次往上一級：1.5 → 3 → 6 → 12 → 24 kbps（2／4／8／16／32 層）
  d  DAC 24 kHz，每按一次往上一級：1 → 2 → 4 → 8 層
  m  換輸入（LibriSpeech 那句 ↔ 我錄的華語句）        f  畫兩條 F0 曲線（原句與目前的重建）
  r  再播一次目前的設定            t  印指標表（bitrate、LSD、SNR、F0 RMSE、voicing 一致率）    h 選單    q 離開

「只留前 n_q 層」的做法：每個 codec 對每段輸入 encode 一次（全部的層），把 (N_q, T) 的 code 矩陣切成前 n_q 列再 decode——
重建 = 前 n_q 層 codeword 的和，這就是 residual VQ 的定義，也是 quantizer dropout 訓練出來的那個性質。check 對過：切列 decode 與
用該類別自己的層數參數（bandwidth／n_quantizers／num_quantizers）encode 得到同一組 code。
bitrate B = f_r · n_q · log2 V，f_r 與 V 從載入的 config 讀、並以 encode 出來的 T 對輸入長度量一次（兩者都印）。
客觀指標只是參考——判準是耳朵（p34 的 claim）：LSD、SNR（GAN 訓練的 codec 不保相位，SNR 接近 0 dB 是正常的）、有聲 frame 的 F0 RMSE 與 voicing 一致率。
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
import time
import tomllib
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import common as C  # noqa: E402
import dsp          # noqa: E402

CFG = tomllib.loads((HERE / "demo_config.toml").read_text(encoding="utf-8"))
FS = CFG["audio"]["fs"]
CODECS = CFG["codecs"]["order"]
INPUTS = CFG["inputs"]["order"]
MEDIA = HERE.parent / "_media" / "w06_d1"
MODEL_LOCK = HERE / "model.lock"
METRICS_JSON = HERE.parents[1] / "slides" / "assets" / "w06" / "w06-metrics.json"     # parents[1] = speech_ai/
MODEL_PATTERNS = ["*.json", "*.txt", "*.safetensors", "*.bin", "*.model"]
# Mimi 的 model card 寫 CC-BY 4.0；EnCodec、DAC 的權重授權以各自的 model card 為準（fetch 時印 card 上的 license 欄位）。

MIMI_STAGES = CFG["codecs"]["mimi"]["stages"]


# ── 音訊 ─────────────────────────────────────────────────────
def input_path(key: str) -> Path:
    return (HERE / CFG["inputs"][key]["path"]).resolve()


def load_input(key: str) -> np.ndarray:
    """mono float64、24 kHz。"""
    return C.load_wav(input_path(key), FS)


def trim_silence(x: np.ndarray, fs: int) -> np.ndarray:
    """裁掉前後的靜音：門檻 = max(峰值 − 35 dB, 底噪 + 15 dB)，連續 50 ms（W3 Demo 1 的規則）。"""
    c = CFG["inputs"]["mandarin"]
    hop = int(fs * 0.010)
    n = len(x) // hop
    if n < 3:
        return x
    e = 20 * np.log10(np.sqrt(np.mean(x[:n * hop].reshape(n, hop) ** 2, axis=1)) + 1e-9)
    thr = max(e.max() - c["trim_db_below_peak"], np.percentile(e, 10) + c["trim_db_above_floor"])
    need = max(1, int(round(c["trim_min_ms"] / 10)))
    on = e > thr
    runs = np.convolve(on.astype(int), np.ones(need, dtype=int), mode="valid") >= need
    idx = np.flatnonzero(runs)
    if len(idx) == 0:
        return x
    a = max(0, (idx[0] - 5) * hop)
    b = min(len(x), (idx[-1] + need + 5) * hop)
    return x[a:b]


def logmel_db(x: np.ndarray) -> np.ndarray:
    """(n_mels, frames) 的 log-mel（dB），顯示用。"""
    d = CFG["display"]
    L = int(round(d["win_ms"] * FS / 1000)); H = int(round(d["hop_ms"] * FS / 1000)); N = d["n_fft"]
    X = dsp.stft(x, L, H, N)
    M = dsp.mel_filterbank(FS, N, d["n_mels"], d["scale"])
    S = 10 * np.log10(M @ (np.abs(X) ** 2) + 1e-10)
    return np.maximum(S, S.max() + d["db_floor"])


def f0_track(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(時間 s, F0 Hz 或 nan)，W2 的 dsp.f0_autocorr 逐 frame。"""
    c = CFG["f0"]
    flen = int(round(c["frame_ms"] * FS / 1000)); hop = int(round(c["hop_ms"] * FS / 1000))
    ts, f0 = [], []
    for a in range(0, max(1, len(x) - flen), hop):
        v, _ = dsp.f0_autocorr(x[a:a + flen], FS, c["floor_hz"], c["ceiling_hz"], c["voicing"])
        ts.append((a + flen / 2) / FS); f0.append(v)
    return np.array(ts), np.array(f0, dtype=float)


def metrics(ref: np.ndarray, est: np.ndarray, f0_ref: np.ndarray | None = None) -> dict:
    """客觀參考：LSD、SNR、F0 RMSE（兩邊都有聲的 frame）、voicing 一致率。est 先對齊到 ref 的長度。"""
    m = CFG["metrics"]
    n = min(len(ref), len(est))
    ref, est = ref[:n], est[:n]
    L = int(round(m["lsd_win_ms"] * FS / 1000)); H = int(round(m["lsd_hop_ms"] * FS / 1000))
    r = {"lsd_db": dsp.log_spectral_distance(ref, est, L, H, m["lsd_n_fft"]), "snr_db": dsp.snr_db(ref, est)}
    _, fr = f0_track(ref) if f0_ref is None else (None, f0_ref)
    _, fe = f0_track(est)
    k = min(len(fr), len(fe)); fr, fe = fr[:k], fe[:k]
    vr, ve = ~np.isnan(fr), ~np.isnan(fe)
    both = vr & ve
    r["voicing_agree"] = float(np.mean(vr == ve)) if k else float("nan")
    r["voiced_ref"] = int(vr.sum()); r["voiced_both"] = int(both.sum())
    r["f0_rmse_hz"] = float(np.sqrt(np.mean((fr[both] - fe[both]) ** 2))) if both.any() else float("nan")
    r["f0_mean_abs_cents"] = float(np.mean(np.abs(1200 * np.log2(fe[both] / fr[both])))) if both.any() else float("nan")
    return r


def bitrate_bps(frame_rate: float, n_q: int, codebook_size: int) -> float:
    return float(frame_rate * n_q * math.log2(codebook_size))


def fmt_bps(b: float) -> str:
    return f"{b / 1000:.2f} kbps" if b >= 1000 else f"{b:.0f} bps"


# ── 模型 ─────────────────────────────────────────────────────
def read_lock() -> dict[str, str]:
    if not MODEL_LOCK.exists():
        return {}
    out = {}
    for line in MODEL_LOCK.read_text(encoding="utf-8").splitlines():
        if line.strip() and not line.startswith("#"):
            repo, sha = line.split()
            out[repo] = sha
    return out


def model_dir(key: str, offline: bool = True) -> str:
    from huggingface_hub import snapshot_download
    repo = CFG["codecs"][key]["repo"]
    rev = read_lock().get(repo)
    return snapshot_download(repo, revision=rev, local_files_only=offline, allow_patterns=MODEL_PATTERNS)


def versions() -> dict:
    v = C.versions()
    from importlib import metadata
    for pkg in ("torch", "transformers", "huggingface_hub", "soundfile"):
        try:
            v[pkg] = metadata.version(pkg)
        except metadata.PackageNotFoundError:
            v[pkg] = None
    return v


class Codec:
    """transformers 的 MimiModel／EncodecModel／DacModel，同一個介面：encode_all(x) → (N_q, T) 的 code 矩陣；decode(codes[:n_q]) → 24 kHz 波形。"""

    def __init__(self, key: str, device: str):
        import torch
        from transformers import AutoFeatureExtractor
        self.torch, self.key = torch, key
        self.label = CFG["codecs"][key]["label"]
        self.repo = CFG["codecs"][key]["repo"]
        self.path = model_dir(key)
        self.fe = AutoFeatureExtractor.from_pretrained(self.path)
        if key == "mimi":
            from transformers import MimiModel as Cls
        elif key == "encodec":
            from transformers import EncodecModel as Cls
        else:
            from transformers import DacModel as Cls
        self.model = Cls.from_pretrained(self.path).eval()
        if device == "auto":
            device = "mps" if torch.backends.mps.is_available() else "cpu"
        self.device = torch.device(device)
        self.model.to(self.device)
        cfg = self.model.config
        self.n_params = sum(p.numel() for p in self.model.parameters())
        self.fs_model = int(getattr(cfg, "sampling_rate", FS))
        self.codebook_size = int(getattr(cfg, "codebook_size"))
        fr = getattr(cfg, "frame_rate", None)
        if fr is None and hasattr(cfg, "hop_length"):
            fr = self.fs_model / cfg.hop_length
        self.frame_rate_cfg = float(fr) if fr is not None else float("nan")
        self.n_q_cfg = int(getattr(cfg, "num_quantizers", getattr(cfg, "n_codebooks", 0)) or 0)
        self.n_semantic = int(getattr(cfg, "num_semantic_quantizers", 0) or 0)
        self.commit = read_lock().get(self.repo)
        self._scales = None
        self._mask = None

    def _inputs(self, x: np.ndarray):
        inp = self.fe(raw_audio=x.astype(np.float32), sampling_rate=self.fs_model, return_tensors="pt")
        iv = inp["input_values"].to(self.device)
        mask = inp["padding_mask"].to(self.device) if "padding_mask" in inp else None
        return iv, mask

    def encode_all(self, x: np.ndarray) -> np.ndarray:
        """全部的層。回 (N_q, T) 的 int64 numpy。"""
        torch = self.torch
        assert self.fs_model == FS, f"{self.label} 吃 {self.fs_model} Hz，demo 的音訊是 {FS} Hz"
        iv, mask = self._inputs(x)
        with torch.no_grad():
            if self.key == "encodec":
                bw = max(self.model.config.target_bandwidths)
                out = self.model.encode(iv, mask, bandwidth=bw)
                codes = out.audio_codes                                   # (chunks, B, N_q, T)
                self._scales = out.audio_scales
                assert codes.shape[0] == 1, f"EnCodec 切了 {codes.shape[0]} 個 chunk（24 kHz 模型應該是 1）"
                c = codes[0, 0]
            elif self.key == "mimi":
                out = self.model.encode(iv, mask)
                c = out.audio_codes[0]                                    # (N_q, T)
            else:
                out = self.model.encode(iv)
                c = out.audio_codes[0]                                    # (N_q, T)
            self._mask = mask
        if self.device.type == "mps":
            torch.mps.synchronize()
        return c.detach().cpu().numpy().astype(np.int64)

    def encode_with_param(self, x: np.ndarray, n_q: int) -> np.ndarray:
        """用該類別自己的層數參數 encode（check 用來對切列）。"""
        torch = self.torch
        iv, mask = self._inputs(x)
        with torch.no_grad():
            if self.key == "encodec":
                bw = n_q * self.frame_rate_cfg * math.log2(self.codebook_size) / 1000
                tb = self.model.config.target_bandwidths
                bw = min(tb, key=lambda b: abs(b - bw))
                c = self.model.encode(iv, mask, bandwidth=bw).audio_codes[0, 0]
            elif self.key == "mimi":
                c = self.model.encode(iv, mask, num_quantizers=n_q).audio_codes[0]
            else:
                c = self.model.encode(iv, n_quantizers=n_q).audio_codes[0]
        return c.detach().cpu().numpy().astype(np.int64)

    def decode(self, codes: np.ndarray, length: int) -> np.ndarray:
        """codes: (n_q, T)。回 mono float64，裁／補到 length。"""
        torch = self.torch
        t = torch.as_tensor(codes, dtype=torch.long, device=self.device)[None]   # (1, n_q, T)
        with torch.no_grad():
            if self.key == "encodec":
                y = self.model.decode(t[None], self._scales, self._mask).audio_values
            elif self.key == "mimi":
                y = self.model.decode(t, self._mask).audio_values
            else:
                y = self.model.decode(audio_codes=t).audio_values
        if self.device.type == "mps":
            torch.mps.synchronize()
        y = y.detach().float().cpu().numpy().reshape(-1)
        if len(y) >= length:
            return y[:length].astype(np.float64)
        return np.pad(y, (0, length - len(y))).astype(np.float64)

    def info(self) -> dict:
        return {"repo": self.repo, "commit": self.commit, "device": self.device.type, "n_params": self.n_params,
                "sampling_rate": self.fs_model, "frame_rate_cfg": self.frame_rate_cfg, "codebook_size": self.codebook_size,
                "n_q_cfg": self.n_q_cfg, "n_semantic_quantizers": self.n_semantic}


class FakeCodec:
    """沒有 torch 時的替身：低通 + 雜訊，cutoff 隨層數上升。畫面標 FAKE。"""
    DEFAULTS = {"mimi": (12.5, 2048, 8), "encodec": (75.0, 1024, 32), "dac": (75.0, 1024, 32)}

    def __init__(self, key: str, device: str = "cpu"):
        self.key, self.label = key, CFG["codecs"][key]["label"] + " [FAKE]"
        self.repo, self.commit, self.device = "fake", None, type("D", (), {"type": "fake"})()
        self.frame_rate_cfg, self.codebook_size, self.n_q_cfg = self.DEFAULTS[key]
        self.fs_model, self.n_params, self.n_semantic = FS, 0, 1 if key == "mimi" else 0
        self._x = None
        self._rng = np.random.default_rng(20261018)

    def encode_all(self, x: np.ndarray) -> np.ndarray:
        self._x = x
        T = int(math.ceil(len(x) / FS * self.frame_rate_cfg))
        return self._rng.integers(0, self.codebook_size, size=(self.n_q_cfg, T), dtype=np.int64)

    def encode_with_param(self, x: np.ndarray, n_q: int) -> np.ndarray:
        return self.encode_all(x)[:n_q]

    def decode(self, codes: np.ndarray, length: int) -> np.ndarray:
        from scipy.signal import butter, sosfiltfilt
        n_q = codes.shape[0]
        cut = CFG["fake"]["cutoff_hz_per_stage"]
        fc = min(cut[min(n_q, len(cut)) - 1], 0.45 * FS)
        sos = butter(6, fc / (FS / 2), output="sos")
        y = sosfiltfilt(sos, self._x[:length])
        y = y + self._rng.standard_normal(len(y)) * (10 ** (CFG["fake"]["noise_db"] / 20)) * (np.std(y) + 1e-9) / max(1, n_q)
        return y

    def info(self) -> dict:
        return {"repo": "fake", "commit": None, "device": "fake", "n_params": 0, "sampling_rate": FS,
                "frame_rate_cfg": self.frame_rate_cfg, "codebook_size": self.codebook_size, "n_q_cfg": self.n_q_cfg,
                "n_semantic_quantizers": self.n_semantic}


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


def stage_name(codec: str, n_q: int) -> str:
    return f"{codec}_nq{n_q:02d}"


# ── 主體 ─────────────────────────────────────────────────────
class Demo:
    """一個 session：兩段輸入 × 三個 codec × 各自的 stages。recon[(input, codec, n_q)] = {"wav": Path, "metrics": …, "codes_shape": …}。"""

    def __init__(self, out: Out, codecs: dict, folder: Path, tag: str = ""):
        self.out, self.codecs, self.folder, self.tag = out, codecs, folder, tag
        self.inputs: dict[str, np.ndarray] = {}
        self.recon: dict[tuple, dict] = {}
        self.f0_cache: dict[str, tuple] = {}
        self.cur_input = INPUTS[0]
        self.cur = None                          # (codec, n_q) 目前顯示的設定；None = 原句
        self.cycle = {"encodec": -1, "dac": -1}
        self.log: dict = {"time": C.now(), "versions": versions(), "tag": tag, "fs": FS,
                          "codecs": {k: c.info() for k, c in codecs.items()}, "inputs": {}, "recon": {}, "sections": {}}
        self.fig = self.axes = self.plt = None
        self.show_f0 = False

    def input_dir(self, key: str) -> Path:
        """這段輸入的原句與重建放哪：private = true 的輸入（我的華語錄音）放 ../_media/w06_d1/<runs 下的相對路徑>/<key>/——
        不進版控、也不在 publish_demos.py 會走的 runs/rehearsal/ 裡；其餘放 runs/<…>/<key>/。"""
        if CFG["inputs"][key].get("private"):
            return MEDIA / self.folder.relative_to(HERE / "runs") / key
        return self.folder / key

    # ── 準備 ──
    def load_inputs(self) -> None:
        for key in INPUTS:
            p = input_path(key)
            if not p.exists():
                self.out(f"  {C.AMBER}沒有 {key} 的音訊（{p}）{C.RESET}——" + ("先跑 ./present.sh fetch" if key == "libri" else "先跑 ./present.sh record；這次跳過華語那段"))
                continue
            x = load_input(key)
            self.inputs[key] = x
            self.log["inputs"][key] = {"label": CFG["inputs"][key]["label"], "path": str(CFG["inputs"][key]["path"]), "seconds": round(len(x) / FS, 3), "peak": float(np.max(np.abs(x)))}
        if not self.inputs:
            sys.exit("兩段輸入都沒有，先跑 ./present.sh fetch")
        if self.cur_input not in self.inputs:
            self.cur_input = next(iter(self.inputs))
        for key in self.inputs:
            self.input_dir(key).mkdir(parents=True, exist_ok=True)
            # 原句與它全部的重建在 compute_all 結束時用同一個增益寫檔（不各自正規化，否則響度差會蓋過音質差）

    def compute(self, key: str, codec: str) -> None:
        """一段輸入 × 一個 codec：encode 一次、每個 stage decode 一次，存 wav 與指標。"""
        o = self.out
        x = self.inputs[key]
        cd = self.codecs[codec]
        t0 = time.perf_counter()
        codes = cd.encode_all(x)
        t_enc = time.perf_counter() - t0
        N_q, T = codes.shape
        fr_meas = T / (len(x) / FS)
        o(f"  {cd.label:<16s} {key:<9s} encode {t_enc * 1000:6.0f} ms   codes {N_q} × {T}   f_r cfg {cd.frame_rate_cfg:g} Hz · 量到 {fr_meas:.2f} Hz   V = {cd.codebook_size}")
        if key not in self.f0_cache:
            self.f0_cache[key] = f0_track(x)
        stages = [s for s in CFG["codecs"][codec]["stages"] if s <= N_q]
        if len(stages) < len(CFG["codecs"][codec]["stages"]):
            o(f"    {C.AMBER}權重只有 {N_q} 層，stages 裡超過的跳過{C.RESET}")
        for n_q in stages:
            t1 = time.perf_counter()
            y = cd.decode(codes[:n_q], len(x))
            t_dec = time.perf_counter() - t1
            m = metrics(x, y, self.f0_cache[key][1])
            B = bitrate_bps(cd.frame_rate_cfg, n_q, cd.codebook_size)
            wav = self.input_dir(key) / f"{stage_name(codec, n_q)}.wav"
            rec = {"codec": codec, "n_q": n_q, "N_q": N_q, "T": T, "frame_rate_cfg": cd.frame_rate_cfg, "frame_rate_measured": round(fr_meas, 3),
                   "codebook_size": cd.codebook_size, "bitrate_bps": round(B, 1), "decode_ms": round(t_dec * 1000, 1), "encode_ms": round(t_enc * 1000, 1),
                   "wav": os.path.relpath(wav, HERE), "metrics": m, "codes_used": int(len(np.unique(codes[:n_q], axis=None)))}
            self.recon[(key, codec, n_q)] = {"wav": wav, "y": y, "rec": rec}
            self.log["recon"].setdefault(key, {})[stage_name(codec, n_q)] = rec
            o(f"    n_q = {n_q:>2d}   B = {fmt_bps(B):>10s}   LSD {m['lsd_db']:5.2f} dB   SNR {m['snr_db']:6.2f} dB   F0 RMSE {m['f0_rmse_hz']:6.1f} Hz   voicing agree {m['voicing_agree']:.2f}   decode {t_dec * 1000:5.0f} ms")
        self.log["recon"][key]["_codes_" + codec] = {"N_q": N_q, "T": T, "frame_rate_measured": round(fr_meas, 3), "encode_ms": round(t_enc * 1000, 1)}

    def compute_all(self) -> None:
        for key in self.inputs:
            self.out(f"\n{C.BOLD}{CFG['inputs'][key]['label']}{C.RESET}  {self.log['inputs'][key]['seconds']} s")
            for codec in CODECS:
                if codec in self.codecs:
                    self.compute(key, codec)
            self.write_wavs(key)

    def write_wavs(self, key: str) -> None:
        """一段輸入的原句與全部重建用同一個增益寫檔：峰值取所有版本的最大值，對到 peak_target。"""
        x = self.inputs[key]
        ys = [r["y"] for (k, _, _), r in self.recon.items() if k == key]
        peak = max([float(np.max(np.abs(x)))] + [float(np.max(np.abs(y))) for y in ys]) / CFG["audio"]["peak_target"] * 0.89
        C.save_wav(self.input_dir(key) / "original.wav", x, FS, peak=peak)
        for (k, _, _), r in self.recon.items():
            if k == key:
                C.save_wav(r["wav"], r["y"], FS, peak=peak)
        self.log["inputs"][key]["gain"] = round(0.89 / peak, 4)

    # ── 視窗 ──
    def make_fig(self):
        import matplotlib.pyplot as plt
        self.plt = plt
        d = CFG["display"]
        self.fig, self.axes = plt.subplots(3, 1, figsize=(d["fig_w"], d["fig_h"]), gridspec_kw={"height_ratios": [1, 1, 0.75]})
        self.fig.suptitle("W6 Demo 1 — adding the stages back, one at a time" + (f"   [{self.tag}]" if self.tag else ""), fontsize=13)
        self.fig.canvas.mpl_connect("key_press_event", self.on_key)
        self.draw()
        return self.fig

    def redraw(self):
        if self.fig is not None:
            self.fig.canvas.draw_idle()
            self.plt.pause(0.05)

    def _mel_panel(self, ax, x: np.ndarray, title: str):
        ax.clear()
        S = logmel_db(x)
        hop = CFG["display"]["hop_ms"] / 1000
        ax.imshow(S, origin="lower", aspect="auto", cmap="magma", extent=[0, S.shape[1] * hop, 0, S.shape[0]])
        ax.set_title(title, fontsize=11, loc="left")
        ax.set_ylabel("mel bin"); ax.set_xlabel("time (s)")

    def draw(self):
        if self.fig is None:
            return
        key = self.cur_input
        x = self.inputs[key]
        self._mel_panel(self.axes[0], x, f"original — {CFG['inputs'][key]['label']}  ({len(x) / FS:.2f} s)")
        ax = self.axes[1]
        if self.cur is None:
            ax.clear(); ax.set_title("reconstruction — press 1 / 2 / 4 / 8 (Mimi), e (EnCodec), d (DAC)", fontsize=11, loc="left")
            ax.set_xticks([]); ax.set_yticks([])
        else:
            codec, n_q = self.cur
            r = self.recon[(key, codec, n_q)]
            rec = r["rec"]
            cd = self.codecs[codec]
            sem = "  · stage 1 = distilled semantic" if codec == "mimi" and n_q == 1 and cd.n_semantic else ""
            self._mel_panel(ax, r["y"], f"{cd.label}   n_q = {n_q} of {rec['N_q']}   f_r = {rec['frame_rate_cfg']:g} Hz   V = {rec['codebook_size']}   B = {fmt_bps(rec['bitrate_bps'])}{sem}")
        ax = self.axes[2]
        ax.clear()
        if self.show_f0:
            ts, f0 = self.f0_cache.get(key) or f0_track(x)
            self.f0_cache[key] = (ts, f0)
            ax.plot(ts, f0, ".", ms=3, color="#333333", label="original")
            if self.cur is not None:
                codec, n_q = self.cur
                r = self.recon[(key, codec, n_q)]
                if "f0" not in r:
                    r["f0"] = f0_track(r["y"])
                t2, f2 = r["f0"]
                ax.plot(t2, f2, ".", ms=3, color="#c8742a", label=f"{self.codecs[codec].label} n_q = {n_q}")
                m = r["rec"]["metrics"]
                ax.set_title(f"$F_0$ (autocorrelation, W2 dsp.py) — RMSE {m['f0_rmse_hz']:.1f} Hz on frames voiced in both · voicing agree {m['voicing_agree']:.2f} · a reference, not the judge", fontsize=10, loc="left")
            else:
                ax.set_title("$F_0$ — original (press a stage key to overlay the reconstruction)", fontsize=10, loc="left")
            ax.set_ylim(CFG["f0"]["floor_hz"] - 10, CFG["f0"]["ceiling_hz"] + 10); ax.set_xlim(0, len(x) / FS)
            ax.set_ylabel("Hz"); ax.set_xlabel("time (s)"); ax.legend(loc="upper right", fontsize=9); ax.grid(alpha=0.3)
        else:
            ax.set_title("press f for the $F_0$ tracks", fontsize=10, loc="left"); ax.set_xticks([]); ax.set_yticks([])
        self.fig.tight_layout()
        self.redraw()

    # ── 鍵 ──
    def _play(self, path: Path, what: str) -> None:
        self.out(f"  ▶ {what}")
        C.play(path)

    def original(self) -> None:
        key = self.cur_input
        self.cur = None
        self.draw()
        self._play(self.input_dir(key) / "original.wav", f"原句 {CFG['inputs'][key]['label']}")
        self.log["sections"].setdefault("played", []).append([key, "original"])

    def stage(self, codec: str, n_q: int) -> None:
        key = self.cur_input
        if (key, codec, n_q) not in self.recon:
            self.out(f"  {C.AMBER}沒有 {codec} n_q = {n_q} 的重建（這段輸入／這個 codec 沒算，或權重層數不夠）{C.RESET}")
            return
        self.cur = (codec, n_q)
        self.draw()
        rec = self.recon[(key, codec, n_q)]["rec"]
        m = rec["metrics"]
        self.out(f"  {self.codecs[codec].label}  n_q = {n_q}/{rec['N_q']}  f_r = {rec['frame_rate_cfg']:g} Hz  V = {rec['codebook_size']}  B = {fmt_bps(rec['bitrate_bps'])}   LSD {m['lsd_db']:.2f} dB  SNR {m['snr_db']:.2f} dB  F0 RMSE {m['f0_rmse_hz']:.1f} Hz")
        self._play(self.recon[(key, codec, n_q)]["wav"], f"{codec} n_q = {n_q}")
        self.log["sections"].setdefault("played", []).append([key, codec, n_q])

    def cycle_codec(self, codec: str) -> None:
        stages = [s for s in CFG["codecs"][codec]["stages"] if (self.cur_input, codec, s) in self.recon]
        if not stages:
            self.out(f"  {C.AMBER}{codec} 沒有可播的設定{C.RESET}"); return
        self.cycle[codec] = (self.cycle[codec] + 1) % len(stages)
        self.stage(codec, stages[self.cycle[codec]])

    def switch_input(self) -> None:
        keys = [k for k in INPUTS if k in self.inputs]
        self.cur_input = keys[(keys.index(self.cur_input) + 1) % len(keys)]
        self.cur = None
        self.cycle = {"encodec": -1, "dac": -1}
        self.out(f"\n  輸入 → {C.BOLD}{CFG['inputs'][self.cur_input]['label']}{C.RESET}")
        self.draw()

    def toggle_f0(self) -> None:
        self.show_f0 = not self.show_f0
        self.draw()

    def again(self) -> None:
        if self.cur is None:
            self.original()
        else:
            self.stage(*self.cur)

    def table(self) -> None:
        o = self.out
        key = self.cur_input
        o(f"\n{C.BOLD}{CFG['inputs'][key]['label']}{C.RESET}  — objective references (the ear is the judge)")
        o(f"  {'codec':<16s}{'n_q':>5s}{'f_r':>9s}{'V':>6s}{'B':>12s}{'LSD dB':>9s}{'SNR dB':>9s}{'F0 RMSE':>9s}{'cents':>8s}{'voicing':>9s}{'used':>6s}")
        for codec in CODECS:
            for n_q in CFG["codecs"][codec]["stages"]:
                r = self.recon.get((key, codec, n_q))
                if not r:
                    continue
                rec, m = r["rec"], r["rec"]["metrics"]
                o(f"  {self.codecs[codec].label:<16s}{n_q:>5d}{rec['frame_rate_cfg']:>9g}{rec['codebook_size']:>6d}{fmt_bps(rec['bitrate_bps']):>12s}{m['lsd_db']:>9.2f}{m['snr_db']:>9.2f}{m['f0_rmse_hz']:>9.1f}{m['f0_mean_abs_cents']:>8.0f}{m['voicing_agree']:>9.2f}{rec['codes_used']:>6d}")
        o("  B = f_r · n_q · log2 V（f_r、V 從 config 讀）；F0 RMSE 在兩邊都有聲的 frame 上；used = 前 n_q 層一共用到幾個不同的 code index")
        self.log["sections"]["table"] = key

    def menu(self) -> None:
        o = self.out
        o(f"\n{C.BOLD}p33／p34{C.RESET}  o 原句   1 2 4 8 Mimi 前 n 層   e EnCodec 1.5→3→6→12→24 kbps   d DAC 1→2→4→8 層")
        o(f"          m 換輸入（現在 {CFG['inputs'][self.cur_input]['label']}）   f F0 曲線   r 再播   t 指標表   h 選單   q 離開")

    def dispatch(self, k: str) -> bool:
        if k == "o":
            self.original()
        elif k in ("1", "2", "4", "8") and int(k) in MIMI_STAGES:
            self.stage("mimi", int(k))
        elif k == "e":
            self.cycle_codec("encodec")
        elif k == "d":
            self.cycle_codec("dac")
        elif k == "m":
            self.switch_input()
        elif k == "f":
            self.toggle_f0()
        elif k == "r":
            self.again()
        elif k == "t":
            self.table()
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
        """每段輸入一張 log-mel 的格子圖（原句 + 全部設定）、華語那段加一張 F0 圖。"""
        import matplotlib.pyplot as plt
        hop = CFG["display"]["hop_ms"] / 1000
        for key in self.inputs:
            items = [("original", self.inputs[key])] + [(f"{self.codecs[c].label}  n_q = {n}  {fmt_bps(self.recon[(key, c, n)]['rec']['bitrate_bps'])}", self.recon[(key, c, n)]["y"])
                                                         for c in CODECS for n in CFG["codecs"][c]["stages"] if (key, c, n) in self.recon]
            cols = 3; rows = math.ceil(len(items) / cols)
            fig, axes = plt.subplots(rows, cols, figsize=(15, 2.4 * rows), squeeze=False)
            for ax, (title, y) in zip(axes.ravel(), items):
                S = logmel_db(y)
                ax.imshow(S, origin="lower", aspect="auto", cmap="magma", extent=[0, S.shape[1] * hop, 0, S.shape[0]])
                ax.set_title(title, fontsize=9, loc="left"); ax.set_xticks([]); ax.set_yticks([])
            for ax in axes.ravel()[len(items):]:
                ax.axis("off")
            fig.suptitle(f"W6 Demo 1 — {CFG['inputs'][key]['label']} — log-mel of every setting", fontsize=12)
            fig.tight_layout()
            fig.savefig(self.input_dir(key) / "logmel.png", dpi=110); plt.close(fig)
            # F0
            ts, f0 = self.f0_cache[key]
            fig, ax = plt.subplots(figsize=(13, 3.6))
            ax.plot(ts, f0, ".", ms=4, color="#333333", label="original")
            for (c, n, col) in (("mimi", 1, "#c8742a"), ("mimi", 8, "#5b3fa0"), ("encodec", 2, "#2a7fc8")):
                r = self.recon.get((key, c, n))
                if r:
                    t2, f2 = f0_track(r["y"])
                    ax.plot(t2, f2, ".", ms=3, color=col, alpha=0.8, label=f"{self.codecs[c].label} n_q = {n}  (RMSE {r['rec']['metrics']['f0_rmse_hz']:.1f} Hz)")
            ax.set_ylim(CFG["f0"]["floor_hz"] - 10, CFG["f0"]["ceiling_hz"] + 10); ax.set_xlabel("time (s)"); ax.set_ylabel("$F_0$ (Hz)"); ax.grid(alpha=0.3); ax.legend(fontsize=9)
            ax.set_title(f"$F_0$ tracks — {CFG['inputs'][key]['label']} — a reference, not the judge", loc="left", fontsize=11)
            fig.tight_layout(); fig.savefig(self.input_dir(key) / "f0.png", dpi=110); plt.close(fig)

    def save(self) -> None:
        stem = "rehearsal" if self.folder.name == "rehearsal" else "session"
        (self.folder / f"{stem}.json").write_text(json.dumps(self.log, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
        (self.folder / f"{stem}.txt").write_text(self.out.text(), encoding="utf-8")
        C.note(f"  存到 {self.folder}/（{stem}.json、{stem}.txt）；各輸入的 .wav 與圖在 " + "、".join(str(self.input_dir(k)) for k in self.inputs))


# ── replay ───────────────────────────────────────────────────
class ReplayCodec:
    def __init__(self, key: str, info: dict):
        self.key, self.label = key, CFG["codecs"][key]["label"]
        self.n_semantic = info.get("n_semantic_quantizers", 0)
        self.frame_rate_cfg, self.codebook_size = info.get("frame_rate_cfg"), info.get("codebook_size")

    def info(self):
        return {}


def build_replay(out: Out) -> Demo:
    folder = HERE / "runs" / "rehearsal"
    rec_path = folder / "rehearsal.json"
    if not rec_path.exists():
        sys.exit("沒有 runs/rehearsal/rehearsal.json：先 ./present.sh rehearse")
    rec = json.loads(rec_path.read_text(encoding="utf-8"))
    codecs = {k: ReplayCodec(k, v) for k, v in rec["codecs"].items()}
    d = Demo(out, codecs, folder, tag="replay " + rec["time"])
    d.log["codecs"] = rec["codecs"]
    for key in INPUTS:
        p = d.input_dir(key) / "original.wav"
        if not p.exists():
            out(f"  {C.AMBER}沒有 {key} 的彩排音訊（{p}）{C.RESET}")
            continue
        d.inputs[key] = C.load_wav(p, FS)
        d.log["inputs"][key] = rec["inputs"].get(key, {})
        for name, r in rec["recon"].get(key, {}).items():
            if name.startswith("_"):
                continue
            wav = (HERE / r["wav"]).resolve()
            if wav.exists():
                d.recon[(key, r["codec"], r["n_q"])] = {"wav": wav, "y": C.load_wav(wav, FS), "rec": r}
    if not d.inputs:
        sys.exit("runs/rehearsal/ 裡沒有音訊")
    if d.cur_input not in d.inputs:
        d.cur_input = next(iter(d.inputs))
    out(f"  replay：{rec['time']} 的彩排，{sum(1 for v in rec['recon'].values() for n in v if not n.startswith('_'))} 個設定")
    return d


def build(cmd: str, out: Out, device: str | None = None) -> Demo:
    folder = HERE / "runs" / ("rehearsal" if cmd == "rehearse" else "live" / Path(C.now().replace(":", "")))
    if cmd == "fake":
        folder = HERE / "runs" / "live" / (C.now().replace(":", "") + ".fake")
        codecs = {k: FakeCodec(k) for k in CODECS}
        tag = "FAKE"
    else:
        try:
            import torch  # noqa: F401
        except ImportError:
            sys.exit("沒有 torch：先跑 ../w03_d3_blank_heatmap/setup.sh；或 ./present.sh fake 演練流程")
        dev = device or CFG["infer"]["device"]
        codecs = {}
        for k in CODECS:
            t0 = time.perf_counter()
            try:
                codecs[k] = Codec(k, dev)
                out(f"  載入 {codecs[k].label:<16s} {codecs[k].n_params / 1e6:6.1f} M 參數   {codecs[k].device.type}   {1000 * (time.perf_counter() - t0):5.0f} ms   f_r {codecs[k].frame_rate_cfg:g} Hz · V {codecs[k].codebook_size} · N_q {codecs[k].n_q_cfg}")
            except Exception as e:
                out(f"  {C.RED}{CFG['codecs'][k]['label']} 載入失敗：{e}{C.RESET}（跳過；先 ./present.sh check）")
        if not codecs:
            sys.exit("三個 codec 都載不起來")
        tag = "rehearsal" if cmd == "rehearse" else ""
    d = Demo(out, codecs, folder, tag=tag)
    d.load_inputs()
    d.compute_all()
    return d


# ── check ────────────────────────────────────────────────────
def check(device: str | None = None) -> None:
    C.banner("課前體檢", C.AMBER)
    res = {"time": C.now(), "versions": versions()}
    for k, v in res["versions"].items():
        print(f"  {k:<16s} {v}")
    try:
        import torch
    except ImportError:
        sys.exit("  沒有 torch：先跑 ../w03_d3_blank_heatmap/setup.sh")
    mps = bool(torch.backends.mps.is_available())
    print(f"  {'mps':<16s} built={torch.backends.mps.is_built()} available={mps}")
    res["mps_available"] = mps
    x = load_input("libri") if input_path("libri").exists() else None
    if x is None:
        print("  沒有 audio/utterance.wav，先跑 ./present.sh fetch；下面只檢查模型載入")
    res["codecs"] = {}
    for k in CODECS:
        repo = CFG["codecs"][k]["repo"]
        r = {"repo": repo, "commit": read_lock().get(repo)}
        print(f"\n  {CFG['codecs'][k]['label']}  {repo} @ {str(r['commit'])[:12]}")
        try:
            r["path"] = model_dir(k)
        except Exception as e:
            print(f"    不在快取：{e}\n    先跑 ./present.sh fetch")
            r["error"] = str(e); res["codecs"][k] = r; continue
        outs = {}
        devs = [device] if device else (["cpu"] + (["mps"] if mps else []))
        for dev in devs:
            try:
                t0 = time.perf_counter()
                cd = Codec(k, dev)
                r.update({kk: vv for kk, vv in cd.info().items() if kk not in ("device",)})
                r[f"load_{dev}_ms"] = round(1000 * (time.perf_counter() - t0), 1)
                print(f"    {dev}: 載入 {r[f'load_{dev}_ms']:.0f} ms   {cd.n_params / 1e6:.1f} M   config: fs {cd.fs_model} · f_r {cd.frame_rate_cfg:g} Hz · V {cd.codebook_size} · N_q {cd.n_q_cfg} · semantic {cd.n_semantic}")
                if x is None:
                    continue
                cd.encode_all(x[:FS])                                                   # 暖機
                t0 = time.perf_counter(); codes = cd.encode_all(x); t_enc = time.perf_counter() - t0
                t0 = time.perf_counter(); y = cd.decode(codes, len(x)); t_dec = time.perf_counter() - t0
                N_q, T = codes.shape
                fr_meas = T / (len(x) / FS)
                rr = {"N_q": N_q, "T": T, "frame_rate_measured": round(fr_meas, 3), "encode_ms": round(t_enc * 1000, 1), "decode_all_ms": round(t_dec * 1000, 1),
                      "codes_min": int(codes.min()), "codes_max": int(codes.max()), "y_len": int(len(y))}
                # 切列 decode 對該類別自己的層數參數：code 矩陣應該逐格相同（前 n 層的 code 不依賴後面幾層）
                n_small = CFG["codecs"][k]["stages"][0] if k != "encodec" else 2
                c2 = cd.encode_with_param(x, n_small)
                rr["param_encode_n_q"] = int(c2.shape[0])
                rr["param_vs_slice_equal"] = bool(c2.shape[0] <= N_q and np.array_equal(c2, codes[:c2.shape[0]]))
                y1 = cd.decode(codes[:n_small], len(x))
                rr["decode_sliced_ok"] = bool(np.isfinite(y1).all() and np.std(y1) > 1e-4)
                m_all = metrics(x, y); m_1 = metrics(x, y1)
                rr["lsd_all_layers_db"] = round(m_all["lsd_db"], 3); rr["lsd_first_stages_db"] = round(m_1["lsd_db"], 3)
                rr["snr_all_layers_db"] = round(m_all["snr_db"], 3)
                outs[dev] = codes
                r[f"infer_{dev}"] = rr
                print(f"    {dev}: encode {t_enc * 1000:6.0f} ms   decode {t_dec * 1000:6.0f} ms   codes {N_q} × {T}（量到 f_r {fr_meas:.2f} Hz）   "
                      f"參數 encode n_q={n_small} 對切列：{'相同' if rr['param_vs_slice_equal'] else C.RED + '不同' + C.RESET}   "
                      f"LSD 全層 {rr['lsd_all_layers_db']:.2f} / 前 {n_small} 層 {rr['lsd_first_stages_db']:.2f} dB   SNR 全層 {rr['snr_all_layers_db']:.2f} dB")
            except Exception as e:
                r[f"infer_{dev}"] = {"error": repr(e)}
                print(f"    {dev}: {C.RED}失敗{C.RESET} {e!r}")
        if "cpu" in outs and "mps" in outs:
            a, b = outs["cpu"], outs["mps"]
            same = a.shape == b.shape and np.array_equal(a, b)
            frac = float(np.mean(a != b)) if a.shape == b.shape else float("nan")
            r["mps_vs_cpu"] = {"codes_identical": bool(same), "frac_codes_differ": frac}
            print(f"    mps 對 cpu：code 矩陣{'相同' if same else f'不同（{100 * frac:.1f}% 的格子）'}")
        res["codecs"][k] = r
    (HERE / "runs").mkdir(exist_ok=True)
    (HERE / "runs" / "check.json").write_text(json.dumps(res, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    print(f"\n  → runs/check.json\n下一步：./present.sh record（錄華語句子）→ ./present.sh rehearse")


# ── fetch ────────────────────────────────────────────────────
def fetch() -> None:
    from huggingface_hub import HfApi, snapshot_download
    C.banner("下載三個 codec、複製 W3 那句", C.AMBER)
    lock = read_lock()
    api = HfApi()
    for k in CODECS:
        repo = CFG["codecs"][k]["repo"]
        info = api.model_info(repo)
        sha = lock.get(repo) or info.sha
        path = snapshot_download(repo, revision=sha, allow_patterns=MODEL_PATTERNS)
        lock[repo] = sha
        lic = (info.card_data or {}).get("license") if getattr(info, "card_data", None) else None
        size = sum(f.stat().st_size for f in Path(path).rglob("*") if f.is_file()) / 1e6
        print(f"  {repo} @ {sha[:12]}   {size:.0f} MB   license (model card): {lic}   → {path}")
    MODEL_LOCK.write_text("".join(f"{r} {s}\n" for r, s in lock.items()), encoding="utf-8")
    src = (HERE / CFG["inputs"]["libri"]["source"]).resolve()
    dst = input_path("libri")
    if not src.exists():
        print(f"  {C.AMBER}找不到 {src}：先跑 ../w03_d3_blank_heatmap/present.sh fetch{C.RESET}")
    else:
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dst)
        txt = src.with_suffix(".txt")
        if txt.exists():
            shutil.copyfile(txt, dst.with_suffix(".txt"))
        print(f"  {src.name} → {dst.relative_to(HERE)}（16 kHz；載入時重取樣到 {FS} Hz）")
    print(f"  → model.lock\n下一步：./present.sh check")


# ── record ───────────────────────────────────────────────────
def record(src: str | None = None, seconds: float | None = None) -> None:
    C.banner("華語句子（不進版控、不上站）", C.AMBER)
    c = CFG["inputs"]["mandarin"]
    dst = input_path("mandarin")
    if src:
        p = Path(src).expanduser()
        if p.suffix.lower() != ".wav":
            sys.exit(f"只吃 WAV。先轉：afconvert -f WAVE -d LEI16@{FS} {p} {p.with_suffix('.wav')}")
        x = C.load_wav(p, FS)
        print(f"  匯入 {p}（{len(x) / FS:.2f} s）")
    else:
        sec = seconds or c["record_seconds"]
        print("  念一句含四聲對比的話，例如：媽媽罵馬，馬不理媽媽。（錄音前先深呼吸，開始後一秒再念）")
        x = C.record(sec, FS)
    x0 = len(x)
    x = trim_silence(x, FS)
    peak = float(np.max(np.abs(x)) + 1e-9)
    print(f"  裁靜音 {x0 / FS:.2f} s → {len(x) / FS:.2f} s；峰值 {20 * np.log10(peak):.1f} dBFS" + (f"  {C.AMBER}（太小，離麥克風近一點再錄）{C.RESET}" if peak < 0.05 else ""))
    dst.parent.mkdir(parents=True, exist_ok=True)
    C.save_wav(dst, x, FS)
    C.play(dst)
    ts, f0 = f0_track(x)
    v = f0[~np.isnan(f0)]
    print(f"  → {dst}   有聲 frame {len(v)}/{len(f0)}   F0 中位數 {np.median(v):.0f} Hz（範圍 {v.min():.0f}–{v.max():.0f}）" if len(v) else f"  → {dst}   {C.RED}沒有偵測到有聲 frame{C.RESET}")
    print("再錄一次就再跑一次；滿意之後 ./present.sh rehearse")


# ── export ───────────────────────────────────────────────────
def export(rec_path: Path) -> None:
    rec = json.loads(rec_path.read_text(encoding="utf-8"))
    d = json.loads(METRICS_JSON.read_text(encoding="utf-8")) if METRICS_JSON.exists() else {
        "_note": "量測數字（進版控的出處，speech_ai/CLAUDE.md 4.8）。w06-data.json 是 make_figs_w06.py 算出來的、每次重產；量測的放這裡。"}
    demo1 = {"source": "demos/w06_d1_layer_listen/runs/rehearsal/rehearsal.json", "time": rec["time"], "tag": rec.get("tag", ""),
             "fs": rec["fs"], "codecs": rec["codecs"], "inputs": rec["inputs"],
             "definitions": {"bitrate_bps": "f_r · n_q · log2 V with f_r and V read from the loaded config; frame_rate_measured = T / seconds",
                             "lsd_db": "log-spectral distance, 25 ms / 10 ms / 1024-point, dB", "snr_db": "waveform SNR (phase-sensitive; near 0 dB is normal for GAN-trained codecs)",
                             "f0_rmse_hz": "RMSE of autocorrelation F0 on frames voiced in both original and reconstruction", "voicing_agree": "fraction of frames with the same voiced/unvoiced decision",
                             "what_we_heard": "filled by hand after rehearsal (see w06.qmd p33/p35)"},
             "recon": {}}
    for key, v in rec["recon"].items():
        demo1["recon"][key] = {}
        for name, r in v.items():
            if name.startswith("_"):
                continue
            demo1["recon"][key][name] = {"codec": r["codec"], "n_q": r["n_q"], "N_q": r["N_q"], "T": r["T"], "frame_rate_cfg": r["frame_rate_cfg"], "frame_rate_measured": r["frame_rate_measured"],
                                         "codebook_size": r["codebook_size"], "bitrate_bps": r["bitrate_bps"], "codes_used": r["codes_used"],
                                         **{kk: (round(vv, 3) if isinstance(vv, float) else vv) for kk, vv in r["metrics"].items()}}
    d["demo1"] = demo1
    METRICS_JSON.parent.mkdir(parents=True, exist_ok=True)
    METRICS_JSON.write_text(json.dumps(d, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"  → {METRICS_JSON}（demo1）")


# ── selftest ─────────────────────────────────────────────────
def selftest() -> None:
    C.banner("selftest", C.AMBER)
    fails = []

    def ok(cond: bool, msg: str):
        print(f"  {'OK ' if cond else C.RED + 'FAIL' + C.RESET}  {msg}")
        if not cond:
            fails.append(msg)

    # 1. bitrate 公式對原文的數字（Moshi §3.3.1：12.5 Hz、Q = 8、N_A = 2048 → 1.1 kbps；EnCodec §3.1–3.2：75 Hz、1024 → 750 bps 一層）
    ok(abs(bitrate_bps(12.5, 1, 2048) - 137.5) < 1e-9, "Mimi 一層 = 137.5 bps")
    ok(abs(bitrate_bps(12.5, 8, 2048) - 1100) < 1e-9, "Mimi 八層 = 1.1 kbps")
    ok(abs(bitrate_bps(75, 1, 1024) - 750) < 1e-9, "EnCodec 一層 = 750 bps")
    ok(abs(bitrate_bps(75, 32, 1024) - 24000) < 1e-9, "EnCodec 32 層 = 24 kbps")
    # 2. F0 估計：合成母音 F0 220 → 110 Hz 的滑音，估計在 5% 內
    f0 = np.linspace(220, 110, 50)
    x = dsp.synth_vowel(FS, 1.0, f0, dsp.VOWELS["a"])
    ts, est = f0_track(x)
    truth = np.interp(ts, np.linspace(0, 1.0, 50), f0)
    good = ~np.isnan(est)
    rel = np.abs(est[good] - truth[good]) / truth[good]
    ok(good.mean() > 0.9 and np.median(rel) < 0.05, f"F0 滑音：有聲 {100 * good.mean():.0f}%，中位相對誤差 {100 * np.median(rel):.1f}%")
    # 3. 指標恆等：自己對自己 LSD = 0、voicing 一致 1、F0 RMSE 0；靜音對自己不炸
    m = metrics(x, x)
    ok(m["lsd_db"] < 1e-9 and m["voicing_agree"] == 1.0 and m["f0_rmse_hz"] < 1e-9, "metrics(x, x)：LSD 0、voicing 1、F0 RMSE 0")
    m2 = metrics(x, np.zeros_like(x))
    ok(np.isfinite(m2["lsd_db"]) and np.isnan(m2["f0_rmse_hz"]) and m2["voiced_both"] == 0, "metrics(x, 0)：LSD 有限、沒有共同有聲 frame")
    # 4. 裁靜音：前後各 0.5 s 靜音的母音，裁完長度接近 1 s
    pad = np.zeros(int(0.5 * FS))
    y = trim_silence(np.concatenate([pad, x, pad]), FS)
    ok(0.9 * FS <= len(y) <= 1.25 * FS, f"trim_silence：2.0 s → {len(y) / FS:.2f} s")
    # 5. fake 流程（沒有模型）：兩段輸入用合成訊號，不開視窗
    import matplotlib
    matplotlib.use("Agg")
    out = Out()
    codecs = {k: FakeCodec(k) for k in CODECS}
    CFG["inputs"]["mandarin"]["private"] = False                 # selftest 不往 ../_media 寫
    d = Demo(out, codecs, HERE / "runs" / "live" / "selftest.fake", tag="selftest FAKE")
    d.inputs = {"libri": x, "mandarin": dsp.synth_vowel(FS, 0.8, np.linspace(150, 250, 20), dsp.VOWELS["i"])}
    for key, xx in d.inputs.items():
        d.log["inputs"][key] = {"label": key, "path": "", "seconds": round(len(xx) / FS, 3), "peak": float(np.max(np.abs(xx)))}
        d.input_dir(key).mkdir(parents=True, exist_ok=True)
    d.compute_all()
    n_expected = sum(len(CFG["codecs"][c]["stages"]) for c in CODECS) * 2
    ok(len(d.recon) == n_expected, f"fake：{len(d.recon)} 個重建（預期 {n_expected}）")
    ok(all(np.isfinite(r["rec"]["metrics"]["lsd_db"]) for r in d.recon.values()), "fake：指標全部有限")
    lsd = [d.recon[("libri", "mimi", n)]["rec"]["metrics"]["lsd_db"] for n in MIMI_STAGES]
    ok(all(a >= b for a, b in zip(lsd, lsd[1:])), f"fake：Mimi 層數越多 LSD 越小 {[round(v, 2) for v in lsd]}")
    d.make_fig()
    d.cur_input = "libri"
    d.stage("mimi", 1); d.cycle_codec("encodec"); d.cycle_codec("dac"); d.toggle_f0(); d.stage("mimi", 8); d.switch_input(); d.stage("mimi", 1); d.table()
    d.rehearse_figs()
    d.save()
    ok((d.folder / "libri" / "logmel.png").exists() and (d.folder / "mandarin" / "f0.png").exists(), "fake：圖存了")
    shutil.rmtree(d.folder, ignore_errors=True)
    # 6. 有彩排紀錄：對 w06-metrics.json 的 bitrate 與 f_r
    rec_path = HERE / "runs" / "rehearsal" / "rehearsal.json"
    if rec_path.exists() and METRICS_JSON.exists():
        rec = json.loads(rec_path.read_text(encoding="utf-8"))
        mj = json.loads(METRICS_JSON.read_text(encoding="utf-8")).get("demo1", {})
        n_bad = 0
        for key, v in rec["recon"].items():
            for name, r in v.items():
                if name.startswith("_"):
                    continue
                e = mj.get("recon", {}).get(key, {}).get(name)
                if not e or abs(e["bitrate_bps"] - bitrate_bps(r["frame_rate_cfg"], r["n_q"], r["codebook_size"])) > 0.1 or abs(e["lsd_db"] - r["metrics"]["lsd_db"]) > 1e-3:
                    n_bad += 1
        ok(n_bad == 0, f"彩排紀錄對 w06-metrics.json → demo1：{n_bad} 處不一致")
    else:
        print("  --   沒有彩排紀錄或 w06-metrics.json，跳過比對")
    print(f"\n  {'全部通過' if not fails else str(len(fails)) + ' 項失敗'}")
    sys.exit(1 if fails else 0)


# ── main ─────────────────────────────────────────────────────
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["show", "rehearse", "replay", "record", "check", "fetch", "fake", "selftest", "export"])
    ap.add_argument("--device", help="show／rehearse／check：蓋過 demo_config.toml 的 device（cpu／mps／auto）")
    ap.add_argument("--from", dest="src", help="record：匯入一個 WAV（24 kHz 最好；其他取樣率會重取樣）")
    ap.add_argument("--seconds", type=float, help="record：錄幾秒")
    ap.add_argument("--no-plot", action="store_true", help="show／replay／fake：只在終端機，不開圖")
    args = ap.parse_args()

    if args.cmd == "check":
        check(args.device); return
    if args.cmd == "fetch":
        fetch(); return
    if args.cmd == "record":
        record(args.src, args.seconds); return
    if args.cmd == "selftest":
        selftest(); return
    if args.cmd == "export":
        export(HERE / "runs" / "rehearsal" / "rehearsal.json"); return

    if args.cmd == "rehearse" or args.no_plot:
        import matplotlib
        matplotlib.use("Agg")

    out = Out()
    out.banner("W6 Demo 1：adding the stages back, one at a time")
    d = build_replay(out) if args.cmd == "replay" else build(args.cmd, out, args.device)

    if args.cmd == "rehearse":
        for key in d.inputs:
            d.cur_input = key; d.table()
        d.rehearse_figs()
        d.save()
        export(d.folder / "rehearsal.json")
        print("\n下一步：./present.sh 自己按一遍 o 1 2 4 8 e e e e e d d d d m o 1 8 e f t；把聽到的寫進 p33／p34 的表（what we heard），"
              "DAC 的 f_r／V、Mimi 的 N_q 從上面的 config 列填進 p33 表格與【提示】。")
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
    if args.cmd == "show":
        d.save()


if __name__ == "__main__":
    main()

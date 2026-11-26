# -*- coding: utf-8 -*-
# 此檔由 bin/publish_demos.py 從 private repo 的 <course>/demos/ 複製而來，請勿在公開 repo 直接編輯。
"""W9 demo 1：cloning 的三種條件與洩漏——同一位說話人的 prompt 做成四版，過三種給條件的方式，十二段聲音。

  show       課堂：載模型、把十二段（加上有錄音時的「加一格」四段）全部算好，開一個視窗；鍵在視窗裡按。離開時存 runs/live/<時間>/
  rehearse   課前：不互動，全部算一遍，.wav、圖、數字存 runs/rehearsal/（進版控 = 現場的退路），寫 slides/assets/w09/w09-metrics.json → demo1
  replay     現場出狀況時：從 runs/rehearsal/ 讀回，同一個視窗、同樣的鍵，不碰 torch 也不開 worker
  check      課前體檢：版本；SpeechBrain 的 x-vector 與 WavLMForXVector 在 cpu／mps 上載不載得起、兩個裝置的向量差多少；
             SIM 的兩個參考點（同一位說話人的另一句、另一個人）；Whisper 對真人錄音的 WER；mlx-audio 有沒有 qwen3_tts → runs/check.json
  fetch      課前一週：下載 x-vector encoder 與 SIM 的 encoder（鎖 commit → model.lock）、Whisper（commit 讀 W7 Demo 1 的）；
             確認 W8 Demo 1／2 的模型都在（要網路）
  record     錄「加一格」的兩段自己的 prompt：record mood、record accent → audio/own_<名字>.wav（不上站）
  fake       沒有模型也能演練流程（三個系統與三個裁判都是替身，畫面標明 FAKE）
  selftest   不開視窗、不需要模型：四版 prompt 的 SNR 與頻寬、兩個自己定的量、WER、fake 流程；有彩排紀錄時對 w09-metrics.json
  export     只重寫 w09-metrics.json → demo1（從 runs/rehearsal/rehearsal.json）

畫面上的鍵（順序照投影片 p16、p17 講稿的【操作】）：
  p  四版 prompt 的 mel，依序播：乾淨、加噪、加音樂、電話頻寬
  1  加噪那一版的三個輸出（SpeechT5、F5-TTS、Qwen3-TTS），依序播；畫面是那一版 prompt 與三個輸出的 mel
  2  加音樂那一版的三個輸出
  3  電話頻寬那一版的三個輸出（mel 上畫 4 kHz 的線）
  c  乾淨那一版的三個輸出（對照）
  s f k  目前這一版只再播 SpeechT5／F5-TTS／Qwen3-TTS；  o 再播這一版的 prompt；  g 同一位說話人念這一句的真人錄音
  t  數字：停頓處的噪音底、4 kHz 以上的能量、兩種 SIM、WER（終端機是表，視窗是長條圖）
  e  加一格：語氣明顯的 prompt 念中性的句子（F5-TTS、Qwen3-TTS）
  a  加一格：華語的 prompt 念英文
  r  再播一次     h  選單     q  離開

三種條件（投影片 p10–p13）：
  (a) SpeechT5      參考音 → x-vector（512 個數字）→ 模型。瓶頸。
  (b) F5-TTS        參考音的 mel 整段放進去，模型把後面補完（flow matching）。沒有瓶頸。
  (b) Qwen3-TTS     參考音的 codec token 整段放進去，模型往下續寫（AR）。沒有瓶頸。
量的四樣：
  停頓處的噪音底   10 ms 一格的 RMS 位準（dB）第 5 百分位 − 第 95 百分位，只看句子頭尾之間（最安靜的格子比最響的格子低多少；越負越乾淨）
  4 kHz 以上的能量 重取樣到 16 kHz 後 4–8 kHz 的能量 ÷ 0–8 kHz 的能量（dB）
  SIM              輸出與參考音經同一個 speaker encoder（WavLM-SV）的向量的 cosine；對乾淨的 prompt 算一次、對「那一版」prompt 算一次
  WER              Whisper small 轉錄後對要念的句子（兩邊過 Whisper 的 English normalizer）
前兩樣是這支程式自己定的量，不是文獻指標。
"""
from __future__ import annotations

import argparse
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


def _load_d1():
    p = (HERE / CFG["d1"]["dir"] / "demo.py").resolve()
    if not p.exists():
        sys.exit(f"找不到 W8 Demo 1 的程式 {p}（demo_config.toml [d1].dir）")
    spec = importlib.util.spec_from_file_location("w08_d1_demo", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


D1 = _load_d1()
D1_DIR = (HERE / CFG["d1"]["dir"]).resolve()
D2_DIR = (HERE / CFG["d2"]["dir"]).resolve()
D2_CFG = tomllib.loads((D2_DIR / "demo_config.toml").read_text(encoding="utf-8")) if (D2_DIR / "demo_config.toml").exists() else {}
WORKER_PY = D2_DIR / "mlx_worker.py"
WORKER_PY_ENV = HERE.parent / ".venv" / "bin" / "python"           # mlx-audio 住在共用環境
QWEN_REPO = D2_CFG.get("qwen3", {}).get("repo", "mlx-community/Qwen3-TTS-12Hz-0.6B-Base-bf16")

PROMPT_WAV = D1_DIR / D1.CFG["audio"]["prompt_path"]
PROMPT_TEXT = CFG["text"]["prompt_text"] or D1.TXT["prompt_text"]
PROMPT_UTT = D1.TXT["prompt_utt"]
TARGET = CFG["text"]["target"] or D1.TXT["target"]
MODEL_LOCK = HERE / "model.lock"
METRICS_JSON = HERE.parents[1] / "slides" / "assets" / "w09" / "w09-metrics.json"     # parents[1] = speech_ai/
REHEARSAL = HERE / "runs" / "rehearsal"
FS = 16000
VERSIONS = CFG["versions"]["order"]
SYSTEMS = ["speecht5", "f5", "qwen3"]
SYS_LABEL = {"speecht5": "SpeechT5 (x-vector)", "f5": "F5-TTS (prompt mel)", "qwen3": "Qwen3-TTS (prompt tokens)"}
SYS_SHORT = {"speecht5": "SpeechT5", "f5": "F5-TTS", "qwen3": "Qwen3-TTS"}
SYS_COLOR = {"prompt": "#888888", "speecht5": "#2a7fc8", "f5": "#5b3fa0", "qwen3": "#c8742a"}
VER_LABEL = {"clean": "clean", "noise": "+ noise", "music": "+ music", "phone": "telephone band"}
VER_KEY = {"c": "clean", "1": "noise", "2": "music", "3": "phone"}
OWN = [k for k in ("mood", "accent") if k in CFG["own"]]
OWN_SYSTEMS = ["f5", "qwen3"]
REPOS = {"xvector": CFG["xvector"]["repo"], "sim": CFG["sim"]["repo"], "whisper": CFG["whisper"]["repo"]}
PATTERNS = {"xvector": ["hyperparams.yaml", "embedding_model.ckpt", "mean_var_norm_emb.ckpt", "classifier.ckpt", "label_encoder.txt"],
            "sim": ["*.json", "pytorch_model.bin"], "whisper": ["*.json", "*.txt", "*.safetensors"]}


def cell(version: str, system: str) -> str:
    return f"{version}_{system}"


def own_cell(k: str, system: str) -> str:
    return f"own_{k}_{system}"


def own_wav(k: str) -> Path:
    return HERE / "audio" / f"own_{k}.wav"


# ── 四版 prompt ──────────────────────────────────────────────
def rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(x))) + 1e-12)


def mix_at_snr(x: np.ndarray, n: np.ndarray, snr_db: float) -> np.ndarray:
    """x ＋ g·n，g 讓 20 log10(rms(x) / rms(g·n)) = snr_db（整段的 RMS）。"""
    g = rms(x) / (rms(n) * 10 ** (snr_db / 20))
    return x + g * n


def synth_music(n: int, fs: int = FS) -> np.ndarray:
    """一段用 numpy 合成的和弦（C–G–Am–F，每個和弦 1.2 s，四分音符刷四下，加一個低音）。沒有授權問題；聽起來像電子琴。"""
    chords = [(261.63, 329.63, 392.00), (196.00, 246.94, 293.66), (220.00, 261.63, 329.63), (174.61, 220.00, 261.63)]
    beat, per = 0.3, 4
    y = np.zeros(n)
    t = np.arange(int(beat * fs)) / fs
    env = np.exp(-t / 0.18) * np.minimum(1.0, t / 0.005)
    i = 0
    while i * len(t) < n:
        ch = chords[(i // per) % len(chords)]
        note = np.zeros(len(t))
        for f0 in ch + (ch[0] / 2,):
            for h, a in ((1, 1.0), (2, 0.5), (3, 0.25), (4, 0.12)):
                note += a * np.sin(2 * np.pi * f0 * h * t)
        a0 = i * len(t)
        seg = (note * env)[: max(0, n - a0)]
        y[a0:a0 + len(seg)] += seg
        i += 1
    return y / (np.max(np.abs(y)) + 1e-12)


def phone_band(x: np.ndarray, fs: int = FS) -> np.ndarray:
    from scipy.signal import butter, sosfiltfilt
    v = CFG["versions"]
    sos = butter(int(v["phone_order"]), float(v["phone_lowpass_hz"]), btype="low", fs=fs, output="sos")
    y = sosfiltfilt(sos, x)
    y = dsp.resample(dsp.resample(y, fs, 8000), 8000, fs)
    return y[: len(x)] if len(y) >= len(x) else np.concatenate([y, np.zeros(len(x) - len(y))])


def make_prompts(base: np.ndarray) -> tuple[dict[str, np.ndarray], dict]:
    """base：裁過前後靜音的乾淨 prompt（16 kHz）。回四版（同長度）與每一版怎麼做的。"""
    v = CFG["versions"]
    n = len(base)
    out, how = {"clean": base.copy()}, {"clean": {"how": "edges below −42 dBFS trimmed"}}
    noise = np.random.default_rng(int(v["noise_seed"])).standard_normal(n)
    out["noise"] = mix_at_snr(base, noise, float(v["noise_snr_db"]))
    how["noise"] = {"how": "white Gaussian noise (numpy)", "snr_db": float(v["noise_snr_db"]), "seed": int(v["noise_seed"])}
    mp = HERE / v["music_path"]
    if mp.exists():
        m = C.load_wav(mp, FS)
        m = np.tile(m, int(math.ceil(n / max(1, len(m)))))[:n]
        src = f"file {mp.name}"
    else:
        m = synth_music(n)
        src = "synthesised in demo.py (C–G–Am–F, numpy)"
    out["music"] = mix_at_snr(base, m, float(v["music_snr_db"]))
    how["music"] = {"how": "background music", "source": src, "snr_db": float(v["music_snr_db"])}
    out["phone"] = phone_band(base)
    how["phone"] = {"how": f"zero-phase Butterworth low-pass (order {v['phone_order']}) at {v['phone_lowpass_hz']:g} Hz, then 16 → 8 → 16 kHz"}
    pk = max(float(np.max(np.abs(y))) for y in out.values())
    if pk > 0.99:                                            # 四版用同一個增益，SNR 不變
        out = {k: y * 0.99 / pk for k, y in out.items()}
    return out, how


# ── 兩個自己定的量 ───────────────────────────────────────────
def frame_levels(x: np.ndarray, fs: int) -> np.ndarray:
    hop = int(fs * 0.010)
    n = len(x) // hop
    if n == 0:
        return np.array([CFG["measure"]["level_floor_dbfs"]])
    e = np.sqrt(np.mean(x[: n * hop].reshape(n, hop) ** 2, axis=1))
    return np.maximum(20 * np.log10(e + 1e-12), CFG["measure"]["level_floor_dbfs"])


def floor_db(x: np.ndarray, fs: int) -> float:
    """停頓處的噪音底（dB re speech）：10 ms 位準的第 5 百分位 − 第 95 百分位，只看第一格到最後一格「有聲音」的 frame 之間
    （有聲音 = 位準在第 95 百分位以下 active_db 之內）——句子前後的靜音不算，算的是句子裡面的停頓。"""
    m = CFG["measure"]
    e = frame_levels(x, fs)
    hi = float(np.percentile(e, m["floor_high_pct"]))
    on = np.flatnonzero(e > hi - m["floor_active_db"])
    if len(on) == 0:
        return 0.0
    return float(np.percentile(e[on[0]:on[-1] + 1], m["floor_low_pct"]) - hi)


def hf_db(x: np.ndarray, fs: int) -> float:
    """4 kHz 以上的能量（dB re total）：重取樣到 16 kHz，4–8 kHz ÷ 0–8 kHz。"""
    y = dsp.resample(x, fs, FS)
    P = np.abs(dsp.stft(y, 400, 160, 512)) ** 2
    f = np.linspace(0, FS / 2, P.shape[0])
    hi = P[f >= CFG["measure"]["hf_split_hz"]].sum()
    return float(10 * np.log10((hi + 1e-20) / (P.sum() + 1e-20)))


def f0_stats(x: np.ndarray, fs: int) -> dict:
    c = CFG["f0"]
    flen = int(round(c["frame_ms"] * fs / 1000)); hop = int(round(c["hop_ms"] * fs / 1000))
    f0 = np.array([dsp.f0_autocorr(x[a:a + flen], fs, c["floor_hz"], c["ceiling_hz"], c["voicing"])[0] for a in range(0, max(1, len(x) - flen), hop)], dtype=float)
    v = f0[~np.isnan(f0)]
    if len(v) < 5:
        return {"f0_median_hz": None, "f0_p10_p90_hz": None, "f0_span_semitones": None}
    p10, p90 = np.percentile(v, 10), np.percentile(v, 90)
    return {"f0_median_hz": round(float(np.median(v)), 1), "f0_p10_p90_hz": [round(float(p10), 1), round(float(p90), 1)],
            "f0_span_semitones": round(float(12 * np.log2(p90 / p10)), 2)}


_FB = {}


def disp_mel(x: np.ndarray, fs: int) -> np.ndarray:
    """畫面用的 log-mel（dB）：一律重取樣到 16 kHz、25 ms／10 ms、0–8 kHz——所有 prompt 與輸出同一個前端，只給眼睛看。"""
    n_mels = int(CFG["display"]["n_mels"])
    if n_mels not in _FB:
        _FB[n_mels] = dsp.mel_filterbank(FS, 512, n_mels, "slaney", area_norm=False, fmin=0.0, fmax=FS / 2)
    y = dsp.resample(x, fs, FS)
    P = np.abs(dsp.stft(y, 400, 160, 512)) ** 2
    return 10 * np.log10(np.maximum(_FB[n_mels] @ P, 1e-10))


def mel_bin_of(hz: float) -> float:
    n_mels = int(CFG["display"]["n_mels"])
    m = np.linspace(dsp.hz_to_mel(0.0, "slaney"), dsp.hz_to_mel(FS / 2, "slaney"), n_mels + 2)[1:-1]
    return float(np.interp(dsp.hz_to_mel(hz, "slaney"), m, np.arange(n_mels)))


# ── WER ──────────────────────────────────────────────────────
def edit_distance(ref: list, hyp: list) -> int:
    d = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, 1):
        prev, d[0] = d[0], i
        for j, h in enumerate(hyp, 1):
            prev, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, prev + (r != h))
    return d[-1]


def basic_norm_en(s: str) -> str:
    """沒有 Whisper 的 normalizer 時的退路（selftest／fake）：小寫、去標點、Mr → mister。"""
    s = re.sub(r"[^a-z0-9' ]+", " ", s.lower().replace("mr.", "mister"))
    return " ".join(s.split())


def zh_units(s: str) -> list[str]:
    """華語的比對單位：每個漢字的帶調拼音（繁簡、同音字不算錯，聲調錯算錯）；沒有 pypinyin 就退到逐字。"""
    chars = re.findall(r"[一-鿿]", s)
    try:
        from pypinyin import Style, lazy_pinyin
        return lazy_pinyin("".join(chars), style=Style.TONE3, neutral_tone_with_five=True)
    except ImportError:
        return chars


def error_rate(ref: str, hyp: str, lang: str, norm=None) -> dict:
    if lang == "zh":
        r, h = zh_units(ref), zh_units(hyp)
        unit = "tone-numbered pinyin syllables"
    else:
        f = norm or basic_norm_en
        r, h = f(ref).split(), f(hyp).split()
        unit = "words"
    e = edit_distance(r, h)
    return {"errors": e, "n_ref": len(r), "pct": round(100 * e / max(1, len(r)), 1), "unit": unit}


# ── 模型 ─────────────────────────────────────────────────────
def read_lock() -> dict[str, str]:
    """model.lock：每行 `repo sha`。Whisper 沒有就退到 W7 Demo 1 的 lock。"""
    out = {}
    if MODEL_LOCK.exists():
        for line in MODEL_LOCK.read_text(encoding="utf-8").splitlines():
            if line.strip() and not line.startswith("#"):
                repo, sha = line.split()[:2]
                out[repo] = sha
    sib = (HERE / CFG["whisper"]["sibling_lock"]).resolve()
    if REPOS["whisper"] not in out and sib.exists():
        for line in sib.read_text(encoding="utf-8").splitlines():
            p = line.split()
            if len(p) == 2 and p[0] == REPOS["whisper"]:
                out[p[0]] = p[1]
    return out


def model_dir(key: str, offline: bool = True) -> str:
    from huggingface_hub import snapshot_download
    return snapshot_download(REPOS[key], revision=read_lock().get(REPOS[key]), local_files_only=offline, allow_patterns=PATTERNS[key])


def unit(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64).reshape(-1)
    return v / (np.linalg.norm(v) + 1e-12)


def cos(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(unit(a), unit(b)))


class XVector:
    """SpeechBrain 的 x-vector（Snyder et al. 2018 的架構，VoxCeleb 上訓）：24 維 fbank → 五層 TDNN → statistics pooling → 512 維。
    SpeechT5 的 TTS 訓練時用的就是這個 encoder 的向量（HF 文件的做法），所以 (a) 的條件從這裡來。"""
    label = "x-vector (SpeechBrain, VoxCeleb)"

    def __init__(self, device: str = "cpu"):
        import torch
        self.torch = torch
        self.device = D1.resolve_device(device)
        with warnings.catch_warnings(), D1._quiet():
            warnings.simplefilter("ignore")
            from speechbrain.inference.speaker import EncoderClassifier
            self._tmp = tempfile.mkdtemp(prefix="w09_d1_sb_")
            src = model_dir("xvector")
            # hyperparams.yaml 裡的 pretrained_path 寫的是 Hub 上的 repo 名：不蓋掉的話，權重會另外向 Hub 要「main」，
            # 而 fetch 只釘了 commit、快取裡沒有 refs/main，離線時就找不到。蓋成同一個本機快照。
            self.enc = EncoderClassifier.from_hparams(source=src, savedir=self._tmp, overrides={"pretrained_path": src}, run_opts={"device": self.device})
        self.enc.eval()
        self.n_params = sum(p.numel() for p in self.enc.mods.embedding_model.parameters())

    def embed(self, x16: np.ndarray) -> np.ndarray:
        torch = self.torch
        with torch.no_grad(), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            e = self.enc.encode_batch(torch.tensor(x16, dtype=torch.float32)[None].to(self.device))
        return e.detach().float().cpu().numpy().reshape(-1)

    def info(self) -> dict:
        return {"repo": REPOS["xvector"], "commit": read_lock().get(REPOS["xvector"]), "device": self.device, "n_params": self.n_params, "dim": 512}


class WavLMSV:
    """SIM 的裁判：WavLM Base+ 接 x-vector head（WavLMForXVector），整段聲音 → 512 維。"""
    label = "WavLM-SV"

    def __init__(self, device: str = "auto"):
        import torch
        from transformers import Wav2Vec2FeatureExtractor, WavLMForXVector
        self.torch = torch
        self.device = D1.resolve_device(device)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.fe = Wav2Vec2FeatureExtractor.from_pretrained(model_dir("sim"))
            self.model = WavLMForXVector.from_pretrained(model_dir("sim")).eval().to(self.device)
        self.n_params = sum(p.numel() for p in self.model.parameters())

    def embed(self, x16: np.ndarray) -> np.ndarray:
        torch = self.torch
        inp = self.fe([x16.astype(np.float32)], sampling_rate=FS, return_tensors="pt", padding=True)
        with torch.no_grad(), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            e = self.model(**{k: v.to(self.device) for k, v in inp.items()}).embeddings
        return e.detach().float().cpu().numpy().reshape(-1)

    def info(self) -> dict:
        return {"repo": REPOS["sim"], "commit": read_lock().get(REPOS["sim"]), "device": self.device, "n_params": self.n_params,
                "threshold_from_card": CFG["sim"]["threshold_from_card"]}


class Whisper:
    """WER 的裁判：W7 的 Whisper small（transformers），溫度 0、不用前文、不出時間戳。"""
    label = "Whisper small"

    def __init__(self, device: str = "cpu"):
        import torch
        from transformers import WhisperForConditionalGeneration, WhisperProcessor
        self.torch = torch
        self.device = D1.resolve_device(device)
        path = model_dir("whisper")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.proc = WhisperProcessor.from_pretrained(path)
            self.model = WhisperForConditionalGeneration.from_pretrained(path).eval().to(self.device)

    def transcribe(self, x16: np.ndarray, lang: str = "en", hint: str | None = None) -> str:
        torch = self.torch
        inp = self.proc(x16.astype(np.float32), sampling_rate=FS, return_tensors="pt", return_attention_mask=True)
        with torch.no_grad(), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ids = self.model.generate(inp.input_features.to(self.device), attention_mask=inp.attention_mask.to(self.device), language=lang, task="transcribe",
                                      return_timestamps=False, do_sample=False, num_beams=1, max_new_tokens=int(CFG["whisper"]["max_new_tokens"]))
        return self.proc.batch_decode(ids, skip_special_tokens=True)[0].strip()

    def norm(self, s: str) -> str:
        return self.proc.tokenizer.normalize(s)

    def info(self) -> dict:
        return {"repo": REPOS["whisper"], "commit": read_lock().get(REPOS["whisper"]), "device": self.device, "decoding": "greedy, temperature 0, no timestamps"}


class Worker:
    """W8 Demo 2 的 mlx_worker.py（共用 ../.venv 的 mlx-audio），一行 JSON 進、一行 JSON 出。這裡只用 Qwen3-TTS。"""

    def __init__(self, out_dir: Path):
        self.out_dir = out_dir
        py = WORKER_PY_ENV if WORKER_PY_ENV.exists() else Path(sys.executable)
        env = dict(os.environ, HF_HUB_OFFLINE=os.environ.get("HF_HUB_OFFLINE", "1"), HF_HUB_DISABLE_PROGRESS_BARS="1", TOKENIZERS_PARALLELISM="false")
        self.p = subprocess.Popen([str(py), str(WORKER_PY), "--out", str(out_dir)], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=sys.stderr,
                                  text=True, encoding="utf-8", bufsize=1, env=env)
        self.ready = self._read()
        if not self.ready.get("ok"):
            raise RuntimeError(f"mlx worker 起不來：{self.ready.get('error')}")
        self.versions = self.ready.get("versions", {})
        self.loaded: dict = {}

    def _read(self) -> dict:
        line = self.p.stdout.readline()
        if not line:
            raise RuntimeError("mlx worker 沒有回應（看 stderr）")
        return json.loads(line)

    def call(self, **req) -> dict:
        self.p.stdin.write(json.dumps(req, ensure_ascii=False) + "\n")
        self.p.stdin.flush()
        return self._read()

    def load(self) -> dict:
        lock = D2_lock()
        r = self.call(cmd="load", qwen3=QWEN_REPO, kokoro=None, revision={"qwen3": lock.get(QWEN_REPO)})
        if not r.get("ok"):
            raise RuntimeError(f"Qwen3-TTS 載入失敗：{r.get('error')}")
        self.loaded = r.get("loaded", {})
        return r

    def qwen3(self, name: str, text: str, ref_audio: Path, ref_text: str, seed: int) -> dict:
        q = CFG["qwen3"]
        return self.call(cmd="qwen3", name=name, text=text, ref_audio=str(ref_audio), ref_text=ref_text, interval=q["streaming_interval"],
                         seed=seed, temperature=q["temperature"], lang_code=q["lang_code"], measure_ref_encode=False)

    def close(self) -> None:
        try:
            self.call(cmd="quit")
        except Exception:
            pass
        try:
            self.p.stdin.close(); self.p.wait(timeout=10)
        except Exception:
            self.p.kill()


def D2_lock() -> dict[str, str]:
    out = {}
    p = D2_DIR / "model.lock"
    if p.exists():
        for line in p.read_text(encoding="utf-8").splitlines():
            if line.strip() and not line.startswith("#"):
                repo, sha = line.split()[:2]
                out[repo] = sha
    return out


class Systems:
    """三個系統的真的版本。synth(system, name, prompt16, prompt_text, text, wav_path_of_prompt) → {"wav", "fs", "gen_s", …}"""
    fake = False

    def __init__(self, out, folder: Path, device: str | None = None, no_mlx: bool = False, nfe: int | None = None):
        c = CFG["infer"]
        self.out, self.folder, self.nfe = out, folder, nfe
        self.torch = __import__("torch")
        D1.quiet_hf()
        t0 = time.perf_counter()
        self.xv = XVector(c["xvector_device"])
        out(f"  載入 x-vector encoder  {self.xv.n_params / 1e6:6.1f} M 參數   {self.xv.device}   {time.perf_counter() - t0:5.1f} s")
        t0 = time.perf_counter()
        self.st5 = D1.SpeechT5(device or c["speecht5_device"], device or c["vocoder_device"])
        out(f"  載入 SpeechT5          {self.st5.n_params / 1e6:6.1f} M 參數   {self.st5.device} / vocoder {self.st5.vdev}   {time.perf_counter() - t0:5.1f} s")
        t0 = time.perf_counter()
        self.f5 = D1.F5(device or c["f5_device"], device or c["vocoder_device"])
        out(f"  載入 F5-TTS            {self.f5.n_params / 1e6:6.1f} M 參數   {self.f5.device} / vocoder {self.f5.vdev}   {time.perf_counter() - t0:5.1f} s")
        self.worker = None
        if not no_mlx:
            try:
                t0 = time.perf_counter()
                self.worker = Worker(folder / "_worker")
                self.worker.load()
                ld = self.worker.loaded.get("qwen3", {})
                out(f"  載入 Qwen3-TTS         {ld.get('n_params_M', float('nan')):6.1f} M 參數   MLX（mlx-audio {self.worker.versions.get('mlx-audio')}）   {time.perf_counter() - t0:5.1f} s")
                self.worker.qwen3("_warm", "Hi there.", PROMPT_WAV, PROMPT_TEXT, 0)                       # 第一次呼叫含 Metal 編譯
            except Exception as e:
                out(f"  {C.AMBER}Qwen3-TTS 起不來：{e}（第三欄留空；先 ./present.sh check）{C.RESET}")
                if self.worker is not None:
                    self.worker.close()
                self.worker = None

    def available(self) -> list[str]:
        return ["speecht5", "f5"] + (["qwen3"] if self.worker is not None else [])

    def synth(self, system: str, name: str, prompt16: np.ndarray, prompt_text: str, text: str, prompt_path: Path) -> dict:
        t0 = time.perf_counter()
        if system == "speecht5":
            xv = self.xv.embed(prompt16)
            self.st5.xvector = self.torch.tensor(unit(xv), dtype=self.torch.float32)[None]
            r = self.st5.synth(text, int(CFG["speecht5"]["seed"]))
            return {"wav": r["wav"], "fs": r["fs"], "gen_s": time.perf_counter() - t0, "seed": int(CFG["speecht5"]["seed"]), "T": r["T"],
                    "xvector_norm_before_unit": round(float(np.linalg.norm(xv)), 2)}
        if system == "f5":
            p25 = np.concatenate([dsp.resample(prompt16, FS, 24000), np.zeros(int(24000 * D1.PROMPT_PAD_MS / 1000))])
            r = self.f5.synth(p25, prompt_text, text, int(CFG["f5"]["seed"]), nfe=self.nfe)
            return {"wav": r["wav"], "fs": r["fs"], "gen_s": time.perf_counter() - t0, "seed": int(CFG["f5"]["seed"]), "T": r["T"], "nfe": r["nfe"],
                    "w_slide": r["w_slide"], "sway": r["sway"], "length": {k: v for k, v in r["length"].items() if k != "ref_text_used"}}
        q = self.worker.qwen3(name, text, prompt_path, prompt_text, int(CFG["qwen3"]["seed"]))
        if not q.get("ok"):
            raise RuntimeError(q.get("error"))
        y = C.load_wav(Path(q["wav"]), q["sr"])
        return {"wav": y, "fs": int(q["sr"]), "gen_s": time.perf_counter() - t0, "seed": int(CFG["qwen3"]["seed"]), "n_tokens": q.get("n_tokens"),
                "temperature": CFG["qwen3"]["temperature"]}

    def info(self) -> dict:
        i = {"speecht5": self.st5.info(), "f5": self.f5.info(), "xvector": self.xv.info()}
        i["speecht5"]["xvector"] = {"from": "the prompt, through [xvector] (unit-normalised)"}
        if self.nfe:
            i["f5"]["nfe"] = self.nfe
        i["qwen3"] = ({"repo": QWEN_REPO, "commit": D2_lock().get(QWEN_REPO), "mlx": self.worker.versions, "loaded": self.worker.loaded.get("qwen3"),
                       "temperature": CFG["qwen3"]["temperature"], "streaming_interval": CFG["qwen3"]["streaming_interval"]} if self.worker else None)
        return i

    def close(self) -> None:
        if self.worker is not None:
            self.worker.close()


class FakeSystems:
    """替身：輸出＝同一位說話人念這一句的真人錄音；兩個「續寫」的系統把 prompt 那一版的污染原樣帶進去，「瓶頸」的不帶。畫面標 FAKE。"""
    fake = True

    def __init__(self, real_target: np.ndarray):
        self.real = real_target

    def available(self) -> list[str]:
        return list(SYSTEMS)

    def synth(self, system: str, name: str, prompt16: np.ndarray, prompt_text: str, text: str, prompt_path: Path) -> dict:
        fs = 16000 if system == "speecht5" else 24000
        y = self.real.copy()
        if system != "speecht5":
            v = name.split("_")[0]
            if v in ("noise", "music", "phone"):
                y = make_prompts(y)[0][v]
        return {"wav": dsp.resample(y, FS, fs) * (0.9 if system == "qwen3" else 1.0), "fs": fs, "gen_s": 0.0, "seed": 0}

    def info(self) -> dict:
        return {"fake": True}

    def close(self) -> None:
        pass


class FakeEmbed:
    """替身的 speaker encoder：log-mel 的時間平均與標準差接起來。只為了讓流程走得過。"""
    label = "FAKE encoder"

    def embed(self, x16: np.ndarray) -> np.ndarray:
        M = disp_mel(x16, FS)
        act = M.mean(axis=0) > M.mean(axis=0).max() - 25
        M = M[:, act] if act.any() else M
        return np.concatenate([M.mean(axis=1) - M.mean(), M.std(axis=1)])

    def info(self) -> dict:
        return {"fake": True}


class FakeASR:
    label = "FAKE ASR"

    def transcribe(self, x16: np.ndarray, lang: str = "en", hint: str | None = None) -> str:
        return hint or ""

    norm = staticmethod(basic_norm_en)

    def info(self) -> dict:
        return {"fake": True}


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


def versions() -> dict:
    v = C.versions()
    from importlib import metadata
    for pkg in ("torch", "torchaudio", "transformers", "huggingface_hub", "f5-tts", "vocos", "speechbrain", "pypinyin", "soundfile"):
        try:
            v[pkg] = metadata.version(pkg)
        except metadata.PackageNotFoundError:
            v[pkg] = None
    return v


# ── 主體 ─────────────────────────────────────────────────────
class Demo:
    """items[name] = {"wav", "fs", "wav_path", "kind": prompt｜cell｜ref｜own_prompt｜own_cell, 量到的數字…}。
    名字：prompt_<版>、<版>_<系統>、real_target、other_voice、own_<k>_prompt、own_<k>_<系統>。"""

    def __init__(self, out: Out, systems, sim, sim2, asr, folder: Path, tag: str = ""):
        self.out, self.sys, self.sim, self.sim2, self.asr, self.folder, self.tag = out, systems, sim, sim2, asr, folder, tag
        self.items: dict[str, dict] = {}
        self.emb: dict[str, dict] = {}
        self.log: dict = {"time": C.now(), "versions": versions(), "tag": tag,
                          "text": {"target": TARGET, "prompt_text": PROMPT_TEXT, "prompt_utt": PROMPT_UTT, "target_utt": D1.TXT.get("target_utt")},
                          "env": {"PYTORCH_ENABLE_MPS_FALLBACK_at_start": os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK")},
                          "models": None, "judges": None, "prompt": {}, "items": {}, "own": {}, "played": [], "config": {k: CFG[k] for k in ("versions", "measure", "sim", "f0")}}
        self.fig = self.plt = None
        self.mode, self.ver, self.own_k = "prompts", "noise", OWN[0] if OWN else None
        self.numbers = False
        self.last = None
        p = (HERE / CFG["audio"]["match_rms_to"]).resolve()
        self.ref_rms = rms(C.load_wav(p, FS)) if p.exists() else 0.05
        self.folder.mkdir(parents=True, exist_ok=True)

    # ── 存一段聲音、量它 ──
    def _add(self, name: str, kind: str, wav: np.ndarray, fs: int, **extra) -> dict:
        y = wav * (self.ref_rms / rms(wav)) if len(wav) else wav
        pk = float(np.max(np.abs(y))) if len(y) else 0.0
        if pk > 0.99:
            y = y * 0.99 / pk
        p = self.folder / f"{name}.wav"
        C.save_wav(p, y, fs)
        r = {"name": name, "kind": kind, "wav": wav, "fs": fs, "wav_path": p, "seconds": round(len(wav) / fs, 3),
             "floor_db": round(floor_db(wav, fs), 1), "hf_db": round(hf_db(wav, fs), 1), **extra}
        self.items[name] = r
        return r

    def _embed(self, name: str) -> None:
        r = self.items[name]
        x16 = dsp.resample(r["wav"], r["fs"], FS)
        self.emb[name] = {"sim": self.sim.embed(x16)}
        if self.sim2 is not None:
            self.emb[name]["sim2"] = self.sim2.embed(x16)

    def _sim(self, a: str, b: str, which: str = "sim") -> float | None:
        if a not in self.emb or b not in self.emb or which not in self.emb[a] or which not in self.emb[b]:
            return None
        return round(cos(self.emb[a][which], self.emb[b][which]), 3)

    def _asr(self, name: str, ref: str, lang: str) -> None:
        r = self.items[name]
        hyp = self.asr.transcribe(dsp.resample(r["wav"], r["fs"], FS), lang, hint=ref)
        r["hyp"] = hyp
        r["wer"] = error_rate(ref, hyp, lang, getattr(self.asr, "norm", None))

    def _record(self, name: str) -> None:
        r = self.items[name]
        rec = {k: v for k, v in r.items() if k not in ("wav", "wav_path", "name")}
        if "gen_s" in rec:
            rec["gen_s"] = round(rec["gen_s"], 2)
        rec["wav"] = os.path.relpath(r["wav_path"], HERE)
        self.log["items"][name] = rec

    # ── 全部算好 ──
    def compute(self) -> None:
        o = self.out
        if not PROMPT_WAV.exists():
            sys.exit(f"沒有 {PROMPT_WAV}：先跑 ../w08_d1_mean_vs_sample/present.sh fetch")
        base = D1.trim_edges(C.load_wav(PROMPT_WAV, FS), FS)
        prompts, how = make_prompts(base)
        self.log["models"], self.log["judges"] = self.sys.info(), {"sim": self.sim.info(), "sim2": self.sim2.info() if self.sim2 is not None else None, "asr": self.asr.info()}
        o(f"\n{C.BOLD}prompt{C.RESET}  {PROMPT_UTT}  「{PROMPT_TEXT}」  {len(base) / FS:.2f} s（裁過前後靜音）")
        o(f"{C.BOLD}要念的句子{C.RESET}  「{TARGET}」")
        self.log["prompt"] = {"seconds": round(len(base) / FS, 3), "versions": how}
        for v in VERSIONS:
            r = self._add(f"prompt_{v}", "prompt", prompts[v], FS, version=v)
            self._embed(r["name"])
            r["sim_clean"] = self._sim(r["name"], "prompt_clean")
            r["sim2_clean"] = self._sim(r["name"], "prompt_clean", "sim2")
            if v in ("noise", "music"):
                r["snr_db_measured"] = round(20 * math.log10(rms(prompts["clean"]) / rms(prompts[v] - prompts["clean"])), 2)
            self._record(r["name"])
            o(f"  prompt · {VER_LABEL[v]:<15s} 噪音底 {r['floor_db']:6.1f} dB   4 kHz 以上 {r['hf_db']:6.1f} dB   SIM 對乾淨的 prompt {r['sim_clean']:.3f}")
        # 兩個參考點
        for name, key, what in (("real_target", "real_target", "同一位說話人念這一句的真人錄音"), ("other_voice", "other_voice", "上週 SpeechT5 用 ARCTIC bdl 的聲音念的同一句（另一個人）")):
            p = (HERE / CFG["audio"][key]).resolve()
            if not p.exists():
                o(f"  {C.GREY}（沒有 {os.path.relpath(p, HERE)}，跳過參考點 {name}）{C.RESET}")
                continue
            r = self._add(name, "ref", C.load_wav(p, FS), FS, what=what)
            self._embed(name)
            r["sim_clean"], r["sim2_clean"] = self._sim(name, "prompt_clean"), self._sim(name, "prompt_clean", "sim2")
            self._asr(name, TARGET, "en")
            self._record(name)
            o(f"  參考 · {what}：SIM 對乾淨的 prompt {r['sim_clean']:.3f}   WER {r['wer']['pct']:.1f}%")
        # 十二格
        avail = self.sys.available()
        for v in VERSIONS:
            for s in SYSTEMS:
                if s not in avail:
                    continue
                name = cell(v, s)
                try:
                    g = self.sys.synth(s, name, prompts[v], PROMPT_TEXT, TARGET, self.items[f"prompt_{v}"]["wav_path"])
                except Exception as e:
                    o(f"  {C.RED}{name} 失敗：{e!r}{C.RESET}")
                    continue
                r = self._add(name, "cell", g.pop("wav"), g.pop("fs"), version=v, system=s, **g)
                self._embed(name)
                r["sim_clean"], r["sim_version"] = self._sim(name, "prompt_clean"), self._sim(name, f"prompt_{v}")
                r["sim2_clean"], r["sim2_version"] = self._sim(name, "prompt_clean", "sim2"), self._sim(name, f"prompt_{v}", "sim2")
                self._asr(name, TARGET, "en")
                self._record(name)
                o(f"  {VER_LABEL[v]:<15s} → {SYS_SHORT[s]:<10s} {r['seconds']:5.2f} s   噪音底 {r['floor_db']:6.1f} dB   4 kHz 以上 {r['hf_db']:6.1f} dB   "
                  f"SIM {r['sim_clean']:.3f} / {r['sim_version']:.3f}   WER {r['wer']['pct']:4.1f}%   生成 {r['gen_s']:5.1f} s")
        # 加一格
        for k in OWN:
            p = own_wav(k)
            c = CFG["own"][k]
            if not p.exists():
                o(f"  {C.GREY}（沒有 audio/own_{k}.wav：./present.sh record {k} 之後才有鍵 {'e' if k == 'mood' else 'a'} 的內容）{C.RESET}")
                continue
            x = D1.trim_edges(C.load_wav(p, FS), FS)
            pr = self._add(f"own_{k}_prompt", "own_prompt", x, FS, own=k, private=True, **f0_stats(x, FS))
            self._embed(pr["name"])
            self._record(pr["name"])
            self.log["own"][k] = {kk: c[kk] for kk in ("prompt_text", "prompt_lang", "target", "target_lang")}
            o(f"\n{C.BOLD}加一格 · {k}{C.RESET}  prompt「{c['prompt_text']}」{len(x) / FS:.2f} s  F0 中位數 {pr['f0_median_hz']} Hz、起伏 {pr['f0_span_semitones']} 半音   →  「{c['target']}」")
            for s in OWN_SYSTEMS:
                if s not in avail:
                    continue
                name = own_cell(k, s)
                try:
                    g = self.sys.synth(s, name, x, c["prompt_text"], c["target"], pr["wav_path"])
                except Exception as e:
                    o(f"  {C.RED}{name} 失敗：{e!r}{C.RESET}")
                    continue
                r = self._add(name, "own_cell", g.pop("wav"), g.pop("fs"), own=k, system=s, private=True, **g)
                r.update(f0_stats(r["wav"], r["fs"]))
                self._embed(name)
                r["sim_prompt"], r["sim2_prompt"] = self._sim(name, pr["name"]), self._sim(name, pr["name"], "sim2")
                self._asr(name, c["target"], c["target_lang"])
                self._record(name)
                o(f"  {SYS_SHORT[s]:<10s} {r['seconds']:5.2f} s   F0 中位數 {r['f0_median_hz']} Hz、起伏 {r['f0_span_semitones']} 半音   SIM 對這段 prompt {r['sim_prompt']:.3f}   "
                  f"錯誤率 {r['wer']['pct']:.1f}%（{r['wer']['unit']}）")

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

    def _mel(self, ax, name: str, title: str, color: str = "black", hl: bool = False):
        if name not in self.items:
            self._hint(ax, title); return
        r = self.items[name]
        S = disp_mel(r["wav"], r["fs"])
        vmax = float(S.max())
        ax.imshow(S, origin="lower", aspect="auto", cmap="magma", vmin=vmax - CFG["display"]["db_range"], vmax=vmax,
                  extent=[0, S.shape[1] / 100.0, -0.5, S.shape[0] - 0.5], interpolation="nearest")
        ticks = [250, 500, 1000, 2000, 4000, 8000]
        ax.set_yticks([mel_bin_of(f) for f in ticks]); ax.set_yticklabels([f"{f / 1000:g}k" for f in ticks], fontsize=8)
        ax.axhline(mel_bin_of(CFG["measure"]["hf_split_hz"]), color="white", lw=0.7, ls=(0, (4, 4)), alpha=0.7)
        if self.numbers:
            title += f"\nfloor {r['floor_db']:.0f} dB · above 4 kHz {r['hf_db']:.0f} dB"
            if r.get("sim_clean") is not None and r["kind"] == "cell":
                title += f" · SIM {r['sim_clean']:.2f} / {r['sim_version']:.2f}"
        ax.set_title(title, fontsize=9.5, loc="left", color=color)
        ax.tick_params(labelsize=8); ax.set_xlabel("time (s)", fontsize=8)
        if hl:
            for s in ax.spines.values():
                s.set_color("#d62728"); s.set_linewidth(2.5)

    def _hint(self, ax, text: str):
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_color("#cccccc")
        ax.text(0.5, 0.5, text, ha="center", va="center", fontsize=10, color="#777777", transform=ax.transAxes, wrap=True)

    def draw(self, hl: str | None = None):
        if self.fig is None:
            return
        self.fig.clf()
        tag = f"   [{self.tag}]" if self.tag else ""
        if self.mode == "prompts":
            self.fig.suptitle("W9 Demo 1 — one prompt, four versions (same speaker, same words, same length)" + tag, fontsize=12)
            gs = self.fig.add_gridspec(2, 2)
            for i, v in enumerate(VERSIONS):
                n = f"prompt_{v}"
                self._mel(self.fig.add_subplot(gs[i // 2, i % 2]), n, f"prompt · {VER_LABEL[v]}", hl=(hl == n))
        elif self.mode == "version":
            v = self.ver
            self.fig.suptitle(f"W9 Demo 1 — the {VER_LABEL[v]} prompt, through three ways to condition" + tag, fontsize=12)
            gs = self.fig.add_gridspec(2, 2)
            n = f"prompt_{v}"
            self._mel(self.fig.add_subplot(gs[0, 0]), n, f"the prompt · {VER_LABEL[v]}", color="#555555", hl=(hl == n))
            for i, s in enumerate(SYSTEMS, 1):
                n = cell(v, s)
                self._mel(self.fig.add_subplot(gs[i // 2, i % 2]), n, f"{SYS_LABEL[s]}" + ("" if n in self.items else " — not available"), color=SYS_COLOR[s], hl=(hl == n))
        elif self.mode == "own":
            k = self.own_k
            self.fig.suptitle(f"W9 Demo 1, the extra row — {'a mood' if k == 'mood' else 'an accent'} rides along (the lecturer's own voice)" + tag, fontsize=12)
            gs = self.fig.add_gridspec(1, 3)
            n = f"own_{k}_prompt"
            self._mel(self.fig.add_subplot(gs[0, 0]), n, "the prompt" if n in self.items else f"record it first: ./present.sh record {k}", color="#555555", hl=(hl == n))
            for i, s in enumerate(OWN_SYSTEMS, 1):
                n = own_cell(k, s)
                self._mel(self.fig.add_subplot(gs[0, i]), n, f"{SYS_SHORT[s]} reads {'a neutral sentence' if k == 'mood' else 'an English sentence'}", color=SYS_COLOR[s], hl=(hl == n))
        else:
            self.fig.suptitle("W9 Demo 1 — what the three outputs carry over from each prompt version" + tag, fontsize=12)
            gs = self.fig.add_gridspec(1, 3)
            x = np.arange(len(VERSIONS))
            cols = ["prompt"] + SYSTEMS
            w = 0.8 / len(cols)
            for j, (key, ttl) in enumerate((("floor_db", "noise floor in the pauses (dB re speech) — higher = noisier"), ("hf_db", "energy above 4 kHz (dB re total)"))):
                ax = self.fig.add_subplot(gs[0, j])
                for i, cname in enumerate(cols):
                    vals = [self.items.get(f"prompt_{v}" if cname == "prompt" else cell(v, cname), {}).get(key, np.nan) for v in VERSIONS]
                    ax.bar(x + (i - (len(cols) - 1) / 2) * w, vals, w, color=SYS_COLOR[cname], label="the prompt" if cname == "prompt" else SYS_SHORT[cname])
                ax.set_xticks(x); ax.set_xticklabels([VER_LABEL[v] for v in VERSIONS], fontsize=9)
                ax.set_title(ttl, fontsize=9.5, loc="left"); ax.grid(alpha=0.3, axis="y"); ax.tick_params(labelsize=8)
                if j == 0:
                    ax.legend(fontsize=8.5, loc="upper center", bbox_to_anchor=(0.5, -0.07), ncol=4, frameon=False)
            ax = self.fig.add_subplot(gs[0, 2])
            w = 0.8 / (2 * len(SYSTEMS))
            for i, s in enumerate(SYSTEMS):
                a = [self.items.get(cell(v, s), {}).get("sim_clean") or np.nan for v in VERSIONS]
                b = [self.items.get(cell(v, s), {}).get("sim_version") or np.nan for v in VERSIONS]
                ax.bar(x + (2 * i - len(SYSTEMS) + 0.5) * w, a, w, color=SYS_COLOR[s], label=f"{SYS_SHORT[s]} vs the clean prompt")
                ax.bar(x + (2 * i + 1 - len(SYSTEMS) + 0.5) * w, b, w, color=SYS_COLOR[s], alpha=0.45, hatch="//", label=f"{SYS_SHORT[s]} vs that version")
            for nm, ls, lab in (("real_target", "--", "same speaker, real recording"), ("other_voice", ":", "another voice (ARCTIC bdl)")):
                if nm in self.items and self.items[nm].get("sim_clean") is not None:
                    ax.axhline(self.items[nm]["sim_clean"], color="#444444", lw=0.9, ls=ls, label=lab)
            ax.set_xticks(x); ax.set_xticklabels([VER_LABEL[v] for v in VERSIONS], fontsize=9)
            sims = [r[k] for r in self.items.values() for k in ("sim_clean", "sim_version") if r.get(k) is not None and r["kind"] in ("cell", "ref")]
            if sims:
                ax.set_ylim(max(0.0, min(sims) - 0.05), 1.0)
            ax.set_title(f"SIM ({self.sim.label}): solid vs clean, hatched vs that version", fontsize=9.5, loc="left")
            ax.grid(alpha=0.3, axis="y"); ax.tick_params(labelsize=8)
            ax.legend(fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.07), ncol=2, frameon=False)
        self.fig.tight_layout()
        self.redraw()

    # ── 播 ──
    def _play(self, name: str, what: str) -> None:
        if name not in self.items:
            self.out(f"  {C.AMBER}沒有 {name}{C.RESET}"); return
        self.draw(hl=name)
        self.out(f"  ▶ {what}")
        self.last = (name, what)
        self.log["played"].append(what)
        C.play(self.items[name]["wav_path"])

    def show_prompts(self) -> None:
        self.mode = "prompts"
        self.out(f"\n{C.BOLD}四版 prompt{C.RESET}  {PROMPT_UTT}「{PROMPT_TEXT}」")
        for v in VERSIONS:
            self._play(f"prompt_{v}", f"prompt · {VER_LABEL[v]}")
        self.draw()

    def show_version(self, v: str, with_prompt: bool = False) -> None:
        self.mode, self.ver = "version", v
        self.out(f"\n{C.BOLD}{VER_LABEL[v]} 的 prompt → 三個系統{C.RESET}  「{TARGET}」")
        if with_prompt:
            self._play(f"prompt_{v}", f"prompt · {VER_LABEL[v]}")
        for s in SYSTEMS:
            if cell(v, s) in self.items:
                self._play(cell(v, s), f"{VER_LABEL[v]} → {SYS_SHORT[s]}")
        self.draw()

    def show_own(self, k: str) -> None:
        self.mode, self.own_k = "own", k
        c = CFG["own"].get(k)
        if c is None or f"own_{k}_prompt" not in self.items:
            self.out(f"\n  {C.AMBER}沒有 audio/own_{k}.wav：先 ./present.sh record {k}（然後重新 rehearse）{C.RESET}")
            self.draw(); return
        self.out(f"\n{C.BOLD}加一格 · {k}{C.RESET}  prompt「{c['prompt_text']}」→「{c['target']}」")
        self._play(f"own_{k}_prompt", f"{k} · the prompt")
        for s in OWN_SYSTEMS:
            if own_cell(k, s) in self.items:
                r = self.items[own_cell(k, s)]
                self._play(own_cell(k, s), f"{k} → {SYS_SHORT[s]}")
                self.out(f"      Whisper 聽到：{r.get('hyp', '')}")
        self.draw()
        self.own_table(k)

    # ── 表 ──
    def _row(self, label: str, vals: list[str]) -> str:
        return f"    {label:<16s}" + "".join(f"{v:>16s}" for v in vals)

    def table(self) -> None:
        o = self.out
        self.numbers = True
        avail = [s for s in SYSTEMS if any(cell(v, s) in self.items for v in VERSIONS)]
        head = self._row("prompt version", ["the prompt"] + [SYS_SHORT[s] for s in avail])
        o(f"\n{C.BOLD}W9 Demo 1 — numbers{C.RESET}   prompt {PROMPT_UTT}（{self.log['prompt'].get('seconds')} s）  「{TARGET}」")

        def block(title: str, key: str, fmt: str):
            o(f"  {title}")
            o(head)
            for v in VERSIONS:
                vals = [fmt.format(self.items[f"prompt_{v}"][key]) if f"prompt_{v}" in self.items else "—"]
                vals += [fmt.format(self.items[cell(v, s)][key]) if cell(v, s) in self.items else "—" for s in avail]
                o(self._row(VER_LABEL[v], vals))

        block("noise floor in the pauses (dB re speech: 5th − 95th percentile of 10 ms frame levels inside the utterance; closer to 0 = noisier)", "floor_db", "{:.1f}")
        block("energy above 4 kHz (dB re total, at 16 kHz)", "hf_db", "{:.1f}")
        for which, lab in (("sim", self.sim.label), ("sim2", getattr(self.sim2, "label", None))):
            if lab is None:
                continue
            o(f"  SIM — {lab} cosine: vs the clean prompt / vs that version of the prompt" + ("" if which == "sim" else "   (a second encoder: the numbers are not comparable with the first)"))
            o(head)
            kc, kv = ("sim_clean", "sim_version") if which == "sim" else ("sim2_clean", "sim2_version")
            for v in VERSIONS:
                p = self.items.get(f"prompt_{v}", {})
                vals = [f"{p[kc]:.3f} / 1.000" if p.get(kc) is not None else "—"]
                for s in avail:
                    r = self.items.get(cell(v, s), {})
                    vals.append(f"{r[kc]:.3f} / {r[kv]:.3f}" if r.get(kc) is not None else "—")
                o(self._row(VER_LABEL[v], vals))
            for nm in ("real_target", "other_voice"):
                if nm in self.items and self.items[nm].get(kc) is not None:
                    o(f"    reference · {self.items[nm]['what']}：{self.items[nm][kc]:.3f}")
        o(f"    （{self.sim.label} 的 model card 給的同一人門檻：{CFG['sim']['threshold_from_card']}——在 VoxCeleb1 的真人錄音上定的）")
        o(f"  WER % — {self.asr.label}, both sides through Whisper's English normaliser")
        o(head)
        for v in VERSIONS:
            vals = ["—"] + [f"{self.items[cell(v, s)]['wer']['pct']:.1f}" if cell(v, s) in self.items and "wer" in self.items[cell(v, s)] else "—" for s in avail]
            o(self._row(VER_LABEL[v], vals))
        bad = [(n, r) for n, r in self.items.items() if r["kind"] == "cell" and r.get("wer", {}).get("errors")]
        for n, r in bad:
            o(f"    {n}: heard “{r['hyp']}”")
        o("  One draw per cell (seed 1 for each system). The first two quantities are this demo's own, not standard metrics.")
        self.mode = "numbers"
        self.draw()

    def own_table(self, k: str) -> None:
        o = self.out
        pr = self.items.get(f"own_{k}_prompt")
        if not pr:
            return
        o(f"    {'':<12s}{'seconds':>9s}{'F0 median':>11s}{'F0 span (st)':>14s}{'SIM vs prompt':>15s}{'error %':>9s}")
        o(f"    {'the prompt':<12s}{pr['seconds']:>9.2f}{str(pr['f0_median_hz']):>11s}{str(pr['f0_span_semitones']):>14s}{'—':>15s}{'—':>9s}")
        for s in OWN_SYSTEMS:
            r = self.items.get(own_cell(k, s))
            if r:
                o(f"    {SYS_SHORT[s]:<12s}{r['seconds']:>9.2f}{str(r['f0_median_hz']):>11s}{str(r['f0_span_semitones']):>14s}{r['sim_prompt']:>15.3f}{r['wer']['pct']:>9.1f}")

    def menu(self) -> None:
        self.out(f"\n{C.BOLD}p16{C.RESET}  p 四版 prompt   1 加噪   2 加音樂   3 電話頻寬   c 乾淨   （s f k 只播一個系統，o 這一版的 prompt，g 真人錄音）   t 數字"
                 f"     {C.BOLD}p17{C.RESET}  e 語氣   a 腔     r 再播   h 選單   q 離開")

    def dispatch(self, k: str) -> bool:
        if k == "p":
            self.show_prompts()
        elif k in VER_KEY:
            self.show_version(VER_KEY[k])
        elif k in ("s", "f", "k"):
            s = {"s": "speecht5", "f": "f5", "k": "qwen3"}[k]
            self.mode = "version"
            self._play(cell(self.ver, s), f"{VER_LABEL[self.ver]} → {SYS_SHORT[s]}")
        elif k == "o":
            self.mode = "version"
            self._play(f"prompt_{self.ver}", f"prompt · {VER_LABEL[self.ver]}")
        elif k == "g":
            self._play("real_target", "the same speaker's real recording of this sentence")
        elif k == "t":
            self.table()
        elif k == "e" and "mood" in OWN:
            self.show_own("mood")
        elif k == "a" and "accent" in OWN:
            self.show_own("accent")
        elif k == "r":
            if self.last:
                self._play(*self.last)
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
        jobs = [("prompts", None, None, "prompts.png")] + [("version", v, None, f"version_{v}.png") for v in VERSIONS] + [("numbers", None, None, "numbers.png")]
        jobs += [("own", None, k, f"own_{k}.png") for k in OWN if f"own_{k}_prompt" in self.items]
        for mode, v, k, nm in jobs:
            self.fig = plt.figure(figsize=(d["fig_w"], d["fig_h"]))
            self.mode = mode
            if v:
                self.ver = v
            if k:
                self.own_k = k
            self.numbers = True
            self.draw(); self.fig.savefig(self.folder / nm, dpi=100); plt.close(self.fig)
        self.fig = None

    def save(self) -> None:
        stem = "rehearsal" if self.folder.name == "rehearsal" else "session"
        (self.folder / f"{stem}.json").write_text(json.dumps(self.log, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
        (self.folder / f"{stem}.txt").write_text(self.out.text(), encoding="utf-8")
        shutil.rmtree(self.folder / "_worker", ignore_errors=True)
        C.note(f"  存到 {os.path.relpath(self.folder, HERE)}/（{stem}.json、{stem}.txt、.wav）")


# ── replay ───────────────────────────────────────────────────
class _Label:
    def __init__(self, label):
        self.label = label


def build_replay(out: Out) -> Demo:
    rec_path = REHEARSAL / "rehearsal.json"
    if not rec_path.exists():
        sys.exit("沒有 runs/rehearsal/rehearsal.json：先 ./present.sh rehearse")
    rec = json.loads(rec_path.read_text(encoding="utf-8"))
    d = Demo(out, None, _Label("WavLM-SV"), _Label("x-vector (SpeechBrain, VoxCeleb)") if rec["judges"].get("sim2") else None, _Label("Whisper small"), REHEARSAL, tag="replay " + rec["time"])
    d.log = rec
    d.log["played"] = []
    missing = []
    for name, r in rec["items"].items():
        p = (HERE / r["wav"]).resolve()
        if not p.exists():
            missing.append(name); continue
        item = dict(r)
        item.update(name=name, wav_path=p, wav=C.load_wav(p, r["fs"]))
        d.items[name] = item
    out(f"  replay：{rec['time']} 的彩排，{len(d.items)} 段聲音" + (f"（{len(missing)} 段的 .wav 不在：{', '.join(missing[:4])}…）" if missing else ""))
    return d


def build(cmd: str, out: Out, device: str | None = None, no_mlx: bool = False, nfe: int | None = None) -> Demo:
    folder = REHEARSAL if cmd == "rehearse" else HERE / "runs" / "live" / (C.now().replace(":", "") + (".fake" if cmd == "fake" else ""))
    folder.mkdir(parents=True, exist_ok=True)
    if cmd == "fake":
        real = C.load_wav((HERE / CFG["audio"]["real_target"]).resolve(), FS)
        d = Demo(out, FakeSystems(real), FakeEmbed(), None, FakeASR(), folder, tag="FAKE")
    else:
        try:
            import torch  # noqa: F401
        except ImportError:
            sys.exit("沒有 torch：先跑 ../w08_d1_mean_vs_sample/setup.sh 與 ./setup.sh（../.venv-tts）；或 ./present.sh fake 演練流程")
        D1.quiet_hf()
        c = CFG["infer"]
        systems = Systems(out, folder, device, no_mlx, nfe)
        t0 = time.perf_counter()
        sim = WavLMSV(device or c["sim_device"])
        out(f"  載入 SIM 的裁判        {sim.n_params / 1e6:6.1f} M 參數   {sim.device}   {time.perf_counter() - t0:5.1f} s   （{REPOS['sim']}）")
        t0 = time.perf_counter()
        asr = Whisper(c["whisper_device"])
        out(f"  載入 WER 的裁判        Whisper small   {asr.device}   {time.perf_counter() - t0:5.1f} s")
        d = Demo(out, systems, sim, systems.xv if CFG["sim"].get("second") == "xvector" else None, asr, folder, tag="rehearsal" if cmd == "rehearse" else "")
    d.compute()
    return d


# ── check ────────────────────────────────────────────────────
def check(device: str | None = None) -> None:
    C.banner("課前體檢", C.AMBER)
    res = {"time": C.now(), "versions": versions(), "env": {"PYTORCH_ENABLE_MPS_FALLBACK": os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK")}}
    for k, v in res["versions"].items():
        print(f"  {k:<16s} {v}")
    try:
        import torch
    except ImportError:
        sys.exit("  沒有 torch：先跑 ../w08_d1_mean_vs_sample/setup.sh 與 ./setup.sh")
    if res["versions"].get("speechbrain") is None:
        sys.exit("  沒有 speechbrain：先跑 ./setup.sh（裝進 ../.venv-tts）")
    D1.quiet_hf()
    mps = bool(torch.backends.mps.is_available())
    res["mps_available"] = mps
    print(f"  {'mps':<16s} available={mps}   PYTORCH_ENABLE_MPS_FALLBACK={res['env']['PYTORCH_ENABLE_MPS_FALLBACK']}")
    lock = read_lock()
    for k, repo in REPOS.items():
        print(f"  {repo:<36s} @ {str(lock.get(repo))[:12]}")
    devs = [device] if device else (["cpu"] + (["mps"] if mps else []))
    caught: list[str] = []
    p17 = D1.trim_edges(C.load_wav(PROMPT_WAV, FS), FS)
    t16 = C.load_wav((HERE / CFG["audio"]["real_target"]).resolve(), FS)
    op = (HERE / CFG["audio"]["other_voice"]).resolve()
    o16 = C.load_wav(op, FS) if op.exists() else None
    prompts, _ = make_prompts(p17)

    def run(cls, name: str) -> dict:
        r, embs = {}, {}
        for dev in devs:
            try:
                with warnings.catch_warnings(record=True) as ws:
                    warnings.simplefilter("always")
                    t0 = time.perf_counter(); m = cls(dev); load = time.perf_counter() - t0
                    m.embed(p17)                                                        # 暖機
                    t0 = time.perf_counter(); e = {"prompt": m.embed(p17), "target": m.embed(t16), "noise": m.embed(prompts["noise"])}
                    if o16 is not None:
                        e["other"] = m.embed(o16)
                    dt = (time.perf_counter() - t0) / len(e)
                for w in ws:
                    if "fall back" in str(w.message) or "not currently supported on the MPS" in str(w.message):
                        caught.append(f"{name} {dev}: {str(w.message)[:160]}")
                embs[dev] = e
                r[dev] = {"load_s": round(load, 2), "embed_s": round(dt, 4), "dim": int(len(e["prompt"])), "norm": round(float(np.linalg.norm(e["prompt"])), 3),
                          "sim_same_speaker_other_sentence": round(cos(e["prompt"], e["target"]), 4),
                          "sim_other_voice": round(cos(e["prompt"], e["other"]), 4) if "other" in e else None,
                          "sim_clean_vs_noisy_prompt": round(cos(e["prompt"], e["noise"]), 4)}
                x = r[dev]
                print(f"  {name} {dev}: 載入 {x['load_s']} s · 一段 {x['embed_s'] * 1000:.0f} ms · {x['dim']} 維 · SIM 同一人另一句 {x['sim_same_speaker_other_sentence']:.3f}"
                      f" · 另一個聲音 {x['sim_other_voice']} · 乾淨對加噪的 prompt {x['sim_clean_vs_noisy_prompt']:.3f}")
            except Exception as e:
                r[dev] = {"error": repr(e)}
                print(f"  {name} {dev}: {C.RED}失敗{C.RESET} {e!r}")
        if "cpu" in embs and "mps" in embs:
            r["mps_vs_cpu_cosine"] = {k: round(cos(embs["cpu"][k], embs["mps"][k]), 6) for k in embs["cpu"]}
            print(f"  {name}：同一段聲音 mps 對 cpu 的向量 cosine {r['mps_vs_cpu_cosine']}（要 ≈ 1）")
        return r

    # 1. SpeechBrain 的 x-vector（speech_ai/CLAUDE.md §6 第 6 項：SpeechBrain 在 MPS 上的推論）
    res["xvector"] = run(XVector, "x-vector (SpeechBrain)")
    # 2. WavLMForXVector
    res["sim"] = run(WavLMSV, "WavLM-SV")
    # 3. Whisper 對真人錄音
    try:
        t0 = time.perf_counter(); asr = Whisper(CFG["infer"]["whisper_device"]); load = time.perf_counter() - t0
        t0 = time.perf_counter(); hyp = asr.transcribe(t16, "en"); dt = time.perf_counter() - t0
        res["whisper"] = {"load_s": round(load, 2), "transcribe_s": round(dt, 2), "hyp": hyp, "wer": error_rate(TARGET, hyp, "en", asr.norm),
                          "ref_normalised": asr.norm(TARGET), "hyp_normalised": asr.norm(hyp)}
        print(f"  Whisper {asr.device}: 真人錄音 → “{hyp}”  WER {res['whisper']['wer']['pct']}%（{dt:.2f} s）")
    except Exception as e:
        res["whisper"] = {"error": repr(e)}
        print(f"  Whisper: {C.RED}失敗{C.RESET} {e!r}")
    # 4. SpeechT5 吃這位說話人的 x-vector：對上週的 bdl
    try:
        xv = XVector(CFG["infer"]["xvector_device"])
        st = D1.SpeechT5(device or CFG["infer"]["speecht5_device"], device or CFG["infer"]["vocoder_device"])
        e = xv.embed(p17)
        bdl = st.xvector.numpy().reshape(-1)
        st.xvector = torch.tensor(unit(e), dtype=torch.float32)[None]
        y = st.synth(TARGET, int(CFG["speecht5"]["seed"]))
        (HERE / "runs" / "check").mkdir(parents=True, exist_ok=True)
        C.save_wav(HERE / "runs" / "check" / "speecht5_prompt_xvector.wav", y["wav"] / max(1e-9, np.max(np.abs(y["wav"]))) * 0.9, 16000)
        res["speecht5_with_prompt_xvector"] = {"xvector_norm": round(float(np.linalg.norm(e)), 2), "cos_with_arctic_bdl_mean": round(cos(e, bdl), 4), "T": y["T"], "seconds": round(y["seconds"], 2)}
        print(f"  SpeechT5 吃 prompt 的 x-vector：‖x‖ = {res['speecht5_with_prompt_xvector']['xvector_norm']}（單位化後才給模型）、對上週 bdl 平均的 cosine {res['speecht5_with_prompt_xvector']['cos_with_arctic_bdl_mean']:.3f}、"
              f"T = {y['T']}（{y['seconds']:.2f} s）→ runs/check/speecht5_prompt_xvector.wav")
    except Exception as e:
        res["speecht5_with_prompt_xvector"] = {"error": repr(e)}
        print(f"  SpeechT5 吃 prompt 的 x-vector: {C.RED}失敗{C.RESET} {e!r}")
    # 5. MLX 那一半
    try:
        py = WORKER_PY_ENV if WORKER_PY_ENV.exists() else Path(sys.executable)
        p = subprocess.run([str(py), str(WORKER_PY), "--selfcheck"], capture_output=True, text=True, timeout=120)
        res["mlx"] = json.loads(p.stdout.strip().splitlines()[-1]) if p.stdout.strip() else {"ok": False, "stderr": p.stderr[-500:]}
    except Exception as e:
        res["mlx"] = {"ok": False, "error": repr(e)}
    mv = res["mlx"].get("versions", {})
    print(f"  mlx-audio {mv.get('mlx-audio')}、mlx {mv.get('mlx')}；qwen3_tts import：{mv.get('qwen3_tts_in_mlx_audio')}；{QWEN_REPO} @ {str(D2_lock().get(QWEN_REPO))[:12]}")
    res["mps_fallback_warnings"] = caught
    if caught:
        print(f"  {C.AMBER}有運算退回 CPU：{caught[:3]}{C.RESET}")
    (HERE / "runs").mkdir(exist_ok=True)
    (HERE / "runs" / "check.json").write_text(json.dumps(res, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    print("\n  → runs/check.json。要看的：兩個 encoder 都載得起、mps 對 cpu 的 cosine ≈ 1、「同一人另一句」明顯高於「另一個聲音」、Whisper 對真人錄音 WER 0。")
    print("  下一步：./present.sh record mood；./present.sh record accent（加一格，可以晚點做）→ ./present.sh rehearse")


# ── fetch ────────────────────────────────────────────────────
def fetch() -> None:
    from huggingface_hub import HfApi, snapshot_download
    C.banner("下載 x-vector encoder、SIM 的 encoder、Whisper；確認 W8 的模型都在", C.AMBER)
    lock = read_lock()
    api = HfApi()
    for key, repo in REPOS.items():
        info = api.model_info(repo)
        sha = lock.get(repo) or info.sha
        path = snapshot_download(repo, revision=sha, allow_patterns=PATTERNS[key])
        lock[repo] = sha
        card = getattr(info, "card_data", None)
        size = sum(f.stat().st_size for f in Path(path).rglob("*") if f.is_file()) / 1e6
        print(f"  {repo} @ {sha[:12]}   {size:.0f} MB   license (card): {card.get('license') if card else None}")
    MODEL_LOCK.write_text("".join(f"{r} {s}\n" for r, s in lock.items()), encoding="utf-8")
    print("  → model.lock")
    for k in D1.REPOS:
        try:
            D1.model_dir(k, offline=True)
            print(f"  W8 Demo 1 · {D1.REPOS[k]}：在")
        except Exception:
            print(f"  {C.AMBER}W8 Demo 1 · {D1.REPOS[k]} 不在快取：先跑 ../w08_d1_mean_vs_sample/present.sh fetch{C.RESET}")
    print(f"  W8 Demo 1 · audio/prompt.wav：{'在' if PROMPT_WAV.exists() else C.AMBER + '不在，先跑它的 fetch' + C.RESET}")
    sha = D2_lock().get(QWEN_REPO)
    try:
        snapshot_download(QWEN_REPO, revision=sha, local_files_only=True)
        print(f"  W8 Demo 2 · {QWEN_REPO} @ {str(sha)[:12]}：在")
    except Exception:
        print(f"  {C.AMBER}W8 Demo 2 · {QWEN_REPO} 不在快取：先跑 ../w08_d2_nfe_cfg_first_packet/present.sh fetch{C.RESET}")
    print("下一步：./present.sh check")


# ── record ───────────────────────────────────────────────────
def record(k: str) -> None:
    if k not in CFG["own"] or not isinstance(CFG["own"][k], dict):
        sys.exit(f"demo_config.toml 的 [own] 沒有 {k}（有：{', '.join(OWN)}）")
    c = CFG["own"][k]
    C.banner(f"錄「加一格」的 prompt：{k}", C.AMBER)
    print(f"  照這一句念（{'用很明顯的語氣' if k == 'mood' else '平常說華語的樣子'}）：\n\n      {C.BOLD}{c['prompt_text']}{C.RESET}\n")
    print(f"  要念別的句子就先改 demo_config.toml [own.{k}] prompt_text——文字要與實際念的一致。按 Enter 開始錄 {CFG['own']['record_s']:g} 秒。")
    input()
    x = C.record(float(CFG["own"]["record_s"]), FS)
    y = D1.trim_edges(x, FS)
    if len(y) < FS:
        sys.exit(f"  {C.RED}裁掉靜音後只剩 {len(y) / FS:.2f} s——沒收到聲音？（麥克風權限、輸入裝置：../README.md 末段）{C.RESET}")
    p = own_wav(k)
    C.save_wav(p, y / max(1e-9, np.max(np.abs(y))) * 0.7, FS)
    print(f"  → {os.path.relpath(p, HERE)}（{len(y) / FS:.2f} s）。回放：")
    C.play(p)
    print("  不滿意就再跑一次。這個檔不上站。下一步：./present.sh rehearse")


# ── export ───────────────────────────────────────────────────
def export(rec_path: Path) -> None:
    rec = json.loads(rec_path.read_text(encoding="utf-8"))
    d = json.loads(METRICS_JSON.read_text(encoding="utf-8")) if METRICS_JSON.exists() else {
        "_note": "量測數字（進版控的出處，speech_ai/CLAUDE.md 4.8）。w09-data.json 是 make_figs_w09.py 算出來的、每次重產；量測的放這裡。"}
    drop = ("wav", "kind", "name")
    it = rec["items"]
    cells = {v: {s: {k: x for k, x in it[cell(v, s)].items() if k not in drop} for s in SYSTEMS if cell(v, s) in it} for v in VERSIONS}
    prompts = {v: {k: x for k, x in it[f"prompt_{v}"].items() if k not in drop} for v in VERSIONS if f"prompt_{v}" in it}
    refs = {n: {k: x for k, x in it[n].items() if k not in drop} for n in ("real_target", "other_voice") if n in it}
    own = {}
    for k, texts in rec.get("own", {}).items():
        own[k] = {"texts": texts, "prompt": {kk: x for kk, x in it.get(f"own_{k}_prompt", {}).items() if kk not in drop},
                  "systems": {s: {kk: x for kk, x in it[own_cell(k, s)].items() if kk not in drop} for s in OWN_SYSTEMS if own_cell(k, s) in it}}
    d["demo1"] = {"source": "demos/w09_d1_cloning_leak/runs/rehearsal/rehearsal.json", "time": rec["time"], "tag": rec.get("tag", ""), "text": rec["text"],
                  "models": rec["models"], "judges": rec["judges"], "prompt": rec["prompt"], "prompts": prompts, "cells": cells, "references": refs, "own": own,
                  "definitions": {"floor_db": "noise floor in the pauses, dB re speech: 5th percentile minus 95th percentile of the 10 ms frame RMS levels (dB) of the raw output, taken between the first and the last frame within 30 dB of that 95th percentile (leading and trailing silence excluded); closer to 0 = noisier. This demo's own quantity.",
                                  "hf_db": "energy above 4 kHz, dB re total: after resampling to 16 kHz, power in 4–8 kHz over power in 0–8 kHz. This demo's own quantity.",
                                  "sim_clean / sim_version": "cosine between WavLM-SV (microsoft/wavlm-base-plus-sv) embeddings of the output and of the clean prompt / of that version of the prompt",
                                  "sim2_*": "the same with a second encoder, the SpeechBrain x-vector that also conditions SpeechT5; not comparable with sim_*",
                                  "wer": "Whisper small (greedy) against the text to read; English: words after Whisper's English normaliser; Mandarin: tone-numbered pinyin syllables",
                                  "f0_span_semitones": "12 log2 (90th / 10th percentile of the autocorrelation F0 over voiced frames)",
                                  "one draw": "each cell is a single sample (seed 1 per system)", "what_we_heard": "filled by hand after the rehearsal (w09.qmd p16, p17)"}}
    METRICS_JSON.parent.mkdir(parents=True, exist_ok=True)
    METRICS_JSON.write_text(json.dumps(d, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    print(f"  → {os.path.relpath(METRICS_JSON, HERE)}（demo1）")


# ── selftest ─────────────────────────────────────────────────
def selftest() -> None:
    C.banner("selftest", C.AMBER)
    fails = []

    def ok(cond: bool, msg: str):
        print(f"  {'OK ' if cond else C.RED + 'FAIL' + C.RESET}  {msg}")
        if not cond:
            fails.append(msg)

    # 1. 四版 prompt：用合成的「語音」（有停頓的調幅鋸齒波，頻譜到 8 kHz）
    rng = np.random.default_rng(0)
    t = np.arange(int(4.0 * FS)) / FS
    src = np.sign(np.sin(2 * np.pi * 120 * t)) * 0.2 + 0.02 * rng.standard_normal(len(t))
    gate = ((t % 1.0) < 0.7).astype(float)
    base = src * gate + 1e-4 * rng.standard_normal(len(t))
    P, how = make_prompts(base)
    ok(set(P) == set(VERSIONS) and len({len(y) for y in P.values()}) == 1, f"四版 prompt 同長度（{len(base) / FS:.1f} s）：{', '.join(VERSIONS)}")
    for v in ("noise", "music"):
        snr = 20 * math.log10(rms(P["clean"]) / rms(P[v] - P["clean"]))
        ok(abs(snr - CFG["versions"][f"{v}_snr_db"]) < 0.05, f"{v}：量到的 SNR {snr:.2f} dB（設定 {CFG['versions'][f'{v}_snr_db']:g}）")
    ok(hf_db(P["phone"], FS) < hf_db(P["clean"], FS) - 30, f"電話頻寬：4 kHz 以上 {hf_db(P['clean'], FS):.1f} → {hf_db(P['phone'], FS):.1f} dB")
    lo = np.abs(np.fft.rfft(P["phone"]))[: int(3000 * len(base) / FS)]; lo0 = np.abs(np.fft.rfft(P["clean"]))[: int(3000 * len(base) / FS)]
    ok(abs(20 * math.log10(np.linalg.norm(lo) / np.linalg.norm(lo0))) < 0.5, "電話頻寬：3 kHz 以下的能量不變（< 0.5 dB）")
    # 2. 噪音底
    f_clean, f_noise = floor_db(P["clean"], FS), floor_db(P["noise"], FS)
    ok(f_clean < -45 and -25 < f_noise < -5, f"噪音底：乾淨 {f_clean:.1f} dB、加噪（SNR 10）{f_noise:.1f} dB")
    ok(abs(floor_db(P["noise"] * 0.1, FS) - f_noise) < 0.2, "噪音底不隨整體音量變")
    ok(floor_db(np.zeros(FS), FS) == 0.0, "數位靜音：位準夾在 −100 dBFS，不會是 −inf")
    pad = np.concatenate([np.zeros(FS), base, np.zeros(FS // 2)])
    ok(abs(floor_db(pad, FS) - f_clean) < 1.0, f"句子前後的靜音不算（前後各墊靜音：{floor_db(pad, FS):.1f} 對 {f_clean:.1f} dB）")
    m = synth_music(FS * 3)
    ok(abs(np.max(np.abs(m)) - 1) < 1e-9 and floor_db(m, FS) > -40, "合成的音樂：一直有聲音（沒有停頓）")
    # 3. WER 與正規化
    ok(edit_distance("a b c d".split(), "a x c".split()) == 2, "edit distance：一個替換加一個刪除 = 2")
    e = error_rate("Mister Quilter is glad.", "mr. quilter is  GLAD", "en")
    ok(e["errors"] == 0 and e["n_ref"] == 4, f"英文：大小寫、標點、Mr. 不算錯（{e}）")
    z = error_rate("會議改到下週三。", "会议改到下周三", "zh")
    z2 = error_rate("會議改到下週三。", "會議改到下週山", "zh")
    ok(z["n_ref"] == 7 and z2["errors"] == 1, f"華語：{z['unit']}——繁簡 {z['errors']} 個錯、三→山 {z2['errors']} 個錯")
    # 4. cosine、設定
    ok(abs(cos(np.array([1.0, 0, 0]), np.array([2.0, 0, 0])) - 1) < 1e-9 and abs(cos(np.array([1.0, 0]), np.array([0, 1.0]))) < 1e-9, "cosine")
    ok(all(k in CFG["own"] for k in OWN) and all(len(CFG["own"][k]["prompt_text"]) > 0 for k in OWN), f"加一格的設定：{OWN}")
    ok(mel_bin_of(4000) > mel_bin_of(1000) > mel_bin_of(250) >= 0, "畫面的 mel 軸")
    # 5. fake 流程：替身、全部鍵、表、圖、存檔、export
    import matplotlib
    matplotlib.use("Agg")
    C.play = lambda *a, **k: None
    global METRICS_JSON
    real = (HERE / CFG["audio"]["real_target"]).resolve()
    if not real.exists() or not PROMPT_WAV.exists():
        print("  --   沒有 W8 Demo 1 的 audio/（先跑它的 fetch），跳過 fake 流程")
    else:
        folder = HERE / "runs" / "_selftest"
        shutil.rmtree(folder, ignore_errors=True)
        folder.mkdir(parents=True)
        d = Demo(Out(), FakeSystems(C.load_wav(real, FS)), FakeEmbed(), None, FakeASR(), folder, tag="selftest FAKE")
        d.compute()
        n_cells = sum(1 for r in d.items.values() if r["kind"] == "cell")
        ok(n_cells == len(VERSIONS) * len(SYSTEMS), f"fake：{n_cells} 格（{len(VERSIONS)} 版 × {len(SYSTEMS)} 個系統）")
        it = d.items
        ok(it["noise_f5"]["floor_db"] > it["noise_speecht5"]["floor_db"] + 10 and it["noise_qwen3"]["floor_db"] > it["noise_speecht5"]["floor_db"] + 10,
           f"fake：把噪音帶進去的替身量得出來（噪音底 {it['noise_speecht5']['floor_db']} 對 {it['noise_f5']['floor_db']} dB）")
        ok(it["phone_f5"]["hf_db"] < it["phone_speecht5"]["hf_db"] - 20, f"fake：把頻寬帶進去的替身量得出來（4 kHz 以上 {it['phone_speecht5']['hf_db']} 對 {it['phone_f5']['hf_db']} dB）")
        ok(it["noise_f5"]["sim_version"] > it["noise_f5"]["sim_clean"], "fake：帶了噪音的輸出，對加噪的 prompt 比對乾淨的 prompt 像（替身的 encoder）")
        ok(all(r["wer"]["errors"] == 0 for r in it.values() if r["kind"] == "cell"), "fake：WER 欄都有")
        d.make_fig()
        for k in "p123csfkogtearh":
            d.dispatch(k)
        d.rehearse_figs(); d.save()
        ok(all((folder / n).exists() for n in ["prompts.png", "numbers.png", "session.json"] + [f"version_{v}.png" for v in VERSIONS]), "fake：圖與紀錄存了")
        ok("/Users/" not in (folder / "session.json").read_text(encoding="utf-8") and str(HERE) not in (folder / "session.json").read_text(encoding="utf-8"), "紀錄裡沒有絕對路徑")
        keep, METRICS_JSON = METRICS_JSON, folder / "metrics.json"
        export(folder / "session.json")
        mj = json.loads(METRICS_JSON.read_text(encoding="utf-8"))["demo1"]
        ok(set(mj["cells"]) == set(VERSIONS) and all(set(mj["cells"][v]) == set(SYSTEMS) for v in VERSIONS), "export：demo1.cells 是 版 × 系統")
        METRICS_JSON = keep
        shutil.rmtree(folder, ignore_errors=True)
    # 6. 有彩排紀錄：對 w09-metrics.json、replay 讀回
    rec_path = REHEARSAL / "rehearsal.json"
    if rec_path.exists() and METRICS_JSON.exists():
        rec = json.loads(rec_path.read_text(encoding="utf-8"))
        mj = json.loads(METRICS_JSON.read_text(encoding="utf-8")).get("demo1", {})
        bad = [n for n, r in rec["items"].items() if r["kind"] == "cell" and mj.get("cells", {}).get(r["version"], {}).get(r["system"], {}).get("floor_db") != r["floor_db"]]
        ok(not bad, f"彩排紀錄對 w09-metrics.json → demo1：{len(bad)} 處不一致")
        pub = [n for n, r in rec["items"].items() if not r.get("private")]
        ok(all((HERE / rec["items"][n]["wav"]).exists() for n in pub), "彩排的 .wav 都在（replay 用）")
        d = build_replay(Out())
        d.make_fig()
        for k in "p123ctea":
            d.dispatch(k)
        ok(len(d.items) >= len(pub), f"replay：讀回 {len(d.items)} 段、鍵都走得過")
    else:
        print("  --   沒有彩排紀錄或 w09-metrics.json，跳過比對")
    print(f"\n  {'全部通過' if not fails else str(len(fails)) + ' 項失敗'}")
    sys.exit(1 if fails else 0)


# ── main ─────────────────────────────────────────────────────
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["show", "rehearse", "replay", "check", "fetch", "record", "fake", "selftest", "export"])
    ap.add_argument("which", nargs="?", help="record：mood 或 accent")
    ap.add_argument("--device", help="show／rehearse／check：蓋過 demo_config.toml 的 device（cpu／mps／auto）")
    ap.add_argument("--no-plot", action="store_true", help="show／replay／fake：只在終端機，不開圖")
    ap.add_argument("--no-mlx", action="store_true", help="show／rehearse：不開 MLX worker（Qwen3-TTS 那一欄留空）")
    ap.add_argument("--nfe", type=int, help="show／rehearse：F5 的 NFE（預設 32；CPU 上試跑用小一點的）")
    args = ap.parse_args()

    if args.cmd == "check":
        check(args.device); return
    if args.cmd == "fetch":
        fetch(); return
    if args.cmd == "selftest":
        selftest(); return
    if args.cmd == "export":
        export(REHEARSAL / "rehearsal.json"); return
    if args.cmd == "record":
        record(args.which or "mood"); return

    if args.cmd == "rehearse" or args.no_plot:
        import matplotlib
        matplotlib.use("Agg")

    out = Out()
    out.banner("W9 Demo 1：four prompts, three ways to condition")
    d = build_replay(out) if args.cmd == "replay" else build(args.cmd, out, args.device, args.no_mlx, args.nfe)

    if args.cmd == "rehearse":
        C.play = lambda *a, **k: None
        d.table()
        for k in OWN:
            d.own_table(k)
        d.rehearse_figs()
        d.sys.close()
        d.save()
        export(d.folder / "rehearsal.json")
        print("\n下一步：./present.sh 自己按一遍 p 1 2 3 c t e a；聽到的寫進 p16、p17 的表（what we heard），runs/rehearsal/*.png 看過。")
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
        if d.sys is not None:
            d.sys.close()
        if args.cmd in ("show", "fake"):
            d.save()


if __name__ == "__main__":
    main()

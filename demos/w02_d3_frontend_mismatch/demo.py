# -*- coding: utf-8 -*-
# 此檔由 bin/publish_demos.py 從 private repo 的 <course>/demos/ 複製而來，請勿在公開 repo 直接編輯。
"""W2 demo 3：前端定義不匹配——同一段音、同一個預訓練辨識模型，只換 log-mel 的配方，WER 動多少。

  show       課堂：載入 Whisper small（transformers，CPU），先用它自己的前端轉錄 40 句當基準，然後終端機單鍵操作（下面的鍵）；
             離開時存 runs/live/<時間>/（不進版控）
  rehearse   課前：不互動，基準加全部 recipe 跑一遍 → runs/rehearsal/（進版控 = 現場的退路；含 wer.png、features_<id>.png）
  replay     現場模型出狀況時：從彩排紀錄讀回，同樣的鍵、同樣的畫面，不碰 torch
  check      課前體檢：版本、模型在不在快取、自己算的特徵對 transformers 的 WhisperFeatureExtractor（同配方要一致）、
             一句要多久、mel 第 64 個濾波器兩種尺度的中心頻率對 w02-data.json → runs/check.json
  fetch      課前一週：下載 openai/whisper-small（commit 沿 W4 Demo 3 的 model.lock）；從 ../_media/dev-clean.tar.gz 讀參考文字、
             每位說話人挑一句、解出 16 kHz wav → ../_media/w02_d3/wav/，清單 → data/sentences.json；要網路（模型）
  fake       沒有模型也能演練流程（合成的「辨識器」，畫面標明 FAKE，不是給學生看的）
  selftest   不需要模型：自己的前端對 transformers 的 numpy 實作、兩種 mel 尺度對 w02-data.json、WER、FLAC 表頭、挑句、fake 整條流程
  export     runs/rehearsal/ 的摘要 → slides/assets/w02/w02-metrics.json（鍵 demo3）

畫面上的鍵（順序照投影片 p66 講稿的【操作】）：
  r  基準：Whisper 自己的前端（transformers 的 WhisperFeatureExtractor）→ 轉錄 40 句 → WER（show 啟動時自動跑一次）
  0  同一個配方、自己的程式碼算（dsp.py）——每個旋鈕都對上時，應該拿到同一個 WER
  1  mel 尺度 HTK ↔ Slaney            2  濾波器正規化：峰值 1 ↔ 面積        3  magnitude ↔ power
  4  log 的 floor 1e-5 ↔ 1e-10        5  dither                              6  pre-emphasis 0.97
  7  輸出正規化：每句 CMVN ↔ clamp + affine                                  8  五個旋鈕一起轉
  a  全部 recipe 依序跑       t  到目前為止的表（投影片 p66 的表）      n  換特徵圖上看的那一句
  s  播目前特徵圖上那一句     h  重印選單      q  離開

判定的定義（課堂上要念出來的）：
  WER        全部參考詞數為分母的 corpus WER（詞級 Levenshtein：替換 S、刪除 D、插入 I）。參考文字與模型輸出都先過
             Whisper 自己的 EnglishTextNormalizer（小寫、去標點、數字與拼法標準化），所以 "Mr." 對 "MISTER" 不算錯
  mean |Δ|   這個 recipe 算出的特徵與 Whisper 自己的特徵逐格相減取絕對值，在有聲音的 frame 上平均；單位是 Whisper 正規化之後的單位
             （它的特徵大致落在 −1 到 1.5 之間）。"cells changed" 是有多少格不一樣（|Δ| > 1e-6）
  changed    40 句裡有幾句的轉錄與基準的轉錄不同——不管對錯，只看模型有沒有被動到
"""
from __future__ import annotations

import argparse
import io
import json
import math
import os
import re
import struct
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
FS = int(CFG["audio"]["fs"])
FE = CFG["frontend"]
N_FFT, HOP, N_MELS = int(FE["n_fft"]), int(FE["hop"]), int(FE["n_mels"])
N_SAMPLES = int(FE["chunk_s"]) * FS
N_FRAMES = N_SAMPLES // HOP
DIG = int(CFG["display"]["digits"])
MODEL_LOCK = HERE / "model.lock"
SENTENCES = HERE / "data" / "sentences.json"
METRICS = (HERE.parent.parent / "slides" / "assets" / "w02" / "w02-metrics.json").resolve()
DATA_JSON = (HERE.parent.parent / "slides" / "assets" / "w02" / "w02-data.json").resolve()
RECIPES = {r["key"]: r for r in CFG["recipe"]}
RECIPE_BY_ID = {r["id"]: r for r in CFG["recipe"]}
REF = {"key": "r", "id": "ref", "name": "Whisper's own front-end (transformers WhisperFeatureExtractor)", "changes": {}}
PATTERNS = ["*.json", "*.txt", "*.safetensors"]
# W4 Demo 3 的 lock 是單行 sha；W7 的是 `repo sha` 多行。有哪一份就沿哪一份
SIBLING_LOCKS = [HERE.parent / "w04_d3_attention_heads" / "model.lock",
                 HERE.parent / "w07_d1_hallucination" / "model.lock",
                 HERE.parent / "w07_d2_biasing" / "model.lock"]


def _p(rel: str) -> Path:
    return (HERE / rel).resolve()


# ── 純函數：文字與 WER（selftest 測這些）──────────────────────────────────
def norm_simple(text: str) -> str:
    """沒有 Whisper 的 normalizer 時的退路：大寫，只留 A–Z、0–9 與撇號。"""
    return re.sub(r"\s+", " ", re.sub(r"[^A-Z0-9' ]+", " ", text.upper())).strip()


def align_words(ref: list[str], hyp: list[str]) -> dict:
    """詞級 Levenshtein 加回溯：{"errors", "S", "D", "I", "n_ref", "ops"}；ops 是 (op, ref_word|None, hyp_word|None)。"""
    R, H = len(ref), len(hyp)
    d = np.zeros((R + 1, H + 1), dtype=np.int32)
    d[:, 0] = np.arange(R + 1)
    d[0, :] = np.arange(H + 1)
    for i in range(1, R + 1):
        for j in range(1, H + 1):
            d[i, j] = min(d[i - 1, j] + 1, d[i, j - 1] + 1, d[i - 1, j - 1] + (ref[i - 1] != hyp[j - 1]))
    ops = []
    i, j = R, H
    while i > 0 or j > 0:
        if i > 0 and j > 0 and d[i, j] == d[i - 1, j - 1] + (ref[i - 1] != hyp[j - 1]):
            ops.append(("C" if ref[i - 1] == hyp[j - 1] else "S", ref[i - 1], hyp[j - 1]))
            i, j = i - 1, j - 1
        elif i > 0 and d[i, j] == d[i - 1, j] + 1:
            ops.append(("D", ref[i - 1], None))
            i -= 1
        else:
            ops.append(("I", None, hyp[j - 1]))
            j -= 1
    ops.reverse()
    return {"errors": int(d[R, H]), "S": sum(o[0] == "S" for o in ops), "D": sum(o[0] == "D" for o in ops),
            "I": sum(o[0] == "I" for o in ops), "n_ref": R, "ops": ops}


def corpus_wer(refs: list[str], hyps: list[str]) -> dict:
    S = D = I = n = 0
    for r, h in zip(refs, hyps):
        a = align_words(r.split(), h.split())
        S, D, I, n = S + a["S"], D + a["D"], I + a["I"], n + a["n_ref"]
    return {"S": S, "D": D, "I": I, "errors": S + D + I, "n_ref": n, "wer": (S + D + I) / n if n else 0.0}


def diff_pairs(ref: str, hyp: str, limit: int = 3) -> str:
    """一句的錯誤摘要：`THE→A`、`−WORD`（刪）、`+WORD`（插），最多 limit 個，多的用 … 收。"""
    ops = [o for o in align_words(ref.split(), hyp.split())["ops"] if o[0] != "C"]
    parts = []
    for op, r, h in ops[:limit]:
        parts.append(f"{r}→{h}" if op == "S" else (f"−{r}" if op == "D" else f"+{h}"))
    if len(ops) > limit:
        parts.append(f"…(+{len(ops) - limit})")
    return "  ".join(parts)


def colored(ref: str, hyp: str) -> str:
    """輸出那一行：對的詞照印、替換與插入標紅、刪掉的參考詞以灰色 ⟨⟩ 標出位置。"""
    out = []
    for op, r, h in align_words(ref.split(), hyp.split())["ops"]:
        if op == "C":
            out.append(h)
        elif op == "S":
            out.append(f"{C.RED}{h}{C.RESET}")
        elif op == "I":
            out.append(f"{C.RED}+{h}{C.RESET}")
        else:
            out.append(f"{C.GREY}⟨{r}⟩{C.RESET}")
    return " ".join(out)


# ── 純函數：素材（FLAC 表頭、挑句）───────────────────────────────────────
def flac_streaminfo(head: bytes) -> tuple[int, int]:
    """從 FLAC 檔的前 42 個位元組讀 (sample_rate, total_samples)。STREAMINFO 一定是第一個 metadata block。"""
    if head[:4] != b"fLaC" or (head[4] & 0x7F) != 0:
        raise ValueError("不是 FLAC，或第一個 block 不是 STREAMINFO")
    b = head[8:8 + 34]
    sr = (b[10] << 12) | (b[11] << 4) | (b[12] >> 4)
    total = ((b[13] & 0x0F) << 32) | struct.unpack(">I", b[14:18])[0]
    return sr, total


def select_sentences(utts: list[dict], cfg: dict) -> list[dict]:
    """每位說話人一句：長度在 min_s–max_s 之間、最接近 target_s 的（都不在範圍內就取最接近的）。依說話人編號排序。確定性。"""
    by_spk: dict[str, list[dict]] = {}
    for u in utts:
        by_spk.setdefault(u["id"].split("-")[0], []).append(u)
    picked = []
    for spk in sorted(by_spk, key=lambda s: int(s) if s.isdigit() else s):
        us = by_spk[spk]
        ok = [u for u in us if float(cfg["min_s"]) <= float(u["dur_s"]) <= float(cfg["max_s"])] or us
        best = min(ok, key=lambda u: (abs(float(u["dur_s"]) - float(cfg["target_s"])), u["id"]))
        picked.append({"id": best["id"], "spk": spk, "dur_s": round(float(best["dur_s"]), 2), "text": best.get("text", "")})
    return picked


# ── 前端：每個旋鈕一個參數 ─────────────────────────────────────────────────
def recipe_params(changes: dict) -> dict:
    p = dict(FE["reference"])
    p.update(changes)
    return p


def filterbank(p: dict) -> np.ndarray:
    return dsp.mel_filterbank(FS, N_FFT, N_MELS, scale=p["mel_scale"], area_norm=(p["filter_norm"] == "area"),
                              fmin=float(FE["fmin"]), fmax=float(FE["fmax"]))


def filter_centres(scale: str) -> np.ndarray:
    """80 個三角形的中心頻率（Hz）。投影片 p64 的右圖就是兩種尺度的這條曲線相減。"""
    edges = dsp.mel_to_hz(np.linspace(dsp.hz_to_mel(float(FE["fmin"]), scale), dsp.hz_to_mel(float(FE["fmax"]), scale), N_MELS + 2), scale)
    return edges[1:-1]


def n_audio_frames(n_samples: int) -> int:
    return min(N_FRAMES, n_samples // HOP + 1)


def frontend(x: np.ndarray, p: dict, rng: np.random.Generator | None = None) -> np.ndarray:
    """波形（16 kHz，±1）→ (80, 3000) 的 log-mel，照 p 裡的配方。p 是 reference 時逐步對應 openai/whisper audio.py 的 log_mel_spectrogram。

    步驟：補零到 30 s → dither → pre-emphasis → 25 ms periodic Hann、10 ms hop、兩端 reflect padding 的 STFT（丟最後一個 frame）
         → |X|^power → mel filterbank（尺度、正規化）→ floor → log → 輸出正規化。
    """
    x = np.asarray(x, dtype=np.float64)[:N_SAMPLES]
    n_audio = n_audio_frames(len(x))
    x = np.pad(x, (0, N_SAMPLES - len(x)))
    if float(p["dither"]) > 0:
        rng = rng or np.random.default_rng(int(CFG["fake"]["seed"]))
        x = x + float(p["dither"]) * rng.standard_normal(len(x))
    if float(p["preemphasis"]) > 0:
        a = float(p["preemphasis"])
        x = np.concatenate(([x[0] * (1 - a)], x[1:] - a * x[:-1]))
    w = dsp.hann(N_FFT)
    xp = np.pad(x, N_FFT // 2, mode="reflect")
    frames = np.lib.stride_tricks.sliding_window_view(xp, N_FFT)[::HOP][:N_FRAMES]
    X = np.fft.rfft(frames * w, axis=1).T                      # (201, 3000)
    S = np.abs(X) ** float(p["power"])
    M = filterbank(p)
    mel = np.maximum(M @ S, float(p["floor"]))
    y = np.log10(mel) if p["log"] == "log10" else np.log(mel)
    if p["output"] == "whisper":
        y = np.maximum(y, y.max() - 8.0)
        y = (y + 4.0) / 4.0
    elif p["output"] == "cmvn":
        seg = y[:, :n_audio]                                   # 統計量只看有聲音的 frame，套到全部 3000 個 frame
        y = (y - seg.mean(axis=1, keepdims=True)) / (seg.std(axis=1, keepdims=True) + 1e-8)
    else:
        raise ValueError(f"不認得的 output：{p['output']}")
    return y


def feature_diff(a: np.ndarray, b: np.ndarray, n_audio: int) -> dict:
    d = np.abs(a[:, :n_audio] - b[:, :n_audio])
    return {"mean_abs": float(d.mean()), "max_abs": float(d.max()), "cells_changed": float((d > 1e-6).mean())}


# ── 模型 ───────────────────────────────────────────────────────────────────
def read_lock() -> str | None:
    for path in [MODEL_LOCK] + SIBLING_LOCKS:
        if not path.exists():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            parts = line.split()
            if len(parts) == 2 and parts[0] == CFG["model"]["repo"]:
                return parts[1]
            if len(parts) == 1 and len(parts[0]) == 40:
                return parts[0]
    return None


def model_dir(offline: bool = True) -> str:
    from huggingface_hub import snapshot_download
    return snapshot_download(CFG["model"]["repo"], revision=read_lock(), local_files_only=offline, allow_patterns=PATTERNS)


def strip_prefix(seq: list[int], sot: int, eot: int, n_prefix: int) -> list[int]:
    """generate 的輸出去掉前綴（<|startoftranscript|>、語言、任務、<|notimestamps|>）與結尾，只留文字 token。"""
    if sot in seq:
        seq = seq[seq.index(sot) + n_prefix:]
    else:
        while seq and seq[0] >= eot:
            seq = seq[1:]
    if eot in seq:
        seq = seq[: seq.index(eot)]
    return [int(t) for t in seq if int(t) < eot]


def quiet_transformers() -> None:
    try:
        from transformers.utils import logging as hf_logging
        hf_logging.set_verbosity_error()
        hf_logging.disable_progress_bar()
    except Exception:
        pass


class Whisper:
    """transformers 的 WhisperForConditionalGeneration。兩個入口：它自己的前端（features_hf）、任何 (80, 3000) 的特徵（transcribe）。"""

    def __init__(self):
        import torch
        from transformers import WhisperForConditionalGeneration, WhisperProcessor
        quiet_transformers()
        self.torch = torch
        m = CFG["model"]
        path = model_dir()
        self.path = path
        self.proc = WhisperProcessor.from_pretrained(path)
        self.fe, self.tok = self.proc.feature_extractor, self.proc.tokenizer
        self.model = WhisperForConditionalGeneration.from_pretrained(path).eval()
        self.device = torch.device(m["device"])
        self.model.to(self.device)
        self.n_params = sum(p.numel() for p in self.model.parameters())
        self._norm_ok = getattr(self.tok, "english_spelling_normalizer", None) is not None

        gen, cfg = self.model.generation_config, self.model.config
        self.sot = int(gen.decoder_start_token_id)
        self.eot = int(gen.eos_token_id if gen.eos_token_id is not None else cfg.eos_token_id)
        self.prefix = [self.sot, int(gen.lang_to_id[f"<|{m['language']}|>"]), int(gen.task_to_id[m["task"]]), int(gen.no_timestamps_token_id)]

    def features_hf(self, x: np.ndarray) -> np.ndarray:
        out = self.fe(np.asarray(x, dtype=np.float32), sampling_rate=FS, return_tensors="np")
        return np.asarray(out["input_features"][0], dtype=np.float64)

    def transcribe(self, feats: np.ndarray) -> dict:
        """(80, 3000) 的特徵 → {"text", "n_tokens", "avg_logprob"}。

        avg_logprob：生成之後再做一次 teacher-forced 前向，取每個輸出 token（含 <|endoftext|>）的 log P 平均——
        openai/whisper decoding.py 的 avg_logprob 是同一個定義。字沒變、這個數字也會先動。
        """
        torch, m = self.torch, CFG["model"]
        t = torch.from_numpy(np.ascontiguousarray(feats, dtype=np.float32))[None].to(self.device)
        with torch.no_grad():
            ids = self.model.generate(input_features=t, language=m["language"], task=m["task"], return_timestamps=False,
                                      max_new_tokens=int(m["max_new_tokens"]), do_sample=False, num_beams=int(m["num_beams"]))
        text_ids = strip_prefix([int(v) for v in ids[0].tolist()], self.sot, self.eot, len(self.prefix))
        full = self.prefix + text_ids + [self.eot]
        with torch.no_grad():
            lg = self.model(input_features=t, decoder_input_ids=torch.tensor([full], dtype=torch.long, device=self.device)).logits[0].float()
        logp = torch.log_softmax(lg, dim=-1).cpu().numpy()
        lps = [float(logp[j, full[j + 1]]) for j in range(len(self.prefix) - 1, len(full) - 1)]
        return {"text": self.tok.decode(text_ids, skip_special_tokens=True).strip(), "n_tokens": len(text_ids),
                "avg_logprob": float(np.mean(lps)) if lps else 0.0}

    def normalize(self, text: str) -> str:
        if CFG["scoring"]["normalizer"] == "whisper" and self._norm_ok:
            try:
                return norm_simple(self.tok.normalize(text))
            except Exception:
                pass
        return norm_simple(text)


class FakeWhisper:
    """沒有模型時的替身：特徵差多少，就把參考文字裡的幾個詞換掉。畫面標明 FAKE。"""

    def __init__(self, texts: list[str]):
        self.texts = texts
        self.rng = np.random.default_rng(int(CFG["fake"]["seed"]))
        self.path, self.n_params, self.device = "FAKE", 0, "fake"
        self._ref_feats: dict[int, np.ndarray] = {}
        self._k = 0

    def features_hf(self, x: np.ndarray) -> np.ndarray:
        return frontend(x, recipe_params({}))

    def set_sentence(self, k: int, ref_feats: np.ndarray) -> None:
        self._k, self._ref_feats[k] = k, ref_feats

    def transcribe(self, feats: np.ndarray) -> dict:
        words = self.texts[self._k].split()
        ref = self._ref_feats.get(self._k)
        d = 0.0 if ref is None else float(np.abs(feats - ref).mean())
        n_err = int(min(len(words), round(d * float(CFG["fake"]["errors_per_unit_diff"]) * len(words) / 10)))
        for i in self.rng.choice(len(words), size=n_err, replace=False) if n_err else []:
            words[i] = words[i][::-1]
        return {"text": " ".join(words), "n_tokens": len(words), "avg_logprob": -0.1 - d}

    def normalize(self, text: str) -> str:
        return norm_simple(text)


# ── 輸出（同時印到終端機與存進紀錄）──────────────────────────────────────
class Out:
    def __init__(self):
        self.lines: list[str] = []

    def __call__(self, s: str = "") -> None:
        print(s)
        self.lines.append(re.sub(r"\033\[[0-9;]*m", "", s))

    def banner(self, s: str, color: str = C.BLUE) -> None:
        C.banner(s, color)
        self.lines.extend(["", "─" * 72, s, "─" * 72])

    def text(self) -> str:
        return "\n".join(self.lines) + "\n"


# ── 素材 ───────────────────────────────────────────────────────────────────
def load_sentences() -> list[dict]:
    if not SENTENCES.exists():
        sys.exit(f"沒有 {SENTENCES.relative_to(HERE)}，先跑 ./present.sh fetch（會從 dev-clean.tar.gz 讀參考文字、解出 wav）")
    return json.loads(SENTENCES.read_text(encoding="utf-8"))["sentences"]


def wav_path(utt: str) -> Path:
    return _p(CFG["audio"]["wav_dir"]) / f"{utt}.wav"


def load_audio(sents: list[dict]) -> list[np.ndarray]:
    xs = []
    for s in sents:
        p = wav_path(s["id"])
        if not p.exists():
            sys.exit(f"沒有 {p}，先跑 ./present.sh fetch")
        xs.append(C.load_wav(p, FS))
    return xs


def fake_material(n: int = 6) -> tuple[list[dict], list[np.ndarray]]:
    """合成母音串當句子，參考文字是假的詞。只用來測流程。"""
    rng = np.random.default_rng(int(CFG["fake"]["seed"]))
    words = "ALPHA BRAVO CHARLIE DELTA ECHO FOXTROT GOLF HOTEL INDIA JULIET KILO LIMA MIKE NOVEMBER OSCAR PAPA".split()
    sents, xs = [], []
    for k in range(n):
        parts = [dsp.synth_vowel(FS, 0.35, np.linspace(110 + 20 * k, 150 + 20 * k, 40), dsp.VOWELS[v]) for v in "aiuai"]
        x = np.concatenate(parts + [0.02 * rng.standard_normal(FS // 5)])
        xs.append(x)
        sents.append({"id": f"fake-{k:02d}", "spk": f"{k}", "dur_s": round(len(x) / FS, 2), "text": " ".join(rng.choice(words, 8))})
    return sents, xs


# ── 主體 ───────────────────────────────────────────────────────────────────
class Demo:
    def __init__(self, model, sents: list[dict], xs: list[np.ndarray], out: Out, fake: bool = False, fig: bool = True):
        self.model, self.sents, self.xs, self.out, self.fake = model, sents, xs, out, fake
        self.refs = [self.model.normalize(s["text"]) for s in sents]
        self.n_audio = [n_audio_frames(len(x)) for x in xs]
        self.ref_feats: list[np.ndarray] | None = None
        self.results: dict[str, dict] = {}          # id → {"hyps", "wer", ...}
        self.order: list[str] = []                 # 跑過的順序（畫表、畫圖用）
        self.focus = 0                             # 特徵圖上看的那一句
        self.fig = None
        if fig:
            self._open_fig()

    # ── 跑一個 recipe ──
    def run(self, r: dict) -> dict:
        o, m = self.out, self.model
        tag = f"{C.AMBER}[FAKE] {C.RESET}" if self.fake else ""
        o.banner(f"{tag}{r['key']}  {r['name']}", C.AMBER if r["id"] == "ref" else C.BLUE)
        p = recipe_params(r["changes"])
        if r["id"] != "ref":
            o("  " + "  ".join(f"{k} = {v}" for k, v in p.items() if k in r["changes"]) if r["changes"] else "  （每個旋鈕都與基準相同）")
            if "mel_scale" in r["changes"]:
                c_ref, c_new = filter_centres(FE["reference"]["mel_scale"]), filter_centres(p["mel_scale"])
                i = int(np.argmax(np.abs(c_new - c_ref)))
                o(f"  filterbank 中心頻率差最大在第 {i + 1} 個濾波器：{c_new[i]:.1f} Hz 對 {c_ref[i]:.1f} Hz（差 {c_new[i] - c_ref[i]:+.1f} Hz）")
        if self.ref_feats is None and r["id"] != "ref":
            o(f"  {C.GREY}先跑基準（r）{C.RESET}")
            self.run(REF)
        rng = np.random.default_rng(int(CFG["fake"]["seed"]))
        hyps, raw, lps, feats_all, diffs, t_fe, t_asr = [], [], [], [], [], 0.0, 0.0
        t_start = time.perf_counter()
        for k, (s, x) in enumerate(zip(self.sents, self.xs)):
            t0 = time.perf_counter()
            f = m.features_hf(x) if r["id"] == "ref" else frontend(x, p, rng)
            t1 = time.perf_counter()
            if self.fake:
                if r["id"] == "ref":
                    m.set_sentence(k, f)
                else:
                    m.set_sentence(k, self.ref_feats[k])
            tr = m.transcribe(f)
            hyp = m.normalize(tr["text"])
            t2 = time.perf_counter()
            t_fe, t_asr = t_fe + (t1 - t0), t_asr + (t2 - t1)
            hyps.append(hyp)
            raw.append(tr["text"])
            lps.append(float(tr["avg_logprob"]))
            feats_all.append(f)
            if self.ref_feats is not None:
                diffs.append(feature_diff(f, self.ref_feats[k], self.n_audio[k]))
            a = align_words(self.refs[k].split(), hyp.split())
            mark = f"{C.GREEN}✓{C.RESET}" if a["errors"] == 0 else f"{C.RED}✗ {a['errors']}{C.RESET}"
            changed = "" if r["id"] == "ref" else ("" if hyp == self.results["ref"]["hyps"][k] else f"  {C.AMBER}≠ ref{C.RESET}")
            o(f"  [{k + 1:2d}/{len(self.sents)}] {mark:<14} {C.GREY}logP {tr['avg_logprob']:+.2f}{C.RESET}  {diff_pairs(self.refs[k], hyp) if a['errors'] else ''}{changed}")
        wer = corpus_wer(self.refs, hyps)
        res = {"id": r["id"], "key": r["key"], "name": r["name"], "changes": r["changes"], "params": p, "hyps": hyps, "raw": raw, "avg_logprob": lps,
               "mean_avg_logprob": float(np.mean(lps)), "wer": wer,
               "per_sentence_errors": [align_words(rf.split(), h.split())["errors"] for rf, h in zip(self.refs, hyps)],
               "seconds": {"frontend": t_fe, "asr": t_asr, "total": time.perf_counter() - t_start}}
        if r["id"] == "ref":
            self.ref_feats = feats_all
            res["feature_diff"] = {"mean_abs": 0.0, "max_abs": 0.0, "cells_changed": 0.0}
            res["changed_sentences"] = 0
        else:
            res["feature_diff"] = {k2: float(np.mean([d[k2] for d in diffs])) if k2 != "max_abs" else float(max(d[k2] for d in diffs))
                                   for k2 in ("mean_abs", "max_abs", "cells_changed")}
            res["changed_sentences"] = int(sum(h != h0 for h, h0 in zip(hyps, self.results["ref"]["hyps"])))
        self.results[r["id"]] = res
        if r["id"] not in self.order:
            self.order.append(r["id"])
        self._feats_cache = feats_all
        self.summary(res)
        if r["id"] != "ref":
            worst = int(np.argmax(np.array(res["per_sentence_errors"]) - np.array(self.results["ref"]["per_sentence_errors"])))
            self.focus = worst
            self.worst(res)
        self.draw(res["id"], feats_all)
        return res

    def summary(self, res: dict) -> None:
        o, w, fd = self.out, res["wer"], res["feature_diff"]
        ref = self.results.get("ref")
        d_wer = "" if res["id"] == "ref" or ref is None else f"   Δ vs. ref {100 * (w['wer'] - ref['wer']['wer']):+.{DIG}f} 點"
        d_lp = "" if res["id"] == "ref" or ref is None else f"（基準 {ref['mean_avg_logprob']:+.3f}）"
        o(f"  {C.BOLD}WER {100 * w['wer']:.{DIG}f} %{C.RESET}  (S {w['S']}  D {w['D']}  I {w['I']} / {w['n_ref']} words){d_wer}"
          f"   每句平均 log P {res['mean_avg_logprob']:+.3f}{d_lp}")
        if res["id"] != "ref":
            o(f"  特徵對基準：mean |Δ| {fd['mean_abs']:.3f}   max |Δ| {fd['max_abs']:.2f}   cells changed {100 * fd['cells_changed']:.1f} %"
              f"   轉錄與基準不同：{res['changed_sentences']} / {len(self.sents)} 句")
        o(f"  {C.GREY}前端 {res['seconds']['frontend']:.1f} s、辨識 {res['seconds']['asr']:.1f} s{C.RESET}")

    def worst(self, res: dict) -> None:
        o = self.out
        ref = self.results["ref"]
        added = np.array(res["per_sentence_errors"]) - np.array(ref["per_sentence_errors"])
        idx = [int(i) for i in np.argsort(-added)[: int(CFG["scoring"]["worst_n"])] if added[i] > 0]
        if not idx:
            o(f"  沒有任何一句比基準多錯")
            return
        o(f"  比基準多錯最多的 {len(idx)} 句（⟨⟩ 刪、紅字替換或插入）：")
        for i in idx:
            o(f"    {self.sents[i]['id']}  +{int(added[i])}")
            o(f"      ref  {self.refs[i]}")
            o(f"      hyp  {colored(self.refs[i], res['hyps'][i])}")

    def table(self) -> None:
        o = self.out
        o.banner("到目前為止（投影片 p66 的表）", C.GREEN)
        o(f"  {'key':<4}{'recipe':<50}{'mean |Δ|':>9}{'WER %':>8}{'Δ pts':>8}{'changed':>9}{'avg logP':>10}")
        for rid in self.order:
            r = self.results[rid]
            ref = self.results["ref"]["wer"]["wer"]
            d = "" if rid == "ref" else f"{100 * (r['wer']['wer'] - ref):+.{DIG}f}"
            o(f"  {r['key']:<4}{r['name'][:49]:<50}{r['feature_diff']['mean_abs']:>9.3f}{100 * r['wer']['wer']:>8.{DIG}f}{d:>8}"
              f"{(str(r['changed_sentences']) + '/' + str(len(self.sents))) if rid != 'ref' else '—':>9}{r['mean_avg_logprob']:>+10.3f}")
        o(f"  {C.GREY}{len(self.sents)} 句、{self.results['ref']['wer']['n_ref']} 個參考詞；WER 的分母都一樣。avg logP：每句輸出 token 的平均 log 機率，再對句子平均{C.RESET}")

    # ── 圖 ──
    def _open_fig(self) -> None:
        try:
            import matplotlib
            if not os.environ.get("DISPLAY") and sys.platform != "darwin":
                matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            self.plt = plt
            plt.ion()
            d = CFG["display"]
            self.fig = plt.figure(figsize=(float(d["fig_w"]), float(d["fig_h"])))
            try:
                self.fig.canvas.manager.set_window_title("W2 demo 3 · front-end mismatch")
            except Exception:
                pass
        except Exception as e:
            print(f"{C.GREY}沒有繪圖視窗（{e}），只用終端機{C.RESET}")
            self.fig = None

    def draw(self, rid: str, feats_all: list[np.ndarray] | None = None, save: Path | None = None) -> None:
        if self.fig is None:
            return
        plt, d = self.plt, CFG["display"]
        self.fig.clf()
        gs = self.fig.add_gridspec(2, 3, height_ratios=[1.0, 0.9], hspace=0.5, wspace=0.3, left=0.24, right=0.97, top=0.93, bottom=0.08)
        k = self.focus
        n = self.n_audio[k]
        ref = self.ref_feats[k] if self.ref_feats is not None else None
        cur = feats_all[k] if feats_all is not None else ref
        s = self.sents[k]
        res = self.results.get(rid)
        if ref is not None:
            vmin, vmax = np.percentile(ref[:, :n], 1), np.percentile(ref[:, :n], 99)
            ext = [0, n * HOP / FS, 0, N_MELS]
            ax = self.fig.add_subplot(gs[0, 0])
            ax.imshow(ref[:, :n], origin="lower", aspect="auto", cmap=d["cmap"], vmin=vmin, vmax=vmax, extent=ext)
            ax.set_title(f"Whisper's own features · {s['id']}", fontsize=10)
            ax.set_xlabel("time (s)"); ax.set_ylabel("mel bin")
            ax = self.fig.add_subplot(gs[0, 1])
            ax.imshow(cur[:, :n], origin="lower", aspect="auto", cmap=d["cmap"], vmin=vmin, vmax=vmax, extent=ext)
            ax.set_title((res["name"] if res else rid)[:60], fontsize=10)
            ax.set_xlabel("time (s)")
            ax = self.fig.add_subplot(gs[0, 2])
            dd = cur[:, :n] - ref[:, :n]
            lim = max(float(np.percentile(np.abs(dd), 99)), 1e-3)
            im = ax.imshow(dd, origin="lower", aspect="auto", cmap="RdBu_r", vmin=-lim, vmax=lim, extent=ext)
            ax.set_title(f"difference (mean |Δ| = {np.abs(dd).mean():.3f})", fontsize=10)
            ax.set_xlabel("time (s)")
            self.fig.colorbar(im, ax=ax, fraction=0.046)
        ids = [r for r in self.order]
        wers = [100 * self.results[r]["wer"]["wer"] for r in ids]
        lps = [self.results[r]["mean_avg_logprob"] for r in ids]
        labels = [f"{self.results[r]['key']}  {self.results[r]['name'][:44]}" for r in ids]
        colors = ["#555555" if r == "ref" else ("#4C72B0" if self.results[r]["changes"] else "#8CB369") for r in ids]
        ax = self.fig.add_subplot(gs[1, 0:2])
        ax.barh(range(len(ids)), wers, color=colors)
        ax.set_yticks(range(len(ids)), labels, fontsize=9)
        ax.invert_yaxis()
        if "ref" in self.results:
            ax.axvline(100 * self.results["ref"]["wer"]["wer"], color="#555555", ls="--", lw=1)
        for i, w in enumerate(wers):
            ax.text(w, i, f" {w:.{DIG}f}", va="center", fontsize=9)
        ax.set_xlabel(f"WER % on {len(self.sents)} dev-clean sentences ({self.results['ref']['wer']['n_ref'] if 'ref' in self.results else '?'} words)")
        ax.set_xlim(0, max(wers + [5.0]) * 1.18)
        ax2 = self.fig.add_subplot(gs[1, 2], sharey=ax)
        ax2.barh(range(len(ids)), [-v for v in lps], color=colors)
        ax2.tick_params(labelleft=False)
        if "ref" in self.results:
            ax2.axvline(-self.results["ref"]["mean_avg_logprob"], color="#555555", ls="--", lw=1)
        for i, v in enumerate(lps):
            ax2.text(-v, i, f" {v:+.2f}", va="center", fontsize=9)
        ax2.set_xlabel("− mean avg log P per token (larger = less sure)")
        ax2.set_xlim(0, max([-v for v in lps] + [0.5]) * 1.3)
        if self.fake:
            self.fig.text(0.5, 0.5, "FAKE", fontsize=60, color="red", alpha=0.25, ha="center", va="center")
        self.fig.canvas.draw_idle()
        try:
            plt.pause(0.05)
        except Exception:
            pass
        if save is not None:
            self.fig.savefig(save, dpi=110)

    def refocus(self) -> None:
        """換特徵圖上看的那一句（n 鍵）：往下一句。"""
        self.focus = (self.focus + 1) % len(self.sents)
        rid = self.order[-1] if self.order else "ref"
        rng = np.random.default_rng(int(CFG["fake"]["seed"]))
        p = recipe_params(self.results[rid]["changes"]) if rid in self.results else recipe_params({})
        feats = [None] * len(self.sents)
        feats[self.focus] = (self.model.features_hf(self.xs[self.focus]) if rid == "ref" else frontend(self.xs[self.focus], p, rng))
        self.out(f"  特徵圖：{self.sents[self.focus]['id']}  「{self.refs[self.focus]}」")
        self.draw(rid, feats)

    def play(self) -> None:
        p = wav_path(self.sents[self.focus]["id"])
        if p.exists():
            self.out(f"  播 {p.name}：{self.refs[self.focus]}")
            C.play(p)
        else:
            self.out("  沒有音檔可播（fake）")

    # ── 紀錄 ──
    def record(self) -> dict:
        return {"time": C.now(), "fake": self.fake, "model": {"repo": CFG["model"]["repo"], "commit": read_lock(), "path": str(getattr(self.model, "path", "")),
                                                              "device": str(getattr(self.model, "device", "")), "n_params": int(getattr(self.model, "n_params", 0))},
                "versions": versions(), "frontend": FE, "sentences": [{**s, "ref_normalized": r, "n_audio_frames": n} for s, r, n in zip(self.sents, self.refs, self.n_audio)],
                "n_words": self.results["ref"]["wer"]["n_ref"] if "ref" in self.results else None,
                "order": self.order, "results": {k: {kk: vv for kk, vv in v.items()} for k, v in self.results.items()}}

    def save(self, folder: Path, name: str) -> None:
        folder.mkdir(parents=True, exist_ok=True)
        (folder / f"{name}.json").write_text(json.dumps(self.record(), ensure_ascii=False, indent=1), encoding="utf-8")
        (folder / f"{name}.txt").write_text(self.out.text(), encoding="utf-8")
        if self.fig is not None and self.order:
            self.draw(self.order[-1], getattr(self, "_feats_cache", None), save=folder / "wer.png")


def versions() -> dict:
    v = C.versions()
    from importlib import metadata
    for pkg in ("torch", "transformers", "soundfile", "huggingface_hub"):
        try:
            v[pkg] = metadata.version(pkg)
        except metadata.PackageNotFoundError:
            v[pkg] = None
    return v


# ── 互動 ───────────────────────────────────────────────────────────────────
MENU = """
  r 基準（Whisper 自己的前端）  0 同配方、自己的程式碼
  1 HTK 尺度  2 峰值正規化  3 magnitude  4 floor 1e-5  5 dither  6 pre-emphasis  7 CMVN  8 五個一起
  a 全部依序  t 表  n 換特徵圖的句子  s 播那一句  h 選單  q 離開"""


def interactive(demo: Demo, replay_from: dict | None = None) -> None:
    o = demo.out
    o(MENU)
    while True:
        k = C.getkey()
        if k == "q":
            break
        if k == "h":
            o(MENU)
        elif k == "t":
            if demo.results:
                demo.table()
            else:
                o("  還沒有結果")
        elif k == "n":
            if demo.results:
                demo.refocus()
        elif k == "s":
            demo.play()
        elif k == "r":
            if replay_from:
                replay_one(demo, replay_from, "ref")
            else:
                demo.run(REF)
        elif k in RECIPES:
            if replay_from:
                replay_one(demo, replay_from, RECIPES[k]["id"])
            else:
                demo.run(RECIPES[k])
        elif k == "a":
            for r in [REF] + [RECIPES[kk] for kk in sorted(RECIPES)]:
                if replay_from:
                    replay_one(demo, replay_from, r["id"])
                else:
                    demo.run(r)
            demo.table()
        elif k.strip():
            o(f"  不認得的鍵 {k!r}（h 看選單）")


def replay_one(demo: Demo, rec: dict, rid: str) -> None:
    """replay：轉錄從彩排紀錄讀回；特徵（只是 numpy）有音檔就當場重算來畫圖。"""
    o = demo.out
    res = rec["results"].get(rid)
    if res is None:
        o(f"  彩排紀錄裡沒有 {rid}")
        return
    r = REF if rid == "ref" else RECIPE_BY_ID[rid]
    o.banner(f"{C.GREY}[非現場，{rec['time']} 的彩排] {C.RESET}{r['key']}  {r['name']}", C.GREY)
    lps = res.get("avg_logprob") or [float("nan")] * len(res["hyps"])
    for k, (rf, hyp, lp) in enumerate(zip(demo.refs, res["hyps"], lps)):
        a = align_words(rf.split(), hyp.split())
        mark = f"{C.GREEN}✓{C.RESET}" if a["errors"] == 0 else f"{C.RED}✗ {a['errors']}{C.RESET}"
        changed = "" if rid == "ref" else ("" if hyp == rec["results"]["ref"]["hyps"][k] else f"  {C.AMBER}≠ ref{C.RESET}")
        o(f"  [{k + 1:2d}/{len(demo.sents)}] {mark:<14} {C.GREY}logP {lp:+.2f}{C.RESET}  {diff_pairs(rf, hyp) if a['errors'] else ''}{changed}")
    demo.results[rid] = res
    if rid not in demo.order:
        demo.order.append(rid)
    demo.summary(res)
    feats = None
    if demo.xs:
        p = recipe_params(r["changes"])
        rng = np.random.default_rng(int(CFG["fake"]["seed"]))
        if demo.ref_feats is None:
            demo.ref_feats = [frontend(x, recipe_params({}), rng) for x in demo.xs]
        feats = demo.ref_feats if rid == "ref" else [frontend(x, p, rng) for x in demo.xs]
        if rid != "ref":
            demo.focus = int(np.argmax(np.array(res["per_sentence_errors"]) - np.array(rec["results"]["ref"]["per_sentence_errors"])))
            demo.worst(res)
    demo.draw(rid, feats)


# ── 子命令 ─────────────────────────────────────────────────────────────────
def build(fake: bool, n: int | None, fig: bool, out: Out) -> Demo:
    if fake:
        sents, xs = fake_material()
        model = FakeWhisper([s["text"] for s in sents])
    else:
        sents = load_sentences()
        if n:
            sents = sents[:n]
        xs = load_audio(sents)
        t0 = time.perf_counter()
        model = Whisper()
        out(f"  {CFG['model']['repo']} @ {(read_lock() or '?')[:12]}，{model.n_params / 1e6:.0f} M 參數，{model.device}，載入 {time.perf_counter() - t0:.1f} s；"
            f"normalizer：{'Whisper EnglishTextNormalizer' if model._norm_ok else '退回簡單版'}")
    out(f"  {len(sents)} 句、共 {sum(len(x) for x in xs) / FS:.0f} s 音訊")
    return Demo(model, sents, xs, out, fake=fake, fig=fig)


def cmd_show(args) -> None:
    out = Out()
    out.banner("W2 demo 3：前端定義不匹配" + ("（FAKE）" if args.fake else ""), C.AMBER if args.fake else C.BLUE)
    demo = build(args.fake, args.n, not args.no_fig, out)
    demo.run(REF)
    interactive(demo)
    folder = HERE / "runs" / ("_fake" if args.fake else "live") / C.now().replace(":", "")
    demo.save(folder, "live")
    out(f"  存到 {folder.relative_to(HERE)}/")


def cmd_rehearse(args) -> None:
    out = Out()
    out.banner("彩排：基準加全部 recipe" + ("（FAKE）" if args.fake else ""), C.AMBER)
    demo = build(args.fake, args.n, not args.no_fig, out)
    folder = HERE / "runs" / ("_fake" if args.fake else "rehearsal")
    folder.mkdir(parents=True, exist_ok=True)
    for r in [REF] + [RECIPES[k] for k in sorted(RECIPES)]:
        demo.run(r)
        if demo.fig is not None and r["id"] != "ref":
            demo.draw(r["id"], demo._feats_cache, save=folder / f"features_{r['id']}.png")
    demo.table()
    demo.save(folder, "rehearsal")
    out(f"  存到 {folder.relative_to(HERE)}/（rehearsal.json、rehearsal.txt、wer.png、features_<id>.png）")
    if not args.fake:
        out("\n下一步：./present.sh export → 把數字回填到投影片 p66；上課用 ./present.sh")


def cmd_replay(args) -> None:
    out = Out()
    path = HERE / "runs" / "rehearsal" / "rehearsal.json"
    if not path.exists():
        sys.exit("沒有彩排紀錄 runs/rehearsal/rehearsal.json，先跑 ./present.sh rehearse")
    rec = json.loads(path.read_text(encoding="utf-8"))
    out.banner(f"重播 {rec['time']} 的彩排（不碰模型）", C.GREY)
    sents = [{k: v for k, v in s.items() if k not in ("ref_normalized", "n_audio_frames")} for s in rec["sentences"]]
    xs = []
    if all(wav_path(s["id"]).exists() for s in sents):
        xs = [C.load_wav(wav_path(s["id"]), FS) for s in sents]
    else:
        out(f"  {C.GREY}沒有音檔，特徵圖不畫、s 不能播{C.RESET}")

    class _Norm:                       # replay 不載 tokenizer；參考文字用彩排時正規化過的那份
        path, n_params, device = "replay", 0, "none"

        def normalize(self, t):
            return t

        def features_hf(self, x):
            return frontend(x, recipe_params({}))

    demo = Demo(_Norm(), sents, xs if xs else [np.zeros(FS)] * len(sents), out, fake=False, fig=not args.no_fig and bool(xs))
    demo.refs = [s["ref_normalized"] for s in rec["sentences"]]
    demo.n_audio = [s["n_audio_frames"] for s in rec["sentences"]]
    replay_one(demo, rec, "ref")
    interactive(demo, replay_from=rec)


def cmd_check(args) -> None:
    out = Out()
    out.banner("課前體檢", C.AMBER)
    rep = {"time": C.now(), "versions": versions()}
    out("  " + "  ".join(f"{k} {v}" for k, v in rep["versions"].items()))
    import torch
    rep["mps_available"] = bool(torch.backends.mps.is_available())
    out(f"  MPS 可用：{rep['mps_available']}（Whisper 照設定放 {CFG['model']['device']}）")
    sents = load_sentences()[: max(1, args.n or 3)]
    xs = load_audio(sents)
    t0 = time.perf_counter()
    m = Whisper()
    rep["model"] = {"repo": CFG["model"]["repo"], "commit": read_lock(), "path": m.path, "n_params": m.n_params, "load_s": time.perf_counter() - t0}
    out(f"  模型：{m.path}（{m.n_params / 1e6:.0f} M，載入 {rep['model']['load_s']:.1f} s）")
    # 1. 自己的前端對 WhisperFeatureExtractor
    diffs = []
    for s, x in zip(sents, xs):
        a, b = m.features_hf(x), frontend(x, recipe_params({}))
        n = n_audio_frames(len(x))
        diffs.append({"id": s["id"], "max_abs_audio": float(np.abs(a[:, :n] - b[:, :n]).max()), "max_abs_all": float(np.abs(a - b).max()),
                      "shape_hf": list(a.shape), "shape_own": list(b.shape)})
        out(f"  {s['id']}  特徵 shape {a.shape} 對 {b.shape}；|差| 最大（有聲音的 frame）{diffs[-1]['max_abs_audio']:.2e}，全部 {diffs[-1]['max_abs_all']:.2e}")
    rep["features_vs_hf"] = diffs
    # 2. 同配方的轉錄要相同；量一句多久
    same = []
    for s, x in zip(sents, xs):
        t0 = time.perf_counter(); h0 = m.transcribe(m.features_hf(x)); t1 = time.perf_counter()
        h1 = m.transcribe(frontend(x, recipe_params({}))); t2 = time.perf_counter()
        same.append({"id": s["id"], "hf": h0, "own": h1, "identical": h0["text"] == h1["text"], "seconds_hf": t1 - t0, "seconds_own": t2 - t1,
                     "ref": m.normalize(s["text"]), "errors_hf": align_words(m.normalize(s["text"]).split(), m.normalize(h0["text"]).split())["errors"]})
        out(f"  {s['id']}  {'同' if same[-1]['identical'] else C.RED + '不同' + C.RESET}  {t1 - t0:.2f} s／{t2 - t1:.2f} s（generate + teacher-forced）"
            f"   錯 {same[-1]['errors_hf']}   logP {h0['avg_logprob']:+.3f}／{h1['avg_logprob']:+.3f}   「{h0['text']}」")
    rep["same_recipe_transcripts"] = same
    # 3. mel 第 64 個濾波器對 w02-data.json
    rep["mel_scales"] = mel_scale_check(out)
    # 4. 每個 recipe 的特徵差（只算一句，不轉錄）
    x = xs[0]
    rng = np.random.default_rng(int(CFG["fake"]["seed"]))
    f_ref = m.features_hf(x)
    rep["feature_diff_first_sentence"] = {}
    for k in sorted(RECIPES):
        r = RECIPES[k]
        d = feature_diff(frontend(x, recipe_params(r["changes"]), rng), f_ref, n_audio_frames(len(x)))
        rep["feature_diff_first_sentence"][r["id"]] = d
        out(f"  {k} {r['id']:<11} mean |Δ| {d['mean_abs']:.3f}  max {d['max_abs']:.2f}  cells {100 * d['cells_changed']:.1f} %")
    (HERE / "runs").mkdir(exist_ok=True)
    (HERE / "runs" / "check.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1), encoding="utf-8")
    out("  → runs/check.json")


def mel_scale_check(out: Out) -> dict:
    c_htk, c_sl = filter_centres("htk"), filter_centres("slaney")
    i = int(np.argmax(np.abs(c_htk - c_sl)))
    rep = {"max_diff_hz": float(abs(c_htk[i] - c_sl[i])), "at_filter": i + 1, "htk_hz": float(c_htk[i]), "slaney_hz": float(c_sl[i])}
    line = f"  兩種 mel 尺度的中心頻率差最大在第 {i + 1} 個濾波器：HTK {c_htk[i]:.1f} Hz、Slaney {c_sl[i]:.1f} Hz，差 {rep['max_diff_hz']:.1f} Hz"
    if DATA_JSON.exists():
        ms = json.loads(DATA_JSON.read_text(encoding="utf-8")).get("mel_scales", {})
        ok = (ms.get("at_filter_index") == i + 1 and abs(ms.get("max_centre_difference_hz", 0) - rep["max_diff_hz"]) < 0.1
              and abs(ms["centres_at_that_index_hz"]["htk"] - c_htk[i]) < 0.1 and abs(ms["centres_at_that_index_hz"]["slaney"] - c_sl[i]) < 0.1)
        rep["matches_w02_data_json"] = bool(ok)
        line += f"（對 w02-data.json → mel_scales：{'一致' if ok else C.RED + '不一致' + C.RESET}）"
    out(line)
    return rep


def cmd_fetch(args) -> None:
    out = Out()
    out.banner("下載模型、挑句、解音檔", C.AMBER)
    from huggingface_hub import HfApi, snapshot_download
    sha = read_lock() or HfApi().model_info(CFG["model"]["repo"]).sha
    path = snapshot_download(CFG["model"]["repo"], revision=sha, allow_patterns=PATTERNS)
    MODEL_LOCK.write_text(f"{CFG['model']['repo']} {sha}\n", encoding="utf-8")
    out(f"  {CFG['model']['repo']} @ {sha[:12]} → {path}（commit 寫進 model.lock）")
    tar = _p(CFG["audio"]["dev_clean_tar"])
    if not tar.exists():
        sys.exit(f"沒有 {tar}（W3 Demo 3 的 fetch 會下載 dev-clean.tar.gz；或自己從 openslr.org/12 下載放到那裡）")
    utts, texts = scan_tar(tar, out)
    for u in utts:
        u["text"] = texts.get(u["id"], "")
    sel = select_sentences([u for u in utts if u["text"]], CFG["audio"])
    out(f"  {len(utts)} 句、{len({u['id'].split('-')[0] for u in utts})} 位說話人 → 挑 {len(sel)} 句，"
        f"長度 {min(s['dur_s'] for s in sel):.1f}–{max(s['dur_s'] for s in sel):.1f} s")
    want = [s["id"] for s in sel if not wav_path(s["id"]).exists()]
    if want:
        extract_tar(tar, want, out)
    SENTENCES.parent.mkdir(exist_ok=True)
    SENTENCES.write_text(json.dumps({"source": {"audio": "https://www.openslr.org/resources/12/dev-clean.tar.gz", "license": "CC BY 4.0", "time": C.now(),
                                                "selection": {k: CFG["audio"][k] for k in ("min_s", "max_s", "target_s")}, "n_candidates": len(utts)},
                                     "sentences": sel}, ensure_ascii=False, indent=1), encoding="utf-8")
    out(f"  → {SENTENCES.relative_to(HERE)}；wav 在 {_p(CFG['audio']['wav_dir'])}")
    out("\n下一步：./present.sh check → ./present.sh rehearse")


def scan_tar(tar: Path, out: Out) -> tuple[list[dict], dict[str, str]]:
    """一次走過 dev-clean.tar.gz：每個 flac 只讀表頭拿長度，*.trans.txt 讀參考文字。"""
    import tarfile
    utts, texts = [], {}
    t0 = time.perf_counter()
    with tarfile.open(tar, "r|gz") as tf:
        for m in tf:
            if m.name.endswith(".trans.txt"):
                for line in tf.extractfile(m).read().decode("utf-8").splitlines():
                    parts = line.strip().split(" ", 1)
                    if len(parts) == 2:
                        texts[parts[0]] = parts[1]
            elif m.name.endswith(".flac"):
                sr, total = flac_streaminfo(tf.extractfile(m).read(42))
                utts.append({"id": Path(m.name).stem, "dur_s": total / sr})
    out(f"  讀完 {tar.name}：{len(utts)} 個 flac、{len(texts)} 行參考文字（{time.perf_counter() - t0:.0f} s）")
    return utts, texts


def extract_tar(tar: Path, want: list[str], out: Out) -> None:
    import tarfile
    import soundfile as sf
    names = {f"{u}.flac": u for u in want}
    done = 0
    with tarfile.open(tar, "r|gz") as tf:
        for m in tf:
            u = names.get(Path(m.name).name)
            if u is None:
                continue
            x, fs_in = sf.read(io.BytesIO(tf.extractfile(m).read()), dtype="float64")
            C.save_wav(wav_path(u), dsp.resample(dsp.to_mono_float(x), fs_in, FS), FS)
            done += 1
            if done == len(names):
                break
    out(f"  解出 {done} 個 wav → {_p(CFG['audio']['wav_dir'])}")


def cmd_export(args) -> None:
    path = HERE / "runs" / "rehearsal" / "rehearsal.json"
    if not path.exists():
        sys.exit("沒有 runs/rehearsal/rehearsal.json，先跑 ./present.sh rehearse")
    rec = json.loads(path.read_text(encoding="utf-8"))
    ref = rec["results"]["ref"]
    rows = []
    for rid in rec["order"]:
        r = rec["results"][rid]
        rows.append({"key": r["key"], "id": rid, "name": r["name"], "changes": r["changes"], "wer_pct": round(100 * r["wer"]["wer"], DIG),
                     "S": r["wer"]["S"], "D": r["wer"]["D"], "I": r["wer"]["I"], "delta_wer_pts": round(100 * (r["wer"]["wer"] - ref["wer"]["wer"]), DIG),
                     "mean_avg_logprob": round(r["mean_avg_logprob"], 3), "delta_avg_logprob": round(r["mean_avg_logprob"] - ref["mean_avg_logprob"], 3),
                     "changed_sentences": r["changed_sentences"], "feature_mean_abs_diff": round(r["feature_diff"]["mean_abs"], 3),
                     "feature_max_abs_diff": round(r["feature_diff"]["max_abs"], 2), "cells_changed_pct": round(100 * r["feature_diff"]["cells_changed"], 1),
                     "asr_seconds": round(r["seconds"]["asr"], 1)})
    entry = {"kind": "measured", "source": "demos/w02_d3_frontend_mismatch/runs/rehearsal/rehearsal.json", "recorded": rec["time"],
             "model": {k: v for k, v in rec["model"].items() if k != "path"},  # 絕對路徑不進投影片的 assets（會原樣上站）
             "versions": {k: rec["versions"].get(k) for k in ("python", "platform", "torch", "transformers", "numpy")},
             "n_sentences": len(rec["sentences"]), "n_words": rec["n_words"], "data": "LibriSpeech dev-clean, one sentence per speaker (CC BY 4.0)",
             "normalizer": "Whisper EnglishTextNormalizer on both reference and hypothesis", "recipes": rows,
             "caveat": f"{len(rec['sentences'])} sentences from one corpus, one model size; one error is {100 / rec['n_words']:.2f} WER points"}
    data = json.loads(METRICS.read_text(encoding="utf-8")) if METRICS.exists() else {"_about": "W2 投影片上來自 demo 彩排的數字；由各 demo 的 ./present.sh export 寫入"}
    data["demo3"] = entry
    METRICS.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"→ {METRICS}（demo3：{len(rows)} 列）")
    for r in rows:
        print(f"  {r['key']}  {r['id']:<11} WER {r['wer_pct']:6.{DIG}f} %  Δ {r['delta_wer_pts']:+6.{DIG}f}  mean |Δ| {r['feature_mean_abs_diff']:.3f}"
              f"  changed {r['changed_sentences']:>2}  avg logP {r['mean_avg_logprob']:+.3f}")


def cmd_selftest(args) -> None:
    out = Out()
    out.banner("selftest（不需要模型）", C.GREEN)
    fails = 0

    def check(name: str, ok: bool, detail: str = "") -> None:
        nonlocal fails
        fails += 0 if ok else 1
        out(f"  {'✓' if ok else C.RED + '✗' + C.RESET}  {name}" + (f"  {C.GREY}{detail}{C.RESET}" if detail else ""))

    # 1. WER
    a = corpus_wer(["THE CAT SAT", "A B C D"], ["THE CAT SAT", "A X C"])
    check("corpus WER：S 1、D 1、I 0 / 7 詞", a["S"] == 1 and a["D"] == 1 and a["I"] == 0 and a["n_ref"] == 7 and abs(a["wer"] - 2 / 7) < 1e-12, str({k: a[k] for k in ("S", "D", "I", "n_ref")}))
    ops = align_words("A B C D E".split(), "A C D E F".split())
    check("回溯：刪 B、插 F（距離 2）", ops["errors"] == 2 and ("D", "B", None) in ops["ops"] and ("I", None, "F") in ops["ops"], diff_pairs("A B C D E", "A C D E F"))
    check("norm_simple", norm_simple("Mr. Quilter's  gospel!") == "MR QUILTER'S GOSPEL")
    # 2. FLAC 表頭
    head = b"fLaC" + bytes([0x80, 0, 0, 34]) + struct.pack(">HH", 4096, 4096) + b"\x00\x00\x00" * 2
    sr, total = 16000, 93680
    head += bytes([(sr >> 12) & 0xFF, (sr >> 4) & 0xFF, ((sr & 0xF) << 4) | (0 << 1) | 0, (15 << 4) | ((total >> 32) & 0xF)]) + struct.pack(">I", total & 0xFFFFFFFF) + b"\x00" * 16
    check("FLAC STREAMINFO：16000 Hz、93680 samples", flac_streaminfo(head) == (16000, 93680), str(flac_streaminfo(head)))
    # 3. 挑句
    utts = [{"id": "1-1-0", "dur_s": 2.0, "text": "a"}, {"id": "1-1-1", "dur_s": 6.5, "text": "b"}, {"id": "1-1-2", "dur_s": 7.4, "text": "c"},
            {"id": "22-3-0", "dur_s": 1.0, "text": "d"}, {"id": "22-3-1", "dur_s": 1.5, "text": "e"}]
    sel = select_sentences(utts, {"min_s": 4.0, "max_s": 10.0, "target_s": 7.0})
    check("挑句：每位說話人一句、最接近 7 s（7.4 贏 6.5）、範圍外取最接近", [s["id"] for s in sel] == ["1-1-2", "22-3-1"], str([s["id"] for s in sel]))
    # 4. 兩種 mel 尺度對 w02-data.json
    ms = mel_scale_check(out)
    check("filterbank 中心頻率對 w02-data.json", ms.get("matches_w02_data_json", True))
    # 5. 自己的前端對 transformers 的 numpy 實作（有 transformers 才測；不需要 torch）
    wav = HERE.parent / "w03_d3_blank_heatmap" / "runs" / "rehearsal" / "utterance.wav"
    x = C.load_wav(wav, FS) if wav.exists() else dsp.synth_vowel(FS, 2.0, 120.0, dsp.VOWELS["a"])
    own = frontend(x, recipe_params({}))
    check("特徵 shape (80, 3000)", own.shape == (N_MELS, N_FRAMES), str(own.shape))
    try:
        from transformers import WhisperFeatureExtractor
        fe = WhisperFeatureExtractor()
        xp = np.pad(np.asarray(x, dtype=np.float32)[:N_SAMPLES], (0, max(0, N_SAMPLES - len(x))))
        hf = fe._np_extract_fbank_features(xp[None], "cpu")[0]
        d = np.abs(hf - own)
        n = n_audio_frames(len(x))
        check("同配方對 WhisperFeatureExtractor（numpy 路徑）|差| < 1e-4", float(d.max()) < 1e-4, f"max {d.max():.2e}（有聲音的 frame {d[:, :n].max():.2e}）；{'W3 那句' if wav.exists() else '合成母音'}")
        M_hf = fe.mel_filters.T
        check("filterbank 對 transformers 的 mel_filter_bank（Slaney、面積正規化）", float(np.abs(M_hf - filterbank(recipe_params({}))).max()) < 1e-9)
    except ImportError:
        out(f"  {C.GREY}–  沒有 transformers，跳過與 WhisperFeatureExtractor 的比對{C.RESET}")
    # 6. 每個旋鈕都真的改到東西；只動一個旋鈕
    for k in sorted(RECIPES):
        r = RECIPES[k]
        f = frontend(x, recipe_params(r["changes"]), np.random.default_rng(1))
        d = feature_diff(f, own, n_audio_frames(len(x)))
        if r["id"] == "same":
            check(f"{k} {r['id']}：與基準逐格相同", d["max_abs"] == 0.0)
        else:
            check(f"{k} {r['id']}：有改到（mean |Δ| {d['mean_abs']:.3f}，cells {100 * d['cells_changed']:.0f} %）", d["cells_changed"] > 0,
                  "、".join(f"{kk}={vv}" for kk, vv in r["changes"].items()))
    check("recipe 鍵不重複、id 不重複", len(RECIPES) == len(CFG["recipe"]) == len(RECIPE_BY_ID))
    check("dither 可重現（同 seed）", np.array_equal(frontend(x, recipe_params({"dither": 1e-4}), np.random.default_rng(3)), frontend(x, recipe_params({"dither": 1e-4}), np.random.default_rng(3))))
    # 7. fake 整條流程
    sents, xs = fake_material(4)
    demo = Demo(FakeWhisper([s["text"] for s in sents]), sents, xs, Out(), fake=True, fig=False)
    demo.out.lines = []
    _quiet = demo.out
    import contextlib
    with contextlib.redirect_stdout(io.StringIO()):
        demo.run(REF)
        demo.run(RECIPES["1"])
        demo.run(RECIPES["0"])
        demo.table()
    check("fake 流程：基準、同配方、HTK 都跑完", set(demo.results) == {"ref", "htk", "same"} and demo.results["same"]["wer"] == demo.results["ref"]["wer"])
    folder = HERE / "runs" / "_selftest"
    demo.save(folder, "selftest")
    check("存檔", (folder / "selftest.json").exists() and (folder / "selftest.txt").exists())
    out(f"\n  {'全部通過' if fails == 0 else C.RED + f'{fails} 項沒過' + C.RESET}")
    sys.exit(1 if fails else 0)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["show", "rehearse", "replay", "check", "fetch", "fake", "selftest", "export"])
    ap.add_argument("--n", type=int, default=None, help="只用前 n 句（試跑用；彩排與上課不要加）")
    ap.add_argument("--no-fig", action="store_true", help="不開繪圖視窗")
    ap.add_argument("--fake", action="store_true", help="沒有模型時的替身（fake 子命令會自動加）")
    args = ap.parse_args()
    if args.cmd == "fake":
        args.fake = True
        cmd_show(args)
    else:
        {"show": cmd_show, "rehearse": cmd_rehearse, "replay": cmd_replay, "check": cmd_check, "fetch": cmd_fetch,
         "selftest": cmd_selftest, "export": cmd_export}[args.cmd](args)


if __name__ == "__main__":
    main()

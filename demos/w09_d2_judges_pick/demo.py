# -*- coding: utf-8 -*-
# 此檔由 bin/publish_demos.py 從 private repo 的 <course>/demos/ 複製而來，請勿在公開 repo 直接編輯。
"""W9 demo 2：同一句抽十二次，三位裁判各自挑誰——WER（Whisper）、SIM（speaker encoder）、MOS predictor（UTMOS）。

  show       課堂：載 Qwen3-TTS 與三位裁判，讀回彩排的那一組（沒有彩排就現場算），開一個視窗；鍵在視窗裡按。
             鍵 n 現場再抽一組新的（seed 接著往下數）。離開時存 runs/live/<時間>/
  rehearse   課前：不互動，兩句各抽一組（加上 F5-TTS 的對照組），.wav、圖、數字存 runs/rehearsal/（進版控 = 現場的退路），
             寫 slides/assets/w09/w09-metrics.json → demo2
  listen     彩排之後：把一組的十二段打亂順序、不顯示編號地播給你聽，你挑一段 → runs/rehearsal/picks.json（鍵 h 用）
  replay     現場出狀況時：從 runs/rehearsal/ 讀回，同一個視窗、同樣的鍵（除了 n），不碰 torch 也不開 worker
  check      課前體檢：版本；MOS predictor 載不載得起、cpu／mps 的分數差多少、分數隨音量變多少；
             三位裁判對真人錄音與上週幾段輸出各給幾分；mlx-audio 有沒有 qwen3_tts → runs/check.json
  fetch      課前一週：下載 MOS predictor 的程式（git clone 一個 tag、對 commit）與權重（對 sha256）；
             確認 W8、W9 Demo 1 的模型都在（要網路）
  fake       沒有模型也能演練流程（生成器與裁判都是替身，畫面標明 FAKE）；--real-judges 換成真的三位裁判
  selftest   不開視窗、不需要模型：對齊與 S／D／I、音節數、語速、F0 起伏、advantage、名次、fake 流程；有彩排紀錄時對 w09-metrics.json
  export     只重寫 w09-metrics.json → demo2（從 runs/rehearsal/rehearsal.json 與 picks.json）

畫面上的鍵（順序照投影片 p56、p57 講稿的【操作】）：
  g  這一組的表：十二段 × （WER、SIM、predicted MOS ｜ 語速、F0 起伏）
  l  長尾：有錯字或長度離群的段標出來，播最糟的那一段
  a  每位裁判多一欄 advantage：(r − 組平均) ÷ 組標準差；標準差是 0 的那一欄沒有訊號
  w s m  播 WER／SIM／MOS 這位裁判的第一名；  h 播授課者自己挑的（listen 存的）
  p  散點圖：橫軸是一位裁判給的名次，縱軸是語速與 F0 起伏；再按換下一位裁判，第四次回到表
  z  換另一句（英文 ↔ 華語）     f  英文那一句的 F5-TTS 對照組 ↔ Qwen3-TTS
  n  現場再抽一組（只有 show）   1–9 0 - =  播第 1–12 段     o  播 prompt
  t  投影片上那兩張表要的摘要     r  再播一次     ?  選單     q  離開

三位裁判（投影片 p55–p57）：
  WER   Whisper small 聽寫後對要念的句子。英文：詞，兩邊過 Whisper 的 English normalizer；華語：帶調拼音的音節。reward = −WER
  SIM   輸出與 prompt 經同一個 speaker encoder（WavLM-SV）的向量的 cosine
  MOS   UTMOS22 strong learner 預測的 1–5 分（BVCC 上訓的：英語、2022 年以前的系統）
兩個不是裁判的描述量（沒有人拿它們當 reward，只是描述這段聲音；都是這支程式自己定的算法）：
  語速       要念的句子的音節數 ÷ 說話的時間（第一格到最後一格有聲音的 frame，句中停頓算在內）
  F0 起伏    有聲 frame 的 log F0 的標準差，以半音為單位
這不是 RL，也不訓練任何東西：每一組只是從同一個模型抽十二次，再讓裁判排名（best-of-12）。
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import random
import re
import shutil
import subprocess
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


def _load_prev():
    p = (HERE / CFG["prev"]["dir"] / "demo.py").resolve()
    if not p.exists():
        sys.exit(f"找不到 W9 Demo 1 的程式 {p}（demo_config.toml [prev].dir）")
    spec = importlib.util.spec_from_file_location("w09_d1_demo", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


P = _load_prev()          # W9 Demo 1：Worker、WavLMSV、Whisper、error_rate、Out…
D1 = P.D1                 # W8 Demo 1：F5、trim_edges、resolve_device…
FS = 16000
N = int(CFG["group"]["n"])
SIDS = list(CFG["sentences"]["order"])
METRICS_JSON = HERE.parents[1] / "slides" / "assets" / "w09" / "w09-metrics.json"     # parents[1] = speech_ai/
REHEARSAL = HERE / "runs" / "rehearsal"
PICKS = REHEARSAL / "picks.json"
MOS_DIR = (HERE / CFG["mos"]["dir"]).resolve()
MOS_CODE = MOS_DIR / "SpeechMOS"
MOS_WEIGHTS = MOS_DIR / CFG["mos"]["weights_url"].rsplit("/", 1)[-1]
JUDGES = ["wer", "sim", "mos"]
J_LABEL = {"wer": "WER (Whisper)", "sim": "SIM", "mos": "predicted MOS"}
J_KEY = {"w": "wer", "s": "sim", "m": "mos"}
GEN_LABEL = {"qwen3": "Qwen3-TTS", "f5": "F5-TTS (control)", "live": "Qwen3-TTS, drawn just now"}
GEN_TAG = {"qwen3": "q", "f5": "f", "live": "n"}
DRAW_KEYS = list("1234567890-=")
COLOR = {"wer": "#2a7fc8", "sim": "#5b3fa0", "mos": "#c8742a", "ear": "#2a9d4b", "bad": "#d62728"}


def fmt2(v) -> str:
    return "—" if v is None else f"{v:.2f}"


def draw_name(sid: str, gen: str, i: int) -> str:
    return f"{sid}_{GEN_TAG[gen]}{i:02d}"


# ── 文字：對齊、音節 ─────────────────────────────────────────
def align_counts(ref: list, hyp: list) -> dict:
    """Levenshtein 對齊，回 S（替換）、D（漏）、I（多）——W7 的記號。平手時優先算成替換。"""
    n, m = len(ref), len(hyp)
    d = np.zeros((n + 1, m + 1), dtype=int)
    d[:, 0] = np.arange(n + 1)
    d[0, :] = np.arange(m + 1)
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            d[i, j] = min(d[i - 1, j - 1] + (ref[i - 1] != hyp[j - 1]), d[i - 1, j] + 1, d[i, j - 1] + 1)
    i, j, S, D, I = n, m, 0, 0, 0
    while i > 0 or j > 0:
        if i > 0 and j > 0 and d[i, j] == d[i - 1, j - 1] + (ref[i - 1] != hyp[j - 1]):
            S += int(ref[i - 1] != hyp[j - 1]); i -= 1; j -= 1
        elif i > 0 and d[i, j] == d[i - 1, j] + 1:
            D += 1; i -= 1
        else:
            I += 1; j -= 1
    return {"S": S, "D": D, "I": I, "N": n, "errors": S + D + I}


def units(text: str, lang: str, norm=None) -> list[str]:
    if lang == "zh":
        return P.zh_units(text)
    return (norm or P.basic_norm_en)(text).split()


def error_detail(ref: str, hyp: str, lang: str, norm=None) -> dict:
    a = align_counts(units(ref, lang, norm), units(hyp, lang, norm))
    a["pct"] = round(100 * a["errors"] / max(1, a["N"]), 1)
    a["unit"] = "tone-numbered pinyin syllables" if lang == "zh" else "words"
    return a


def syllables_en(text: str) -> int:
    """英文音節數的啟發式（母音群；字尾不發音的 e、-ed 不算，pickled 這種 -le 加 d 的照算）。只給 selftest 對設定檔手數的數、與 fake 模式用。"""
    total = 0
    for w in re.findall(r"[a-z]+", text.lower().replace("'s", "")):
        g = len(re.findall(r"[aeiouy]+", w))
        if w.endswith("e") and not w.endswith(("le", "ee")) and g > 1:
            g -= 1
        if w.endswith("ed") and not w.endswith(("ted", "ded")) and not re.search(r"[^aeiouy]led$", w) and g > 1:
            g -= 1
        total += max(1, g)
    return total


def n_syllables(sid: str, text: str, lang: str) -> int:
    if lang == "zh":
        return len(re.findall(r"[一-鿿]", text))
    return int(CFG["sentences"].get(sid, {}).get("syllables") or 0) if text == CFG["sentences"].get(sid, {}).get("text") else syllables_en(text)


# ── 兩個描述量 ───────────────────────────────────────────────
def speech_span(x: np.ndarray, fs: int) -> float:
    """說話的時間（秒）：第一格到最後一格「有聲音」的 10 ms frame（位準在第 95 百分位以下 active_db 之內）。"""
    e = P.frame_levels(x, fs)
    on = np.flatnonzero(e > float(np.percentile(e, 95)) - CFG["measure"]["active_db"])
    return 0.0 if len(on) == 0 else (on[-1] - on[0] + 1) * 0.010


def f0_desc(x: np.ndarray, fs: int) -> dict:
    """F0 的中位數與起伏。起伏 = 有聲 frame 的 12·log2 F0 的標準差（半音）；離中位數超過 ±1.8 倍的 frame 當成 tracker 的倍頻錯、不算。"""
    c = CFG["f0"]
    y = dsp.resample(x, fs, FS)
    flen = int(round(c["frame_ms"] * FS / 1000)); hop = int(round(c["hop_ms"] * FS / 1000))
    f0 = np.array([dsp.f0_autocorr(y[a:a + flen], FS, c["floor_hz"], c["ceiling_hz"], c["voicing"])[0] for a in range(0, max(1, len(y) - flen), hop)], dtype=float)
    v = f0[~np.isnan(f0)]
    if len(v) < 10:
        return {"f0_median_hz": None, "f0_sd_st": None, "voiced_frames": int(len(v))}
    med = float(np.median(v))
    v = v[(v > med / 1.8) & (v < med * 1.8)]
    return {"f0_median_hz": round(med, 1), "f0_sd_st": round(float(np.std(12 * np.log2(v))), 2), "voiced_frames": int(len(v))}


# ── 一組的統計 ───────────────────────────────────────────────
def reward(row: dict, j: str) -> float | None:
    if j == "wer":
        return None if row.get("wer") is None else -float(row["wer"]["pct"])
    return row.get(j)


def avg_rank(values: list[float]) -> list[float]:
    """名次（1 = reward 最高）；平手取平均名次。"""
    order = sorted(range(len(values)), key=lambda i: -values[i])
    rank = [0.0] * len(values)
    i = 0
    while i < len(order):
        k = i
        while k + 1 < len(order) and values[order[k + 1]] == values[order[i]]:
            k += 1
        for t in range(i, k + 1):
            rank[order[t]] = (i + k) / 2 + 1
        i = k + 1
    return rank


def spearman(a: list[float], b: list[float]) -> float | None:
    if len(a) < 3 or len(set(a)) < 2 or len(set(b)) < 2:
        return None
    ra, rb = np.array(avg_rank(a)), np.array(avg_rank(b))
    return round(float(np.corrcoef(ra, rb)[0, 1]), 2)


def group_stats(rows: list[dict]) -> dict:
    """rows：一組裡的每一段（照 seed 順序）。回每位裁判的 reward、advantage、名次、第一名（可能平手），以及名次與兩個描述量的 Spearman 相關。
    advantage 照 GRPO：(r − mean) ÷ std，組內、母體標準差；std = 0（全部平手）時沒有定義，回 None。"""
    out: dict = {"n": len(rows), "judges": {}}
    desc = {"rate": [r.get("rate") for r in rows], "f0_sd_st": [r.get("f0_sd_st") for r in rows]}
    for j in JUDGES:
        r = [reward(x, j) for x in rows]
        if any(v is None for v in r) or not r:
            out["judges"][j] = None
            continue
        mu, sd = float(np.mean(r)), float(np.std(r))
        best = max(r)
        first = [i for i, v in enumerate(r) if v == best]
        rk = avg_rank(r)
        jj = {"reward": r, "mean": round(mu, 4), "std": round(sd, 4), "advantage": None if sd < 1e-9 else [round((v - mu) / sd, 2) for v in r],
              "rank": rk, "first": first, "n_tied_first": len(first), "spearman": {}}
        for k, d in desc.items():
            ok = [i for i, v in enumerate(d) if v is not None]
            # 名次 1 = 最好；相關為正 = 這位裁判越喜歡的段，這個量越小
            jj["spearman"][k] = spearman([rk[i] for i in ok], [d[i] for i in ok]) if len(ok) == len(rows) else None
            rest = [d[i] for i in ok if i not in first]
            mine = [d[i] for i in ok if i in first]
            jj[f"first_{k}"] = round(float(np.mean(mine)), 2) if mine else None
            jj[f"rest_{k}"] = round(float(np.mean(rest)), 2) if rest else None
        out["judges"][j] = jj
    for k, d in desc.items():
        v = [x for x in d if x is not None]
        out[k] = {"mean": round(float(np.mean(v)), 2), "std": round(float(np.std(v)), 2), "min": round(float(min(v)), 2), "max": round(float(max(v)), 2)} if v else None
    sec = [r["speech_s"] for r in rows if r.get("speech_s")]
    med = float(np.median(sec)) if sec else 0.0
    tail = []
    errs = [(r.get("wer") or {}).get("errors", 0) for r in rows]
    base = min(errs) if errs else 0                 # 每一段都有的錯（多半是 ASR 自己的）不算長尾；比這一組最好的那一段多錯的才算
    out["min_errors"] = base
    for i, r in enumerate(rows):
        why = []
        w = r.get("wer") or {}
        if w.get("errors", 0) <= base:
            w = {}
        if w.get("D"):
            why.append(f"{w['D']} dropped")
        if w.get("I"):
            why.append(f"{w['I']} extra")
        if w.get("S"):
            why.append(f"{w['S']} wrong")
        if med and r.get("speech_s", 0) > CFG["measure"]["long_ratio"] * med:
            why.append(f"runs long ({r['speech_s']:.1f} s vs median {med:.1f} s)")
        if why:
            tail.append({"i": i, "why": why})
    out["tail"] = tail
    out["median_speech_s"] = round(med, 2)
    return out


# ── MOS predictor ────────────────────────────────────────────
def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


class UTMOS:
    """UTMOS22 strong learner（SpeechMOS v1.2.0 的重寫版）：wav2vec 2.0 → BLSTM → 每個 frame 一個分數 → 平均。吃 16 kHz 波形，回 1–5 分。"""
    label = "UTMOS22 strong"

    def __init__(self, device: str = "auto"):
        import torch
        if not (MOS_CODE / "speechmos").exists() or not MOS_WEIGHTS.exists():
            sys.exit(f"沒有 MOS predictor（{os.path.relpath(MOS_DIR, HERE)}/）：先 ./present.sh fetch")
        self.torch = torch
        self.device = D1.resolve_device(device)
        if str(MOS_CODE) not in sys.path:
            sys.path.insert(0, str(MOS_CODE))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            from speechmos.utmos22.strong.model import UTMOS22Strong
            self.model = UTMOS22Strong()
            self.model.load_state_dict(torch.load(str(MOS_WEIGHTS), map_location="cpu", weights_only=True))
        self.model.eval().to(self.device)
        self.n_params = sum(p.numel() for p in self.model.parameters())

    def score(self, x16: np.ndarray, level: bool = True) -> float:
        torch = self.torch
        x = x16 * (CFG["mos"]["input_rms"] / P.rms(x16)) if level else x16
        with torch.no_grad(), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            s = self.model(torch.tensor(x, dtype=torch.float32)[None].to(self.device), FS)
        return float(s.detach().float().cpu().reshape(-1)[0])

    def info(self) -> dict:
        m = CFG["mos"]
        return {"model": "UTMOS22 strong learner", "code": m["repo"], "tag": m["tag"], "commit": m["commit"], "weights_sha256": m["weights_sha256"],
                "device": self.device, "n_params": self.n_params, "input": f"16 kHz, scaled to RMS {m['input_rms']}"}


class FakeMOS:
    """替身：停頓越乾淨分數越高。只為了讓流程走得過。"""
    label = "FAKE MOS"

    def score(self, x16: np.ndarray, level: bool = True) -> float:
        return round(float(np.clip(1.0 + (-P.floor_db(x16, FS)) / 12.0, 1.0, 5.0)), 3)

    def info(self) -> dict:
        return {"fake": True}


# ── 生成器 ───────────────────────────────────────────────────
class Generators:
    """Qwen3-TTS（W8 Demo 2 的 worker）與 F5-TTS（W8 Demo 1 的類別；只在 need_f5 時載）。"""
    fake = False

    def __init__(self, out, folder: Path, device: str | None = None, no_mlx: bool = False, need_f5: bool = False, nfe: int | None = None):
        c = CFG["infer"]
        self.out, self.nfe, self.f5, self.worker = out, nfe, None, None
        D1.quiet_hf()
        if need_f5:
            t0 = time.perf_counter()
            self.f5 = D1.F5(device or c["f5_device"], device or c["vocoder_device"])
            out(f"  載入 F5-TTS            {self.f5.n_params / 1e6:6.1f} M 參數   {self.f5.device} / vocoder {self.f5.vdev}   {time.perf_counter() - t0:5.1f} s")
        if not no_mlx:
            try:
                t0 = time.perf_counter()
                self.worker = P.Worker(folder / "_worker")
                self.worker.load()
                ld = self.worker.loaded.get("qwen3", {})
                out(f"  載入 Qwen3-TTS         {ld.get('n_params_M', float('nan')):6.1f} M 參數   MLX（mlx-audio {self.worker.versions.get('mlx-audio')}）   {time.perf_counter() - t0:5.1f} s")
                self.worker.qwen3("_warm", "Hi there.", P.PROMPT_WAV, P.PROMPT_TEXT, 0)                   # 第一次呼叫含 Metal 編譯
            except Exception as e:
                out(f"  {C.AMBER}Qwen3-TTS 起不來：{e}（先 ./present.sh check）{C.RESET}")
                if self.worker is not None:
                    self.worker.close()
                self.worker = None

    def available(self) -> list[str]:
        return (["qwen3", "live"] if self.worker is not None else []) + (["f5"] if self.f5 is not None else [])

    def synth(self, gen: str, name: str, text: str, prompt16: np.ndarray, prompt_text: str, prompt_path: Path, seed: int) -> dict:
        t0 = time.perf_counter()
        if gen == "f5":
            p24 = np.concatenate([dsp.resample(prompt16, FS, 24000), np.zeros(int(24000 * D1.PROMPT_PAD_MS / 1000))])
            r = self.f5.synth(p24, prompt_text, text, seed, nfe=self.nfe)
            return {"wav": r["wav"], "fs": r["fs"], "gen_s": time.perf_counter() - t0, "seed": seed, "T": r["T"], "nfe": r["nfe"]}
        q = CFG["qwen3"]
        r = self.worker.call(cmd="qwen3", name=name, text=text, ref_audio=str(prompt_path), ref_text=prompt_text, interval=q["streaming_interval"],
                             seed=seed, temperature=q["temperature"], lang_code=q["lang_code"], measure_ref_encode=False)
        if not r.get("ok"):
            raise RuntimeError(r.get("error"))
        return {"wav": C.load_wav(Path(r["wav"]), r["sr"]), "fs": int(r["sr"]), "gen_s": time.perf_counter() - t0, "seed": seed, "n_tokens": r.get("n_tokens")}

    def info(self) -> dict:
        i: dict = {"qwen3": None, "f5": None}
        if self.worker is not None:
            i["qwen3"] = {"repo": P.QWEN_REPO, "commit": P.D2_lock().get(P.QWEN_REPO), "mlx": self.worker.versions, "loaded": self.worker.loaded.get("qwen3"),
                          "temperature": CFG["qwen3"]["temperature"], "streaming_interval": CFG["qwen3"]["streaming_interval"]}
        if self.f5 is not None:
            i["f5"] = self.f5.info()
            if self.nfe:
                i["f5"]["nfe"] = self.nfe
        return i

    def close(self) -> None:
        if self.worker is not None:
            self.worker.close()


class FakeGenerators:
    """替身：每一「段」是同一段真人錄音改一點——快一點或慢一點、加一點底噪；其中三段故意做壞（漏一截、多一截、停不下來）。畫面標 FAKE。
    回傳裡的 said 是「這一段實際念了什麼」，給替身的 ASR 用。"""
    fake = True

    def __init__(self, with_f5: bool = True):
        self.with_f5 = with_f5

    def available(self) -> list[str]:
        return ["qwen3", "live"] + (["f5"] if self.with_f5 else [])

    def synth(self, gen: str, name: str, text: str, prompt16: np.ndarray, prompt_text: str, prompt_path: Path, seed: int) -> dict:
        rng = np.random.default_rng(seed + (1000 if gen == "f5" else 0))
        y = prompt16.copy()                                   # fake 模式下「要念的句子」就是 prompt 那一段錄音的文字
        words = text.split() if " " in text else list(text)
        n, k = len(y), len(words)
        if gen != "f5":
            if seed % 12 == 5:                                # 漏一截
                a, b = int(0.40 * n), int(0.65 * n)
                y = np.concatenate([y[:a], y[b:]]); words = words[: int(0.40 * k)] + words[int(0.65 * k):]
            elif seed % 12 == 9:                              # 多一截
                a, b = int(0.30 * n), int(0.50 * n)
                y = np.concatenate([y[:b], y[a:]]); words = words[: int(0.50 * k)] + words[int(0.30 * k):]
            elif seed % 12 == 11:                             # 停不下來
                y = np.concatenate([y, y[int(0.2 * n):]]); words = words + words[int(0.2 * k):]
            sp = float(rng.uniform(0.88, 1.14))
            y = dsp.resample(y, int(round(1000 * sp)), 1000)   # 快慢（音高跟著動——替身而已）
        y = y + 10 ** (float(rng.uniform(-60, -34)) / 20) * P.rms(y) * rng.standard_normal(len(y))
        fs = 24000
        return {"wav": dsp.resample(y, FS, fs), "fs": fs, "gen_s": 0.0, "seed": seed, "said": (" " if " " in text else "").join(words)}

    def info(self) -> dict:
        return {"fake": True}

    def close(self) -> None:
        pass


def versions() -> dict:
    v = P.versions()
    v["git"] = shutil.which("git") is not None
    return v


# ── 主體 ─────────────────────────────────────────────────────
class Demo:
    """items[name] = 一段聲音與量到的數字；groups[(sid, gen)] = 那一組的名字（照 seed 順序）。
    名字：<sid>_q01…（Qwen3-TTS）、<sid>_f01…（F5 對照）、<sid>_n13…（現場再抽的）、prompt_<sid>。"""

    def __init__(self, out, gens, sim, mos, asr, folder: Path, tag: str = "", n: int = N):
        self.out, self.gens, self.sim, self.mos, self.asr, self.folder, self.tag, self.n = out, gens, sim, mos, asr, folder, tag, n
        self.items: dict[str, dict] = {}
        self.groups: dict[tuple[str, str], list[str]] = {}
        self.sent: dict[str, dict] = {}
        self.emb: dict[str, np.ndarray] = {}
        self.picks: dict[str, int] = {}
        self.log: dict = {"time": C.now(), "versions": versions(), "tag": tag, "n": n, "sentences": {}, "models": None, "judges": None, "items": {}, "groups": {},
                          "played": [], "config": {k: CFG[k] for k in ("group", "qwen3", "measure", "f0")}}
        self.fig = self.plt = None
        self.sid, self.gen = SIDS[0], "qwen3"
        self.mode, self.revealed, self.adv, self.tail_on, self.xj = "wait", set(), False, False, "wer"
        self.last = None
        self.live_seed = n
        self.rehearsal_time = None
        p = (P.HERE / P.CFG["audio"]["match_rms_to"]).resolve()
        self.ref_rms = P.rms(C.load_wav(p, FS)) if p.exists() else 0.05
        self.folder.mkdir(parents=True, exist_ok=True)

    # ── 句子與 prompt ──
    def setup_sentences(self, fake: bool = False) -> None:
        o = self.out
        for sid in SIDS:
            c = CFG["sentences"][sid]
            own = P.own_wav("accent")
            if c["prompt"] == "own_accent" and own.exists():
                path, ptext, private, what = own, P.CFG["own"]["accent"]["prompt_text"], True, "the lecturer's own recording (W9 Demo 1's extra row)"
            else:
                path, ptext, private, what = P.PROMPT_WAV, P.PROMPT_TEXT, False, f"LibriSpeech dev-clean {P.PROMPT_UTT}"
                if c["prompt"] == "own_accent":
                    if fake:
                        continue                                # 替身要一段「念這一句」的錄音；沒有就不做這一句
                    o(f"  {C.AMBER}{sid}：沒有 {os.path.relpath(own, HERE)}，prompt 退到英文那一段（腔與聲調的觀察要打折）{C.RESET}")
            if not path.exists():
                sys.exit(f"沒有 {path}：先跑 ../w08_d1_mean_vs_sample/present.sh fetch")
            x = D1.trim_edges(C.load_wav(path, FS), FS)
            text = ptext if fake else c["text"]
            s = {"text": text, "lang": c["lang"] if not fake else ("zh" if private else "en"), "prompt_text": ptext, "prompt_what": what, "private": private,
                 "prompt_seconds": round(len(x) / FS, 2)}
            s["syllables"] = n_syllables(sid, text, s["lang"])
            self.sent[sid] = s
            pr = self._store(f"prompt_{sid}", "prompt", x, FS, sid=sid, private=private)
            self.emb[sid] = self.sim.embed(x)
            pr["mos"] = round(self.mos.score(x), 3)
            self._record(pr["name"])
            self.log["sentences"][sid] = s
            o(f"\n{C.BOLD}{sid}{C.RESET}  「{text}」（{s['syllables']} 個音節）")
            o(f"  prompt：{what}  「{ptext}」  {s['prompt_seconds']:.2f} s   predicted MOS {pr['mos']:.2f}")
        if self.sid not in self.sent:
            self.sid = next(iter(self.sent))

    def _store(self, name: str, kind: str, wav: np.ndarray, fs: int, **extra) -> dict:
        y = wav * (self.ref_rms / P.rms(wav)) if len(wav) else wav
        pk = float(np.max(np.abs(y))) if len(y) else 0.0
        if pk > 0.99:
            y = y * 0.99 / pk
        p = self.folder / f"{name}.wav"
        C.save_wav(p, y, fs)
        r = {"name": name, "kind": kind, "wav": wav, "fs": fs, "wav_path": p, "seconds": round(len(wav) / fs, 2), **extra}
        self.items[name] = r
        return r

    def _record(self, name: str) -> None:
        r = self.items[name]
        rec = {k: v for k, v in r.items() if k not in ("wav", "wav_path", "name", "said")}
        if "gen_s" in rec:
            rec["gen_s"] = round(rec["gen_s"], 2)
        rec["wav"] = os.path.relpath(r["wav_path"], HERE)
        self.log["items"][name] = rec

    # ── 抽一段、過三位裁判 ──
    def draw(self, sid: str, gen: str, seed: int) -> dict | None:
        s = self.sent[sid]
        name = draw_name(sid, gen, seed)
        try:
            g = self.gens.synth(gen, name, s["text"], self.items[f"prompt_{sid}"]["wav"], s["prompt_text"], self.items[f"prompt_{sid}"]["wav_path"], seed)
        except Exception as e:
            self.out(f"  {C.RED}{name} 失敗：{e!r}{C.RESET}")
            return None
        r = self._store(name, "draw", g.pop("wav"), g.pop("fs"), sid=sid, gen=gen, private=s["private"], **g)
        x16 = dsp.resample(r["wav"], r["fs"], FS)
        hyp = self.asr.transcribe(x16, s["lang"], hint=r.get("said", s["text"]))
        r["hyp"] = hyp
        r["wer"] = error_detail(s["text"], hyp, s["lang"], getattr(self.asr, "norm", None))
        r["sim"] = round(P.cos(self.sim.embed(x16), self.emb[sid]), 3)
        r["mos"] = round(self.mos.score(x16), 3)
        r["speech_s"] = round(speech_span(r["wav"], r["fs"]), 2)
        r["rate"] = round(s["syllables"] / r["speech_s"], 2) if r["speech_s"] > 0 else None
        r.update(f0_desc(r["wav"], r["fs"]))
        self._record(name)
        self.groups.setdefault((sid, gen), []).append(name)
        self.log["groups"][f"{sid}/{gen}"] = self.groups[(sid, gen)]
        return r

    def row_text(self, i: int, r: dict, st: dict | None = None, adv: bool = False) -> str:
        w = r["wer"]
        s = (f"   {i + 1:>3d}  {r['seconds']:6.2f}  {w['pct']:6.1f}  {w['S']:>2d} {w['D']:>2d} {w['I']:>2d}   {r['sim']:.3f}   {r['mos']:5.2f}  │ "
             f"{r['rate'] if r['rate'] is not None else float('nan'):6.2f}   {r['f0_sd_st'] if r['f0_sd_st'] is not None else float('nan'):6.2f}")
        if adv and st:
            for j in JUDGES:
                a = (st["judges"].get(j) or {}).get("advantage")
                s += f"   {a[i]:+6.2f}" if a else "       —"
        return s

    HEAD = "  draw  seconds   WER %   S  D  I     SIM     MOS  │  syl/s   F0 sd (st)"

    def compute_group(self, sid: str, gen: str, seeds: list[int]) -> None:
        o = self.out
        o(f"\n{C.BOLD}{GEN_LABEL[gen]} · {sid}{C.RESET}  {len(seeds)} draws, seeds {seeds[0]}–{seeds[-1]}")
        o(self.HEAD)
        for seed in seeds:
            r = self.draw(sid, gen, seed)
            if r is None:
                continue
            i = len(self.groups[(sid, gen)]) - 1
            o(self.row_text(i, r) + f"    {C.GREY}生成 {r['gen_s']:.1f} s{C.RESET}")
            if (sid, gen) == (self.sid, self.gen) and self.fig is not None and self.mode == "table":
                self.revealed.add(r["name"])
                self.draw_fig()

    def compute(self, f5: bool = True) -> None:
        self.log["models"] = self.gens.info()
        self.log["judges"] = {"sim": self.sim.info(), "mos": self.mos.info(), "asr": self.asr.info()}
        avail = self.gens.available()
        seeds = list(range(1, self.n + 1))
        for sid in self.sent:
            if "qwen3" in avail:
                self.compute_group(sid, "qwen3", seeds)
        fs = CFG["f5"]["sentence"]
        if f5 and "f5" in avail and fs in self.sent:
            self.compute_group(fs, "f5", seeds)

    # ── 目前這一組 ──
    def rows(self, sid: str | None = None, gen: str | None = None) -> list[dict]:
        return [self.items[n] for n in self.groups.get((sid or self.sid, gen or self.gen), []) if n in self.items]

    def stats(self, sid: str | None = None, gen: str | None = None) -> dict | None:
        rows = self.rows(sid, gen)
        return group_stats(rows) if rows else None

    def pick_index(self, which: str) -> tuple[int | None, str]:
        st = self.stats()
        if st is None:
            return None, ""
        if which == "ear":
            i = self.picks.get(self.sid) if self.gen == "qwen3" else None
            return (i - 1 if i else None), ""
        j = st["judges"].get(which)
        if not j:
            return None, ""
        note = f"（{j['n_tied_first']} 段平手——這位裁判分不出來；播編號最小的）" if j["n_tied_first"] > 1 else ""
        return j["first"][0], note

    # ── 視窗 ──
    def make_fig(self):
        import matplotlib.pyplot as plt
        self.plt = plt
        for k in list(plt.rcParams):
            if k.startswith("keymap."):
                plt.rcParams[k] = []                         # matplotlib 內建的快捷鍵（s 存檔、p 平移…）與這裡的鍵衝突
        d = CFG["display"]
        self.fig = plt.figure(figsize=(d["fig_w"], d["fig_h"]))
        self.fig.canvas.mpl_connect("key_press_event", self.on_key)
        self.draw_fig()
        return self.fig

    def redraw(self):
        if self.fig is not None:
            self.fig.canvas.draw_idle()
            self.plt.pause(0.05)

    def _title(self) -> str:
        s = self.sent[self.sid]
        tag = f"   [{self.tag}]" if self.tag else ""
        return f"W9 Demo 2 — one sentence, {len(self.rows())} draws, three judges · {GEN_LABEL[self.gen]} · {'English' if s['lang'] == 'en' else 'Mandarin'}" + tag

    def draw_fig(self, hl: str | None = None):
        if self.fig is None:
            return
        self.fig.clf()
        self.fig.suptitle(self._title(), fontsize=12)
        if self.mode == "scatter":
            self._draw_scatter()
        else:
            self._draw_table(hl)
        self.redraw()

    def _draw_table(self, hl: str | None):
        ax = self.fig.add_axes([0.02, 0.03, 0.96, 0.88])
        ax.set_axis_off()
        rows, st = self.rows(), self.stats()
        s = self.sent[self.sid]
        shown = "“" + s["text"] + "”" if s["lang"] == "en" else f"(the Mandarin sentence is in the terminal: {s['syllables']} syllables)"
        ax.text(0.0, 1.0, shown if len(shown) < 215 else shown[:212] + "…”", fontsize=9.5, color="#555555", va="top", wrap=True, transform=ax.transAxes)
        cols = [("draw", 0.03), ("seconds", 0.11), ("WER %", 0.20), ("SIM", 0.29), ("predicted MOS", 0.40), ("syllables / s", 0.55), ("F0 spread (st)", 0.67)]
        if self.adv:
            cols += [("Â  WER", 0.79), ("Â  SIM", 0.87), ("Â  MOS", 0.95)]
        y0, dy = 0.88, 0.058
        for name, x in cols:
            ax.text(x, y0, name, fontsize=11, weight="bold", ha="center", transform=ax.transAxes,
                    color=COLOR.get({"WER %": "wer", "SIM": "sim", "predicted MOS": "mos", "Â  WER": "wer", "Â  SIM": "sim", "Â  MOS": "mos"}.get(name, ""), "black"))
        ax.plot([0.0, 1.0], [y0 - 0.02, y0 - 0.02], color="#999999", lw=0.8, transform=ax.transAxes)
        ax.plot([0.475, 0.475], [y0 + 0.03, y0 - dy * (self.n + 2.2)], color="#bbbbbb", lw=0.8, ls=(0, (4, 4)), transform=ax.transAxes)
        ax.text(0.26, y0 + 0.045, "three judges", fontsize=9, color="#777777", ha="center", transform=ax.transAxes)
        ax.text(0.61, y0 + 0.045, "two descriptors — not judges", fontsize=9, color="#777777", ha="center", transform=ax.transAxes)
        if not rows:
            ax.text(0.5, 0.5, "no draws for this group" + ("" if self.gen != "live" else " — press n to draw"), ha="center", fontsize=12, color="#777777", transform=ax.transAxes)
            return
        if not self.revealed.intersection(r["name"] for r in rows):
            ax.text(0.5, 0.5, "press g", ha="center", fontsize=14, color="#999999", transform=ax.transAxes)
            return
        firsts = {w: self.pick_index(w)[0] for w in JUDGES + ["ear"]}
        tail = {t["i"] for t in st["tail"]} if self.tail_on else set()
        for i, r in enumerate(rows):
            if r["name"] not in self.revealed:
                continue
            y = y0 - dy * (i + 1)
            col = COLOR["bad"] if i in tail else "black"
            if r["name"] == hl:
                ax.add_patch(self.plt.Rectangle((0.0, y - 0.018), 1.0, dy * 0.92, color="#fff2a8", transform=ax.transAxes, zorder=0))
            vals = [f"{i + 1}", f"{r['seconds']:.2f}", f"{r['wer']['pct']:.1f}", f"{r['sim']:.3f}", f"{r['mos']:.2f}",
                    "—" if r["rate"] is None else f"{r['rate']:.2f}", "—" if r["f0_sd_st"] is None else f"{r['f0_sd_st']:.2f}"]
            if self.adv:
                for j in JUDGES:
                    a = (st["judges"].get(j) or {}).get("advantage")
                    vals.append(f"{a[i]:+.2f}" if a else "—")
            for (name, x), v in zip(cols, vals):
                ax.text(x, y, v, fontsize=11.5, ha="center", color=col, family="monospace", transform=ax.transAxes)
            marks = [(w, lab) for w, lab in (("wer", "W"), ("sim", "S"), ("mos", "M"), ("ear", "ear")) if self.mode == "picks" and firsts.get(w) == i]
            if marks:
                ax.text(0.055, y, " ".join(lab for _, lab in marks), fontsize=9, ha="left", weight="bold", color="#2a9d4b", transform=ax.transAxes)
        if len(self.revealed.intersection(r["name"] for r in rows)) == len(rows):
            y = y0 - dy * (len(rows) + 1.25)
            ax.plot([0.0, 1.0], [y + dy * 0.8, y + dy * 0.8], color="#999999", lw=0.8, transform=ax.transAxes)
            mean = ["mean", f"{np.mean([r['seconds'] for r in rows]):.2f}", f"{np.mean([r['wer']['pct'] for r in rows]):.1f}", f"{np.mean([r['sim'] for r in rows]):.3f}",
                    f"{np.mean([r['mos'] for r in rows]):.2f}", f"{st['rate']['mean']:.2f}" if st["rate"] else "—", f"{st['f0_sd_st']['mean']:.2f}" if st["f0_sd_st"] else "—"]
            sd = ["std", f"{np.std([r['seconds'] for r in rows]):.2f}", f"{np.std([r['wer']['pct'] for r in rows]):.1f}", f"{np.std([r['sim'] for r in rows]):.3f}",
                  f"{np.std([r['mos'] for r in rows]):.2f}", f"{st['rate']['std']:.2f}" if st["rate"] else "—", f"{st['f0_sd_st']['std']:.2f}" if st["f0_sd_st"] else "—"]
            for k, vals in enumerate((mean, sd)):
                for (name, x), v in zip(cols, vals):
                    ax.text(x, y - dy * k, v, fontsize=11, ha="center", color="#444444", family="monospace", transform=ax.transAxes)
            if self.adv:
                dead = [J_LABEL[j] for j in JUDGES if st["judges"].get(j) and st["judges"][j]["advantage"] is None]
                if dead:
                    ax.text(0.87, y, "std = 0 → no signal:\n" + ", ".join(dead), fontsize=9.5, ha="center", color=COLOR["bad"], transform=ax.transAxes)

    def _draw_scatter(self):
        rows, st = self.rows(), self.stats()
        j = st["judges"].get(self.xj) if st else None
        gs = self.fig.add_gridspec(1, 2, left=0.07, right=0.97, bottom=0.11, top=0.86, wspace=0.22)
        firsts = {w: self.pick_index(w)[0] for w in JUDGES + ["ear"]}
        for k, (key, lab) in enumerate((("rate", "syllables per second"), ("f0_sd_st", "F0 spread: std of log F0 (semitones)"))):
            ax = self.fig.add_subplot(gs[0, k])
            if not j:
                ax.text(0.5, 0.5, "no draws", ha="center", transform=ax.transAxes); continue
            xs, ys = j["rank"], [r.get(key) for r in rows]
            ok = [i for i, v in enumerate(ys) if v is not None]
            ax.scatter([xs[i] for i in ok], [ys[i] for i in ok], s=70, color=COLOR[self.xj], alpha=0.85, zorder=3)
            for i in ok:
                ax.annotate(str(i + 1), (xs[i], ys[i]), textcoords="offset points", xytext=(7, 5), fontsize=9)
            for w, mk in (("wer", "W"), ("sim", "S"), ("mos", "M"), ("ear", "ear")):
                i = firsts.get(w)
                if i is not None and ys[i] is not None:
                    ax.annotate(mk, (xs[i], ys[i]), textcoords="offset points", xytext=(-16 - 4 * len(mk), -12), fontsize=9, weight="bold", color=COLOR[w])
            rho = j["spearman"].get(key)
            ax.set_title(f"{lab}\nSpearman ρ with the rank = {'— (all tied)' if rho is None else f'{rho:+.2f}'}   (n = {len(rows)})", fontsize=10, loc="left")
            ax.set_xlabel(f"rank given by the {J_LABEL[self.xj]} judge (1 = its favourite; ties share a rank)", fontsize=9.5)
            ax.grid(alpha=0.3); ax.tick_params(labelsize=9)
            ax.set_xlim(0.3, len(rows) + 0.7)

    # ── 播 ──
    def _play(self, name: str, what: str) -> None:
        if name not in self.items:
            self.out(f"  {C.AMBER}沒有 {name}{C.RESET}"); return
        self.draw_fig(hl=name)
        self.out(f"  ▶ {what}")
        self.last = (name, what)
        self.log["played"].append(what)
        C.play(self.items[name]["wav_path"])

    def play_draw(self, i: int, why: str = "") -> None:
        rows = self.rows()
        if not (0 <= i < len(rows)):
            return
        r = rows[i]
        self.revealed.add(r["name"])
        self._play(r["name"], f"{why}draw {i + 1}   WER {r['wer']['pct']:.1f}%  SIM {r['sim']:.3f}  MOS {r['mos']:.2f}  │  {fmt2(r['rate'])} syl/s  F0 spread {fmt2(r['f0_sd_st'])} st")
        if r["wer"]["errors"]:
            self.out(f"      Whisper 聽到：{r['hyp']}")

    # ── 鍵 ──
    def show_table(self) -> None:
        o = self.out
        rows = self.rows()
        self.mode = "table"
        if not rows:
            o(f"  {C.AMBER}這一組沒有東西（{self.sid}／{self.gen}）{C.RESET}"); self.draw_fig(); return
        s = self.sent[self.sid]
        o(f"\n{C.BOLD}{GEN_LABEL[self.gen]} · {self.sid}{C.RESET}  「{s['text']}」")
        o(self.HEAD + ("     Â WER   Â SIM   Â MOS" if self.adv else ""))
        st = self.stats()
        for i, r in enumerate(rows):
            self.revealed.add(r["name"])
            o(self.row_text(i, r, st, self.adv))
        o(f"  mean  {np.mean([r['seconds'] for r in rows]):6.2f}  {np.mean([r['wer']['pct'] for r in rows]):6.1f}             {np.mean([r['sim'] for r in rows]):.3f}   {np.mean([r['mos'] for r in rows]):5.2f}  │ "
          f"{st['rate']['mean'] if st['rate'] else float('nan'):6.2f}   {st['f0_sd_st']['mean'] if st['f0_sd_st'] else float('nan'):6.2f}")
        o(f"  std   {np.std([r['seconds'] for r in rows]):6.2f}  {np.std([r['wer']['pct'] for r in rows]):6.1f}             {np.std([r['sim'] for r in rows]):.3f}   {np.std([r['mos'] for r in rows]):5.2f}  │ "
          f"{st['rate']['std'] if st['rate'] else float('nan'):6.2f}   {st['f0_sd_st']['std'] if st['f0_sd_st'] else float('nan'):6.2f}")
        self.draw_fig()

    def show_tail(self) -> None:
        o, st, rows = self.out, self.stats(), self.rows()
        if not st:
            return
        self.tail_on, self.mode = True, "table"
        self.revealed.update(r["name"] for r in rows)
        if not st["tail"]:
            o(f"\n  長尾：這 {len(rows)} 段沒有哪一段比最好的那一段多錯（每段 {st['min_errors']} 個錯）、也沒有長度離群的（中位數 {st['median_speech_s']} s）。")
            self.draw_fig(); return
        o(f"\n{C.BOLD}長尾{C.RESET}  {len(st['tail'])} / {len(rows)} 段：")
        for t in st["tail"]:
            o(f"    draw {t['i'] + 1:>2d}: {', '.join(t['why'])}   → “{rows[t['i']]['hyp']}”")
        worst = max(st["tail"], key=lambda t: (rows[t["i"]]["wer"]["pct"], rows[t["i"]]["speech_s"]))
        self.play_draw(worst["i"], "the long tail · ")

    def show_adv(self) -> None:
        o, st = self.out, self.stats()
        if not st:
            return
        self.adv = True
        self.show_table()
        for j in JUDGES:
            jj = st["judges"].get(j)
            if not jj:
                continue
            if jj["advantage"] is None:
                o(f"    {J_LABEL[j]:<16s} reward 全部相同（std = 0）→ advantage 沒有定義：這一組對這位裁判沒有訊號")
            else:
                o(f"    {J_LABEL[j]:<16s} reward 平均 {jj['mean']:.3f}、標準差 {jj['std']:.3f}；advantage 最高 draw {int(np.argmax(jj['advantage'])) + 1}（{max(jj['advantage']):+.2f}）、"
                  f"最低 draw {int(np.argmin(jj['advantage'])) + 1}（{min(jj['advantage']):+.2f}）")

    def show_pick(self, which: str) -> None:
        i, note = self.pick_index(which)
        self.mode = "picks"
        self.revealed.update(r["name"] for r in self.rows())
        if i is None:
            if which == "ear":
                self.out(f"  {C.AMBER}還沒有你自己挑的（{self.sid}）：./present.sh listen {self.sid}{C.RESET}" if self.gen == "qwen3" else f"  {C.AMBER}自己挑的只有彩排的 Qwen3-TTS 那一組有{C.RESET}")
            self.draw_fig(); return
        lab = "the lecturer's ear" if which == "ear" else f"the {J_LABEL[which]} judge"
        if note:
            self.out(f"  {lab} {note}")
        self.play_draw(i, f"first choice of {lab} · ")

    def show_scatter(self) -> None:
        o, st = self.out, self.stats()
        if not st:
            return
        if self.mode != "scatter":
            self.mode, self.xj = "scatter", "wer"
        else:
            k = JUDGES.index(self.xj) + 1
            if k >= len(JUDGES):
                self.mode = "table"; self.draw_fig(); return
            self.xj = JUDGES[k]
        if self.xj == "wer":
            o(f"\n{C.BOLD}名次 × 描述量{C.RESET}  Spearman ρ（名次 1 = 那位裁判最喜歡的；ρ > 0 = 它越喜歡的段，這個量越小）   n = {st['n']}")
            o(f"    {'judge':<16s}{'syl/s':>10s}{'F0 spread':>12s}     first choice(s): syl/s, F0 spread  ｜ the rest")
            for j in JUDGES:
                jj = st["judges"].get(j)
                if not jj:
                    continue
                f = lambda v: "   — (tied)" if v is None else f"{v:+10.2f}"       # noqa: E731
                o(f"    {J_LABEL[j]:<16s}{f(jj['spearman']['rate'])}{f(jj['spearman']['f0_sd_st']):>12s}     "
                  f"{fmt2(jj['first_rate'])} , {fmt2(jj['first_f0_sd_st'])}  ｜ {fmt2(jj['rest_rate'])} , {fmt2(jj['rest_f0_sd_st'])}" + (f"   （{jj['n_tied_first']} 段平手）" if jj["n_tied_first"] > 1 else ""))
            o(f"    {C.GREY}一組只有 {st['n']} 段：|ρ| 要到約 0.59 才過 5% 的雙尾檢定（n = 12）。這是挑選，不是訓練。{C.RESET}")
        self.draw_fig()

    def summary(self) -> None:
        """投影片 p56、p57 兩張表要的列（英文，直接抄）。"""
        o, st, rows = self.out, self.stats(), self.rows()
        if not st:
            return
        o(f"\n{C.BOLD}for the slides{C.RESET}  {GEN_LABEL[self.gen]} · {self.sid} · {len(rows)} draws")
        w = [r["wer"]["pct"] for r in rows]
        o(f"    WER %: {min(w):.1f}–{max(w):.1f}, {sum(1 for v in w if v == 0)} of {len(rows)} at 0   SIM: {min(r['sim'] for r in rows):.3f}–{max(r['sim'] for r in rows):.3f}   "
          f"MOS: {min(r['mos'] for r in rows):.2f}–{max(r['mos'] for r in rows):.2f}   syl/s: {st['rate']['min']:.2f}–{st['rate']['max']:.2f}   F0 spread: {st['f0_sd_st']['min']:.2f}–{st['f0_sd_st']['max']:.2f} st")
        o(f"    group mean; std   WER {np.mean(w):.1f}; {np.std(w):.1f}   SIM {np.mean([r['sim'] for r in rows]):.3f}; {np.std([r['sim'] for r in rows]):.3f}   "
          f"MOS {np.mean([r['mos'] for r in rows]):.2f}; {np.std([r['mos'] for r in rows]):.2f}   syl/s {st['rate']['mean']:.2f}; {st['rate']['std']:.2f}   F0 {st['f0_sd_st']['mean']:.2f}; {st['f0_sd_st']['std']:.2f}")
        for t in st["tail"]:
            r = rows[t["i"]]
            o(f"    long tail · draw {t['i'] + 1}: {', '.join(t['why'])}   WER {r['wer']['pct']:.1f}  SIM {r['sim']:.3f}  MOS {r['mos']:.2f}  {fmt2(r['rate'])} syl/s  {fmt2(r['f0_sd_st'])} st")
        for which in JUDGES + ["ear"]:
            i, _ = self.pick_index(which)
            if i is None:
                continue
            r = rows[i]
            jj = st["judges"].get(which) if which != "ear" else None
            tie = f" (tied with {jj['n_tied_first'] - 1} others: draws {', '.join(str(k + 1) for k in jj['first'])})" if jj and jj["n_tied_first"] > 1 else ""
            o(f"    first choice · {'the lecturer, by ear' if which == 'ear' else J_LABEL[which]:<22s} draw {i + 1}{tie}   {fmt2(r['rate'])} syl/s   {fmt2(r['f0_sd_st'])} st")

    def switch(self, gen: str | None = None) -> None:
        """gen 是 None：換另一句（鍵 z）。否則留在這一句、換生成器（鍵 f）；F5 的對照組只有一句有，沒有就跳到那一句。"""
        sid = self.sid
        if gen is None:
            ids = list(self.sent)
            sid = ids[(ids.index(self.sid) + 1) % len(ids)]
            gen = next((g for g in ("qwen3", "live", "f5") if self.groups.get((sid, g))), "qwen3")
        elif not self.groups.get((sid, gen)):
            f = CFG["f5"]["sentence"]
            if gen == "f5" and self.groups.get((f, "f5")):
                sid = f
            else:
                self.out(f"  {C.AMBER}沒有 {GEN_LABEL[gen]} 這一組{C.RESET}"); return
        self.sid, self.gen = sid, gen
        self.mode, self.adv, self.tail_on = "table", False, False
        s = self.sent[self.sid]
        self.out(f"\n{C.BOLD}→ {GEN_LABEL[self.gen]} · {self.sid}{C.RESET}  「{s['text']}」  （g 看表）")
        self.draw_fig()

    def draw_live(self) -> None:
        if self.gens is None or "live" not in self.gens.available():
            self.out(f"  {C.AMBER}現場抽要 Qwen3-TTS（replay 與 --no-mlx 沒有）{C.RESET}"); return
        self.groups[(self.sid, "live")] = []
        self.gen, self.mode, self.adv, self.tail_on = "live", "table", False, False
        seeds = list(range(self.live_seed + 1, self.live_seed + self.n + 1))
        self.live_seed += self.n
        self.draw_fig()
        self.compute_group(self.sid, "live", seeds)
        self.draw_fig()

    def menu(self) -> None:
        self.out(f"\n{C.BOLD}p56{C.RESET}  g 表   l 長尾   a advantage   z 換另一句   f F5 對照 ↔ Qwen3   n 現場再抽一組     "
                 f"{C.BOLD}p57{C.RESET}  w s m 三位裁判的第一名   h 我挑的   p 散點圖（再按換裁判）\n"
                 f"     1–9 0 - = 播第 1–12 段   o prompt   t 摘要   r 再播   ? 選單   q 離開")

    def dispatch(self, k: str) -> bool:
        if k == "g":
            self.show_table()
        elif k == "l":
            self.show_tail()
        elif k == "a":
            self.show_adv()
        elif k in J_KEY:
            self.show_pick(J_KEY[k])
        elif k == "h":
            self.show_pick("ear")
        elif k == "p":
            self.show_scatter()
        elif k == "z":
            self.switch()
        elif k == "f":
            self.switch(gen="qwen3" if self.gen == "f5" else "f5")
        elif k == "n":
            self.draw_live()
        elif k in DRAW_KEYS:
            self.mode = "table" if self.mode == "scatter" else self.mode
            self.play_draw(DRAW_KEYS.index(k))
        elif k == "o":
            self._play(f"prompt_{self.sid}", f"the prompt · {self.sent[self.sid]['prompt_what']}")
        elif k == "t":
            self.summary()
        elif k == "r":
            if self.last:
                self._play(*self.last)
        elif k == "?":
            self.menu()
        elif k == "q":
            return False
        return True

    def on_key(self, ev) -> None:
        if ev.key is not None and not self.dispatch(ev.key):
            self.plt.close(self.fig)

    # ── 彩排 ──
    def rehearse_figs(self) -> None:
        import matplotlib.pyplot as plt
        self.plt = plt
        d = CFG["display"]
        keep = (self.sid, self.gen)
        for (sid, gen), names in self.groups.items():
            if not names:
                continue
            self.sid, self.gen = sid, gen
            self.revealed.update(names)
            for mode, adv, xj, nm in (("picks", True, "wer", f"table_{sid}_{gen}.png"), ("scatter", False, "wer", f"scatter_{sid}_{gen}_wer.png"),
                                      ("scatter", False, "sim", f"scatter_{sid}_{gen}_sim.png"), ("scatter", False, "mos", f"scatter_{sid}_{gen}_mos.png")):
                self.fig = plt.figure(figsize=(d["fig_w"], d["fig_h"]))
                self.mode, self.adv, self.xj, self.tail_on = mode, adv, xj, True
                self.draw_fig(); self.fig.savefig(self.folder / nm, dpi=100); plt.close(self.fig)
        self.fig = None
        (self.sid, self.gen), self.mode, self.adv, self.tail_on = keep, "wait", False, False

    def save(self) -> None:
        stem = "rehearsal" if self.folder.name == "rehearsal" else "session"
        self.log["stats"] = {f"{sid}/{gen}": self.stats(sid, gen) for (sid, gen) in self.groups if self.groups[(sid, gen)]}
        (self.folder / f"{stem}.json").write_text(json.dumps(self.log, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
        (self.folder / f"{stem}.txt").write_text(self.out.text(), encoding="utf-8")
        shutil.rmtree(self.folder / "_worker", ignore_errors=True)
        C.note(f"  存到 {os.path.relpath(self.folder, HERE)}/（{stem}.json、{stem}.txt、.wav）")


# ── 讀回彩排 ─────────────────────────────────────────────────
def read_picks() -> dict[str, int]:
    if PICKS.exists():
        return {k: int(v) for k, v in json.loads(PICKS.read_text(encoding="utf-8")).get("picks", {}).items()}
    return {}


def load_rehearsal(d: Demo, out) -> bool:
    """把 runs/rehearsal/ 的那一組讀進 d（items、groups、sent、picks）。回有沒有讀到。"""
    rec_path = REHEARSAL / "rehearsal.json"
    if not rec_path.exists():
        return False
    rec = json.loads(rec_path.read_text(encoding="utf-8"))
    missing = []
    for name, r in rec["items"].items():
        p = (HERE / r["wav"]).resolve()
        if not p.exists():
            missing.append(name); continue
        item = dict(r)
        item.update(name=name, wav_path=p, wav=C.load_wav(p, r["fs"]))
        d.items[name] = item
    for key, names in rec["groups"].items():
        sid, gen = key.split("/")
        d.groups[(sid, gen)] = [n for n in names if n in d.items]
    d.sent.update(rec["sentences"])
    d.n = int(rec.get("n", d.n))
    d.live_seed = d.n
    d.picks = read_picks()
    d.rehearsal_time = rec["time"]
    if d.sid not in d.sent and d.sent:
        d.sid = next(iter(d.sent))
    out(f"  讀回 {rec['time']} 的彩排：{sum(len(v) for v in d.groups.values())} 段" + (f"（{len(missing)} 段的 .wav 不在：{', '.join(missing[:4])}…）" if missing else "")
        + (f"；自己挑的：{d.picks}" if d.picks else "；還沒有自己挑的（./present.sh listen en）"))
    return True


class _Label:
    def __init__(self, label):
        self.label = label


def build_replay(out) -> Demo:
    d = Demo(out, None, _Label("WavLM-SV"), _Label("UTMOS22 strong"), _Label("Whisper small"), REHEARSAL, tag="replay")
    if not load_rehearsal(d, out):
        sys.exit("沒有 runs/rehearsal/rehearsal.json：先 ./present.sh rehearse")
    d.tag = "replay " + d.rehearsal_time
    d.log = json.loads((REHEARSAL / "rehearsal.json").read_text(encoding="utf-8"))
    d.log["played"] = []
    return d


def real_judges(out, device: str | None = None):
    c = CFG["infer"]
    D1.quiet_hf()
    t0 = time.perf_counter()
    sim = P.WavLMSV(device or c["sim_device"])
    out(f"  載入 SIM 的裁判        {sim.n_params / 1e6:6.1f} M 參數   {sim.device}   {time.perf_counter() - t0:5.1f} s   （{P.REPOS['sim']}）")
    t0 = time.perf_counter()
    mos = UTMOS(device or c["mos_device"])
    out(f"  載入 MOS predictor     {mos.n_params / 1e6:6.1f} M 參數   {mos.device}   {time.perf_counter() - t0:5.1f} s   （UTMOS22 strong，SpeechMOS {CFG['mos']['tag']}）")
    t0 = time.perf_counter()
    P.CFG["whisper"]["max_new_tokens"] = int(c["whisper_max_new_tokens"])
    asr = P.Whisper(c["whisper_device"])
    out(f"  載入 WER 的裁判        Whisper small   {asr.device}   {time.perf_counter() - t0:5.1f} s")
    return sim, mos, asr


def build(cmd: str, out, args) -> Demo:
    folder = REHEARSAL if cmd == "rehearse" else HERE / "runs" / "live" / (C.now().replace(":", "") + (".fake" if cmd == "fake" else ""))
    folder.mkdir(parents=True, exist_ok=True)
    n = args.n or N
    if cmd == "fake":
        sim, mos, asr = real_judges(out, args.device) if args.real_judges else (P.FakeEmbed(), FakeMOS(), P.FakeASR())
        d = Demo(out, FakeGenerators(), sim, mos, asr, folder, tag="FAKE" + (" generator, real judges" if args.real_judges else ""), n=n)
        d.setup_sentences(fake=True)
        d.compute()
        return d
    try:
        import torch  # noqa: F401
    except ImportError:
        sys.exit("沒有 torch：先跑 ../w08_d1_mean_vs_sample/setup.sh（../.venv-tts）；或 ./present.sh fake 演練流程")
    have = cmd == "show" and (REHEARSAL / "rehearsal.json").exists() and not args.fresh
    gens = Generators(out, folder, args.device, args.no_mlx, need_f5=(not have and CFG["f5"]["enabled"] and not args.no_f5), nfe=args.nfe)
    sim, mos, asr = real_judges(out, args.device)
    d = Demo(out, gens, sim, mos, asr, folder, tag="rehearsal" if cmd == "rehearse" else "", n=n)
    if have:
        load_rehearsal(d, out)
        for sid in d.sent:                                     # 現場再抽要 prompt 的向量
            d.emb[sid] = sim.embed(dsp.resample(d.items[f"prompt_{sid}"]["wav"], d.items[f"prompt_{sid}"]["fs"], FS))
        d.log["models"], d.log["judges"] = gens.info(), {"sim": sim.info(), "mos": mos.info(), "asr": asr.info()}
        d.log["sentences"], d.log["from_rehearsal"] = d.sent, d.rehearsal_time
    else:
        d.setup_sentences()
        d.compute(f5=not args.no_f5)
    return d


# ── listen：自己挑一段 ───────────────────────────────────────
def listen(sid: str) -> None:
    out = P.Out()
    d = build_replay(out)
    if sid not in d.sent or not d.groups.get((sid, "qwen3")):
        sys.exit(f"彩排裡沒有 {sid} 的 Qwen3-TTS 那一組（有：{', '.join(d.sent)}）")
    names = list(d.groups[(sid, "qwen3")])
    order = list(range(len(names)))
    random.Random(time.time_ns()).shuffle(order)
    letters = "ABCDEFGHIJKLMNOPQRSTUVWX"[: len(names)]
    C.banner(f"自己挑一段：{sid}，{len(names)} 段，順序打亂、不顯示編號與分數", C.AMBER)
    print(f"  「{d.sent[sid]['text']}」\n  先整輪播一次；之後打字母重聽（例如 c），打 `pick c` 定案，打 `q` 放棄。挑你覺得最自然的一段。\n")
    for k, i in enumerate(order):
        print(f"  ▶ {letters[k]}")
        C.play(d.items[names[i]]["wav_path"])
    while True:
        try:
            s = input("  > ").strip().lower()
        except EOFError:
            return
        if s == "q":
            return
        m = re.fullmatch(r"(pick\s+)?([a-x])", s)
        if not m or m.group(2).upper() not in letters:
            print(f"  字母 {letters[0]}–{letters[-1]}，或 pick <字母>，或 q"); continue
        k = letters.index(m.group(2).upper())
        if not m.group(1):
            print(f"  ▶ {letters[k]}")
            C.play(d.items[names[order[k]]]["wav_path"]); continue
        picks = read_picks()
        picks[sid] = order[k] + 1
        PICKS.write_text(json.dumps({"picks": picks, "time": C.now(), "how": "blind: shuffled order, no numbers or scores shown", "rehearsal": d.rehearsal_time},
                                    ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"  {letters[k]} 是 draw {order[k] + 1} → {os.path.relpath(PICKS, HERE)}。接著 ./present.sh export，把它寫進 w09-metrics.json。")
        return


# ── check ────────────────────────────────────────────────────
def check(device: str | None = None) -> None:
    C.banner("課前體檢", C.AMBER)
    res: dict = {"time": C.now(), "versions": versions(), "env": {"PYTORCH_ENABLE_MPS_FALLBACK": os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK")}}
    for k, v in res["versions"].items():
        print(f"  {k:<16s} {v}")
    try:
        import torch
    except ImportError:
        sys.exit("  沒有 torch：先跑 ../w08_d1_mean_vs_sample/setup.sh")
    D1.quiet_hf()
    mps = bool(torch.backends.mps.is_available())
    res["mps_available"] = mps
    devs = [device] if device else (["cpu"] + (["mps"] if mps else []))
    clips: dict[str, np.ndarray] = {}
    for label, p in (("real recording (target)", P.HERE / P.CFG["audio"]["real_target"]), ("real recording (prompt)", P.PROMPT_WAV),
                     ("W8 SpeechT5, bdl voice", P.HERE / P.CFG["audio"]["other_voice"]), ("W9 D1 Qwen3-TTS, clean prompt", P.REHEARSAL / "clean_qwen3.wav"),
                     ("W9 D1 F5-TTS, clean prompt", P.REHEARSAL / "clean_f5.wav"), ("W9 D1 SpeechT5, noisy prompt", P.REHEARSAL / "noise_speecht5.wav")):
        p = Path(p).resolve()
        if p.exists():
            clips[label] = C.load_wav(p, FS)
    caught: list[str] = []
    # 1. MOS predictor：cpu／mps、音量
    res["mos"] = {"code_commit": None, "weights_sha256_ok": None}
    if (MOS_CODE / ".git").exists():
        try:
            res["mos"]["code_commit"] = subprocess.run(["git", "-C", str(MOS_CODE), "rev-parse", "HEAD"], capture_output=True, text=True, timeout=20).stdout.strip()
        except Exception as e:
            res["mos"]["code_commit"] = repr(e)
    if MOS_WEIGHTS.exists():
        res["mos"]["weights_sha256_ok"] = sha256(MOS_WEIGHTS) == CFG["mos"]["weights_sha256"]
    print(f"  MOS predictor：程式 @ {str(res['mos']['code_commit'])[:12]}（設定 {CFG['mos']['commit'][:12]}）   權重 sha256 {'對' if res['mos']['weights_sha256_ok'] else '不對或不在'}")
    scores: dict[str, dict] = {}
    for dev in devs:
        try:
            with warnings.catch_warnings(record=True) as ws:
                warnings.simplefilter("always")
                t0 = time.perf_counter(); m = UTMOS(dev); load = time.perf_counter() - t0
                m.score(next(iter(clips.values())))
                t0 = time.perf_counter(); sc = {k: round(m.score(x), 3) for k, x in clips.items()}; dt = (time.perf_counter() - t0) / max(1, len(clips))
                x = clips.get("real recording (target)", next(iter(clips.values())))
                lv = {f"rms {g:g}": round(m.score(x * (g / P.rms(x)), level=False), 3) for g in (0.01, 0.05, 0.2)}
            for w in ws:
                if "fall back" in str(w.message) or "not currently supported on the MPS" in str(w.message):
                    caught.append(f"UTMOS {dev}: {str(w.message)[:160]}")
            scores[dev] = sc
            res["mos"][dev] = {"load_s": round(load, 2), "score_s": round(dt, 3), "n_params": m.n_params, "scores": sc, "same_clip_three_levels": lv}
            print(f"  UTMOS {dev}: 載入 {load:.1f} s · 一段 {dt * 1000:.0f} ms · 同一段三種音量 {lv}")
        except Exception as e:
            res["mos"][dev] = {"error": repr(e)}
            print(f"  UTMOS {dev}: {C.RED}失敗{C.RESET} {e!r}")
    if "cpu" in scores and "mps" in scores:
        res["mos"]["mps_minus_cpu_max_abs"] = round(max(abs(scores["mps"][k] - scores["cpu"][k]) for k in scores["cpu"]), 4)
        print(f"  UTMOS：mps 對 cpu 的分數最大差 {res['mos']['mps_minus_cpu_max_abs']}（要 ≈ 0；差很多就把 [infer] mos_device 改成 \"cpu\"）")
    # 2. 三位裁判對幾段已知的聲音
    try:
        out = P.Out()
        sim, mos, asr = real_judges(out, device)
        ref = sim.embed(D1.trim_edges(C.load_wav(P.PROMPT_WAV, FS), FS))
        print(f"\n  {'':<32s}{'predicted MOS':>14s}{'SIM vs prompt':>15s}{'WER %':>8s}")
        res["known_clips"] = {}
        for label, x in clips.items():
            w = error_detail(P.TARGET, asr.transcribe(x, "en"), "en", asr.norm)["pct"] if label != "real recording (prompt)" else None
            row = {"mos": round(mos.score(x), 3), "sim": round(P.cos(sim.embed(x), ref), 3), "wer_pct": w}
            res["known_clips"][label] = row
            print(f"  {label:<32s}{row['mos']:>14.2f}{row['sim']:>15.3f}{'—' if w is None else format(w, '.1f'):>8s}")
    except SystemExit as e:
        res["known_clips"] = {"error": str(e)}
        print(f"  {C.RED}{e}{C.RESET}")
    except Exception as e:
        res["known_clips"] = {"error": repr(e)}
        print(f"  三位裁判：{C.RED}失敗{C.RESET} {e!r}")
    # 3. MLX 那一半
    try:
        py = P.WORKER_PY_ENV if P.WORKER_PY_ENV.exists() else Path(sys.executable)
        p = subprocess.run([str(py), str(P.WORKER_PY), "--selfcheck"], capture_output=True, text=True, timeout=120)
        res["mlx"] = json.loads(p.stdout.strip().splitlines()[-1]) if p.stdout.strip() else {"ok": False, "stderr": p.stderr[-500:]}
    except Exception as e:
        res["mlx"] = {"ok": False, "error": repr(e)}
    mv = res["mlx"].get("versions", {})
    print(f"\n  mlx-audio {mv.get('mlx-audio')}、mlx {mv.get('mlx')}；qwen3_tts import：{mv.get('qwen3_tts_in_mlx_audio')}；{P.QWEN_REPO} @ {str(P.D2_lock().get(P.QWEN_REPO))[:12]}")
    own = P.own_wav("accent")
    res["zh_prompt"] = "own_accent" if own.exists() else "falls back to the English prompt"
    print(f"  華語那一句的 prompt：{'W9 Demo 1 錄的 own_accent.wav' if own.exists() else C.AMBER + '沒有 own_accent.wav，會退到英文的 prompt' + C.RESET}")
    res["mps_fallback_warnings"] = caught
    if caught:
        print(f"  {C.AMBER}有運算退回 CPU：{caught[:3]}{C.RESET}")
    (HERE / "runs").mkdir(exist_ok=True)
    (HERE / "runs" / "check.json").write_text(json.dumps(res, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    print("\n  → runs/check.json。要看的：MOS predictor 載得起、mps 對 cpu 的分數差 ≈ 0、三種音量的分數差不大、Whisper 對真人錄音 WER 0。")
    print("  下一步：./present.sh rehearse → ./present.sh listen en；./present.sh listen zh → ./present.sh export")


# ── fetch ────────────────────────────────────────────────────
def fetch() -> None:
    import urllib.request
    C.banner("下載 MOS predictor（程式與權重）；確認前面 demo 的模型都在", C.AMBER)
    m = CFG["mos"]
    MOS_DIR.mkdir(parents=True, exist_ok=True)
    if not (MOS_CODE / "speechmos").exists():
        if shutil.which("git") is None:
            sys.exit("  需要 git（xcode-select --install）")
        subprocess.run(["git", "clone", "--depth", "1", "--branch", m["tag"], m["repo"], str(MOS_CODE)], check=True)
    sha = subprocess.run(["git", "-C", str(MOS_CODE), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    if sha != m["commit"]:
        sys.exit(f"  {C.RED}{m['repo']} 的 {m['tag']} 現在是 {sha}，設定檔鎖的是 {m['commit']}——tag 被移動過？先看過程式再改設定。{C.RESET}")
    print(f"  {m['repo']} @ {m['tag']}（{sha[:12]}）→ {os.path.relpath(MOS_CODE, HERE)}/")
    if not MOS_WEIGHTS.exists():
        print(f"  下載權重（約 411 MB）：{m['weights_url']}")
        tmp = MOS_WEIGHTS.with_suffix(".part")
        urllib.request.urlretrieve(m["weights_url"], tmp)
        tmp.rename(MOS_WEIGHTS)
    h = sha256(MOS_WEIGHTS)
    if h != m["weights_sha256"]:
        sys.exit(f"  {C.RED}權重的 sha256 是 {h}，設定檔鎖的是 {m['weights_sha256']}——不要載入，先查原因。{C.RESET}")
    print(f"  權重 {MOS_WEIGHTS.name}  {MOS_WEIGHTS.stat().st_size / 1e6:.0f} MB  sha256 對")
    for key in ("sim", "whisper"):
        try:
            P.model_dir(key, offline=True)
            print(f"  W9 Demo 1 · {P.REPOS[key]}：在")
        except Exception:
            print(f"  {C.AMBER}W9 Demo 1 · {P.REPOS[key]} 不在快取：先跑 ../w09_d1_cloning_leak/present.sh fetch{C.RESET}")
    try:
        D1.model_dir("f5", offline=True)
        print(f"  W8 Demo 1 · {D1.REPOS['f5']}：在")
    except Exception:
        print(f"  {C.AMBER}W8 Demo 1 · {D1.REPOS['f5']} 不在快取：先跑 ../w08_d1_mean_vs_sample/present.sh fetch{C.RESET}")
    try:
        from huggingface_hub import snapshot_download
        snapshot_download(P.QWEN_REPO, revision=P.D2_lock().get(P.QWEN_REPO), local_files_only=True)
        print(f"  W8 Demo 2 · {P.QWEN_REPO}：在")
    except Exception:
        print(f"  {C.AMBER}W8 Demo 2 · {P.QWEN_REPO} 不在快取：先跑 ../w08_d2_nfe_cfg_first_packet/present.sh fetch{C.RESET}")
    print(f"  prompt（英文）：{'在' if P.PROMPT_WAV.exists() else C.AMBER + '不在' + C.RESET}   prompt（華語，own_accent.wav）：{'在' if P.own_wav('accent').exists() else '不在（會退到英文的 prompt）'}")
    print("下一步：./present.sh check")


# ── export ───────────────────────────────────────────────────
def export(rec_path: Path) -> None:
    rec = json.loads(rec_path.read_text(encoding="utf-8"))
    d = json.loads(METRICS_JSON.read_text(encoding="utf-8")) if METRICS_JSON.exists() else {
        "_note": "量測數字（進版控的出處，speech_ai/CLAUDE.md 4.8）。w09-data.json 是 make_figs_w09.py 算出來的、每次重產；量測的放這裡。"}
    drop = ("wav", "kind", "name", "sid", "gen")
    groups = {}
    for key, names in rec["groups"].items():
        rows = [rec["items"][n] for n in names if n in rec["items"]]
        if rows:
            groups[key] = {"draws": [{k: v for k, v in r.items() if k not in drop} for r in rows], "stats": group_stats(rows)}
    picks = read_picks() if rec_path.parent == REHEARSAL else {}
    d["demo2"] = {"source": "demos/w09_d2_judges_pick/runs/rehearsal/rehearsal.json", "time": rec["time"], "tag": rec.get("tag", ""), "n": rec.get("n"),
                  "sentences": rec["sentences"], "models": rec["models"], "judges": rec["judges"], "groups": groups,
                  "lecturer_pick": {"draw": picks, "how": "blind listening (./present.sh listen): shuffled order, no numbers or scores shown"},
                  "definitions": {"wer": "Whisper small (greedy) against the sentence; English: words after Whisper's English normaliser; Mandarin: tone-numbered pinyin syllables. S, D, I from a Levenshtein alignment. Reward = −WER.",
                                  "sim": "cosine between WavLM-SV (microsoft/wavlm-base-plus-sv) embeddings of the draw and of the prompt",
                                  "mos": "UTMOS22 strong learner (tarepan/SpeechMOS v1.2.0 port of the official checkpoint), 16 kHz input scaled to a fixed RMS; a prediction of a 1–5 naturalness score, trained on BVCC (English)",
                                  "rate": "syllables of the sentence to read / seconds between the first and the last 10 ms frame within 30 dB of the 95th-percentile level (pauses inside the utterance included). The numerator is the reference sentence, not what was actually said. This demo's own descriptor.",
                                  "f0_sd_st": "standard deviation of 12·log2(F0) over voiced frames (autocorrelation F0, 40 ms / 10 ms; frames more than a factor 1.8 from the median dropped as octave errors), in semitones. This demo's own descriptor.",
                                  "advantage": "(r − group mean) / group standard deviation (population), per judge; null when the group's rewards are all equal",
                                  "rank / first": "rank 1 = highest reward; ties share the average rank; 'first' lists every draw tied for the best reward (0-based indices)",
                                  "spearman": "Spearman correlation between a judge's rank and a descriptor; positive = the draws the judge prefers have smaller values",
                                  "tail": "draws with more word errors than the best draw of the group, or whose speech span exceeds 1.5 × the group median",
                                  "what_we_heard": "filled by hand after the rehearsal (w09.qmd p56, p57)"}}
    METRICS_JSON.parent.mkdir(parents=True, exist_ok=True)
    METRICS_JSON.write_text(json.dumps(d, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    print(f"  → {os.path.relpath(METRICS_JSON, HERE)}（demo2）")


# ── selftest ─────────────────────────────────────────────────
def selftest() -> None:
    C.banner("selftest", C.AMBER)
    fails = []

    def ok(cond: bool, msg: str):
        print(f"  {'OK ' if cond else C.RED + 'FAIL' + C.RESET}  {msg}")
        if not cond:
            fails.append(msg)

    # 1. 對齊
    a = align_counts("a b c d e".split(), "a x c e".split())
    ok((a["S"], a["D"], a["I"], a["N"]) == (1, 1, 0, 5), f"對齊：一個替換、一個漏（{a}）")
    a = align_counts("a b c".split(), "a b c b c".split())
    ok((a["S"], a["D"], a["I"]) == (0, 0, 2), "對齊：多念一截 = 2 個 I")
    a = align_counts("a b c d".split(), "a d".split())
    ok((a["S"], a["D"], a["I"]) == (0, 2, 0), "對齊：漏一截 = 2 個 D")
    e = error_detail("Where's the peck?", "where is the PECK", "en")
    ok(e["errors"] <= 2 and e["unit"] == "words", f"英文的退路 normalizer（沒有 Whisper 時）：{e}")
    z = error_detail("媽媽騎馬，馬慢，媽媽罵馬。", "妈妈骑马马慢妈妈骂马", "zh")
    z2 = error_detail("媽媽騎馬，馬慢，媽媽罵馬。", "媽媽騎馬，馬慢，媽媽媽馬。", "zh")
    ok(z["errors"] == 0 and z["N"] == 10, f"華語：繁簡不算錯（{z['N']} 個音節、{z['errors']} 個錯）")
    try:
        import pypinyin  # noqa: F401
        ok(z2["S"] == 1 and z2["errors"] == 1, f"華語：罵 → 媽（只差聲調）算 {z2['S']} 個替換")
    except ImportError:
        print("  --   沒有 pypinyin：華語退到逐字比對，聲調錯看不到")
    # 2. 音節
    en = CFG["sentences"]["en"]
    h = syllables_en(en["text"])
    ok(abs(h - en["syllables"]) <= 2, f"英文句子的音節數：設定 {en['syllables']}、啟發式 {h}")
    ok(n_syllables("zh", CFG["sentences"]["zh"]["text"], "zh") == len(re.findall(r"[一-鿿]", CFG["sentences"]["zh"]["text"])) > 20, f"華語句子：{n_syllables('zh', CFG['sentences']['zh']['text'], 'zh')} 個音節")
    # 3. 語速與 F0 起伏：合成訊號
    fs = 24000
    t = np.arange(int(3.0 * fs)) / fs
    f_flat = np.full(len(t), 150.0)
    f_live = 150.0 * 2 ** (4 * np.sin(2 * np.pi * 0.7 * t) / 12)
    tone = lambda f: 0.2 * np.sign(np.sin(2 * np.pi * np.cumsum(f) / fs))          # noqa: E731
    pad = np.zeros(fs // 2)
    x_flat, x_live = np.concatenate([pad, tone(f_flat), pad]), np.concatenate([pad, tone(f_live), pad])
    sp = speech_span(x_flat, fs)
    ok(abs(sp - 3.0) < 0.05, f"說話的時間：前後各 0.5 s 靜音的 3 s 訊號 → {sp:.2f} s")
    a, b = f0_desc(x_flat, fs), f0_desc(x_live, fs)
    ok(a["f0_sd_st"] < 0.3 and abs(a["f0_median_hz"] - 150) < 3, f"F0 起伏：平的 150 Hz → 中位數 {a['f0_median_hz']} Hz、{a['f0_sd_st']} 半音")
    ok(abs(b["f0_sd_st"] - 4 / math.sqrt(2)) < 0.4, f"F0 起伏：±4 半音的正弦起伏 → {b['f0_sd_st']} 半音（理論 {4 / math.sqrt(2):.2f}）")
    # 4. 一組的統計
    mk = lambda w, s, m, r, f, sec=5.0: {"wer": {"pct": w, "S": int(w > 0), "D": 0, "I": 0, "errors": int(w > 0)}, "sim": s, "mos": m, "rate": r, "f0_sd_st": f, "speech_s": sec}   # noqa: E731
    rows = [mk(0.0, 0.90, 4.0, 5.0, 3.0), mk(0.0, 0.92, 3.5, 4.0, 2.0), mk(10.0, 0.95, 4.5, 6.0, 4.0), mk(0.0, 0.91, 3.0, 4.5, 2.5, sec=9.0)]
    st = group_stats(rows)
    jw = st["judges"]["wer"]
    ok(jw["first"] == [0, 1, 3] and jw["n_tied_first"] == 3 and jw["rank"] == [2.0, 2.0, 4.0, 2.0], f"WER：三段平手第一（名次 {jw['rank']}）")
    ok(st["judges"]["sim"]["first"] == [2] and st["judges"]["mos"]["first"] == [2], "SIM 與 MOS 的第一名")
    adv = st["judges"]["sim"]["advantage"]
    ok(abs(sum(adv)) < 0.02 and abs(float(np.std(adv)) - 1) < 0.02, f"advantage：組內平均 0、標準差 1（{adv}）")
    st0 = group_stats([mk(0.0, 0.9, 4.0, 5.0, 3.0), mk(0.0, 0.8, 3.0, 4.5, 2.0), mk(0.0, 0.7, 3.5, 4.0, 2.5)])
    ok(st0["judges"]["wer"]["advantage"] is None and st0["judges"]["wer"]["spearman"]["rate"] is None, "WER 全部平手：advantage 與相關都沒有定義")
    ok(st0["judges"]["sim"]["spearman"]["rate"] == -1.0, f"名次與語速的 Spearman：SIM 越高語速越快 → ρ = {st0['judges']['sim']['spearman']['rate']}")
    ok([t["i"] for t in st["tail"]] == [2, 3] and "runs long" in st["tail"][1]["why"][0], f"長尾：有錯字的與長度離群的（{st['tail']}）")
    ok(avg_rank([3.0, 1.0, 3.0, 2.0]) == [1.5, 4.0, 1.5, 3.0], "平手取平均名次")
    ok(len(DRAW_KEYS) >= N and len(set(DRAW_KEYS)) == len(DRAW_KEYS), f"{N} 段各有一個鍵")
    ok(re.fullmatch(r"[0-9a-f]{40}", CFG["mos"]["commit"]) is not None and re.fullmatch(r"[0-9a-f]{64}", CFG["mos"]["weights_sha256"]) is not None, "MOS predictor 的 commit 與 sha256 有鎖")
    # 5. fake 流程
    import matplotlib
    matplotlib.use("Agg")
    C.play = lambda *a, **k: None
    global METRICS_JSON
    if not P.PROMPT_WAV.exists():
        print("  --   沒有 W8 Demo 1 的 audio/prompt.wav（先跑它的 fetch），跳過 fake 流程")
    else:
        folder = HERE / "runs" / "_selftest"
        shutil.rmtree(folder, ignore_errors=True)
        folder.mkdir(parents=True)
        d = Demo(P.Out(), FakeGenerators(), P.FakeEmbed(), FakeMOS(), P.FakeASR(), folder, tag="selftest FAKE")
        d.setup_sentences(fake=True)
        d.compute()
        sid = next(iter(d.sent))
        ok(len(d.groups[(sid, "qwen3")]) == N and len(d.groups.get((CFG["f5"]["sentence"], "f5"), [])) in (0, N), f"fake：{sid} 一組 {N} 段")
        st = d.stats(sid, "qwen3")
        ok(len(st["tail"]) >= 3 and st["judges"]["wer"]["advantage"] is not None, f"fake：做壞的三段被抓到（長尾 {[t['i'] + 1 for t in st['tail']]}）")
        ok(any("dropped" in w for t in st["tail"] for w in t["why"]) and any("extra" in w for t in st["tail"] for w in t["why"]), "fake：漏一截算 D、多一截算 I")
        ok(any("runs long" in w for t in st["tail"] for w in t["why"]), "fake：停不下來的那一段長度離群")
        d.make_fig()
        for k in "glawsmhpppp1=otz?fgafrz":
            d.dispatch(k)
        d.dispatch("n"); d.dispatch("g"); d.dispatch("a")
        ok(len(d.groups[(d.sid, "live")]) == N and d.items[d.groups[(d.sid, "live")][0]]["seed"] == N + 1, f"fake：現場再抽一組，seed 從 {N + 1} 起")
        d.rehearse_figs(); d.save()
        ok((folder / f"table_{sid}_qwen3.png").exists() and (folder / f"scatter_{sid}_qwen3_wer.png").exists() and (folder / "session.json").exists(), "fake：圖與紀錄存了")
        txt = (folder / "session.json").read_text(encoding="utf-8")
        ok("/Users/" not in txt and str(HERE) not in txt, "紀錄裡沒有絕對路徑")
        keep, METRICS_JSON = METRICS_JSON, folder / "metrics.json"
        export(folder / "session.json")
        mj = json.loads(METRICS_JSON.read_text(encoding="utf-8"))["demo2"]
        ok(f"{sid}/qwen3" in mj["groups"] and len(mj["groups"][f"{sid}/qwen3"]["draws"]) == N, "export：demo2.groups")
        METRICS_JSON = keep
        shutil.rmtree(folder, ignore_errors=True)
    # 6. 有彩排紀錄：對 w09-metrics.json、replay 讀回
    rec_path = REHEARSAL / "rehearsal.json"
    if rec_path.exists() and METRICS_JSON.exists():
        rec = json.loads(rec_path.read_text(encoding="utf-8"))
        mj = json.loads(METRICS_JSON.read_text(encoding="utf-8")).get("demo2", {})
        bad = [k for k, names in rec["groups"].items() if [rec["items"][n]["mos"] for n in names] != [x["mos"] for x in mj.get("groups", {}).get(k, {}).get("draws", [])]]
        ok(not bad, f"彩排紀錄對 w09-metrics.json → demo2：{len(bad)} 組不一致")
        d = build_replay(P.Out())
        d.make_fig()
        for k in "glawsmhpppp1z":
            d.dispatch(k)
        ok(sum(len(v) for v in d.groups.values()) > 0, "replay：讀回、鍵都走得過")
    else:
        print("  --   沒有彩排紀錄或 w09-metrics.json 裡沒有 demo2，跳過比對")
    print(f"\n  {'全部通過' if not fails else str(len(fails)) + ' 項失敗'}")
    sys.exit(1 if fails else 0)


# ── main ─────────────────────────────────────────────────────
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["show", "rehearse", "listen", "replay", "check", "fetch", "fake", "selftest", "export"])
    ap.add_argument("which", nargs="?", help="listen：en 或 zh")
    ap.add_argument("--device", help="show／rehearse／check：蓋過 demo_config.toml 的 device（cpu／mps／auto）")
    ap.add_argument("--no-plot", action="store_true", help="show／replay／fake：只在終端機，不開圖")
    ap.add_argument("--no-mlx", action="store_true", help="show／rehearse：不開 MLX worker（沒有 Qwen3-TTS 那兩組）")
    ap.add_argument("--no-f5", action="store_true", help="rehearse：不做 F5-TTS 的對照組")
    ap.add_argument("--fresh", action="store_true", help="show：不讀彩排，現場從 seed 1 抽起")
    ap.add_argument("--real-judges", action="store_true", help="fake：生成器是替身，三位裁判是真的")
    ap.add_argument("--n", type=int, help="一組幾段（預設讀 demo_config.toml；試跑用小一點的）")
    ap.add_argument("--nfe", type=int, help="rehearse：F5 的 NFE（預設 32；CPU 上試跑用小一點的）")
    args = ap.parse_args()

    if args.cmd == "check":
        check(args.device); return
    if args.cmd == "fetch":
        fetch(); return
    if args.cmd == "selftest":
        selftest(); return
    if args.cmd == "export":
        export(REHEARSAL / "rehearsal.json"); return
    if args.cmd == "listen":
        listen(args.which or SIDS[0]); return

    if args.cmd == "rehearse" or args.no_plot:
        import matplotlib
        matplotlib.use("Agg")

    out = P.Out()
    out.banner("W9 Demo 2：one sentence, twelve draws, three judges")
    d = build_replay(out) if args.cmd == "replay" else build(args.cmd, out, args)

    if args.cmd == "rehearse":
        C.play = lambda *a, **k: None
        for (sid, gen) in list(d.groups):
            d.sid, d.gen = sid, gen
            d.show_adv(); d.show_tail(); d.mode = "table"; d.show_scatter(); d.summary()
        d.rehearse_figs()
        d.gens.close()
        d.save()
        export(d.folder / "rehearsal.json")
        print("\n下一步：./present.sh listen en；./present.sh listen zh（不看編號挑一段）→ ./present.sh export → ./present.sh 自己按一遍 g l a w s m h p z。")
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
        if d.gens is not None:
            d.gens.close()
        if args.cmd in ("show", "fake"):
            d.save()


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
# 此檔由 bin/publish_demos.py 從 private repo 的 <course>/demos/ 複製而來，請勿在公開 repo 直接編輯。
"""W10 demo 2：增強之後 WER 往哪裡走——同一批句子、五個 SNR、幾種前端、兩個辨識器，再加一位偵測器。

  show       課堂：讀回彩排算好的數字與聲音，載入模型（給鍵 l 現場重算用），單鍵操作（下面的鍵）
  rehearse   課前：全部算一遍 → runs/rehearsal/（進版控 = 現場的退路）；可以中斷，再跑一次會從算到的地方接下去
             --noise white,babble  只算這幾種噪音     --no-extra  不跑加映的前端     --fresh  不接續、從頭算
             --n 12                只用前 12 句（預覽；寫到 runs/preview/，不碰彩排紀錄）
  replay     現場模型出狀況時：只讀彩排紀錄，同樣的鍵（除了 l），不碰 torch；--from runs/preview 讀別的資料夾
  check      課前體檢：版本、模型在不在快取、MetricGAN+ 自己重寫的那幾行對 SpeechBrain 的 enhance_batch、
             各模型在 cpu／mps 上跑不跑與要多久、pystoi／pesq 裝了沒 → runs/check.json
  fetch      課前一週：下載 MetricGAN+ 與 SepFormer 的權重（commit 寫進 model.lock）；確認前幾週的模型與 40 句都在（要網路）
  fake       沒有模型也能演練流程（替身；畫面標明 FAKE，數字沒有意義）
  selftest   不需要模型：加噪音的 SNR、三種噪音、兩次投影、observation adding 的單調性、Wiener、fake 整條流程
  export     runs/rehearsal/ 的摘要 → slides/assets/w10/w10-metrics.json（鍵 demo2）

畫面上的鍵（順序照投影片 p49、p55 講稿的【操作】）：
  0  播乾淨的原句        1  播同一句的三個版本：帶噪的 → 統計式增強 → MetricGAN+      u  統計式、不平滑（musical noise）
  x  加映的前端（SepFormer）  s  換要聽的 SNR      n  換要聽的句子      b  換噪音的種類
  2  CTC 模型：WER 對 SNR，每個前端一條線            3  Whisper：同一張圖
  4  同一句增強過的聲音，兩個辨識器的逐字稿並排；這個 SNR 上兩個辨識器的錯誤種類（替換／刪除／插入）
  w  最糟的一句：MetricGAN+ 之後 Whisper 錯最多的那一句
  5  散點圖：每個（前端，SNR）一個點，橫軸 SAR、縱軸 WER
  6  observation adding：ŝ + ω·y，掃 ω——SNR、SAR 與兩個 WER
  7  同一批聲音送進 W9 Demo 3 的偵測器：被叫成 spoof 的比例
  m  每一位裁判各說什麼：SI-SDR、SNR、SAR、STOI、PESQ 與兩個 WER 排成一張表
  e  只做增強、不加噪音：乾淨的句子過每個前端
  l  現場重算：這一句重新抽一段噪音，三個前端、兩個辨識器（show 才有）
  t  投影片上那兩張表     ?  選單     q  離開

定義（課堂上要念出來的）：
  SNR（輸入） 整句的功率比：10 log10(Σ s² / Σ v²)。語音不動，只調噪音的音量。五個 SNR 用同一段噪音。
  WER         corpus WER（詞級 Levenshtein，分母是全部參考詞數）。參考與兩個辨識器的輸出都過 Whisper 的 EnglishTextNormalizer。
  SDR／SNR／SAR  把前端的輸出 ŝ 投影到 s、再投影到 s 與 v 張成的平面（enhance.decompose）；不含延遲版本的子空間。
  spoof       AASIST 的 bona fide logit − spoof logit 小於 0（W9 Demo 3 的判準，原封不動）。
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
import sys
import time
import tomllib
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
import common as C   # noqa: E402
import enhance as E  # noqa: E402

CFG = tomllib.loads((HERE / "demo_config.toml").read_text(encoding="utf-8"))
FS = int(CFG["audio"]["fs"])
NOISE, STAT, OA, LISTEN = CFG["noise"], CFG["stat"], CFG["oa"], CFG["listen"]
SNRS = [int(v) for v in NOISE["snrs"]]
MODEL_LOCK = HERE / "model.lock"
REHEARSAL = HERE / "runs" / "rehearsal"
PREVIEW = HERE / "runs" / "preview"
METRICS_JSON = HERE.parents[1] / "slides" / "assets" / "w10" / "w10-metrics.json"     # parents[1] = speech_ai/
DIG = int(CFG["display"]["digits"])
FE_LABEL = {"none": "none", "stat": "statistical", "mgan": CFG["neural"]["label"], "sep": CFG["extra"]["label"]}
FE_COLOR = {"none": "#6b7280", "stat": "#c8742a", "mgan": "#5b3fa0", "sep": "#2a7fc8"}
FE_MARK = {"none": "o", "stat": "s", "mgan": "^", "sep": "D"}
REC_LABEL = {"ctc": "CTC, clean-trained (wav2vec2-base-960h)", "whisper": "Whisper small"}
REC_COLOR = {"ctc": "#2a9d4b", "whisper": "#c0392b"}
NOISE_LABEL = {"white": "white noise", "pink": "pink noise", "babble": f"babble ({NOISE['babble_k']} other talkers)"}
PATTERNS = ["*.json", "*.txt", "*.safetensors"]


# ── 小工具 ──────────────────────────────────────────────────────────────
def load_sibling(rel: str, name: str):
    p = (HERE / rel / "demo.py").resolve()
    if not p.exists():
        sys.exit(f"找不到 {os.path.relpath(p, HERE)}（demo_config.toml 指過去的那個 demo）")
    spec = importlib.util.spec_from_file_location(name, p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


W2 = load_sibling(CFG["sentences"]["from"], "w02_d3_demo")     # 句子、Whisper、WER 的三個函數


def key_of(noise: str, snr: int, fe: str, omega: float = 0.0) -> str:
    return f"{noise}|{snr:+d}|{fe}" + (f"|oa{omega:g}" if omega else "")


def fmt(v, nd: int = DIG, inf: str = "∞") -> str:
    return inf if v is None else f"{v:.{nd}f}"


def pct(v) -> str:
    return "—" if v is None else f"{100 * v:.1f}"


def mean(vs) -> float | None:
    vs = [v for v in vs if v is not None]
    return float(np.mean(vs)) if vs else None


def spearman(a: list[float], b: list[float]) -> float | None:
    if len(a) < 3:
        return None
    ra, rb = np.argsort(np.argsort(a)).astype(float), np.argsort(np.argsort(b)).astype(float)
    if ra.std() == 0 or rb.std() == 0:
        return None
    return float(np.corrcoef(ra, rb)[0, 1])


def read_lock() -> dict[str, str]:
    lock = {}
    if MODEL_LOCK.exists():
        for line in MODEL_LOCK.read_text(encoding="utf-8").splitlines():
            parts = line.split()
            if len(parts) == 2 and not line.startswith("#"):
                lock[parts[0]] = parts[1]
    repo = CFG["ctc"]["repo"]
    for rel in CFG["ctc"]["lock_from"]:
        p = (HERE / rel).resolve()
        if repo in lock or not p.exists():
            continue
        for line in p.read_text(encoding="utf-8").splitlines():
            parts = line.split()
            if len(parts) == 2 and parts[0] == repo:
                lock[repo] = parts[1]
            elif len(parts) == 1 and len(parts[0]) == 40:
                lock[repo] = parts[0]
    return lock


def model_dir(repo: str, patterns: list[str] | None = None, offline: bool = True) -> str:
    from huggingface_hub import snapshot_download
    return snapshot_download(repo, revision=read_lock().get(repo), local_files_only=offline, allow_patterns=patterns)


def resolve_device(d: str) -> str:
    import torch
    if d == "auto":
        return "mps" if torch.backends.mps.is_available() else "cpu"
    return d


def config_digest(ids: list[str], models: dict) -> str:
    blob = json.dumps({"ids": ids, "noise": {k: NOISE[k] for k in ("seed", "babble_k")}, "stat": STAT, "models": models}, sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


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


# ── 模型 ────────────────────────────────────────────────────────────────
class MetricGAN:
    """SpeechBrain 的 SpectralMaskEnhancement。enhance() 是它的 enhance_batch 那幾行，差別只有最後不把峰值正規化成 1
    （正規化之後 ŝ 與 y 不在同一個音量上，ŝ + ω·y 就沒有意義）。check 會對兩者。"""

    def __init__(self, device: str = "cpu"):
        import torch
        from speechbrain.inference.enhancement import SpectralMaskEnhancement
        self.torch = torch
        path = model_dir(CFG["neural"]["repo"])
        self.m = SpectralMaskEnhancement.from_hparams(source=path, savedir=path, run_opts={"device": device})
        self.device = device
        self.n_params = sum(p.numel() for p in self.m.mods.enhance_model.parameters())

    def enhance(self, y: np.ndarray) -> np.ndarray:
        torch, m = self.torch, self.m
        noisy = torch.tensor(y[None], dtype=torch.float32, device=self.device)
        with torch.no_grad():
            feats = m.compute_features(noisy)                               # log(1 + |Y|)
            mask = m.mods.enhance_model(feats, lengths=torch.tensor([1.0]))
            mag = torch.expm1(mask * feats)
            wav = m.hparams.resynth(mag, noisy, normalize_wavs=False)       # 帶噪的相位
        return wav[0].cpu().numpy().astype(np.float64)[: len(y)]

    def enhance_official(self, y: np.ndarray) -> np.ndarray:
        torch = self.torch
        with torch.no_grad():
            w = self.m.enhance_batch(torch.tensor(y[None], dtype=torch.float32), lengths=torch.tensor([1.0]))
        return w[0].cpu().numpy().astype(np.float64)


class SepFormer:
    def __init__(self, device: str = "auto"):
        import torch
        from speechbrain.inference.separation import SepformerSeparation
        self.torch = torch
        self.device = resolve_device(device)
        path = model_dir(CFG["extra"]["repo"])
        self.m = SepformerSeparation.from_hparams(source=path, savedir=path, run_opts={"device": self.device})
        self.n_params = sum(p.numel() for p in self.m.mods.parameters())

    def enhance(self, y: np.ndarray) -> np.ndarray:
        torch = self.torch
        with torch.no_grad():
            w = self.m.separate_batch(torch.tensor(y[None], dtype=torch.float32, device=self.device))
        est = w[0, :, 0].float().cpu().numpy().astype(np.float64)[: len(y)]
        return est * E.blind_gain(est, y)


class CTC:
    """transformers 的 Wav2Vec2ForCTC，greedy。"""

    def __init__(self, device: str = "auto"):
        import torch
        from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
        W2.quiet_transformers()
        self.torch = torch
        path = model_dir(CFG["ctc"]["repo"], PATTERNS)
        self.proc = Wav2Vec2Processor.from_pretrained(path)
        self.model = Wav2Vec2ForCTC.from_pretrained(path).eval()
        self.device = resolve_device(device)
        self.model.to(self.device)
        self.n_params = sum(p.numel() for p in self.model.parameters())

    def transcribe(self, x: np.ndarray) -> str:
        torch = self.torch
        inp = self.proc(x.astype(np.float32), sampling_rate=FS, return_tensors="pt")
        with torch.no_grad():
            logits = self.model(inp.input_values.to(self.device)).logits[0].float().cpu()
        return self.proc.decode(logits.argmax(-1))


class Real:
    """真的模型，用到才載。"""
    fake = False

    def __init__(self, device: str | None = None, extra: bool = True):
        self.device = device
        self.want_extra = extra and bool(CFG["extra"]["enabled"])
        self._m: dict = {}
        self.times: dict = {}

    def _get(self, name: str, make):
        if name not in self._m:
            t0 = time.perf_counter()
            self._m[name] = make()
            self.times[name] = time.perf_counter() - t0
            C.note(f"  載入 {name}：{self.times[name]:.1f} s")
        return self._m[name]

    @property
    def wh(self):
        return self._get("whisper", W2.Whisper)

    def load_all(self) -> None:
        _ = self.wh
        self._get("ctc", lambda: CTC(self.device or CFG["ctc"]["device"]))
        self._get("mgan", lambda: MetricGAN(CFG["neural"]["device"]))

    def normalize(self, text: str) -> str:
        return self.wh.normalize(text)

    def neural(self, y: np.ndarray) -> np.ndarray:
        return self._get("mgan", lambda: MetricGAN(CFG["neural"]["device"])).enhance(y)

    def extra(self, y: np.ndarray) -> np.ndarray:
        return self._get("sep", lambda: SepFormer(self.device or CFG["extra"]["device"])).enhance(y)

    def ctc(self, x: np.ndarray, i: int, s: np.ndarray) -> str:
        return self._get("ctc", lambda: CTC(self.device or CFG["ctc"]["device"])).transcribe(x)

    def whisper(self, x: np.ndarray, i: int, s: np.ndarray) -> tuple[str, float]:
        r = self.wh.transcribe(self.wh.features_hf(x))
        return r["text"], r["avg_logprob"]

    def spoof(self, x: np.ndarray) -> float:
        def make():
            w9 = load_sibling(CFG["spoof"]["from"], "w09_d3_demo")
            return w9.AASIST(CFG["spoof"]["device"])
        return self._get("aasist", make).score(x)["s"]

    def info(self) -> dict:
        lock = read_lock()
        d = {"ctc": {"repo": CFG["ctc"]["repo"], "commit": lock.get(CFG["ctc"]["repo"])},
             "whisper": {"repo": W2.CFG["model"]["repo"], "commit": W2.read_lock(), "device": W2.CFG["model"]["device"]},
             "mgan": {"repo": CFG["neural"]["repo"], "commit": lock.get(CFG["neural"]["repo"])},
             "spoof": {"from": CFG["spoof"]["from"], "threshold": 0.0}}
        if self.want_extra:
            d["sep"] = {"repo": CFG["extra"]["repo"], "commit": lock.get(CFG["extra"]["repo"])}
        for k in ("ctc", "mgan", "sep"):
            if k in self._m and k in d:
                d[k]["device"], d[k]["n_params"] = self._m[k].device, self._m[k].n_params
        return d


class Fake:
    """沒有模型時的替身：前端是另一組參數的 Wiener；辨識器依訊號品質刪字或換字。數字沒有意義。"""
    fake = True
    want_extra = True

    def __init__(self):
        self.refs: list[str] = []
        self.times = {}

    def load_all(self) -> None:
        pass

    def normalize(self, text: str) -> str:
        return W2.norm_simple(text)

    def neural(self, y: np.ndarray) -> np.ndarray:
        return E.wiener_dd(y, 512, 128, 512, 0.9, 0.03, 0.5)

    def extra(self, y: np.ndarray) -> np.ndarray:
        return E.wiener_dd(y, 512, 128, 512, 0.95, 0.2, 0.4)

    def _words(self, x: np.ndarray, i: int, s: np.ndarray, salt: int) -> tuple[list[str], np.ndarray, np.random.Generator]:
        q = E.si_sdr(x[: len(s)], s)
        p = 1.0 / (1.0 + math.exp((q - 4.0) / 3.0))
        rng = np.random.default_rng([int(CFG["fake"]["seed"]), i, salt, int(round(q * 10)) + 1000])
        words = self.refs[i].split()
        return words, rng.random(len(words)) < p, rng

    def ctc(self, x: np.ndarray, i: int, s: np.ndarray) -> str:
        words, hit, _ = self._words(x, i, s, 1)
        return " ".join(w for w, h in zip(words, hit) if not h)

    def whisper(self, x: np.ndarray, i: int, s: np.ndarray) -> tuple[str, float]:
        words, hit, _ = self._words(x, i, s, 2)
        return " ".join(w[::-1] if h else w for w, h in zip(words, hit)), -0.2 - float(hit.mean())

    def spoof(self, x: np.ndarray) -> float:
        return float(np.random.default_rng(int(abs(float(np.sum(x))) * 1e6) % (2 ** 32)).normal(1.0, 2.5))

    def info(self) -> dict:
        return {"fake": True}


# ── 計算 ────────────────────────────────────────────────────────────────
class Lab:
    """句子、噪音、前端、裁判。一個「條件」= (噪音種類, SNR, 前端, ω)，對全部句子算一次。"""

    def __init__(self, backend, sents: list[dict], xs: list[np.ndarray], out: Out):
        self.b, self.sents, self.xs, self.out = backend, sents, xs, out
        self.refs = [backend.normalize(s["text"]) for s in sents]
        if backend.fake:
            backend.refs = self.refs
        self._stoi = self._pesq = None
        if CFG["metrics"]["stoi"]:
            try:
                from pystoi import stoi
                self._stoi = stoi
            except Exception:
                pass
        if CFG["metrics"]["pesq"]:
            try:
                from pesq import pesq
                self._pesq = pesq
            except Exception:
                pass

    def noise(self, i: int, kind: str, salt: int = 0) -> np.ndarray:
        rng = np.random.default_rng([int(NOISE["seed"]), i, NOISE["types"].index(kind), salt])
        n = len(self.xs[i])
        if kind == "white":
            return E.white(n, rng)
        if kind == "pink":
            return E.pink(n, rng)
        spk = self.sents[i]["spk"]
        others = [x for x, s in zip(self.xs, self.sents) if s["spk"] != spk]
        return E.babble(n, others, int(NOISE["babble_k"]), rng)

    def frontend(self, fe: str, y: np.ndarray) -> np.ndarray:
        if fe == "none":
            return y
        if fe in ("stat", "stat_a0"):
            a = STAT["alpha"] if fe == "stat" else STAT["alpha_musical"]
            return E.wiener_dd(y, int(STAT["win"]), int(STAT["hop"]), int(STAT["n_fft"]), float(a), float(STAT["g_min"]), float(STAT["q"]))
        if fe == "mgan":
            return self.b.neural(y)
        if fe == "sep":
            return self.b.extra(y)
        raise KeyError(fe)

    def signals(self, i: int, noise: str | None, snr: int | None, fe: str, omega: float = 0.0, salt: int = 0):
        s = self.xs[i]
        if noise is None:
            y, gv = s, None
        else:
            y, gv = E.mix(s, self.noise(i, noise, salt), snr)
        est = self.frontend(fe, y)
        if omega:
            est = E.observation_add(est, y, omega)
        return s, gv, y, est

    def judge_signal(self, est: np.ndarray, s: np.ndarray, gv: np.ndarray | None) -> dict:
        r = {"si_sdr": E.si_sdr(est, s) if est is not s else None, "sdr": None, "snr": None, "sar": None, "sar_inf": False, "stoi": None, "pesq": None}
        if gv is not None:
            q = E.ratios(est, s, gv)
            r.update(sdr=q["sdr"], snr=q["snr"], sar=q["sar"], sar_inf=q["sar"] is None)
        if est is not s:
            if self._stoi is not None:
                try:
                    r["stoi"] = float(self._stoi(s, est, FS, extended=False))
                except Exception:
                    pass
            if self._pesq is not None:
                try:
                    r["pesq"] = float(self._pesq(FS, s.astype(np.float32), est.astype(np.float32), "wb"))
                except Exception:
                    pass
        return r

    def score_text(self, hyps: list[str]) -> dict:
        hyps = [self.b.normalize(h) for h in hyps]
        w = W2.corpus_wer(self.refs, hyps)
        errs = [W2.align_words(r.split(), h.split())["errors"] for r, h in zip(self.refs, hyps)]
        return {"wer": w["wer"], "S": w["S"], "D": w["D"], "I": w["I"], "n_ref": w["n_ref"], "hyps": hyps, "errs": errs}

    def run(self, noise: str | None, snr: int | None, fe: str, omega: float = 0.0) -> dict:
        t0 = time.perf_counter()
        sig, ctc_h, wh_h, wh_lp, sp = [], [], [], [], []
        tt = {"front": 0.0, "ctc": 0.0, "whisper": 0.0, "spoof": 0.0, "signal": 0.0}
        for i in range(len(self.xs)):
            t = time.perf_counter()
            s, gv, y, est = self.signals(i, noise, snr, fe, omega)
            tt["front"] += time.perf_counter() - t
            t = time.perf_counter()
            sig.append(self.judge_signal(est, s, gv))
            tt["signal"] += time.perf_counter() - t
            t = time.perf_counter()
            ctc_h.append(self.b.ctc(est, i, s))
            tt["ctc"] += time.perf_counter() - t
            t = time.perf_counter()
            h, lp = self.b.whisper(est, i, s)
            wh_h.append(h)
            wh_lp.append(lp)
            tt["whisper"] += time.perf_counter() - t
            t = time.perf_counter()
            sp.append(self.b.spoof(est))
            tt["spoof"] += time.perf_counter() - t
        n_inf = sum(q["sar_inf"] for q in sig)
        rec = {"noise": noise, "snr": snr, "fe": fe, "omega": omega, "n": len(self.xs),
               "sig": {k: mean(q[k] for q in sig) for k in ("si_sdr", "sdr", "snr", "sar", "stoi", "pesq")},
               "sar_inf": n_inf == len(sig) and noise is not None,
               "per": {k: [q[k] for q in sig] for k in ("si_sdr", "snr", "sar")},
               "ctc": self.score_text(ctc_h), "whisper": self.score_text(wh_h),
               "spoof": {"n_spoof": int(sum(v < 0 for v in sp)), "n": len(sp), "median": float(np.median(sp)), "scores": [float(v) for v in sp]},
               "seconds": {**{k: round(v, 2) for k, v in tt.items()}, "total": round(time.perf_counter() - t0, 2)}}
        rec["whisper"]["avg_logprob"] = float(np.mean(wh_lp))
        return rec


def plan(noises: list[str], extra: bool) -> list[tuple]:
    """算的順序：先是投影片上的表要的（main 那一種噪音），再是其他。每一項 (noise, snr, fe, omega)。"""
    main = NOISE["main"]
    order = [main] + [n for n in noises if n != main] if main in noises else list(noises)
    todo: list[tuple] = [(None, None, "none", 0.0)]
    for k, nz in enumerate(order):
        for fe in ("none", "stat", "mgan"):
            todo += [(nz, snr, fe, 0.0) for snr in SNRS]
        if nz == main:
            todo += [(nz, int(OA["snr"]), OA["fe"], float(w)) for w in OA["omegas"]]
            todo += [(None, None, fe, 0.0) for fe in ("stat", "mgan")]
            if extra:
                todo += [(nz, snr, "sep", 0.0) for snr in SNRS] + [(None, None, "sep", 0.0)]
    return todo


def clean_key(fe: str) -> str:
    return f"clean|{fe}"


def cond_key(noise, snr, fe, omega) -> str:
    return clean_key(fe) if noise is None else key_of(noise, snr, fe, omega)


def write_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    tmp.replace(path)


def compute_all(lab: Lab, folder: Path, noises: list[str], extra: bool, fresh: bool, tag: str, versions: dict) -> dict:
    out = lab.out
    ids = [s["id"] for s in lab.sents]
    models = lab.b.info()
    digest = config_digest(ids, {k: {kk: vv for kk, vv in v.items() if kk in ("repo", "commit")} if isinstance(v, dict) else v for k, v in models.items()})
    path = folder / "rehearsal.json"
    rec = None
    if path.exists() and not fresh:
        old = json.loads(path.read_text(encoding="utf-8"))
        if old.get("config_digest") == digest:
            rec = old
            out(f"  接續 {os.path.relpath(path, HERE)}：已有 {len(rec['conds'])} 個條件（要從頭算就加 --fresh）")
        else:
            out(f"  {C.AMBER}{os.path.relpath(path, HERE)} 是別的設定算的（句子、噪音的種子、前端的參數或模型不同）：從頭算{C.RESET}")
    if rec is None:
        rec = {"time": C.now(), "tag": tag, "config_digest": digest, "conds": {}, "listen": {}, "what_we_heard": ""}
    rec.update({"versions": versions, "models": models, "main_noise": NOISE["main"], "snrs": SNRS, "noise_types": NOISE["types"],
                "sentences": [{"id": s["id"], "spk": s["spk"], "ref": r} for s, r in zip(lab.sents, lab.refs)],
                "stat": dict(STAT), "oa": dict(OA), "metrics_available": {"stoi": lab._stoi is not None, "pesq": lab._pesq is not None}})
    todo = [t for t in plan(noises, extra) if cond_key(*t) not in rec["conds"]]
    out(f"  要算 {len(todo)} 個條件 × {len(lab.xs)} 句（每個條件：前端 → 訊號類指標 → 兩個辨識器 → 偵測器）")
    sep_broken = False
    for j, t in enumerate(todo):
        if t[2] == "sep" and sep_broken:
            continue
        try:
            c = lab.run(*t)
        except Exception as e:
            if t[2] != "sep":
                raise
            sep_broken = True
            out(f"  {C.AMBER}加映的前端跑不起來，跳過（{type(e).__name__}: {str(e)[:160]}）{C.RESET}")
            rec["skipped"] = [{"cond": "sep", "error": f"{type(e).__name__}: {str(e)[:300]}"}]
            continue
        rec["conds"][cond_key(*t)] = c
        rec["time"] = C.now()
        write_json(path, rec)
        out(f"  [{j + 1:>2}/{len(todo)}] {cond_key(*t):<24s} WER  CTC {pct(c['ctc']['wer']):>6s}  Whisper {pct(c['whisper']['wer']):>6s}   "
            f"SNR {fmt(c['sig']['snr'], inf='—'):>5s}  SAR {'∞' if c['sar_inf'] else fmt(c['sig']['sar'], inf='—'):>5s}   spoof {c['spoof']['n_spoof']}/{c['spoof']['n']}   {c['seconds']['total']:.0f} s")
    write_listen(lab, folder, rec, [n for n in NOISE["types"] if any(k.startswith(n + "|") for k in rec["conds"])], extra and any("|sep" in k for k in rec["conds"]))
    write_json(path, rec)
    return rec


def write_listen(lab: Lab, folder: Path, rec: dict, noises: list[str], extra: bool) -> None:
    """要聽的那幾句：每個（句子，噪音，SNR）一組 .wav，同一組用同一個增益。"""
    ids = [s["id"] for s in lab.sents]
    versions = ["none", "stat", "stat_a0", "mgan"] + (["sep"] if extra else [])
    rec["listen"] = {"sentences": [], "snrs": [int(v) for v in LISTEN["snrs"]], "versions": ["clean"] + versions, "files": {}}
    for sid in LISTEN["sentences"]:
        if sid not in ids:
            continue
        i = ids.index(sid)
        rec["listen"]["sentences"].append(sid)
        for nz in noises:
            for snr in LISTEN["snrs"]:
                group = {"clean": lab.xs[i]}
                for fe in versions:
                    if fe == "sep" and nz != NOISE["main"]:
                        continue
                    try:
                        group[fe] = lab.signals(i, nz, int(snr), fe)[3]
                    except Exception:
                        continue
                peak = max(float(np.max(np.abs(v))) for v in group.values())
                for name, x in group.items():
                    fn = f"listen_{sid}_{nz}_{int(snr):+d}_{name}.wav"
                    C.save_wav(folder / fn, x, FS, peak)
                    rec["listen"]["files"][f"{sid}|{nz}|{int(snr):+d}|{name}"] = fn


# ── 畫面 ────────────────────────────────────────────────────────────────
class Demo:
    def __init__(self, out: Out, rec: dict, folder: Path, lab: Lab | None = None, tag: str = ""):
        self.out, self.rec, self.folder, self.lab, self.tag = out, rec, folder, lab, tag or rec.get("tag", "")
        self.conds = rec["conds"]
        self.refs = [s["ref"] for s in rec["sentences"]]
        self.ids = [s["id"] for s in rec["sentences"]]
        self.noises = [n for n in rec["noise_types"] if any(k.startswith(n + "|") for k in self.conds)]
        self.noise = rec["main_noise"] if rec["main_noise"] in self.noises else (self.noises[0] if self.noises else rec["main_noise"])
        self.listen_ids = rec.get("listen", {}).get("sentences", []) or self.ids[:1]
        self.li = 0
        self.listen_snrs = rec.get("listen", {}).get("snrs", [SNRS[-1]])
        self.si = 0
        self.view, self.shown = "blank", set()
        self.fig = self.plt = None
        self.quiet = False
        self.live_n = 0

    # ── 取數 ──
    def c(self, snr: int, fe: str, omega: float = 0.0, noise: str | None = None) -> dict | None:
        return self.conds.get(key_of(noise or self.noise, snr, fe, omega))

    def fes(self, noise: str | None = None) -> list[str]:
        nz = noise or self.noise
        return [fe for fe in ("none", "stat", "mgan", "sep") if any(key_of(nz, s, fe) in self.conds for s in SNRS)]

    def clean(self, fe: str = "none") -> dict | None:
        return self.conds.get(clean_key(fe))

    @property
    def sid(self) -> str:
        return self.listen_ids[self.li % len(self.listen_ids)]

    @property
    def lsnr(self) -> int:
        return int(self.listen_snrs[self.si % len(self.listen_snrs)])

    def head(self, s: str) -> None:
        self.out.banner(f"{s}   ·   {NOISE_LABEL[self.noise]}" + (f"   [{self.tag}]" if self.tag else ""))

    # ── 聽 ──
    def play(self, version: str, say: str) -> None:
        fn = self.rec.get("listen", {}).get("files", {}).get(f"{self.sid}|{self.noise}|{self.lsnr:+d}|{version}")
        if fn is None or not (self.folder / fn).exists():
            self.out(f"  {C.AMBER}（沒有這一段：{version}，{self.noise} {self.lsnr:+d} dB）{C.RESET}")
            return
        self.out(f"  ▶ {say}")
        if not self.quiet:
            C.play(self.folder / fn)

    def listen(self) -> None:
        self.head(f"聽：{self.sid}，{self.lsnr:+d} dB——帶噪的 → 統計式 → {FE_LABEL['mgan']}")
        self.play("none", "帶噪的，原樣")
        self.play("stat", f"統計式增強（decision-directed Wiener，α = {STAT['alpha']:g}）")
        self.play("mgan", FE_LABEL["mgan"])

    # ── 表 ──
    def wer_table(self, which: str) -> None:
        self.head(f"WER (%) · {REC_LABEL[which]}")
        cl = self.clean()
        self.out(f"  {'front end':<18s}" + "".join(f"{s:>+8d} dB" for s in SNRS) + (f"      clean {pct(cl[which]['wer'])}" if cl else ""))
        for fe in self.fes():
            row = [self.c(s, fe) for s in SNRS]
            self.out(f"  {FE_LABEL[fe]:<18s}" + "".join(f"{pct(r[which]['wer']) if r else '—':>11s}" for r in row))
        best = []
        for s in SNRS:
            cand = [(self.c(s, fe)[which]["wer"], fe) for fe in self.fes() if self.c(s, fe)]
            best.append(FE_LABEL[min(cand)[1]] if cand else "—")
        self.out(f"  {C.GREY}{'lowest':<18s}" + "".join(f"{b[:10]:>11s}" for b in best) + C.RESET)

    def side_by_side(self, sid: str | None = None, snr: int | None = None) -> None:
        sid, snr = sid or self.sid, self.lsnr if snr is None else snr
        i = self.ids.index(sid)
        self.head(f"同一句、{snr:+d} dB：兩個辨識器各寫了什麼   （{sid}）")
        self.out(f"  {C.BOLD}ref{C.RESET}       {self.refs[i]}")
        for fe in self.fes():
            r = self.c(snr, fe)
            if not r:
                continue
            self.out(f"  {C.BOLD}{FE_LABEL[fe]}{C.RESET}")
            for which, name in (("ctc", "CTC"), ("whisper", "Whisper")):
                h = r[which]["hyps"][i]
                self.out(f"    {name:<8s}{W2.colored(self.refs[i], h) if h else C.GREY + '（空的）' + C.RESET}   {C.GREY}{r[which]['errs'][i]} errors{C.RESET}")
        self.out(f"\n  這個 SNR 上全部 {len(self.ids)} 句的錯誤種類（S 替換／D 刪除／I 插入，分母 {self.c(snr, 'none')['ctc']['n_ref'] if self.c(snr, 'none') else '—'} 個詞）：")
        self.out(f"    {'front end':<18s}{'CTC  S / D / I':>22s}{'Whisper  S / D / I':>26s}")
        for fe in self.fes():
            r = self.c(snr, fe)
            if r:
                a, b = r["ctc"], r["whisper"]
                self.out(f"    {FE_LABEL[fe]:<18s}{a['S']:>10d} /{a['D']:>4d} /{a['I']:>4d}{b['S']:>14d} /{b['D']:>4d} /{b['I']:>4d}")

    def worst(self) -> None:
        snr = min(SNRS)
        r = self.c(snr, "mgan")
        if not r:
            return
        i = int(np.argmax(r["whisper"]["errs"]))
        self.side_by_side(self.ids[i], snr)

    def scatter_table(self) -> None:
        self.head("SAR 對 WER：每個（前端，SNR）一個點")
        self.out(f"  {'front end':<18s}{'SNR in':>8s}{'SNR out':>9s}{'SAR':>8s}{'WER CTC':>10s}{'WER Whisper':>13s}")
        pts = {"ctc": ([], []), "whisper": ([], [])}
        for fe in self.fes():
            for s in SNRS:
                r = self.c(s, fe)
                if not r:
                    continue
                self.out(f"  {FE_LABEL[fe]:<18s}{s:>+8d}{fmt(r['sig']['snr']):>9s}{'∞' if r['sar_inf'] else fmt(r['sig']['sar']):>8s}{pct(r['ctc']['wer']):>10s}{pct(r['whisper']['wer']):>13s}")
                if fe != "none" and r["sig"]["sar"] is not None:
                    for w in pts:
                        pts[w][0].append(r["sig"]["sar"])
                        pts[w][1].append(r[w]["wer"])
        for w in pts:
            rho = spearman(*pts[w])
            self.out(f"  {C.GREY}Spearman（SAR 對 WER，增強過的 {len(pts[w][0])} 個點）· {REC_LABEL[w]}：{'—' if rho is None else f'{rho:+.2f}'}{C.RESET}")

    def oa_points(self) -> list[tuple[str, dict]]:
        snr, fe = int(OA["snr"]), OA["fe"]
        pts = [("0", self.c(snr, fe))] + [(f"{w:g}", self.c(snr, fe, float(w))) for w in OA["omegas"]] + [("none", self.c(snr, "none"))]
        return [(lab, r) for lab, r in pts if r]

    def oa_table(self) -> None:
        self.head(f"observation adding：ŝ + ω·y   （{int(OA['snr']):+d} dB，{FE_LABEL[OA['fe']]}）")
        pts = self.oa_points()
        self.out(f"  {'ω':<22s}" + "".join(f"{('no enh.' if lab == 'none' else lab):>10s}" for lab, _ in pts))
        self.out(f"  {'SNR (dB)':<22s}" + "".join(f"{fmt(r['sig']['snr']):>10s}" for _, r in pts))
        self.out(f"  {'SAR (dB)':<22s}" + "".join(f"{'∞' if r['sar_inf'] else fmt(r['sig']['sar']):>10s}" for _, r in pts))
        self.out(f"  {'WER % · CTC':<22s}" + "".join(f"{pct(r['ctc']['wer']):>10s}" for _, r in pts))
        self.out(f"  {'WER % · Whisper':<22s}" + "".join(f"{pct(r['whisper']['wer']):>10s}" for _, r in pts))
        self.out(f"  {'called spoof':<22s}" + "".join(f"{str(r['spoof']['n_spoof']) + '/' + str(r['spoof']['n']):>10s}" for _, r in pts))

    def spoof_table(self) -> None:
        self.head("W9 Demo 3 的偵測器（AASIST）：真人的聲音，被叫成 spoof 的句數")
        cl = self.clean()
        if cl:
            self.out(f"  乾淨的原句：{cl['spoof']['n_spoof']}/{cl['spoof']['n']}（中位數 {cl['spoof']['median']:+.2f}）")
        self.out(f"  {'front end':<18s}" + "".join(f"{s:>+8d} dB" for s in SNRS) + f"{'clean in':>12s}")
        for fe in self.fes():
            row = [self.c(s, fe) for s in SNRS]
            ce = self.clean(fe)
            self.out(f"  {FE_LABEL[fe]:<18s}" + "".join(f"{(str(r['spoof']['n_spoof']) + '/' + str(r['spoof']['n'])) if r else '—':>11s}" for r in row)
                     + f"{(str(ce['spoof']['n_spoof']) + '/' + str(ce['spoof']['n'])) if ce else '—':>12s}")

    def judges(self) -> None:
        self.head("每一位裁判各說什麼")
        self.out(f"  {'front end':<18s}{'SNR in':>7s}{'SI-SDR':>8s}{'SNR':>7s}{'SAR':>7s}{'STOI':>7s}{'PESQ':>7s}{'WER CTC':>9s}{'Whisper':>9s}")
        for s in SNRS:
            for fe in self.fes():
                r = self.c(s, fe)
                if not r:
                    continue
                g = r["sig"]
                self.out(f"  {FE_LABEL[fe]:<18s}{s:>+7d}{fmt(g['si_sdr']):>8s}{fmt(g['snr']):>7s}{'∞' if r['sar_inf'] else fmt(g['sar']):>7s}"
                         f"{fmt(g['stoi'], 2, '—'):>7s}{fmt(g['pesq'], 2, '—'):>7s}{pct(r['ctc']['wer']):>9s}{pct(r['whisper']['wer']):>9s}")
            self.out("")
        av = self.rec.get("metrics_available", {})
        if not av.get("pesq") or not av.get("stoi"):
            self.out(f"  {C.GREY}（{'、'.join(k.upper() for k in ('stoi', 'pesq') if not av.get(k))} 沒裝，那一欄是空的：./setup.sh）{C.RESET}")

    def clean_only(self) -> None:
        self.head("只做增強、不加噪音：乾淨的句子過每個前端")
        self.out(f"  {'front end':<18s}{'SI-SDR':>8s}{'WER CTC':>9s}{'Whisper':>9s}{'spoof':>9s}")
        for fe in ("none", "stat", "mgan", "sep"):
            r = self.clean(fe)
            if r:
                self.out(f"  {FE_LABEL[fe]:<18s}{fmt(r['sig']['si_sdr'], inf='—'):>8s}{pct(r['ctc']['wer']):>9s}{pct(r['whisper']['wer']):>9s}{str(r['spoof']['n_spoof']) + '/' + str(r['spoof']['n']):>9s}")

    def slide_tables(self) -> None:
        self.head("投影片 p49 的表：WER (%)")
        self.out(f"  {'recogniser':<26s}{'front end':<16s}" + "".join(f"{s:>+7d}" for s in SNRS))
        for which in ("ctc", "whisper"):
            for fe in ("none", "stat", "mgan"):
                row = [self.c(s, fe) for s in SNRS]
                self.out(f"  {(REC_LABEL[which][:24] if fe == 'none' else ''):<26s}{FE_LABEL[fe]:<16s}" + "".join(f"{pct(r[which]['wer']) if r else '—':>7s}" for r in row))
        snr, fe = int(OA["snr"]), OA["fe"]
        self.out(f"\n  投影片 p55 的表：{snr:+d} dB，{FE_LABEL[fe]}")
        cols = [(f"ω = {w:g}", self.c(snr, fe, float(w))) for w in OA["table"]] + [("no enh.", self.c(snr, "none"))]
        self.out(f"  {'':<34s}" + "".join(f"{lab:>14s}" for lab, _ in cols))
        self.out(f"  {'SNR, SAR (dB)':<34s}" + "".join(f"{(fmt(r['sig']['snr']) + ', ' + ('∞' if r['sar_inf'] else fmt(r['sig']['sar']))) if r else '—':>14s}" for _, r in cols))
        self.out(f"  {'WER — CTC; Whisper':<34s}" + "".join(f"{(pct(r['ctc']['wer']) + '; ' + pct(r['whisper']['wer'])) if r else '—':>14s}" for _, r in cols))
        self.out(f"  {'real speech called spoof':<34s}" + "".join(f"{(str(r['spoof']['n_spoof']) + '/' + str(r['spoof']['n'])) if r else '—':>14s}" for _, r in cols))
        cl = self.clean()
        if cl:
            self.out(f"  {C.GREY}乾淨的原句：WER CTC {pct(cl['ctc']['wer'])}、Whisper {pct(cl['whisper']['wer'])}；被叫成 spoof {cl['spoof']['n_spoof']}/{cl['spoof']['n']}{C.RESET}")

    # ── 現場重算 ──
    def live(self) -> None:
        if self.lab is None:
            self.out(f"  {C.AMBER}（replay 沒有模型：現場重算只在 show）{C.RESET}")
            return
        self.live_n += 1
        salt = int(time.time()) % 100000 + self.live_n
        i = self.ids.index(self.sid)
        self.head(f"現場重算：{self.sid}，{self.lsnr:+d} dB，重新抽一段噪音（salt {salt}）")
        live_dir = HERE / "runs" / "live"
        self.out(f"  {C.BOLD}ref{C.RESET}       {self.refs[i]}")
        for fe in ("none", "stat", "mgan"):
            s, gv, y, est = self.lab.signals(i, self.noise, self.lsnr, fe, salt=salt)
            q = E.ratios(est, s, gv)
            hc = self.lab.b.normalize(self.lab.b.ctc(est, i, s))
            hw = self.lab.b.normalize(self.lab.b.whisper(est, i, s)[0])
            sp = self.lab.b.spoof(est)
            self.out(f"  {C.BOLD}{FE_LABEL[fe]}{C.RESET}   SNR {fmt(q['snr'])}  SAR {fmt(q['sar'])}  偵測器 {sp:+.2f}（{'spoof' if sp < 0 else 'bona fide'}）")
            self.out(f"    {'CTC':<8s}{W2.colored(self.refs[i], hc) if hc else '（空的）'}")
            self.out(f"    {'Whisper':<8s}{W2.colored(self.refs[i], hw) if hw else '（空的）'}")
            p = live_dir / f"live_{fe}.wav"
            C.save_wav(p, est, FS, float(np.max(np.abs(y))) * 1.05)
            if not self.quiet:
                C.play(p)

    # ── 圖 ──
    def make_fig(self):
        import matplotlib.pyplot as plt
        for k in ("save", "quit", "fullscreen", "pan", "yscale", "xscale", "grid", "back", "forward", "home", "zoom"):
            plt.rcParams[f"keymap.{k}"] = []
        self.plt = plt
        self.fig = plt.figure(figsize=(CFG["display"]["fig_w"], CFG["display"]["fig_h"]))
        self.draw()
        return self.fig

    def redraw(self) -> None:
        if self.fig is not None:
            self.draw()
            self.fig.canvas.draw_idle()

    def draw(self) -> None:
        f = self.fig
        f.clear()
        title = {"blank": "Demo 2 — after enhancement, where does WER go?", "wer": "WER against SNR, one line per front end",
                 "scatter": "SAR against WER — one point per (front end, SNR)", "oa": f"Observation adding at {int(OA['snr']):+d} dB, {FE_LABEL[OA['fe']]}:  ŝ + ω·y",
                 "spoof": "A second judge: AASIST (W9 Demo 3) on real speech"}[self.view]
        f.suptitle(f"{title}   ·   {NOISE_LABEL[self.noise]}" + (f"   [{self.tag}]" if self.tag else ""), fontsize=13)
        getattr(self, f"_draw_{self.view}")()
        f.tight_layout(rect=(0, 0, 1, 0.94))

    def _draw_blank(self) -> None:
        ax = self.fig.add_subplot(111)
        ax.axis("off")
        ax.text(0.5, 0.5, "listen first  (keys 0, 1)", ha="center", va="center", fontsize=20, color="#9aa0a6")

    def _draw_wer(self) -> None:
        xs = list(range(len(SNRS)))[::-1]                     # 右邊比較乾淨
        top = 0.0
        axes = []
        for j, which in enumerate(("ctc", "whisper")):
            ax = self.fig.add_subplot(1, 2, j + 1)
            axes.append(ax)
            ax.set_title(REC_LABEL[which], fontsize=12)
            ax.set_xticks(xs, [f"{s:+d}" for s in SNRS])
            ax.set_xlabel("input SNR (dB)")
            ax.set_ylabel("WER (%)")
            ax.grid(alpha=0.25)
            if which not in self.shown:
                continue
            for fe in self.fes():
                ys = [100 * self.c(s, fe)[which]["wer"] if self.c(s, fe) else np.nan for s in SNRS]
                ax.plot(xs, ys, marker=FE_MARK[fe], color=FE_COLOR[fe], lw=2.2 if fe != "sep" else 1.6, ls="-" if fe != "sep" else "--", label=FE_LABEL[fe])
                top = max(top, np.nanmax(ys))
            cl = self.clean()
            if cl:
                ax.axhline(100 * cl[which]["wer"], color="#9aa0a6", ls=":", lw=1.2, label="clean speech")
            ax.legend(fontsize=10, loc="upper right")
        for ax in axes:
            ax.set_ylim(0, min(max(top * 1.08, 10), 160))

    def _draw_scatter(self) -> None:
        ax = self.fig.add_subplot(111)
        sars = [r["sig"]["sar"] for k, r in self.conds.items() if k.startswith(self.noise + "|") and r["sig"]["sar"] is not None and not r["omega"]]
        x_inf = (max(sars) if sars else 20) + 6
        for which in ("ctc", "whisper"):
            for fe in self.fes():
                for s in SNRS:
                    r = self.c(s, fe)
                    if not r:
                        continue
                    x = x_inf if r["sar_inf"] else r["sig"]["sar"]
                    ax.scatter([x], [100 * r[which]["wer"]], marker=FE_MARK[fe], s=70, color=REC_COLOR[which], facecolors="none" if fe == "none" else REC_COLOR[which], zorder=3)
                    ax.annotate(f"{s:+d}", (x, 100 * r[which]["wer"]), textcoords="offset points", xytext=(6, 4), fontsize=8, color="#6b7280")
        ax.axvline(x_inf - 3, color="#d1d5db", lw=1)
        ticks = [v for v in ax.get_xticks() if ax.get_xlim()[0] <= v < x_inf - 3]
        ax.set_xticks(ticks + [x_inf], [f"{v:g}" for v in ticks] + ["∞\n(no front end)"])
        ax.set_xlabel("SAR (dB)   — lower = more artifacts")
        ax.set_ylabel("WER (%)")
        ax.grid(alpha=0.25)
        from matplotlib.lines import Line2D
        hs = [Line2D([], [], marker="o", ls="", color=REC_COLOR[w], label=REC_LABEL[w]) for w in REC_COLOR]
        hs += [Line2D([], [], marker=FE_MARK[fe], ls="", color="#374151", label=FE_LABEL[fe]) for fe in self.fes()]
        ax.legend(handles=hs, fontsize=10, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=len(hs), frameon=False, handletextpad=0.2, columnspacing=1.2)

    def _draw_oa(self) -> None:
        pts = self.oa_points()
        if not pts:
            return self._draw_blank()
        xs = list(range(len(pts)))
        ax = self.fig.add_subplot(111)
        ax.set_xticks(xs, [("no\nenhancement" if lab == "none" else f"ω = {lab}") for lab, _ in pts])
        sar = [np.nan if r["sar_inf"] else r["sig"]["sar"] for _, r in pts]
        ax.plot(xs, sar, marker="o", color="#5b3fa0", lw=2.2, label="SAR (dB)")
        ax.plot(xs, [r["sig"]["snr"] for _, r in pts], marker="s", color="#c8742a", lw=2.2, label="SNR (dB)")
        ax.annotate("SAR = ∞", (xs[-1], np.nanmax(sar)), textcoords="offset points", xytext=(-8, 8), ha="right", fontsize=9, color="#5b3fa0")
        ax.set_ylabel("dB")
        ax.grid(alpha=0.25)
        ax2 = ax.twinx()
        for which in ("ctc", "whisper"):
            ax2.plot(xs, [100 * r[which]["wer"] for _, r in pts], marker="^", color=REC_COLOR[which], lw=2.2, ls="--", label=f"WER · {REC_LABEL[which]}")
        ax2.set_ylabel("WER (%)")
        ax2.set_ylim(0, None)
        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax.legend(h1 + h2, l1 + l2, fontsize=10, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=4, frameon=False)

    def _draw_spoof(self) -> None:
        ax = self.fig.add_subplot(111)
        fes = self.fes()
        w = 0.8 / max(len(fes), 1)
        xs = np.arange(len(SNRS) + 1)
        for j, fe in enumerate(fes):
            rows = [self.c(s, fe) for s in SNRS[::-1]] + [self.clean(fe)]
            ys = [100 * r["spoof"]["n_spoof"] / r["spoof"]["n"] if r else 0 for r in rows]
            ax.bar(xs + (j - (len(fes) - 1) / 2) * w, ys, w * 0.92, color=FE_COLOR[fe], label=FE_LABEL[fe])
        ax.set_xticks(xs, [f"{s:+d} dB" for s in SNRS[::-1]] + ["no noise added"])
        ax.set_ylabel("real sentences called spoof (%)")
        ax.set_ylim(0, 100)
        ax.grid(alpha=0.25, axis="y")
        ax.legend(fontsize=10, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=len(fes), frameon=False)

    def show_view(self, view: str) -> None:
        self.view = view
        self.redraw()

    # ── 鍵 ──
    def menu(self) -> None:
        o = self.out
        o(f"\n  {C.BOLD}聽{C.RESET}  0 原句   1 帶噪 → 統計式 → {FE_LABEL['mgan']}   u 不平滑（musical noise）   x {FE_LABEL['sep']}   s 換 SNR   n 換句   b 換噪音")
        o(f"  {C.BOLD}看{C.RESET}  2 CTC 的 WER   3 Whisper 的 WER   4 逐字稿並排   w 最糟的一句   5 SAR 對 WER   6 ω   7 偵測器")
        o(f"      m 每位裁判   e 只增強不加噪   l 現場重算   t 投影片的表   ? 選單   q 離開")
        o(f"  {C.GREY}現在：{NOISE_LABEL[self.noise]}；聽 {self.sid} 的 {self.lsnr:+d} dB{C.RESET}")

    def dispatch(self, k: str | None) -> bool:
        if k is None:
            return True
        if k in ("q", "\x03", "\x04"):
            return False
        if k == "0":
            self.head(f"聽：{self.sid} 的原句")
            self.play("clean", "乾淨的原句")
        elif k == "1":
            self.listen()
        elif k == "u":
            self.head(f"聽：統計式增強，不平滑（α = {STAT['alpha_musical']:g}）")
            self.play("stat_a0", "每一格只看自己：聽背景裡零零星星的音")
            self.play("stat", f"平滑過的（α = {STAT['alpha']:g}）")
        elif k == "x":
            self.head(f"聽：{FE_LABEL['sep']}")
            self.play("sep", FE_LABEL["sep"])
        elif k == "s":
            self.si += 1
            self.out(f"  要聽的 SNR → {self.lsnr:+d} dB")
        elif k == "n":
            self.li += 1
            self.out(f"  要聽的句子 → {self.sid}")
        elif k == "b" and self.noises:
            self.noise = self.noises[(self.noises.index(self.noise) + 1) % len(self.noises)]
            self.out(f"  噪音 → {NOISE_LABEL[self.noise]}")
            self.redraw()
        elif k in ("2", "3"):
            which = "ctc" if k == "2" else "whisper"
            self.shown.add(which)
            self.wer_table(which)
            self.show_view("wer")
        elif k == "4":
            self.side_by_side()
        elif k == "w":
            self.worst()
        elif k == "5":
            self.scatter_table()
            self.show_view("scatter")
        elif k == "6":
            self.oa_table()
            self.show_view("oa")
        elif k == "7":
            self.spoof_table()
            self.show_view("spoof")
        elif k == "m":
            self.judges()
        elif k == "e":
            self.clean_only()
        elif k == "l":
            self.live()
        elif k == "t":
            self.slide_tables()
        elif k in ("?", "h"):
            self.menu()
        return True

    def on_key(self, ev) -> None:
        if ev.key is None:
            return
        if not self.dispatch(ev.key):
            self.plt.close(self.fig)

    def save_figs(self) -> None:
        import matplotlib
        matplotlib.use("Agg")
        self.make_fig()
        keep = self.noise
        for nz in self.noises:
            self.noise = nz
            self.shown = {"ctc", "whisper"}
            for view in ("wer", "scatter", "spoof") + (("oa",) if nz == self.rec["main_noise"] else ()):
                self.view = view
                self.draw()
                self.fig.savefig(self.folder / f"{view}_{nz}.png", dpi=110)
        self.noise, self.view, self.shown = keep, "blank", set()
        self.plt.close(self.fig)
        self.fig = None

    def print_all(self) -> None:
        keep = self.noise
        for nz in self.noises:
            self.noise = nz
            for which in ("ctc", "whisper"):
                self.wer_table(which)
            self.judges()
            self.scatter_table()
            self.spoof_table()
            for snr in sorted({min(SNRS), int(OA["snr"])}):
                for sid in self.listen_ids[:1]:
                    self.side_by_side(sid, snr)
        self.noise = keep
        self.oa_table()
        self.clean_only()
        self.worst()
        self.slide_tables()


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


# ── 組裝 ────────────────────────────────────────────────────────────────
def versions() -> dict:
    from importlib import metadata
    v = C.versions()
    for pkg in ("torch", "torchaudio", "transformers", "speechbrain", "pystoi", "pesq", "huggingface_hub"):
        try:
            v[pkg] = metadata.version(pkg)
        except metadata.PackageNotFoundError:
            v[pkg] = None
    return v


def material(n: int | None = None) -> tuple[list[dict], list[np.ndarray]]:
    sents = W2.load_sentences()
    if n:
        keep = [s for s in sents if s["id"] in LISTEN["sentences"]]
        sents = keep + [s for s in sents if s not in keep][: max(n - len(keep), 0)]
    return sents, W2.load_audio(sents)


def fake_material(n: int = 6) -> tuple[list[dict], list[np.ndarray]]:
    """沒有 40 句的 wav 時（selftest 在別的機器上）：合成的母音串當「句子」。"""
    import dsp
    rng = np.random.default_rng(int(CFG["fake"]["seed"]))
    sents, xs = [], []
    for i in range(n):
        f0 = 100 + 25 * i
        x = np.concatenate([dsp.synth_vowel(FS, 0.35, f0 * (1 + 0.1 * rng.standard_normal()), fm) * (rng.random() > 0.15)
                            for fm in ([730, 1090, 2440], [270, 2290, 3010], [300, 870, 2240], [530, 1840, 2480], [660, 1720, 2410], [440, 1020, 2240])])
        sid = LISTEN["sentences"][i] if i < len(LISTEN["sentences"]) else f"fake-{i}"
        sents.append({"id": sid, "spk": f"spk{i}", "text": " ".join(f"WORD{j}" for j in range(8 + i))})
        xs.append(0.1 * x / (np.max(np.abs(x)) + 1e-9))
    return sents, xs


def build_replay(out: Out, folder: Path) -> Demo:
    p = folder / "rehearsal.json"
    if not p.exists():
        sys.exit(f"沒有 {os.path.relpath(p, HERE)}：先跑 ./present.sh rehearse")
    return Demo(out, json.loads(p.read_text(encoding="utf-8")), folder)


# ── fetch ───────────────────────────────────────────────────────────────
def fetch() -> None:
    from huggingface_hub import HfApi, snapshot_download
    C.banner("下載兩個增強模型的權重；確認前幾週的模型與 40 句都在", C.AMBER)
    lock = read_lock()
    lines = []
    for sec in ("neural", "extra"):
        repo = CFG[sec]["repo"]
        if sec == "extra" and not CFG["extra"]["enabled"]:
            continue
        sha = lock.get(repo) or HfApi().model_info(repo).sha
        p = snapshot_download(repo, revision=sha)
        mb = sum(f.stat().st_size for f in Path(p).rglob("*") if f.is_file()) / 1e6
        print(f"  {repo} @ {sha[:12]}  {mb:.0f} MB")
        lines.append(f"{repo} {sha}")
    MODEL_LOCK.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"  → {MODEL_LOCK.name}")
    repo = CFG["ctc"]["repo"]
    try:
        snapshot_download(repo, revision=lock.get(repo), allow_patterns=PATTERNS)
        print(f"  W3 Demo 3 · {repo} @ {(lock.get(repo) or '最新')[:12]}：在")
    except Exception as e:
        print(f"  {C.RED}{repo}：{type(e).__name__}: {str(e)[:160]}{C.RESET}")
    try:
        W2.model_dir(offline=False)
        print(f"  W2 Demo 3 · {W2.CFG['model']['repo']} @ {(W2.read_lock() or '最新')[:12]}：在")
    except Exception as e:
        print(f"  {C.RED}{W2.CFG['model']['repo']}：{type(e).__name__}: {str(e)[:160]}{C.RESET}")
    try:
        sents = W2.load_sentences()
        n = sum(W2.wav_path(s["id"]).exists() for s in sents)
        print(f"  40 句：清單 {len(sents)} 句、wav 在的 {n} 句" + ("" if n == len(sents) else f"  {C.AMBER}← ../w02_d3_frontend_mismatch/present.sh fetch{C.RESET}"))
    except SystemExit as e:
        print(f"  {C.AMBER}{e}{C.RESET}")
    a = (HERE / CFG["spoof"]["from"] / ".." / "_media" / "w09_d3" / "aasist" / "models" / "AASIST.py").resolve()
    print(f"  W9 Demo 3 的偵測器（AASIST）：{'在' if a.exists() else C.AMBER + '不在 ← ../w09_d3_two_defences/present.sh fetch' + C.RESET}")
    for pkg in ("pystoi", "pesq"):
        try:
            __import__(pkg)
            print(f"  {pkg}：在")
        except Exception:
            print(f"  {C.AMBER}{pkg}：沒裝（那一欄會是空的；./setup.sh）{C.RESET}")
    print("下一步：./present.sh check")


# ── check ───────────────────────────────────────────────────────────────
def check() -> None:
    import torch
    C.banner("課前體檢", C.AMBER)
    rec = {"time": C.now(), "versions": versions(), "mps_available": bool(torch.backends.mps.is_available()),
           "mps_fallback_env": os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK")}
    for k, v in rec["versions"].items():
        print(f"  {k:<16s} {v}")
    print(f"  MPS：{rec['mps_available']}   PYTORCH_ENABLE_MPS_FALLBACK={rec['mps_fallback_env']}（0 = 嚴格：不支援的運算直接報錯）")
    sents, xs = material(4)
    b = Real()
    lab = Lab(b, sents, xs, Out(echo=False))
    s, gv, y, _ = lab.signals(0, "white", 0, "none")
    devs = ["cpu"] + (["mps"] if rec["mps_available"] else [])

    def timed(fn):
        t0 = time.perf_counter()
        r = fn()
        return r, time.perf_counter() - t0

    C.banner(f"前端：{FE_LABEL['mgan']}")
    try:
        g = MetricGAN("cpu")
        mine, t = timed(lambda: g.enhance(y))
        off = g.enhance_official(y)
        cos = float(mine @ off[: len(mine)] / (np.linalg.norm(mine) * np.linalg.norm(off[: len(mine)]) + 1e-20))
        half = g.enhance(0.5 * y)
        lin = float(np.linalg.norm(half - 0.5 * mine) / (np.linalg.norm(0.5 * mine) + 1e-20))
        rec["mgan"] = {"ok": True, "n_params": g.n_params, "seconds": t, "cos_vs_enhance_batch": cos, "official_peak": float(np.max(np.abs(off))),
                       "rms_out_over_in": math.sqrt(E.power(mine) / E.power(y)), "half_input_relative_change": lin, **E.ratios(mine, s, gv)}
        print(f"  cpu：{g.n_params / 1e6:.2f} M 個參數，一句 {t:.3f} s")
        print(f"  自己重寫的那幾行對 enhance_batch：cosine {cos:.6f}（1 = 只差一個增益）；官方那一份的峰值 {rec['mgan']['official_peak']:.3f}（它把峰值正規化成 1）")
        print(f"  輸出的 RMS 是輸入的 {rec['mgan']['rms_out_over_in']:.2f} 倍；輸入音量減半，輸出不是剛好減半（相對差 {lin:.3f}）——它對音量敏感")
        print(f"  白噪音 0 dB 的那一句：SNR {fmt(rec['mgan']['snr'])}、SAR {fmt(rec['mgan']['sar'])} dB")
    except Exception as e:
        rec["mgan"] = {"ok": False, "error": f"{type(e).__name__}: {str(e)[:300]}"}
        print(f"  {C.RED}{type(e).__name__}: {str(e)[:200]}{C.RESET}")

    if CFG["extra"]["enabled"]:
        C.banner(f"加映的前端：{FE_LABEL['sep']}")
        rec["sep"] = {}
        ref = None
        for dv in devs:
            try:
                m = SepFormer(dv)
                m.enhance(y[:FS])
                est, t = timed(lambda: m.enhance(y))
                rec["sep"][dv] = {"ok": True, "n_params": m.n_params, "seconds": t, **E.ratios(est, s, gv)}
                print(f"  {dv}：{m.n_params / 1e6:.1f} M 個參數，一句（{len(y) / FS:.1f} s）{t:.2f} s；SNR {fmt(rec['sep'][dv]['snr'])}、SAR {fmt(rec['sep'][dv]['sar'])} dB")
                if dv == "cpu":
                    ref = est
                elif ref is not None:
                    d = float(np.max(np.abs(est - ref)) / (np.max(np.abs(ref)) + 1e-20))
                    rec["sep"]["mps_vs_cpu_rel_max"] = d
                    print(f"  mps 對 cpu：相對最大差 {d:.2e}" + ("" if d < 1e-2 else f"  {C.AMBER}（差得多：demo_config.toml [extra].device 改 \"cpu\"）{C.RESET}"))
            except Exception as e:
                rec["sep"][dv] = {"ok": False, "error": f"{type(e).__name__}: {str(e)[:300]}"}
                print(f"  {C.RED}{dv}：{type(e).__name__}: {str(e)[:200]}{C.RESET}（mps 不行就把 [extra].device 改成 \"cpu\"）")

    C.banner("辨識器")
    rec["ctc"] = {}
    texts = {}
    for dv in devs:
        try:
            m = CTC(dv)
            m.transcribe(xs[0][:FS])
            hs, t = timed(lambda: [m.transcribe(x) for x in xs])
            texts[dv] = hs
            rec["ctc"][dv] = {"ok": True, "seconds_per_sentence": t / len(xs)}
            print(f"  wav2vec2-base-960h · {dv}：一句 {t / len(xs):.3f} s")
        except Exception as e:
            rec["ctc"][dv] = {"ok": False, "error": f"{type(e).__name__}: {str(e)[:300]}"}
            print(f"  {C.RED}wav2vec2 · {dv}：{type(e).__name__}: {str(e)[:200]}{C.RESET}")
    if len(texts) == 2:
        rec["ctc"]["same_text"] = texts["cpu"] == texts["mps"]
        print(f"  cpu 與 mps 的 {len(xs)} 句逐字{'相同' if rec['ctc']['same_text'] else C.AMBER + '不同（[ctc].device 改 cpu）' + C.RESET}")
    try:
        b.whisper(xs[0], 0, xs[0])
        hs, t = timed(lambda: [b.whisper(x, i, x)[0] for i, x in enumerate(xs)])
        w = W2.corpus_wer(lab.refs, [b.normalize(h) for h in hs])
        rec["whisper"] = {"ok": True, "seconds_per_sentence": t / len(xs), "wer_clean_subset": w["wer"]}
        print(f"  Whisper small · {W2.CFG['model']['device']}：一句 {t / len(xs):.2f} s；這 {len(xs)} 句乾淨的 WER {pct(w['wer'])}%")
        n_cond = len(plan(NOISE["types"], bool(CFG["extra"]["enabled"])))
        print(f"  → 彩排估計：{n_cond} 個條件 × 40 句 × {t / len(xs):.2f} s ≈ {n_cond * 40 * t / len(xs) / 60:.0f} 分鐘（只算 Whisper；可以中斷再接）")
    except Exception as e:
        rec["whisper"] = {"ok": False, "error": f"{type(e).__name__}: {str(e)[:300]}"}
        print(f"  {C.RED}Whisper：{type(e).__name__}: {str(e)[:200]}{C.RESET}")

    C.banner("偵測器與兩個給人耳的指標")
    try:
        sc, t = timed(lambda: [b.spoof(x) for x in xs])
        rec["spoof"] = {"ok": True, "seconds_per_sentence": t / len(xs), "scores": sc}
        print(f"  AASIST：一句 {t / len(xs):.2f} s；這 {len(xs)} 句乾淨的分數 {', '.join(f'{v:+.2f}' for v in sc)}")
    except BaseException as e:
        rec["spoof"] = {"ok": False, "error": f"{type(e).__name__}: {str(e)[:300]}"}
        print(f"  {C.RED}AASIST：{type(e).__name__}: {str(e)[:200]}{C.RESET}")
    q = lab.judge_signal(y, s, gv)
    rec["stoi"], rec["pesq"] = q["stoi"], q["pesq"]
    print("  pystoi：" + ("沒裝" if lab._stoi is None else f"白噪音 0 dB 那一句 STOI {fmt(q['stoi'], 3, '—')}"))
    print("  pesq  ：" + ("沒裝（要編譯；./setup.sh 會試）" if lab._pesq is None else f"同一句 PESQ（wb）{fmt(q['pesq'], 2, '—')}"))
    (HERE / "runs").mkdir(exist_ok=True)
    write_json(HERE / "runs" / "check.json", rec)
    print("\n  → runs/check.json\n下一步：./present.sh rehearse")


# ── export ──────────────────────────────────────────────────────────────
def slim(c: dict) -> dict:
    return {"noise": c["noise"], "snr": c["snr"], "fe": c["fe"], "omega": c["omega"], "n": c["n"], "sig": c["sig"], "sar_inf": c["sar_inf"],
            "ctc": {k: v for k, v in c["ctc"].items() if k not in ("hyps", "errs")},
            "whisper": {k: v for k, v in c["whisper"].items() if k not in ("hyps", "errs")},
            "spoof": {k: v for k, v in c["spoof"].items() if k != "scores"}}


def export(rec_path: Path) -> None:
    rec = json.loads(rec_path.read_text(encoding="utf-8"))
    d = json.loads(METRICS_JSON.read_text(encoding="utf-8")) if METRICS_JSON.exists() else {}
    d["demo2"] = {"source": "demos/w10_d2_enhance_wer/runs/rehearsal/rehearsal.json", "time": rec["time"], "tag": rec.get("tag", ""),
                  "models": rec["models"], "versions": rec.get("versions"), "n_sentences": len(rec["sentences"]),
                  "main_noise": rec["main_noise"], "snrs": rec["snrs"], "stat": rec["stat"], "oa": rec["oa"],
                  "metrics_available": rec.get("metrics_available"), "skipped": rec.get("skipped", []),
                  "conds": {k: slim(c) for k, c in rec["conds"].items()},
                  "what_we_heard": rec.get("what_we_heard", ""),
                  "definitions": {
                      "sentences": "one utterance per LibriSpeech dev-clean speaker (the 40 sentences of W2 Demo 3; also W9 Demo 3's real-speech reference set)",
                      "noise": "synthetic: white; pink (power ∝ 1/f); babble = other talkers from the same 40 sentences summed. One draw per sentence and type, reused at every SNR.",
                      "snr (input)": "10 log10(sum s^2 / sum v^2) over the whole utterance; the speech is left untouched and the noise is scaled",
                      "cond key": "noise|SNR|front end[|oa<omega>]; clean|<front end> = the front end applied to clean speech",
                      "front ends": "none; stat = decision-directed Wiener written in numpy (enhance.py); mgan = SpeechBrain metricgan-plus-voicebank without the final peak normalisation; sep = SepFormer DNS4 16 kHz, rescaled by a least-squares gain against the noisy input",
                      "wer": "corpus WER, word-level Levenshtein; references and both recognisers' outputs pass Whisper's EnglishTextNormalizer",
                      "sig.sdr / snr / sar": "mean over sentences (dB) of the ratios from two projections of the front end's output: onto s, then onto span{s, v}. No delayed copies in the subspaces. sar_inf: the artifact term is zero (no front end).",
                      "sig.si_sdr / stoi / pesq": "mean over sentences; PESQ is wide-band (ITU-T P.862.2) via the pesq package, STOI via pystoi; null if the package is missing",
                      "omega": "observation adding: output = enhanced + omega * noisy input",
                      "spoof": "AASIST as configured in W9 Demo 3: score = bona fide logit - spoof logit, called spoof if < 0"}}
    METRICS_JSON.parent.mkdir(parents=True, exist_ok=True)
    METRICS_JSON.write_text(json.dumps(d, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    print(f"  → {os.path.relpath(METRICS_JSON, HERE)}（demo2）")


# ── selftest ────────────────────────────────────────────────────────────
def selftest() -> None:
    import shutil
    import dsp
    C.banner("selftest（不需要模型）", C.AMBER)
    n_ok = [0, 0]

    def ok(cond: bool, msg: str) -> None:
        n_ok[0 if cond else 1] += 1
        print(f"  {C.GREEN + '✓' if cond else C.RED + '✗'}{C.RESET} {msg}")

    rng = np.random.default_rng(0)
    n = 3 * FS
    s = np.concatenate([dsp.synth_vowel(FS, 0.5, f0, fm) for f0, fm in ((110, [730, 1090, 2440]), (130, [270, 2290, 3010]), (0, [300, 870, 2240]),
                                                                          (120, [530, 1840, 2480]), (100, [660, 1720, 2410]), (140, [440, 1020, 2240]))])[:n]
    s[int(1.0 * FS):int(1.5 * FS)] = 0.0
    s = 0.1 * s / np.max(np.abs(s))
    for snr in (20, 0, -5):
        y, gv = E.mix(s, E.white(n, rng), snr)
        got = E.db(E.power(s) / E.power(gv))
        ok(abs(got - snr) < 1e-6 and np.allclose(y - gv, s), f"mix：SNR {got:+.4f} dB（要 {snr:+d}），語音沒被動到")
    v = E.pink(1 << 16, rng)
    P = np.abs(np.fft.rfft(v)) ** 2
    f = np.arange(len(P))
    lo, hi = P[(f >= 100) & (f < 200)].mean(), P[(f >= 1600) & (f < 3200)].mean()
    ok(abs(E.db(lo / hi) - 12.0) < 1.5, f"pink：四個八度掉 {E.db(lo / hi):.1f} dB（1/f 是 12）")
    others = [np.full(1000, float(i + 1)) for i in range(5)]
    bb = E.babble(2500, others, 3, np.random.default_rng(1))
    ok(len(bb) == 2500 and abs(bb[0] - 3.0) < 1e-6, "babble：k 個來源各自調到單位功率再相加，短的繞回去接")

    y, gv = E.mix(s, E.white(n, rng), 0)
    r = E.ratios(y, s, gv)
    ok(r["sar"] is None and abs(r["snr"] - 0.0) < 0.3, f"沒有前端：SAR = ∞、SNR {r['snr']:+.2f} dB（輸入是 0）")
    d = E.decompose(y, s, gv)
    ok(abs(float(d["e_artif"] @ s)) < 1e-6 and abs(float(d["e_artif"] @ gv)) < 1e-6 and np.allclose(d["s_target"] + d["e_noise"] + d["e_artif"], y),
       "decompose：三項加起來是輸入；e_artif 與 s、v 都正交")
    clip = np.tanh(8 * y) / 8
    rc = E.ratios(clip, s, gv)
    ok(rc["sar"] is not None and rc["sar"] < 40, f"非線性（tanh）之後 SAR 變成有限值 {rc['sar']:.1f} dB")
    r2 = E.ratios(3.0 * s + 0.5 * gv, s, gv)
    ok(r2["sar"] is None and abs(r2["snr"] - E.db(9 * E.power(s) / (0.25 * E.power(gv)))) < 0.5, "s 與 v 的線性組合：SAR = ∞，SNR 照係數變")

    est, G, sv = E.wiener_dd(y, 512, 128, 512, 0.98, 0.1, 0.5, True)
    Pv = (np.abs(dsp.stft(gv, 512, 128, 512)) ** 2).mean(1)
    bias = E.db(float(sv.mean() / Pv.mean()))
    ok(abs(bias) < 3.0, f"noise_psd：白噪音的功率估到 {bias:+.1f} dB 以內")
    re_ = E.ratios(est, s, gv)
    ok(re_["snr"] > 5.0 and re_["sar"] is not None, f"wiener_dd：SNR 0 → {re_['snr']:.1f} dB，SAR ∞ → {re_['sar']:.1f} dB（用 SAR 換 SNR）")
    ok(E.si_sdr(est, s) > E.si_sdr(y, s), f"wiener_dd：SI-SDR {E.si_sdr(y, s):.1f} → {E.si_sdr(est, s):.1f} dB")
    _, G0, _ = E.wiener_dd(y, 512, 128, 512, 0.0, 0.1, 0.5, True)
    ok(E.isolated_fraction(G0) > 5 * E.isolated_fraction(G) + 1e-5, f"α = 0 的增益圖有孤立的亮格（{E.isolated_fraction(G0):.4f} 對 {E.isolated_fraction(G):.4f}）——musical noise")
    ok(float(G.min()) >= 0.1 - 1e-12 and float(G.max()) <= 1.0, "增益在 g_min 與 1 之間")

    sars, snrs = [], []
    for w in (0.0, 0.1, 0.3, 1.0, 3.0, 10.0):
        q = E.ratios(E.observation_add(est, y, w), s, gv)
        sars.append(q["sar"])
        snrs.append(q["snr"])
    ok(float(est @ y) > 0 and all(b > a for a, b in zip(sars, sars[1:])), "observation adding：⟨ŝ, y⟩ > 0 時 SAR 隨 ω 單調上升（" + " → ".join(f"{v:.1f}" for v in sars) + "）")
    ok(all(b < a for a, b in zip(snrs, snrs[1:])) and snrs[-1] > 0.0, "observation adding：SNR 隨 ω 往輸入的 SNR 掉回去（" + " → ".join(f"{v:.1f}" for v in snrs) + "）")
    ea0 = E.decompose(est, s, gv)["e_artif"]
    ea1 = E.decompose(E.observation_add(est, y, 0.7), s, gv)["e_artif"]
    ok(np.allclose(ea0, ea1, atol=1e-9), "observation adding：e_artif 一點都沒變（加回去的 y 整個在平面上）")
    g = E.blind_gain(17.0 * est, y)
    ok(abs(g * 17.0 - E.blind_gain(est, y)) < 1e-9, "blind_gain：輸出放大 17 倍，增益就小 17 倍")
    ok(abs(E.si_sdr(5.0 * est, s) - E.si_sdr(est, s)) < 1e-9, "SI-SDR 對音量不敏感")

    a = W2.align_words("A B C D".split(), "A X C".split())
    ok((a["S"], a["D"], a["I"]) == (1, 1, 0), "WER：一個替換、一個刪除")
    ok(key_of("white", -5, "mgan") == "white|-5|mgan" and key_of("white", 0, "mgan", 0.3) == "white|+0|mgan|oa0.3", "條件的鍵")
    p = plan(NOISE["types"], True)
    ok(len(p) == len(set(cond_key(*t) for t in p)) and p[0] == (None, None, "none", 0.0), f"plan：{len(p)} 個條件、沒有重複、第一個是乾淨的原句")
    main_first = [t for t in p if t[0] is not None][0][0] == NOISE["main"]
    ok(main_first and all(sid in [x["id"] for x in W2.load_sentences()] for sid in LISTEN["sentences"]) if W2.SENTENCES.exists() else main_first,
       "plan 先算 main 那一種噪音；要聽的句子在 40 句的清單裡")

    # fake 整條流程
    try:
        sents, xs = material(5)
        src = "真的句子"
    except SystemExit:
        sents, xs = fake_material()
        src = "合成的母音串（40 句的 wav 不在這台機器上）"
    out = Out(echo=False)
    lab = Lab(Fake(), sents, xs, out)
    folder = HERE / "runs" / "_selftest"
    shutil.rmtree(folder, ignore_errors=True)
    rec = compute_all(lab, folder, ["white", "babble"], True, True, "FAKE", {})
    n_expect = len(plan(["white", "babble"], True))
    ok(len(rec["conds"]) == n_expect, f"fake 流程（{src}）：{len(rec['conds'])} 個條件")
    c0 = rec["conds"][key_of(NOISE["main"], 0, "none")]
    ok(c0["sar_inf"] and abs(c0["sig"]["snr"]) < 0.5, "fake：不處理的那一列 SAR = ∞、SNR ≈ 0")
    oa = [rec["conds"][key_of(NOISE["main"], int(OA["snr"]), OA["fe"], float(w))]["sig"]["sar"] for w in [0.0] + list(OA["omegas"])]
    ok(all(b > a for a, b in zip(oa, oa[1:])), "fake：ω 掃過去 SAR 單調上升")
    bab = lab.noise(0, "babble")
    ok(len(bab) == len(xs[0]) and np.allclose(bab, lab.noise(0, "babble")) and not np.allclose(bab, lab.noise(0, "babble", salt=1)), "噪音：同一句重抽是同一段；換 salt 才換")
    rec2 = compute_all(lab, folder, ["white", "babble"], True, False, "FAKE", {})
    ok(len(rec2["conds"]) == n_expect and "接續" in out.text(), "再跑一次：接續、不重算")
    d = Demo(out, json.loads((folder / "rehearsal.json").read_text(encoding="utf-8")), folder, lab)
    d.quiet = True
    for k in "01uxsnb23456w7meltb?":
        d.dispatch(k)
    d.print_all()
    files = rec["listen"]["files"]
    ok(len(files) > 0 and all((folder / f).exists() for f in files.values()), f"要聽的 .wav：{len(files)} 個")
    try:
        import matplotlib
        matplotlib.use("Agg")
        d.save_figs()
        ok((folder / f"wer_{NOISE['main']}.png").exists() and (folder / f"oa_{NOISE['main']}.png").exists(), "四種圖都畫得出來")
    except ImportError:
        print(f"  {C.AMBER}（沒有 matplotlib：圖略過）{C.RESET}")
    json.dumps({k: slim(c) for k, c in rec["conds"].items()})
    ok("/Users/" not in json.dumps(rec) and "/home/" not in json.dumps(rec), "紀錄裡沒有絕對路徑")
    shutil.rmtree(folder, ignore_errors=True)

    rp = REHEARSAL / "rehearsal.json"
    if rp.exists() and METRICS_JSON.exists():
        r = json.loads(rp.read_text(encoding="utf-8"))
        m = json.loads(METRICS_JSON.read_text(encoding="utf-8")).get("demo2")
        if m:
            same = m["time"] == r["time"] and all(abs(m["conds"][k]["whisper"]["wer"] - c["whisper"]["wer"]) < 1e-12 for k, c in r["conds"].items())
            ok(same, "w10-metrics.json → demo2 與 runs/rehearsal/rehearsal.json 一致")
        else:
            print(f"  {C.AMBER}（w10-metrics.json 還沒有 demo2：./present.sh export）{C.RESET}")
    print(f"\n  {n_ok[0]} 項通過、{n_ok[1]} 項失敗")
    if n_ok[1]:
        sys.exit(1)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", nargs="?", default="show", choices=["show", "rehearse", "replay", "check", "fetch", "fake", "selftest", "export"])
    ap.add_argument("--no-plot", action="store_true", help="只在終端機，不開圖")
    ap.add_argument("--device", default=None, help="覆蓋 CTC 與加映前端的裝置（cpu／mps／auto）")
    ap.add_argument("--noise", default=None, help="rehearse：只算這幾種噪音，逗號分隔")
    ap.add_argument("--no-extra", action="store_true", help="rehearse：不跑加映的前端")
    ap.add_argument("--fresh", action="store_true", help="rehearse：不接續，從頭算")
    ap.add_argument("--n", type=int, default=None, help="rehearse：只用前 n 句（預覽 → runs/preview/）")
    ap.add_argument("--from", dest="src", default=None, help="replay／show：讀這個資料夾的紀錄")
    args = ap.parse_args()
    if args.cmd == "fetch":
        return fetch()
    if args.cmd == "check":
        return check()
    if args.cmd == "selftest":
        return selftest()
    if args.cmd == "export":
        return export(REHEARSAL / "rehearsal.json")
    out = Out()
    src = (HERE / args.src).resolve() if args.src else REHEARSAL
    if args.cmd == "replay":
        out.banner("REPLAY：讀回彩排（不碰 torch）", C.AMBER)
        return run(build_replay(out, src), not args.no_plot)
    if args.cmd == "show":
        out.banner("W10 Demo 2 · 增強之後 WER 往哪裡走")
        d = build_replay(out, src)
        try:
            b = Real(args.device)
            sents, xs = material()
            keep = [i for i, s in enumerate(sents) if s["id"] in d.ids]
            d.lab = Lab(b, [sents[i] for i in keep], [xs[i] for i in keep], out)
            b.load_all()
        except BaseException as e:
            d.lab = None
            out(f"  {C.AMBER}模型沒載起來（{type(e).__name__}: {str(e)[:120]}）：照 replay 走，鍵 l 不能用{C.RESET}")
        try:
            run(d, not args.no_plot)
        finally:
            live = HERE / "runs" / "live"
            live.mkdir(parents=True, exist_ok=True)
            (live / f"{C.now().replace(':', '-')}.txt").write_text(out.text(), encoding="utf-8")
        return
    # rehearse／fake
    fake = args.cmd == "fake"
    noises = [n for n in (args.noise.split(",") if args.noise else NOISE["types"]) if n in NOISE["types"]]
    extra = not args.no_extra and bool(CFG["extra"]["enabled"])
    if fake:
        out.banner("FAKE：替身，沒有任何模型——數字沒有意義", C.AMBER)
        try:
            sents, xs = material(args.n or 6)
        except SystemExit:
            sents, xs = fake_material()
        folder, tag, b = HERE / "runs" / "fake", "FAKE", Fake()
    else:
        out.banner("彩排：全部算一遍" + (f"（預覽：只用 {args.n} 句）" if args.n else ""), C.AMBER)
        sents, xs = material(args.n)
        folder, tag, b = (PREVIEW, f"PREVIEW n={len(sents)}", Real(args.device, extra)) if args.n else (REHEARSAL, "", Real(args.device, extra))
    lab = Lab(b, sents, xs, out)
    rec = compute_all(lab, folder, noises, extra, args.fresh or fake, tag, versions() if not fake else {})
    d = Demo(out, rec, folder, lab, tag)
    d.quiet = True
    d.print_all()
    try:
        d.save_figs()
    except ImportError:
        pass
    (folder / "rehearsal.txt").write_text(out.text(), encoding="utf-8")
    print(f"\n  → {os.path.relpath(folder, HERE)}/")
    if folder == REHEARSAL:
        export(folder / "rehearsal.json")
    elif fake and not args.no_plot and sys.stdin.isatty():
        d.quiet = False
        run(d, True)


if __name__ == "__main__":
    main()

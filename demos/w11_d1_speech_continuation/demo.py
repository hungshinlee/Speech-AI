# -*- coding: utf-8 -*-
# 此檔由 bin/publish_demos.py 從 private repo 的 <course>/demos/ 複製而來，請勿在公開 repo 直接編輯。
"""W11 demo 1：序列裡沒有文字的語言模型——聽起來像語音，說了什麼？（只在終端機，單鍵）

一段真人錄音的頭 3.2 秒 → Mimi 的頭 4 層 token（每秒 50 個）→ Llama-Mimi-1.3B 一個一個往下接 → Mimi decoder → 聲音。
同一個開頭接幾次（不同 seed、兩個 temperature）。先聽；之後才把每一段交給 Whisper small 轉成字，
旁邊放一個文字 LLM 從「開頭的字」接同樣長度的文字。

子命令：
  fetch      下載 Llama-Mimi 與 Mimi，確認 Whisper、WavLM-SV、文字 LLM 在快取裡（要網路）→ model.lock
  check      課前體檢：token 的排法、Mimi 與語言模型在 cpu／mps 上的差與速度、兩位裁判、文字 LLM 的兩種接法 → runs/check.json
  rehearse   全部算一遍 → runs/rehearsal/（.wav 與 rehearsal.json，進版控＝現場的退路），寫 w11-metrics.json → demo1
             --preview 寫到 runs/preview/（不進版控、不寫 metrics）；--draws 2 只抽前兩段；--text-repo 換文字 LLM；--no-text 不跑文字 LLM
  show       上課用：讀回彩排的聲音與數字；按 g 才把模型載起來現場再抽一段
  replay     只讀彩排的紀錄，不載任何模型（鍵相同，除了 g）
  fake       沒有模型也能演練流程（畫面標 FAKE，聲音是噪音、數字沒有意義）
  selftest   token 的排法、解析、長度的算術、重複的量法、fake 整條流程；有彩排紀錄時對 w11-metrics.json
  export     只重寫 slides/assets/w11/w11-metrics.json → demo1

畫面上的鍵（順序照投影片 p23 講稿的【操作】）：
  p  播開頭              1 2 3 4  播一段續寫（開頭接著續寫，同一次 Mimi 解碼）      c  切換：只播續寫的部分
  t  三欄的字：開頭、語音 LM 接的（Whisper 轉的）、文字 LLM 接的                     n  數字
  w  Whisper 對每一段有多少把握      r  這位說話人實際上接著說的      m  開頭只過 Mimi 的 4 層（codec 的上限）
  g  現場再抽一段（載模型）          ?  選單      q  離開

數字的定義：
  speech tokens    = frame 數 × 4（Mimi 12.5 Hz、頭 4 層；3.2 s = 40 frames = 160 個）
  tokens / s       = 新生成的 token 數 ÷ 生成花的時間（不含 Mimi 解碼）；real-time factor = 生成花的時間 ÷ 生成出來的聲音長度
  SIM              = 兩段聲音經 WavLM-SV（W9 Demo 1 的那一個 encoder）的向量的 cosine；開頭用原始錄音，不是過了 codec 的
  avg log-prob     = Whisper 對自己轉出來的字的平均 log 機率（W7 Demo 1 的定義）；越低越是在猜
  compression      = 轉錄的 zlib 壓縮比（W7）；重複的迴圈會把它推高
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.util
import json
import sys
import textwrap
import time
import tomllib
import warnings
import zlib
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
DEMOS = HERE.parent
sys.path.insert(0, str(DEMOS))
import common as C  # noqa: E402

CFG = tomllib.loads((HERE / "demo_config.toml").read_text(encoding="utf-8"))
RUNS = HERE / "runs"
LOCK = HERE / "model.lock"
ASSETS = DEMOS.parent / "slides" / "assets" / "w11"
METRICS = ASSETS / "w11-metrics.json"
FS = 16000
WIDTH = 100


def P(rel: str) -> Path:
    return (HERE / rel).resolve()


# ── 純函數（selftest 測這些）─────────────────────────────────────
def n_frames(seconds: float, frame_rate: float) -> int:
    """Mimi 把不滿一格的尾巴補成一格：ceil。3.2 s × 12.5 = 40；3.0 s 會是 38（37.5 進位）。"""
    return int(np.ceil(round(seconds * frame_rate, 6)))


def codes_to_ids(codes: np.ndarray, base: int) -> list[int]:
    """(N_q, T) 的 code 矩陣 → 攤平的 token id：一個 frame 的 N_q 層排完，再排下一個 frame。<v_j> 的 id = base + N_q·v + j。"""
    nq, T = codes.shape
    return [int(base + nq * int(codes[j, t]) + j) for t in range(T) for j in range(nq)]


def parse_ids(ids: list[int], base: int, nq: int, V: int, end_id: int | None = None) -> dict:
    """生成出來的 id → (N_q, T) 的 code 矩陣。照 model card 的解法：遇到層的順序不對（或不是語音 token）就停，
    不滿一個 frame 的尾巴丟掉。回傳停下來的原因，讓「模型有沒有守住排法」看得到。"""
    frames, cur, why = [], [], "length"
    for k, i in enumerate(ids):
        if end_id is not None and i == end_id:
            why = "end tag"
            break
        r = i - base
        if not (0 <= r < nq * V) or r % nq != len(cur):
            why = f"pattern broken at token {k}"
            break
        cur.append(r // nq)
        if len(cur) == nq:
            frames.append(cur)
            cur = []
    codes = np.array(frames, dtype=np.int64).T if frames else np.zeros((nq, 0), dtype=np.int64)
    return {"codes": codes, "n_frames": len(frames), "stop": why, "dropped": len(cur)}


def compression_ratio(text: str) -> float:
    """openai/whisper 的定義（W7 Demo 1）：UTF-8 位元組長度 ÷ zlib 壓縮後的長度。"""
    b = text.encode("utf-8")
    return len(b) / len(zlib.compress(b)) if b else 0.0


def words(text: str) -> list[str]:
    return [w for w in "".join(ch.lower() if ch.isalnum() or ch == "'" else " " for ch in text).split() if w]


def repeat_fraction(text: str, n: int = 3) -> float:
    """連續 n 個詞的組合裡，前面已經出現過的佔多少。0 = 沒有重複；繞圈的句子會很高。"""
    w = words(text)
    grams = [tuple(w[i:i + n]) for i in range(len(w) - n + 1)]
    if not grams:
        return 0.0
    seen, rep = set(), 0
    for g in grams:
        rep += g in seen
        seen.add(g)
    return rep / len(grams)


def match_tokens(seconds: float, text_tokens_per_s: float) -> int:
    """「接同樣長度」：語音續寫幾秒，文字就給幾秒份的 token（W6 Demo 2 量到的每秒文字 token 數）。"""
    return max(1, int(round(seconds * text_tokens_per_s)))


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def resample(x: np.ndarray, fs_in: int, fs_out: int) -> np.ndarray:
    if fs_in == fs_out:
        return x.astype(np.float32)
    from math import gcd
    from scipy.signal import resample_poly
    g = gcd(fs_in, fs_out)
    return resample_poly(x, fs_out // g, fs_in // g).astype(np.float32)


def digest() -> str:
    keys = ("prompt", "speech_lm", "draws", "text_lm")
    return hashlib.sha256(json.dumps({k: CFG[k] for k in keys}, sort_keys=True, ensure_ascii=False).encode()).hexdigest()[:16]


def text_rate() -> tuple[float, str]:
    """每秒語音對應幾個文字 token（英語）：讀投影片的 w11-data.json（來源是 W6 Demo 2）；讀不到用設定檔裡的值。"""
    try:
        d = json.loads((ASSETS / "w11-data.json").read_text(encoding="utf-8"))["token_rate"]
        return float(d["text_tokens_per_s_english"]), "w11-data.json → token_rate (W6 Demo 2)"
    except Exception:
        return float(CFG["text_lm"]["tokens_per_second_fallback"]), "demo_config.toml"


# ── 鎖定檔與快取 ─────────────────────────────────────────────────
def read_lock(path: Path) -> dict:
    out = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            parts = line.split()
            if len(parts) == 2 and not line.startswith("#"):
                out[parts[0]] = parts[1]
    return out


def all_locks() -> dict:
    """自己的 model.lock 優先；Whisper 與 WavLM-SV 的 commit 沿用前面 demo 鎖的那一個。"""
    d = {}
    for rel in (CFG["whisper"]["lock_from"], CFG["sim"]["lock_from"]):
        d.update(read_lock(P(rel)))
    d.update(read_lock(LOCK))
    return d


def local_snapshot(repo: str, sha: str | None = None) -> str:
    """已下載的快照路徑。先問 huggingface_hub；它嫌快照不完整時（新版會檢查沒下載的 README 之類）直接到快取裡找。"""
    from huggingface_hub import constants, snapshot_download
    sha = sha or all_locks().get(repo)
    try:
        return snapshot_download(repo, revision=sha, local_files_only=True)
    except Exception:
        root = Path(constants.HF_HUB_CACHE) / ("models--" + repo.replace("/", "--")) / "snapshots"
        for c in ([root / sha] if sha else sorted(root.glob("*"))):
            if (c / "config.json").exists():
                return str(c)
        raise FileNotFoundError(f"快取裡沒有 {repo}（{sha or '任何版本'}）：先跑 ./present.sh fetch")


def versions() -> dict:
    from importlib import metadata
    v = C.versions()
    for pkg in ("torch", "transformers", "huggingface_hub", "mlx", "mlx-lm"):
        try:
            v[pkg] = metadata.version(pkg)
        except metadata.PackageNotFoundError:
            v[pkg] = None
    return v


def quiet() -> None:
    try:
        from transformers.utils import logging as hf_logging
        hf_logging.set_verbosity_error()
        hf_logging.disable_progress_bar()
    except Exception:
        pass


def pick_device(name: str):
    import torch
    if name == "auto":
        name = "mps" if torch.backends.mps.is_available() else "cpu"
    return torch.device(name)


# ── 模型 ─────────────────────────────────────────────────────────
class SpeechLM:
    """Llama-Mimi（transformers 的 LlamaForCausalLM，詞表裡多了 4 × 2048 個語音 token）加 Mimi。用法照 model card。"""

    def __init__(self, device: str | None = None, dtype: str | None = None):
        import torch
        from transformers import AutoFeatureExtractor, AutoModelForCausalLM, AutoTokenizer, MimiModel
        quiet()
        c = CFG["speech_lm"]
        self.torch = torch
        self.device = pick_device(device or c["device"])
        dt = dtype or c["dtype"]
        if dt == "auto":
            dt = "float32" if self.device.type == "cpu" else "bfloat16"       # card 用 bfloat16；CPU 上 float32 比較快
        self.dtype = dt
        lm_dir, mi_dir = local_snapshot(c["repo"]), local_snapshot(c["mimi_repo"])
        t0 = time.perf_counter()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.tok = AutoTokenizer.from_pretrained(lm_dir)
            self.model = AutoModelForCausalLM.from_pretrained(lm_dir, dtype=getattr(torch, dt)).eval().to(self.device)
            self.mimi = MimiModel.from_pretrained(mi_dir).eval().to(self.device)
            self.fe = AutoFeatureExtractor.from_pretrained(mi_dir)
        self.load_s = time.perf_counter() - t0
        self.nq = int(self.model.config.num_quantizers)
        self.V = int(self.mimi.config.codebook_size)
        self.frame_rate = float(self.mimi.config.frame_rate)
        self.fs = int(self.fe.sampling_rate)
        self.hop = int(round(self.fs / self.frame_rate))
        self.n_params = sum(p.numel() for p in self.model.parameters())
        self.layout = self.token_layout()
        self.top_p = getattr(self.model.generation_config, "top_p", None)     # card 沒有改它，generate 會沿用權重附的設定

    def token_layout(self) -> dict:
        """<v_j> 的 id 是不是 base + N_q·v + j；<audio>、</audio> 的 id。不是就停——整個 demo 靠這個排法。"""
        names = [f"<{v}_{j}>" for v in range(self.V) for j in range(self.nq)]
        ids = self.tok.convert_tokens_to_ids(names)
        base = ids[0]
        if ids != list(range(base, base + self.nq * self.V)):
            raise RuntimeError("語音 token 的 id 不是連續的 base + N_q·v + j：tokenizer 與這支程式的假設不合")
        return {"base": int(base), "start": int(self.tok.convert_tokens_to_ids("<audio>")), "end": int(self.tok.convert_tokens_to_ids("</audio>")),
                "bos": int(self.tok.bos_token_id), "text_vocab": int(base), "vocab": len(self.tok)}

    def encode(self, x: np.ndarray) -> np.ndarray:
        """24 kHz 的波形 → (N_q, T) 的 code。"""
        torch = self.torch
        inp = self.fe(raw_audio=x.astype(np.float32), sampling_rate=self.fs, return_tensors="pt")
        with torch.no_grad():
            out = self.mimi.encode(inp["input_values"].to(self.device), inp["padding_mask"].to(self.device), num_quantizers=self.nq)
        return out.audio_codes[0].detach().cpu().numpy().astype(np.int64)

    def decode(self, codes: np.ndarray) -> np.ndarray:
        torch = self.torch
        with torch.no_grad():
            y = self.mimi.decode(torch.tensor(codes[None], dtype=torch.long, device=self.device))[0]
        return y.detach().float().cpu().numpy().reshape(-1)

    def prompt_ids(self, codes: np.ndarray) -> list[int]:
        L = self.layout
        return [L["bos"], L["start"]] + codes_to_ids(codes, L["base"])

    def continue_(self, codes: np.ndarray, seed: int, temperature: float, frames: int) -> dict:
        """接 frames 個 frame（frames × N_q 個 token）。抽樣的設定照 card：temperature、top_k；top_p 沿用權重附的 generation_config。"""
        torch = self.torch
        L = self.layout
        ids = torch.tensor([self.prompt_ids(codes)], dtype=torch.long, device=self.device)
        torch.manual_seed(int(seed))
        t0 = time.perf_counter()
        with torch.no_grad(), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            out = self.model.generate(ids, attention_mask=torch.ones_like(ids), do_sample=True, temperature=float(temperature),
                                      top_k=int(CFG["speech_lm"]["top_k"]), max_new_tokens=frames * self.nq,
                                      eos_token_id=[L["end"]], pad_token_id=L["end"])
        if self.device.type == "mps":
            torch.mps.synchronize()
        gen_s = time.perf_counter() - t0
        new = [int(t) for t in out[0, ids.shape[1]:].tolist()]
        p = parse_ids(new, L["base"], self.nq, self.V, L["end"])
        full = np.concatenate([codes, p["codes"]], axis=1)
        t1 = time.perf_counter()
        wav = self.decode(full)
        dec_s = time.perf_counter() - t1
        return {"codes": p["codes"], "wav": wav, "n_prompt_frames": int(codes.shape[1]), "n_new_tokens": len(new),
                "n_frames": p["n_frames"], "stop": p["stop"], "gen_s": gen_s, "decode_s": dec_s}

    def text_token_count(self, text: str) -> int:
        """同一串字，用這個模型自己的（Llama 3.2 的）文字詞表數是幾個 token。"""
        return len(self.tok(text, add_special_tokens=False).input_ids)

    def info(self) -> dict:
        lock = all_locks()
        c = CFG["speech_lm"]
        return {"repo": c["repo"], "commit": lock.get(c["repo"]), "mimi_repo": c["mimi_repo"], "mimi_commit": lock.get(c["mimi_repo"]),
                "device": str(self.device), "dtype": self.dtype, "n_params": self.n_params, "num_quantizers": self.nq, "codebook_size": self.V,
                "frame_rate": self.frame_rate, "top_k": c["top_k"], "top_p_from_generation_config": self.top_p, "layout": self.layout,
                "load_s": round(self.load_s, 1)}

    def close(self) -> None:
        self.model = self.mimi = None
        gc.collect()
        if self.device.type == "mps":
            self.torch.mps.empty_cache()


class Judge:
    """兩位裁判：W7 Demo 1 的 Whisper small（importlib 載它的類別，同一份權重、同一個設定）與 W9 Demo 1 用的 WavLM-SV。"""

    def __init__(self):
        import torch
        from transformers import Wav2Vec2FeatureExtractor, WavLMForXVector
        quiet()
        spec = importlib.util.spec_from_file_location("w07_d1_demo", P(CFG["whisper"]["dir"]) / "demo.py")
        self.w7 = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.w7)
        self.whisper = self.w7.Whisper(CFG["whisper"]["device"])
        self.torch = torch
        self.sim_device = pick_device(CFG["sim"]["device"])
        sim_dir = local_snapshot(CFG["sim"]["repo"])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.sim_fe = Wav2Vec2FeatureExtractor.from_pretrained(sim_dir)
            self.sim_model = WavLMForXVector.from_pretrained(sim_dir).eval().to(self.sim_device)

    def transcribe(self, x16: np.ndarray) -> dict:
        ids = self.whisper.generate(x16, temperature=0.0, language="en")[0]
        lp = self.whisper.teacher_logprobs(x16, ids, language="en")
        text = self.whisper.text(ids).strip()
        return {"text": text, "n_words": len(words(text)), "avg_logprob": round(self.w7.avg_logprob(lp), 3),
                "compression_ratio": round(compression_ratio(text), 2), "repeat_3gram": round(repeat_fraction(text), 3)}

    def embed(self, x16: np.ndarray) -> np.ndarray:
        torch = self.torch
        inp = self.sim_fe([x16.astype(np.float32)], sampling_rate=FS, return_tensors="pt", padding=True)
        with torch.no_grad(), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            e = self.sim_model(**{k: v.to(self.sim_device) for k, v in inp.items()}).embeddings
        return e.detach().float().cpu().numpy().reshape(-1)

    def info(self) -> dict:
        lock = all_locks()
        return {"whisper": {"repo": CFG["whisper"]["repo"], "commit": lock.get(CFG["whisper"]["repo"]), "device": str(self.whisper.device)},
                "sim": {"repo": CFG["sim"]["repo"], "commit": lock.get(CFG["sim"]["repo"]), "device": str(self.sim_device),
                        "threshold_from_card": CFG["sim"]["threshold_from_card"]}}


class TextLM:
    """對照用的文字 LLM（mlx-lm；W1 Demo 2 的那一個）。mode = raw：把開頭的字直接當成序列的開頭讓它往下接，
    不套對話模板——與語音那一邊做的是同一件事。mode = chat：用一句指示請它接（模型不肯照 raw 接時的退路）。"""

    def __init__(self, repo: str | None = None, mode: str | None = None):
        from mlx_lm import load
        c = CFG["text_lm"]
        self.repo, self.mode = repo or c["repo"], mode or c["mode"]
        t0 = time.perf_counter()
        try:
            path = local_snapshot(self.repo)
        except Exception:
            path = self.repo
        self.model, self.tok = load(path)
        self.load_s = time.perf_counter() - t0

    def count(self, text: str) -> int:
        ids = list(self.tok.encode(text))
        bos = getattr(self.tok, "bos_token_id", None)
        return len(ids) - (1 if ids and bos is not None and ids[0] == bos else 0)

    def continue_(self, text: str, seed: int, temperature: float, max_tokens: int) -> dict:
        import mlx.core as mx
        from mlx_lm import stream_generate
        from mlx_lm.sample_utils import make_sampler
        c = CFG["text_lm"]
        if self.mode == "chat":
            msgs = [{"role": "user", "content": c["instruction"].format(text=text)}]
            try:
                prompt = self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
            except TypeError:
                prompt = self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        else:
            prompt = list(self.tok.encode(text))
        try:
            sampler = make_sampler(temp=float(temperature), top_k=int(c["top_k"]))
        except TypeError:                                    # 舊版的 make_sampler 沒有 top_k
            sampler = make_sampler(temp=float(temperature))
        mx.random.seed(int(seed))
        out, last, t0 = [], None, time.perf_counter()
        for r in stream_generate(self.model, self.tok, prompt, max_tokens=int(max_tokens), sampler=sampler):
            out.append(r.text)
            last = r
        return {"text": "".join(out).strip(), "n_tokens": int(getattr(last, "generation_tokens", len(out)) or len(out)),
                "tokens_per_s": round(float(getattr(last, "generation_tps", 0.0) or 0.0), 1), "seconds": round(time.perf_counter() - t0, 2)}

    def info(self) -> dict:
        return {"repo": self.repo, "commit": all_locks().get(self.repo), "mode": self.mode, "top_k": CFG["text_lm"]["top_k"], "load_s": round(self.load_s, 1)}

    def close(self) -> None:
        self.model = None
        gc.collect()


# ── 沒有模型時的替身（fake、selftest）────────────────────────────
class FakeSpeechLM:
    nq, V, frame_rate, fs, hop, top_p = 4, 2048, 12.5, 24000, 1920, 0.9
    layout = {"base": 128256, "start": 136448, "end": 136449, "bos": 128000, "text_vocab": 128256, "vocab": 136480}

    def encode(self, x):
        T = n_frames(len(x) / self.fs, self.frame_rate)
        return np.random.default_rng(0).integers(0, self.V, size=(self.nq, T))

    def decode(self, codes):
        rng = np.random.default_rng(int(codes.sum()) % (2 ** 32))
        return (0.02 * rng.standard_normal(codes.shape[1] * self.hop)).astype(np.float32)

    def continue_(self, codes, seed, temperature, frames):
        new = np.random.default_rng(seed).integers(0, self.V, size=(self.nq, frames))
        ids = codes_to_ids(new, self.layout["base"])
        p = parse_ids(ids, self.layout["base"], self.nq, self.V, self.layout["end"])
        return {"codes": p["codes"], "wav": self.decode(np.concatenate([codes, new], axis=1)), "n_prompt_frames": int(codes.shape[1]),
                "n_new_tokens": len(ids), "n_frames": frames, "stop": p["stop"], "gen_s": frames * self.nq / 40.0, "decode_s": 0.1}

    def text_token_count(self, text):
        return len(words(text)) + 2

    def info(self):
        return {"repo": "FAKE", "device": "none", "dtype": "none", "num_quantizers": 4, "frame_rate": 12.5, "top_k": 30, "layout": self.layout}

    def close(self):
        pass


class FakeJudge:
    LINES = ["and the people were very glad to see him and the people were very glad",
             "of the country and he said that it was a great thing to be done in the house",
             "which is the first of the kind that I have ever seen in the middle of the day",
             "and so on and so on and so on and so on and so on and so on"]

    def __init__(self):
        self.k = 0

    def transcribe(self, x16):
        text = self.LINES[self.k % len(self.LINES)]
        self.k += 1
        return {"text": text, "n_words": len(words(text)), "avg_logprob": -0.5 - 0.1 * (self.k % 4),
                "compression_ratio": round(compression_ratio(text), 2), "repeat_3gram": round(repeat_fraction(text), 3)}

    def embed(self, x16):
        return np.random.default_rng(len(x16) % 9973).standard_normal(16)

    def info(self):
        return {"whisper": {"repo": "FAKE"}, "sim": {"repo": "FAKE", "threshold_from_card": CFG["sim"]["threshold_from_card"]}}


class FakeTextLM:
    repo, mode = "FAKE", "raw"

    def count(self, text):
        return len(words(text)) + 1

    def continue_(self, text, seed, temperature, max_tokens):
        t = ", who have long looked to him for a plain account of what they ought to believe about pictures, and he gives it to them."
        return {"text": t, "n_tokens": max_tokens, "tokens_per_s": 100.0, "seconds": 0.4}

    def info(self):
        return {"repo": "FAKE", "mode": "raw"}

    def close(self):
        pass


# ── 彩排：全部算一遍 ─────────────────────────────────────────────
def prompt_audio() -> dict:
    c = CFG["prompt"]
    x = C.load_wav(P(c["wav"]), FS).astype(np.float32)
    n = int(round(float(c["seconds"]) * FS))
    return {"full": x, "head": x[:n], "rest": x[n:]}


def save16(path: Path, x: np.ndarray, fs: int) -> None:
    C.save_wav(path, np.asarray(x, dtype=np.float64), fs)


def draw_record(k: int, d: dict, g: dict, lm, folder: Path) -> dict:
    """一段續寫：存兩個 .wav（開頭接著續寫；只有續寫），回傳數字。"""
    cut = g["n_prompt_frames"] * lm.hop
    cont = g["wav"][cut:]
    save16(folder / f"draw_{k}.wav", g["wav"], lm.fs)
    save16(folder / f"cont_{k}.wav", cont, lm.fs)
    audio_s = g["n_frames"] / lm.frame_rate
    return {"k": k, "seed": d["seed"], "temperature": d["temperature"], "n_new_tokens": g["n_new_tokens"], "n_frames": g["n_frames"],
            "n_speech_tokens": g["n_frames"] * lm.nq,
            "audio_s": round(audio_s, 2), "stop": g["stop"], "gen_s": round(g["gen_s"], 2), "decode_s": round(g["decode_s"], 3),
            "tokens_per_s": round(g["n_new_tokens"] / g["gen_s"], 1) if g["gen_s"] else None,
            "rtf": round(g["gen_s"] / audio_s, 2) if audio_s else None}


def judge_draw(row: dict, judge, e_prompt: np.ndarray, folder: Path, fs: int) -> None:
    x16 = resample(C.load_wav(folder / f"cont_{row['k']}.wav", fs), fs, FS)
    row["whisper"] = judge.transcribe(x16)
    row["sim_to_prompt"] = round(cosine(e_prompt, judge.embed(x16)), 3)


def compute(folder: Path, lm=None, judge=None, text=None, draws: list | None = None, tag: str = "", log=print) -> dict:
    """lm／judge／text 沒給就各自載、用完就放（一次只留一個大模型在記憶體裡）。"""
    folder.mkdir(parents=True, exist_ok=True)
    c, a = CFG["prompt"], prompt_audio()
    draws = draws if draws is not None else CFG["draws"]
    rec = {"date": C.now(), "tag": tag, "digest": digest(), "versions": versions(), "models": {},
           "prompt": {"utterance": c["utterance"], "seconds": float(c["seconds"]), "text": c["text"], "rest_text": c["rest_text"],
                      "rest_seconds": round(len(a["rest"]) / FS, 2)}}
    # 1. 語音 LM
    own = lm is None
    lm = lm or SpeechLM()
    rec["models"]["speech_lm"] = lm.info()
    head24 = resample(a["head"], FS, lm.fs)
    codes = lm.encode(head24)
    frames_new = n_frames(float(CFG["speech_lm"]["continue_seconds"]), lm.frame_rate)
    rec["prompt"].update(n_frames=int(codes.shape[1]), n_speech_tokens=int(codes.size), text_tokens_llama=lm.text_token_count(c["text"]),
                         expected_frames=n_frames(float(c["seconds"]), lm.frame_rate))
    save16(folder / "prompt.wav", a["head"], FS)
    save16(folder / "real_rest.wav", a["rest"], FS)
    save16(folder / "prompt_mimi4.wav", lm.decode(codes), lm.fs)
    rec["draws"] = []
    for k, d in enumerate(draws, start=1):
        log(f"  續寫 {k}／{len(draws)}：seed {d['seed']}、temperature {d['temperature']}、{frames_new} frames……")
        g = lm.continue_(codes, d["seed"], d["temperature"], frames_new)
        rec["draws"].append(draw_record(k, d, g, lm, folder))
        log(f"    {rec['draws'][-1]['n_new_tokens']} tokens、{rec['draws'][-1]['gen_s']} s（{rec['draws'][-1]['tokens_per_s']} tokens/s，RTF {rec['draws'][-1]['rtf']}）、停在 {g['stop']}")
    fs_lm = lm.fs
    if own:
        lm.close()
    # 2. 裁判
    judge = judge or Judge()
    rec["models"].update(judge.info())
    e_prompt = judge.embed(a["head"])
    rec["prompt"]["whisper"] = judge.transcribe(a["head"])
    refs = {"real_rest": {"what": "the same utterance, after the cut (what the speaker actually said next)", "whisper": judge.transcribe(a["rest"]),
                          "sim_to_prompt": round(cosine(e_prompt, judge.embed(a["rest"])), 3)},
            "prompt_mimi4": {"what": "the prompt itself through Mimi, 4 levels (the codec's ceiling)",
                             "sim_to_prompt": round(cosine(e_prompt, judge.embed(resample(C.load_wav(folder / "prompt_mimi4.wav", fs_lm), fs_lm, FS))), 3)}}
    for name, what in (("same_speaker", "another utterance by the same speaker"), ("other_speaker", "an utterance by a different speaker")):
        p = P(CFG["refs"][name])
        if p.exists():
            refs[name] = {"what": what, "file": CFG["refs"][name], "sim_to_prompt": round(cosine(e_prompt, judge.embed(C.load_wav(p, FS).astype(np.float32))), 3)}
    rec["refs"] = refs
    for row in rec["draws"]:
        judge_draw(row, judge, e_prompt, folder, fs_lm)
    del judge
    gc.collect()
    # 3. 文字 LLM
    rate, src = text_rate()
    rec["text_lm"] = {"text_tokens_per_s": rate, "rate_source": src, "draws": []}
    if text is not False:
        try:
            own_t = text is None
            text = text or TextLM()
            rec["models"]["text_lm"] = text.info()
            rec["prompt"]["text_tokens_text_lm"] = text.count(c["text"])
            for row in rec["draws"]:
                mt = match_tokens(row["audio_s"], rate)       # 那一段續寫幾秒，文字就給幾秒份的 token
                rec["text_lm"]["draws"].append({"k": row["k"], "seed": row["seed"], "temperature": row["temperature"], "max_tokens": mt,
                                                **text.continue_(c["text"], row["seed"], row["temperature"], mt)})
            if own_t:
                text.close()
        except Exception as e:                               # 文字 LLM 是對照，不擋語音那一半
            rec["text_lm"]["error"] = repr(e)
            log(f"{C.RED}文字 LLM 沒跑成：{e!r}{C.RESET}")
    (folder / "rehearsal.json").write_text(json.dumps(rec, ensure_ascii=False, indent=1, default=float) + "\n", encoding="utf-8")
    return rec


def summary(rec: dict) -> dict:
    """投影片 p23 那張表用的數字（→ w11-metrics.json 的 demo1）。"""
    dr, p = rec["draws"], rec["prompt"]
    med = lambda v: float(np.median(v)) if v else None  # noqa: E731
    n_txt = p.get("text_tokens_text_lm") or p.get("text_tokens_llama")
    return {"date": rec["date"], "models": {k: {kk: v.get(kk) for kk in ("repo", "commit", "device", "dtype", "mode") if kk in v} for k, v in rec["models"].items()},
            "prompt": {"utterance": p["utterance"], "seconds": p["seconds"], "n_frames": p["n_frames"], "n_speech_tokens": p["n_speech_tokens"],
                       "text": p["text"], "text_tokens_text_lm": p.get("text_tokens_text_lm"), "text_tokens_llama": p.get("text_tokens_llama"),
                       "speech_to_text_token_ratio": round(p["n_speech_tokens"] / n_txt, 1) if n_txt else None},
            "generation": {"continue_seconds": med([d["audio_s"] for d in dr]), "tokens_per_s_median": med([d["tokens_per_s"] for d in dr]),
                           "rtf_median": med([d["rtf"] for d in dr]), "tokens_per_s_needed_for_real_time": 50.0,
                           "stops": [d["stop"] for d in dr]},
            "sim": {"draws": [d.get("sim_to_prompt") for d in dr], **{k: v["sim_to_prompt"] for k, v in rec.get("refs", {}).items()},
                    "threshold_from_card": CFG["sim"]["threshold_from_card"]},
            "whisper": {"avg_logprob_draws": [d["whisper"]["avg_logprob"] for d in dr if "whisper" in d],
                        "avg_logprob_real_speech": rec.get("refs", {}).get("real_rest", {}).get("whisper", {}).get("avg_logprob"),
                        "compression_ratio_draws": [d["whisper"]["compression_ratio"] for d in dr if "whisper" in d]},
            "draws": [{"seed": d["seed"], "temperature": d["temperature"], "transcript": d.get("whisper", {}).get("text")} for d in dr],
            "text_lm_draws": [{"seed": t["seed"], "text": t["text"]} for t in rec.get("text_lm", {}).get("draws", [])]}


def cmd_export(args=None, rec: dict | None = None) -> None:
    rec = rec or load_rec(RUNS / "rehearsal")
    data = json.loads(METRICS.read_text(encoding="utf-8")) if METRICS.exists() else {}
    data["demo1"] = summary(rec)
    METRICS.parent.mkdir(parents=True, exist_ok=True)
    METRICS.write_text(json.dumps(data, ensure_ascii=False, indent=1, default=float) + "\n", encoding="utf-8")
    print(f"→ {METRICS.relative_to(DEMOS.parent)} → demo1")


def load_rec(folder: Path) -> dict:
    p = folder / "rehearsal.json"
    if not p.exists():
        sys.exit(f"沒有 {p.relative_to(HERE)}：先跑 ./present.sh rehearse")
    return json.loads(p.read_text(encoding="utf-8"))


# ── 畫面 ─────────────────────────────────────────────────────────
def para(label: str, text: str, color: str = "") -> None:
    body = textwrap.fill(text or "(nothing)", WIDTH - 22, subsequent_indent=" " * 22)
    print(f"  {color}{label:<20}{C.RESET if color else ''}{body}")


def show_text(rec: dict) -> None:
    C.banner("the same opening, continued two ways — read as text")
    p = rec["prompt"]
    para("the prompt, in words", p["text"], C.BOLD)
    C.note(f"  Whisper on the {p['seconds']} s prompt: {p.get('whisper', {}).get('text', '—')}")
    tl = {t["k"]: t for t in rec.get("text_lm", {}).get("draws", [])}
    for d in rec["draws"]:
        print(f"\n  {C.GREY}draw {d['k']} · seed {d['seed']} · temperature {d['temperature']}{C.RESET}")
        para("speech LM → Whisper", d.get("whisper", {}).get("text", "(not transcribed)"), C.BLUE)
        if d["k"] in tl:
            para("text LLM", tl[d["k"]]["text"], C.AMBER)
    if not tl:
        C.note("\n  （文字 LLM 那一欄沒有：" + rec.get("text_lm", {}).get("error", "這次沒跑") + "）")
    tm = rec["models"].get("text_lm", {})
    if tl:
        C.note(f"\n  text LLM: {tm.get('repo')} (mode {tm.get('mode')}), given the prompt's words and as many tokens as the draw's seconds × "
               f"{rec['text_lm']['text_tokens_per_s']} text tokens/s.")
    C.note("  The speech LM's words are Whisper's reading of the audio — Whisper repairs unclear speech into fluent words (W7). Key w.")


def show_numbers(rec: dict) -> None:
    C.banner("numbers")
    p, s = rec["prompt"], summary(rec)
    print(f"  prompt            {p['seconds']} s = {p['n_frames']} frames × {rec['models']['speech_lm'].get('num_quantizers', 4)} levels = "
          f"{C.BOLD}{p['n_speech_tokens']} speech tokens{C.RESET}")
    if p.get("text_tokens_text_lm") is not None:
        print(f"  the same words    {p['text_tokens_text_lm']} text tokens (the text LLM's tokenizer)   ratio {s['prompt']['speech_to_text_token_ratio']} : 1")
    if p.get("text_tokens_llama") is not None:
        print(f"                    {p['text_tokens_llama']} text tokens (Llama 3.2's tokenizer — the vocabulary this model started from)")
    print(f"\n  {'draw':<6}{'seed':>5}{'temp':>6}{'tokens':>8}{'audio':>8}{'gen':>8}{'tok/s':>8}{'RTF':>7}{'SIM':>7}   stop")
    for d in rec["draws"]:
        print(f"  {d['k']:<6}{d['seed']:>5}{d['temperature']:>6}{d.get('n_speech_tokens', d['n_new_tokens']):>8}{d['audio_s']:>7}s{d['gen_s']:>7}s{d['tokens_per_s']:>8}{d['rtf']:>7}"
              f"{d.get('sim_to_prompt', float('nan')):>7}   {d['stop']}")
    print(f"\n  real time needs {s['generation']['tokens_per_s_needed_for_real_time']:.0f} tokens / s (12.5 frames × 4 levels); RTF < 1 means faster than playback.")
    print(f"\n  speaker similarity to the prompt (WavLM-SV cosine; the model card's same-speaker threshold is {CFG['sim']['threshold_from_card']}):")
    for k, v in rec.get("refs", {}).items():
        print(f"    {v['sim_to_prompt']:>6}   {v['what']}")
    print(f"    {C.BOLD}{' '.join(str(d.get('sim_to_prompt')) for d in rec['draws'])}{C.RESET}   the continuations")
    m = rec["models"]["speech_lm"]
    C.note(f"\n  {m.get('repo')} on {m.get('device')} ({m.get('dtype')}); temperature / top-k {m.get('top_k')} as on the model card; rehearsal of {rec['date']}")


def show_whisper(rec: dict) -> None:
    C.banner("how sure was Whisper? (W7: average log-probability of its own transcript)")
    r = rec.get("refs", {}).get("real_rest", {}).get("whisper")
    print(f"  {'':<22}{'avg log-prob':>13}{'compression':>13}{'repeated 3-grams':>18}{'words':>7}")
    if r:
        print(f"  {'real speech (key r)':<22}{r['avg_logprob']:>13}{r['compression_ratio']:>13}{r['repeat_3gram']:>18}{r['n_words']:>7}")
    for d in rec["draws"]:
        w = d.get("whisper")
        if w:
            print(f"  {'draw ' + str(d['k']):<22}{w['avg_logprob']:>13}{w['compression_ratio']:>13}{w['repeat_3gram']:>18}{w['n_words']:>7}")
    C.note("  Lower log-probability = Whisper is guessing more. W7's fallback rule fires below −1.0 or above a compression ratio of 2.4.")


def show_real(rec: dict) -> None:
    C.banner("what the speaker actually said next")
    p = rec["prompt"]
    para("reference text", p["rest_text"], C.GREEN)
    r = rec.get("refs", {}).get("real_rest", {})
    if r:
        para("Whisper", r["whisper"]["text"])
        print(f"  {p['rest_seconds']} s of real speech; SIM to the prompt {r['sim_to_prompt']}")


MENU = """  p  the prompt            1 2 3 4  a continuation (prompt + continuation)      c  toggle: continuation only
  t  the words: prompt / speech LM (via Whisper) / text LLM                    n  numbers
  w  how sure was Whisper   r  what the speaker really said next   m  the prompt through Mimi, 4 levels
  g  draw one more, live    ?  this menu    q  quit"""


class Session:
    def __init__(self, rec: dict, folder: Path, mode: str):
        self.rec, self.folder, self.mode = rec, folder, mode
        self.cont_only = False
        self.lm = self.judge = None
        self.live_k = 0

    def play(self, name: str, what: str) -> None:
        p = self.folder / name
        if not p.exists():
            print(f"{C.RED}沒有 {name}{C.RESET}")
            return
        print(f"  ▶ {what}")
        if self.mode != "fake" or sys.stdin.isatty():
            C.play(p)

    def live(self) -> None:
        if self.mode == "replay":
            C.note("  replay 不載模型；現場再抽要用 ./present.sh")
            return
        fake = self.mode == "fake"
        if self.lm is None:
            C.note("  載入語音 LM 與兩位裁判（第一次按 g 才載）……")
            self.lm = FakeSpeechLM() if fake else SpeechLM()
            self.judge = FakeJudge() if fake else Judge()
        self.live_k += 1
        seed = int(CFG["live"]["seed_start"]) + self.live_k - 1
        d = {"seed": seed, "temperature": float(CFG["live"]["temperature"])}
        out = RUNS / "live"
        out.mkdir(parents=True, exist_ok=True)
        a = prompt_audio()
        codes = self.lm.encode(resample(a["head"], FS, self.lm.fs))
        frames = n_frames(float(CFG["speech_lm"]["continue_seconds"]), self.lm.frame_rate)
        print(f"  generating {frames * self.lm.nq} tokens (seed {seed}, temperature {d['temperature']})……")
        g = self.lm.continue_(codes, seed, d["temperature"], frames)
        k = 100 + self.live_k
        row = draw_record(k, d, g, self.lm, out)
        print(f"  {row['n_speech_tokens']} speech tokens ({row['audio_s']} s of audio) in {row['gen_s']} s = {C.BOLD}{row['tokens_per_s']} tokens/s{C.RESET}, real-time factor {row['rtf']} (stop: {row['stop']})")
        print("  ▶ live draw")
        if not fake or sys.stdin.isatty():
            C.play(out / f"draw_{k}.wav")
        judge_draw(row, self.judge, self.judge.embed(a["head"]), out, self.lm.fs)
        para("speech LM → Whisper", row["whisper"]["text"], C.BLUE)
        print(f"  SIM to the prompt {row['sim_to_prompt']}; Whisper avg log-prob {row['whisper']['avg_logprob']}")

    def key(self, k: str) -> bool:
        if k in ("q", "Q", "\x03", "\x04", ""):
            return False
        n = len(self.rec["draws"])
        if k == "p":
            self.play("prompt.wav", f"the prompt, {self.rec['prompt']['seconds']} s")
        elif k.isdigit() and 1 <= int(k) <= n:
            d = self.rec["draws"][int(k) - 1]
            name = f"cont_{k}.wav" if self.cont_only else f"draw_{k}.wav"
            self.play(name, f"draw {k} (seed {d['seed']}, temperature {d['temperature']})" + (" — continuation only" if self.cont_only else " — prompt, then the model"))
        elif k == "c":
            self.cont_only = not self.cont_only
            print(f"  keys 1–{n} now play " + ("the continuation only" if self.cont_only else "the prompt followed by the continuation"))
        elif k == "t":
            show_text(self.rec)
        elif k == "n":
            show_numbers(self.rec)
        elif k == "w":
            show_whisper(self.rec)
        elif k == "r":
            show_real(self.rec)
            self.play("real_rest.wav", "the real continuation")
        elif k == "m":
            self.play("prompt_mimi4.wav", "the prompt through Mimi at 4 levels — the best this token stream can sound")
        elif k == "g":
            try:
                self.live()
            except Exception as e:
                print(f"{C.RED}現場生成失敗：{e!r}——彩排的四段還在（1–{n}）{C.RESET}")
        elif k in ("?", "h"):
            print(MENU)
        return True

    def loop(self) -> None:
        tag = {"fake": f"{C.RED}FAKE — 聲音是噪音，數字沒有意義{C.RESET}", "replay": "replay（只讀彩排的紀錄）", "show": ""}[self.mode]
        C.banner(f"W11 Demo 1 · a language model with no text in its sequence   {tag}")
        m = self.rec["models"].get("speech_lm", {})
        C.note(f"  {m.get('repo')} · rehearsal of {self.rec['date']} · {len(self.rec['draws'])} draws\n")
        print(MENU)
        while True:
            print(f"\n{C.GREY}key ›{C.RESET} ", end="", flush=True)
            k = C.getkey()
            print(k if k.strip() else "")
            if not self.key(k):
                break


# ── 子命令 ───────────────────────────────────────────────────────
def cmd_rehearse(args) -> None:
    preview = bool(args.preview)
    folder = RUNS / ("preview" if preview else "rehearsal")
    draws = CFG["draws"][: args.draws] if args.draws else CFG["draws"]
    text = False if args.no_text else (TextLM(repo=args.text_repo) if args.text_repo else None)
    C.banner(f"rehearse → {folder.relative_to(HERE)}/")
    lm = SpeechLM(device=args.device) if args.device else None
    rec = compute(folder, lm=lm, text=text, draws=draws, tag="preview" if preview else "")
    show_text(rec)
    show_numbers(rec)
    show_whisper(rec)
    if not preview:
        cmd_export(rec=rec)
    print(f"\n→ {folder.relative_to(HERE)}/（rehearsal.json 與 {3 + 2 * len(rec['draws'])} 個 .wav）。聽一遍：./present.sh" + (" replay --preview" if preview else ""))


def cmd_show(args, mode: str) -> None:
    if mode == "fake":
        folder = RUNS / "fake"
        rec = compute(folder, lm=FakeSpeechLM(), judge=FakeJudge(), text=FakeTextLM(), tag="FAKE", log=lambda *_: None)
    else:
        folder = RUNS / ("preview" if getattr(args, "preview", False) else "rehearsal")
        rec = load_rec(folder)
        if rec.get("tag"):
            C.note(f"  這一份是 {rec['tag']}（{folder.name}/），不是定案的彩排。")
        if rec.get("digest") != digest():
            C.note("  注意：demo_config.toml 在彩排之後改過；畫面上是彩排當時的設定。")
    Session(rec, folder, mode).loop()


def cmd_fetch(args) -> None:
    from huggingface_hub import HfApi, snapshot_download
    api, lock = HfApi(), read_lock(LOCK)
    inherited = all_locks()
    c = CFG["speech_lm"]
    plan = [(c["repo"], None, "Llama-Mimi（約 5 GB）"), (c["mimi_repo"], ["*.json", "*.safetensors"], "Mimi"),
            (CFG["whisper"]["repo"], ["*.json", "*.txt", "*.safetensors"], "Whisper small（W7 Demo 1 的那一份）"),
            (CFG["sim"]["repo"], ["*.json", "*.txt", "*.safetensors", "*.bin"], "WavLM-SV（W9 Demo 1 的那一份）")]
    for repo, pats, what in plan:
        info = api.model_info(repo, revision=inherited.get(repo))
        sha = inherited.get(repo) or info.sha
        C.note(f"{what}：{repo} @ {sha[:8]} ……")
        snapshot_download(repo, revision=sha, allow_patterns=pats)
        lock[repo] = sha
        lic = getattr(info.card_data, "license", None) if info.card_data else None
        print(f"  {repo} @ {sha[:8]}  license {lic}")
    t = CFG["text_lm"]["repo"]
    try:
        sha = inherited.get(t) or api.model_info(t).sha
        C.note(f"文字 LLM：{t} @ {sha[:8]}（W1 Demo 2 已經下載過的話不會重抓）……")
        snapshot_download(t, revision=sha)
        lock[t] = sha
    except Exception as e:
        print(f"{C.RED}文字 LLM 沒抓到（{e!r}）；語音那一半不受影響，rehearse 會把那一欄留空。{C.RESET}")
    LOCK.write_text("".join(f"{k} {v}\n" for k, v in sorted(lock.items())), encoding="utf-8")
    for name in ("same_speaker", "other_speaker"):
        if not P(CFG["refs"][name]).exists():
            print(f"{C.AMBER}SIM 的參考檔不在：{CFG['refs'][name]}（那一列會留空）{C.RESET}")
    print("→ model.lock。下一步：./present.sh check")


def cmd_check(args) -> None:
    import torch
    out = {"date": C.now(), "versions": versions(), "mps_available": bool(torch.backends.mps.is_available()),
           "mps_fallback_env": __import__("os").environ.get("PYTORCH_ENABLE_MPS_FALLBACK"), "steps": {}}

    def step(name: str, fn):
        t0 = time.perf_counter()
        try:
            r = fn() or {}
            out["steps"][name] = {"ok": True, "t": round(time.perf_counter() - t0, 2), **r}
            print(f"  {C.GREEN}✓{C.RESET} {name:<40} {time.perf_counter() - t0:6.1f} s  {textwrap.shorten(json.dumps(r, ensure_ascii=False, default=float), 160)}")
        except Exception as e:
            out["steps"][name] = {"ok": False, "error": repr(e)}
            print(f"  {C.RED}✗{C.RESET} {name:<40} {e!r}")

    a = prompt_audio()
    box: dict = {}
    devices = ["cpu"] + (["mps"] if out["mps_available"] else [])
    n_probe = int(CFG["check"]["probe_tokens"])

    def probe(dev: str):
        lm = SpeechLM(device=dev)
        codes = lm.encode(resample(a["head"], FS, lm.fs))
        ids = torch.tensor([lm.prompt_ids(codes)], dtype=torch.long, device=lm.device)
        s = "<audio>" + "".join(f"<{int(codes[j, t])}_{j}>" for t in range(codes.shape[1]) for j in range(lm.nq))
        same_as_card = lm.tok(s).input_ids == lm.prompt_ids(codes)
        with torch.no_grad():
            logp = torch.log_softmax(lm.model(ids).logits[0, -1].float(), dim=-1).cpu().numpy()
        L = lm.layout
        audio_mass = float(np.exp(logp[L["base"]: L["base"] + lm.nq * lm.V]).sum())
        level0_mass = float(np.exp(logp[L["base"]: L["base"] + lm.nq * lm.V: lm.nq]).sum())
        g = lm.continue_(codes, 1, 0.8, n_probe // lm.nq)
        box[dev] = {"codes": codes, "logp": logp}
        r = {"dtype": lm.dtype, "load_s": round(lm.load_s, 1), "n_params": lm.n_params, "prompt_frames": int(codes.shape[1]),
             "prompt_ids_same_as_card_string": bool(same_as_card), "layout": L, "top_p_from_generation_config": lm.top_p,
             "next_token_mass_on_speech_tokens": round(audio_mass, 6), "next_token_mass_on_level_0": round(level0_mass, 6),
             "top1": int(logp.argmax()), "tokens_per_s": round(g["n_new_tokens"] / g["gen_s"], 1), "stop": g["stop"]}
        lm.close()
        return r

    for dev in devices:
        step(f"speech LM on {dev}", lambda dev=dev: probe(dev))
    if "cpu" in box and "mps" in box:
        step("mps vs cpu", lambda: {"mimi_codes_equal_frac": round(float((box["cpu"]["codes"] == box["mps"]["codes"]).mean()), 4),
                                    "top1_same": bool(box["cpu"]["logp"].argmax() == box["mps"]["logp"].argmax()),
                                    "top30_overlap": len(set(np.argsort(-box["cpu"]["logp"])[:30]) & set(np.argsort(-box["mps"]["logp"])[:30])),
                                    "max_abs_logp_diff_top30": round(float(np.abs(box["cpu"]["logp"] - box["mps"]["logp"])[np.argsort(-box["cpu"]["logp"])[:30]].max()), 4)})

    def judges():
        j = Judge()
        e = j.embed(a["head"])
        return {"whisper_on_prompt": j.transcribe(a["head"]), "expected": CFG["prompt"]["text"],
                "sim_prompt_vs_rest": round(cosine(e, j.embed(a["rest"])), 3), **j.info()}
    step("Whisper small + WavLM-SV", judges)
    rate, _ = text_rate()
    mt = match_tokens(float(CFG["speech_lm"]["continue_seconds"]), rate)
    for mode in ("raw", "chat"):
        def tl(mode=mode):
            t = TextLM(mode=mode)
            r = {"n_prompt_tokens": t.count(CFG["prompt"]["text"]), **t.continue_(CFG["prompt"]["text"], 1, 0.8, mt), **t.info()}
            t.close()
            return r
        step(f"text LLM, mode = {mode}", tl)
    RUNS.mkdir(exist_ok=True)
    (RUNS / "check.json").write_text(json.dumps(out, ensure_ascii=False, indent=1, default=float) + "\n", encoding="utf-8")
    bad = [k for k, v in out["steps"].items() if not v["ok"]]
    print(f"\n→ runs/check.json；{'全部通過' if not bad else '沒過的：' + '、'.join(bad)}")
    C.note("  看三件事：(1) mps 那一列的 tokens_per_s 有沒有過 50（即時）；(2) mps vs cpu 的 top1_same；(3) 文字 LLM 兩種接法哪一種像在「接下去」——"
           "預設是 raw，不像就把 demo_config.toml [text_lm] mode 改成 chat。")


def cmd_selftest(args) -> None:
    n_ok = n_bad = 0

    def t(name: str, cond: bool, extra: str = "") -> None:
        nonlocal n_ok, n_bad
        n_ok, n_bad = n_ok + bool(cond), n_bad + (not cond)
        print(f"  {C.GREEN + '✓' if cond else C.RED + '✗'}{C.RESET} {name} {C.GREY}{extra}{C.RESET}")

    F = FakeSpeechLM()
    base, nq, V, end = F.layout["base"], F.nq, F.V, F.layout["end"]
    t("3.2 s → 40 frames → 160 tokens", n_frames(3.2, 12.5) == 40 and 40 * nq == 160)
    t("3.0 s → 38 frames（37.5 進位；不是 150 個 token）", n_frames(3.0, 12.5) == 38)
    t("12 s → 150 frames → 600 tokens", n_frames(12.0, 12.5) * nq == 600)
    t("設定檔的開頭長度落在整數個 frame 上", abs(float(CFG["prompt"]["seconds"]) * 12.5 - round(float(CFG["prompt"]["seconds"]) * 12.5)) < 1e-9)
    codes = np.random.default_rng(1).integers(0, V, size=(nq, 7))
    ids = codes_to_ids(codes, base)
    t("id = base + N_q·v + j", ids[0] == base + nq * int(codes[0, 0]) and ids[5] == base + nq * int(codes[1, 1]) + 1)
    t("攤平的順序：一個 frame 的四層排完才換下一個", [(i - base) % nq for i in ids[:8]] == [0, 1, 2, 3, 0, 1, 2, 3])
    p = parse_ids(ids, base, nq, V, end)
    t("攤平 → 解析 還原", np.array_equal(p["codes"], codes) and p["stop"] == "length")
    p = parse_ids(ids[:10] + [end] + ids[10:], base, nq, V, end)
    t("遇到 </audio> 停，不滿一個 frame 的丟掉", p["n_frames"] == 2 and p["stop"] == "end tag" and p["dropped"] == 2)
    bad = ids[:8] + [ids[9]] + ids[9:]
    t("層的順序錯了就停", parse_ids(bad, base, nq, V, end)["n_frames"] == 2 and "broken at token 8" in parse_ids(bad, base, nq, V, end)["stop"])
    t("文字 token 出現就停", parse_ids(ids[:4] + [1234] + ids[4:], base, nq, V, end)["n_frames"] == 1)
    t("最大的 id 還在語音 token 的範圍裡", base + nq * (V - 1) + (nq - 1) == F.layout["start"] - 1)
    t("words()", words("Mister Quilter's gospel, and — we are glad!") == ["mister", "quilter's", "gospel", "and", "we", "are", "glad"])
    t("重複：不重複的句子是 0", repeat_fraction("the quick brown fox jumps over a lazy dog") == 0.0)
    loop = "and so on " * 8
    t("重複：繞圈的句子高", repeat_fraction(loop) > 0.8 and compression_ratio(loop) > 2.4, f"{repeat_fraction(loop):.2f}, {compression_ratio(loop):.2f}")
    t("接同樣長度：12 s × 3.42 ≈ 41 個文字 token", match_tokens(12.0, 3.42) == 41)
    t("cosine", abs(cosine(np.array([1.0, 0.0]), np.array([1.0, 0.0])) - 1) < 1e-9 and abs(cosine(np.array([1.0, 0.0]), np.array([0.0, 1.0]))) < 1e-9)
    x = np.sin(2 * np.pi * 440 * np.arange(FS) / FS)
    t("16 → 24 kHz 的長度", len(resample(x, FS, 24000)) == 24000 and len(resample(resample(x, FS, 24000), 24000, FS)) == FS)
    a = prompt_audio()
    t("開頭的檔案在，切得出 3.2 s 與剩下的", len(a["head"]) == int(round(float(CFG["prompt"]["seconds"]) * FS)) and len(a["rest"]) > FS,
      f"{len(a['head']) / FS:.2f} s + {len(a['rest']) / FS:.2f} s")
    ref = P(CFG["prompt"]["wav"]).with_suffix(".txt")
    if ref.exists():
        full = " ".join(words(ref.read_text(encoding="utf-8")))
        t("開頭的字 + 剩下的字 = 那一句的參考文字", " ".join(words(CFG["prompt"]["text"]) + words(CFG["prompt"]["rest_text"])) == full)
    t("抽幾段：至少三段、seed 不重複、兩個 temperature", len(CFG["draws"]) >= 3 and len({d["seed"] for d in CFG["draws"]}) == len(CFG["draws"])
      and len({d["temperature"] for d in CFG["draws"]}) == 2)
    folder = RUNS / "_selftest"
    rec = compute(folder, lm=FakeSpeechLM(), judge=FakeJudge(), text=FakeTextLM(), tag="FAKE", log=lambda *_: None)
    t("fake 整條流程", len(rec["draws"]) == len(CFG["draws"]) and all((folder / f"draw_{d['k']}.wav").exists() and "whisper" in d for d in rec["draws"]))
    t("fake：開頭 160 個 token、每段 600 個", rec["prompt"]["n_speech_tokens"] == 160 and all(d["n_new_tokens"] == 600 for d in rec["draws"]))
    fs_c, yc = __import__("scipy.io.wavfile", fromlist=["read"]).read(str(folder / "cont_1.wav"))
    fs_d, yd = __import__("scipy.io.wavfile", fromlist=["read"]).read(str(folder / "draw_1.wav"))
    t("兩個 .wav 的長度：續寫 12 s、開頭接續寫 15.2 s", abs(len(yc) / fs_c - 12.0) < 1e-6 and abs(len(yd) / fs_d - 15.2) < 1e-6)
    s = summary(rec)
    t("summary 的欄位", s["prompt"]["n_speech_tokens"] == 160 and len(s["draws"]) == len(rec["draws"]) and s["generation"]["tokens_per_s_needed_for_real_time"] == 50.0)
    sess = Session(rec, folder, "fake")
    import contextlib
    import io
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        alive = all(sess.key(k) for k in "ptnwrc?") and not sess.key("q")
    t("每一個鍵（不播聲音的）", alive and "speech LM → Whisper" in buf.getvalue() and "speech tokens" in buf.getvalue())
    reh = RUNS / "rehearsal" / "rehearsal.json"
    if reh.exists() and METRICS.exists():
        m = json.loads(METRICS.read_text(encoding="utf-8")).get("demo1")
        r = json.loads(reh.read_text(encoding="utf-8"))
        t("w11-metrics.json → demo1 與彩排的紀錄一致", m is not None and json.loads(json.dumps(summary(r), default=float)) == m)
    else:
        C.note("  （還沒有彩排紀錄，不對 w11-metrics.json）")
    print(f"\n{n_ok} 項通過" + (f"、{C.RED}{n_bad} 項沒過{C.RESET}" if n_bad else ""))
    sys.exit(1 if n_bad else 0)


def main() -> None:
    ap = argparse.ArgumentParser(description="W11 demo 1: a speech-only language model continues a prompt")
    sub = ap.add_subparsers(dest="cmd")
    for name in ("fake", "selftest", "export", "fetch", "check"):
        sub.add_parser(name)
    for name in ("show", "replay"):
        sub.add_parser(name).add_argument("--preview", action="store_true", help="讀 runs/preview/ 而不是 runs/rehearsal/")
    r = sub.add_parser("rehearse")
    r.add_argument("--preview", action="store_true", help="寫到 runs/preview/，不寫 w11-metrics.json")
    r.add_argument("--draws", type=int, default=0, help="只抽前 n 段")
    r.add_argument("--device", default=None, help="cpu 或 mps（預設照 demo_config.toml）")
    r.add_argument("--text-repo", default=None, help="換一個文字 LLM（預覽用）")
    r.add_argument("--no-text", action="store_true", help="不跑文字 LLM")
    args = ap.parse_args()
    cmd = args.cmd or "show"
    if cmd in ("show", "replay", "fake"):
        cmd_show(args, cmd)
    else:
        {"rehearse": cmd_rehearse, "fetch": cmd_fetch, "check": cmd_check, "export": cmd_export, "selftest": cmd_selftest}[cmd](args)


if __name__ == "__main__":
    main()

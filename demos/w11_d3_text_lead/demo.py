# -*- coding: utf-8 -*-
# 此檔由 bin/publish_demos.py 從 private repo 的 <course>/demos/ 複製而來，請勿在公開 repo 直接編輯。
"""W11 demo 3：文字領先多少——把輸出一個 token 一個 token 印出來。（只在終端機，單鍵）

一個把文字 token 與聲音 token 寫進同一個輸出的模型（LFM2.5-Audio-1.5B，MLX）。一個用說的問題（英語，錄好的檔案）交給它；
它生成的每一個 token 照順序印出來——文字一種顏色、一個聲音 frame 一格——每一個都帶著時間戳。
前半：轉「每一輪寫幾個字、出幾個 frame」這個旋鈕，看第一個聲音早了多少、念出來的還跟寫的一樣嗎。
後半：同一道算術題，邊寫邊念一次、先在文字裡推理再只念答案一次，看第一個聲音之前等了多久、答案有沒有變好。

子命令：
  fetch      下載 LFM2.5-Audio，確認 Whisper 在快取裡；audio/ 裡沒有問題的聲音就合成（要網路）→ model.lock
  tts        只重新合成問題的聲音（audio/q*.wav；--force 覆蓋）
  record     用自己的聲音錄一題：record q1（寫到 ../_media/w11_d3/，不進版控；有這個檔就優先用它）
  check      課前體檢：排法、速度（對即時的 80 ms／frame）、分塊解碼對整段解碼、sequential 模式在對話裡的行為、
             Whisper、cascade 的三個後端 → runs/check.json
  rehearse   全部算一遍 → runs/rehearsal/（.wav 與 rehearsal.json，進版控＝現場的退路），寫 w11-metrics.json → demo3
             --preview 寫到 runs/preview/（不進版控、不寫 metrics）；--questions 1 只跑第一題；--no-cascade；--no-think
  show       上課用：模型載起來，按鍵現場生成；表格讀彩排的
  replay     只讀彩排的紀錄：token 照當時的時間戳重印一遍、播當時的聲音，不載任何模型
  fake       沒有模型也能演練流程（畫面標 FAKE，聲音是噪音、數字沒有意義）
  selftest   trace.py 的每一條定義、設定檔、fake 整條流程；有彩排紀錄時對 w11-metrics.json
  export     只重寫 slides/assets/w11/w11-metrics.json → demo3

畫面上的鍵（順序照投影片講稿的【操作】）：
  p  播問題                  1  同一題：文字全部先寫完（兩趟）→ 預設的 6 : 12       2  掃比例 2 : 12、6 : 12、12 : 12、24 : 12
  t  表（彩排，三題合計）     3  兩個條件代量到的數字                          4  同一題走 W1 的 cascade（彩排的數字；$ 現場跑）
  5  算術題：第一次邊寫邊念（6 : 12）、第二次先在文字裡推理再只念答案          6  後半的表
  w  上一段：模型寫的字對 Whisper 聽到的字        l  再播上一段        ?  選單        q  離開

數字的定義（細節在 trace.py）：
  first text token   第一個文字 token 算完的時刻；從問題的檔案交給模型起算（不含「等使用者說完」那一格）
  first speech frame 第一個聲音 frame（8 個 code）算完的時刻
  first sound        頭 chunk_frames 個 frame 算完並解成波形的時刻——最早能開始播的時刻
  real-time factor   生成花的時間 ÷ 生成出來的聲音長度（< 1 才播得順）
  WER                模型生成的聲音交給 Whisper small 轉成字，對模型自己寫的字；幾題合計
  blank              文字 token 裡解出來只有空白的比例（模型在沒見過的排法上寫不出字時，這個數字先動）
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.util
import json
import statistics as st
import sys
import textwrap
import time
import tomllib
import warnings
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
DEMOS = HERE.parent
sys.path.insert(0, str(DEMOS))
sys.path.insert(0, str(HERE))
import common as C  # noqa: E402
import trace as T  # noqa: E402

CFG = tomllib.loads((HERE / "demo_config.toml").read_text(encoding="utf-8"))
RUNS = HERE / "runs"
LOCK = HERE / "model.lock"
AUDIO = HERE / "audio"
OWN = DEMOS / "_media" / "w11_d3"
ASSETS = DEMOS.parent / "slides" / "assets" / "w11"
METRICS = ASSETS / "w11-metrics.json"
FS_OUT = 24000
FS_ASR = 16000
WIDTH = 100
F_R = float(CFG["model"]["frame_rate"])
EOS = 2048                                   # 聲音的結束碼（mlx_audio.sts.models.lfm_audio.model.AUDIO_EOS_TOKEN；check 會對）
MAGENTA, CYAN = "\033[35m", "\033[36m"
SETTINGS = {s["key"]: s for s in CFG["settings"]}
QUESTIONS = {q["id"]: q for q in CFG["questions"]}


def P(rel: str) -> Path:
    return (HERE / rel).resolve()


def resample(x: np.ndarray, fs_in: int, fs_out: int) -> np.ndarray:
    if fs_in == fs_out:
        return np.asarray(x, dtype=np.float32)
    from math import gcd
    from scipy.signal import resample_poly
    g = gcd(fs_in, fs_out)
    return resample_poly(x, fs_out // g, fs_in // g).astype(np.float32)


def digest() -> str:
    keys = ("model", "questions", "settings", "think", "tts", "chain")
    return hashlib.sha256(json.dumps({k: CFG[k] for k in keys}, sort_keys=True, ensure_ascii=False).encode()).hexdigest()[:16]


def ms(v: float | None) -> str:
    return "—" if v is None else (f"{v * 1000:.0f} ms" if v < 10 else f"{v:.1f} s")


def pct(v: float | None) -> str:
    return "—" if v is None else f"{100 * v:.1f}%"


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
    d = read_lock(P(CFG["whisper"]["lock_from"]))
    d.update(read_lock(LOCK))
    return d


def local_snapshot(repo: str, sha: str | None = None) -> str:
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
    for pkg in ("mlx", "mlx-audio", "mlx-lm", "torch", "transformers", "huggingface_hub", "pywhispercpp"):
        try:
            v[pkg] = metadata.version(pkg)
        except metadata.PackageNotFoundError:
            v[pkg] = None
    return v


def quiet() -> None:
    warnings.filterwarnings("ignore")
    try:
        from transformers.utils import logging as hf_logging
        hf_logging.set_verbosity_error()
        hf_logging.disable_progress_bar()
    except Exception:
        pass


# ── 問題的聲音 ───────────────────────────────────────────────────
def question_wav(qid: str) -> Path:
    """自己錄的（../_media/w11_d3/）優先；沒有就用合成的（audio/）。"""
    own = OWN / f"{qid}.wav"
    return own if own.exists() else AUDIO / f"{qid}.wav"


def question(qid: str) -> dict:
    q = dict(QUESTIONS.get(qid) or CFG["think"])
    path = question_wav(qid)
    if not path.exists():
        raise FileNotFoundError(f"沒有 {path.name}：先跑 ./present.sh fetch（或 ./present.sh tts）")
    from scipy.io import wavfile
    fs, x = wavfile.read(str(path))
    x = x.astype(np.float32) / (32768.0 if x.dtype == np.int16 else 1.0)
    if x.ndim > 1:
        x = x.mean(axis=1)
    q.update(wav=path, fs=fs, x=x, seconds=len(x) / fs, own=path.parent == OWN)
    return q


def all_question_ids() -> list[str]:
    return [q["id"] for q in CFG["questions"]] + [CFG["think"]["id"]]


# ── 模型 ─────────────────────────────────────────────────────────
class LFM:
    """LFM2.5-Audio（mlx-audio 的 sts/models/lfm_audio）。ask() 自己叫 generate_interleaved——
    mlx-audio 的 generate_from_chat_state 不把交錯的比例傳下去——每拿到一個 token 就 mx.eval、蓋時間戳。"""
    fake = False
    tts_stray_text = 0

    def __init__(self, repo: str | None = None, warm: bool = True):
        import mlx.core as mx
        from mlx_audio.sts.models.lfm_audio import ChatState, LFM2AudioModel, LFM2AudioProcessor, LFMModality
        from mlx_audio.sts.models.lfm_audio import model as lfm_model
        quiet()
        self.mx, self.ChatState, self.Mod = mx, ChatState, LFMModality
        self.repo = repo or CFG["model"]["repo"]
        t0 = time.perf_counter()
        try:
            path = local_snapshot(self.repo)
        except Exception:
            path = self.repo
        self.model = LFM2AudioModel.from_pretrained(path)
        self.proc = LFM2AudioProcessor.from_pretrained(path)
        self.eos = int(lfm_model.AUDIO_EOS_TOKEN)
        self.text_end = int(lfm_model.TEXT_END_TOKEN)
        self.cfg_n_text = int(self.model.config.interleaved_n_text)
        self.cfg_n_audio = int(self.model.config.interleaved_n_audio)
        self.codebooks = int(self.model.config.codebooks)
        if warm:                                         # 第一次前向與 detokenizer 的載入不要算進量測
            self.decode([[0] * self.codebooks] * 2)
            self.ask(None, 0, text="Hi.", max_new=3)
        self.load_s = time.perf_counter() - t0

    def _chat(self, system: str, x, fs: int, text: str | None):
        mx = self.mx
        chat = self.ChatState(self.proc)
        if system:
            chat.new_turn("system")
            chat.add_text(system)
            chat.end_turn()
        chat.new_turn("user")
        if x is not None:
            chat.add_audio(mx.array(np.asarray(x, dtype=np.float32)), sample_rate=int(fs))
        if text:
            chat.add_text(text)
        chat.end_turn()
        chat.new_turn("assistant")
        return chat

    def ask(self, x, fs: int, system: str | None = None, text: str | None = None, n_text: int | None = None, n_audio: int | None = None,
            max_new: int | None = None, seed: int | None = None, on_event=None, sequential: bool = False) -> list[dict]:
        """一個問題（聲音 x，或文字 text）→ 事件串。計時從這裡開始：問題是錄好的檔案，交進來就算「使用者說完了」。"""
        mx, m = self.mx, CFG["model"]
        chat = self._chat(system or m["system"], x, fs, text)
        kw = dict(max_new_tokens=int(max_new or m["max_new_tokens"]), temperature=float(m["text_temperature"]),
                  audio_temperature=float(m["audio_temperature"]), audio_top_k=int(m["audio_top_k"]))
        mx.random.seed(int(m["seed"] if seed is None else seed))
        events, t0 = [], time.perf_counter()
        if sequential:
            gen = self.model.generate_sequential(**dict(chat), **kw)
        else:
            gen = self.model.generate_interleaved(**dict(chat), interleaved_n_text=n_text, interleaved_n_audio=n_audio, **kw)
        for tok, mod in gen:
            mx.eval(tok)
            t = time.perf_counter() - t0
            if mod == self.Mod.TEXT:
                i = int(tok.item())
                e = {"k": "T", "t": round(t, 4), "id": i, "s": self.proc.decode_text([i])}
            else:
                e = {"k": "A", "t": round(t, 4), "c": [int(v) for v in tok.tolist()]}
            events.append(e)
            if on_event:
                on_event(e)
        return events

    def ask_chain(self, x, fs: int, on_event=None, think: bool = False) -> list[dict]:
        """文字全部寫完才出聲音，用這個模型做得到的方式：兩趟，同一個時鐘。
        第一趟：不給 system prompt、sequential——模型像文字 LLM 一樣只寫字（寫到結束標記或上限）。
        第二趟：card 的 TTS 用法，把第一趟寫的字念出來。事件串的格式與 ask() 相同，所以同一套算術可以用。"""
        mx, m, tc = self.mx, CFG["model"], CFG["tts"]
        c = dict(CFG["chain"])
        if think:                                        # 後半：第一趟的字是推理（標 r，不念），第二趟只念最後一句
            c.update(user_text=CFG["think"]["silent_user_text"], max_text_tokens=CFG["think"]["silent_max_text_tokens"])
        events, t0 = [], time.perf_counter()
        chat = self._chat(c.get("system") or None, x, fs, c.get("user_text") or None)
        mx.random.seed(int(m["seed"]))
        for tok, mod in self.model.generate_sequential(**dict(chat), max_new_tokens=int(c["max_text_tokens"]), temperature=float(m["text_temperature"])):
            mx.eval(tok)
            if mod != self.Mod.TEXT:
                break
            i = int(tok.item())
            e = {"k": "T", "t": round(time.perf_counter() - t0, 4), "id": i, "s": self.proc.decode_text([i])}
            if think:
                e["r"] = 1
            events.append(e)
            if on_event:
                on_event(e)
        text = T.whole_sentences(T.text_of(events))
        if think:
            text = T.last_sentence(text)
        chat = self._chat(tc["system"], None, 0, text)
        mx.random.seed(int(tc["seed"]))
        self.tts_stray_text = 0
        for tok, mod in self.model.generate_sequential(**dict(chat), max_new_tokens=int(c["max_frames"]), temperature=float(tc["temperature"]),
                                                       audio_temperature=float(tc["audio_temperature"]), audio_top_k=int(tc["audio_top_k"])):
            mx.eval(tok)
            if mod != self.Mod.AUDIO_OUT:                # TTS 那一趟不該寫字；寫了就記下來（不印、不算進文字）
                self.tts_stray_text += 1
                continue
            e = {"k": "A", "t": round(time.perf_counter() - t0, 4), "c": [int(v) for v in tok.tolist()]}
            events.append(e)
            if on_event:
                on_event(e)
            if e["c"][0] == self.eos:
                break
        return events

    def decode(self, frames: list[list[int]]) -> np.ndarray:
        """frame 的清單（每個 N_q 個 code）→ 24 kHz 的波形。整段一次解。"""
        mx = self.mx
        if not frames:
            return np.zeros(0, dtype=np.float32)
        codes = mx.array(np.asarray(frames, dtype=np.int32).T[None])        # (1, N_q, T)
        w = self.proc.decode_audio(codes)
        mx.eval(w)
        return np.asarray(w, dtype=np.float32).reshape(-1)

    def decode_timed(self, frames: list[list[int]], chunk: int) -> dict:
        """整段解一次（存檔用），另外量：只解頭一塊要多久、只解第二塊要多久（時間軸用）。"""
        t0 = time.perf_counter()
        first = self.decode(frames[:chunk])
        d_first = time.perf_counter() - t0
        t0 = time.perf_counter()
        self.decode(frames[chunk:2 * chunk] or frames[:chunk])
        d_chunk = time.perf_counter() - t0
        t0 = time.perf_counter()
        wav = self.decode(frames)
        d_all = time.perf_counter() - t0
        n = min(len(first), len(wav))
        prefix_diff = float(np.max(np.abs(first[:n] - wav[:n]))) if n else None
        return {"wav": wav, "decode_first_s": d_first, "decode_chunk_s": d_chunk, "decode_all_s": d_all, "prefix_max_abs_diff": prefix_diff}

    def tts(self, text: str) -> np.ndarray:
        """card 的 TTS 用法：sequential、固定的 system prompt。"""
        mx, c = self.mx, CFG["tts"]
        chat = self._chat(c["system"], None, 0, text)
        mx.random.seed(int(c["seed"]))
        frames, self.tts_stray_text = [], 0
        for tok, mod in self.model.generate_sequential(**dict(chat), max_new_tokens=1024, temperature=float(c["temperature"]),
                                                       audio_temperature=float(c["audio_temperature"]), audio_top_k=int(c["audio_top_k"])):
            mx.eval(tok)
            if mod == self.Mod.AUDIO_OUT:
                f = [int(v) for v in tok.tolist()]
                if f[0] == self.eos:
                    break
                frames.append(f)
            else:
                self.tts_stray_text += 1
        return self.decode(frames)

    def info(self) -> dict:
        return {"repo": self.repo, "commit": all_locks().get(self.repo), "backend": "mlx", "load_s": round(self.load_s, 1),
                "config_ratio": [self.cfg_n_text, self.cfg_n_audio], "codebooks": self.codebooks, "eos": self.eos}

    def close(self) -> None:
        self.model = self.proc = None
        gc.collect()


class Judge:
    """W7 Demo 1 的 Whisper small（importlib 載它的類別）。只做一件事：把模型生成的聲音轉成字。"""
    fake = False

    def __init__(self):
        quiet()
        spec = importlib.util.spec_from_file_location("w07_d1_demo", P(CFG["whisper"]["dir"]) / "demo.py")
        self.w7 = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.w7)
        self.whisper = self.w7.Whisper(CFG["whisper"]["device"])

    def transcribe(self, x16: np.ndarray) -> dict:
        if len(x16) < FS_ASR // 10:
            return {"text": "", "avg_logprob": None}
        ids = self.whisper.generate(x16, temperature=0.0, language="en")[0]
        lp = self.whisper.teacher_logprobs(x16, ids, language="en")
        return {"text": self.whisper.text(ids).strip(), "avg_logprob": round(float(self.w7.avg_logprob(lp)), 3)}

    def words(self, text: str) -> list[str]:
        """Whisper 自己的 EnglishTextNormalizer（數字與拼法）再過 trace.norm_simple；前者不能用時只過後者。"""
        try:
            return T.norm_simple(self.whisper.tok.normalize(text))
        except Exception:
            return T.norm_simple(text)

    def info(self) -> dict:
        r = CFG["whisper"]["repo"]
        return {"repo": r, "commit": all_locks().get(r), "device": str(self.whisper.device)}


FAKE_ANSWER = ("The sky looks blue because sunlight scatters in the air, and blue light scatters the most. "
               "At sunset the light crosses more air, so the blue is scattered away and the red is left.")


class FakeLFM:
    """沒有模型時的替身：照排法吐事件、時間戳是算的。畫面標 FAKE。"""
    fake = True
    eos, text_end, cfg_n_text, cfg_n_audio, codebooks, load_s, repo = EOS, 130, 6, 12, 8, 0.0, "FAKE"

    def ask(self, x, fs, system=None, text=None, n_text=None, n_audio=None, max_new=None, seed=None, on_event=None, sequential=False):
        n_text, n_audio = int(n_text or 6), int(n_audio or 12)
        think = "step by step" in (system or "")
        arith = x is not None and len(x) == len(self.tts(CFG["think"]["text"]))
        src = ("48 minus 19 is 29. 29 plus 7 is 36. So there are 36 passengers." if think else "There are 36 passengers.") if arith else FAKE_ANSWER
        pieces = [(" " if i else "") + w for i, w in enumerate(src.split())]
        n_frames = int(len(pieces) / 3.4 * F_R)
        events, t, ti, ai = [], 0.25, 0, 0
        rng = np.random.default_rng(int(seed or 1))
        while ti < len(pieces) or ai < n_frames:
            for _ in range(n_text):
                if ti >= len(pieces):
                    break
                t += 0.012
                events.append({"k": "T", "t": round(t, 4), "id": 1000 + ti, "s": pieces[ti]})
                ti += 1
            k = n_frames - ai if ti >= len(pieces) else min(n_audio, n_frames - ai)
            for _ in range(max(k, 0)):
                t += 0.030
                events.append({"k": "A", "t": round(t, 4), "c": [int(v) for v in rng.integers(0, 2048, 8)]})
                ai += 1
            if len(events) >= int(max_new or CFG["model"]["max_new_tokens"]):
                break
        for e in events:
            if on_event:
                on_event(e)
        return events

    def ask_chain(self, x, fs, on_event=None, think=False):
        ev = self.ask(x, fs, n_text=10 ** 6, system="step by step" if think else None)
        for e in ev:
            if think and e["k"] == "T":
                e["r"] = 1
            if on_event:
                on_event(e)
        if think:                                        # 只念最後一句：聲音縮短
            keep = int(len([e for e in ev if e["k"] == "A"]) * 0.35)
            n = 0
            out = []
            for e in ev:
                if e["k"] == "A":
                    n += 1
                    if n > keep:
                        continue
                out.append(e)
            ev = out
        return ev + [{"k": "A", "t": round(ev[-1]["t"] + 0.03, 4), "c": [EOS] * 8}]

    def decode(self, frames):
        rng = np.random.default_rng(len(frames))
        return (0.02 * rng.standard_normal(int(len(frames) / F_R * FS_OUT))).astype(np.float32)

    def decode_timed(self, frames, chunk):
        return {"wav": self.decode(frames), "decode_first_s": 0.02, "decode_chunk_s": 0.02, "decode_all_s": 0.1, "prefix_max_abs_diff": 0.0}

    def tts(self, text):
        return self.decode([[0] * 8] * int(len(text.split()) / 2.6 * F_R))

    def info(self):
        return {"repo": "FAKE", "commit": None, "backend": "fake", "load_s": 0.0, "config_ratio": [6, 12], "codebooks": 8, "eos": EOS}

    def close(self):
        pass


class FakeJudge:
    """替身：回傳的字由 demo 塞進來的「模型寫的字」每隔幾個詞丟一個做成。"""
    fake = True

    def __init__(self):
        self.next_text, self.drop = "", 0

    def transcribe(self, x16):
        w = self.next_text.split()
        if self.drop:
            w = [v for i, v in enumerate(w) if (i + 1) % self.drop]
        return {"text": " ".join(w), "avg_logprob": -0.2}

    def words(self, text):
        return T.norm_simple(text)

    def info(self):
        return {"repo": "FAKE", "commit": None, "device": "none"}


# ── 一次問答 ─────────────────────────────────────────────────────
class Printer:
    """生成的同時把 token 印出來：文字是紫色的字，一個聲音 frame 是一個藍色的格子；每換一種就印當下的秒數。"""

    def __init__(self, pace: bool = False):
        self.last, self.col, self.pace, self.t_prev = None, 0, pace, 0.0

    def __call__(self, e: dict) -> None:
        if self.pace:                                    # replay：照當時的時間戳走（最長等 0.5 秒）
            time.sleep(min(max(e["t"] - self.t_prev, 0.0), 0.5))
            self.t_prev = e["t"]
        if e["k"] != self.last:
            stamp = f" {C.DIM}{e['t']:.2f}s{C.RESET} "
            if self.col > WIDTH - 24:
                sys.stdout.write("\n  ")
                self.col = 0
            sys.stdout.write(stamp)
            self.col += 8
            self.last = e["k"]
        if e["k"] == "T":
            s = T.SPECIAL.sub("", e["s"]).replace("\n", " ") or "·"
            if not s.strip():
                s = "␣"
            sys.stdout.write(f"{C.AMBER if e.get('r') else MAGENTA}{C.BOLD}{s}{C.RESET}")
            self.col += len(s)
        else:
            sys.stdout.write(f"{CYAN}▮{C.RESET}")
            self.col += 1
        if self.col > WIDTH:
            sys.stdout.write("\n  ")
            self.col = 0
        sys.stdout.flush()


def analyse(events: list[dict], dec: dict, lm, judge, n_text: int, n_audio: int, cap: int | None = None) -> dict:
    """一串事件 → 一列的數字。dec 是 decode_timed 的結果。"""
    chunk = int(CFG["model"]["chunk_frames"])
    tl = T.timeline(events, F_R, chunk, dec["decode_first_s"], dec["decode_chunk_s"], lm.eos)
    steps = T.step_times(events)
    lead = T.lead_curve(events, F_R, lm.eos)
    text = T.text_of(events)
    row = {"n_text": n_text, "n_audio": n_audio, "text": text, "timeline": {k: (round(v, 4) if isinstance(v, float) else v) for k, v in tl.items()},
           "steps": {k: (round(v, 4) if isinstance(v, float) else v) for k, v in steps.items()},
           "lead": {"R_txt": lead["R_txt"] and round(lead["R_txt"], 2), "min_lead": lead["min_lead"] and round(lead["min_lead"], 2), "at_s": lead["at_s"]},
           "blank_fraction": round(T.blank_fraction(events), 3), "schedule": T.schedule(events)[:12],
           "tokens_before_first_frame": T.tokens_before_first_frame(events),
           "decode": {k: (round(v, 4) if isinstance(v, float) else v) for k, v in dec.items() if k != "wav"},
           "ended": bool(events) and len(events) < int(cap or CFG["model"]["max_new_tokens"])}
    if judge is not None:
        if getattr(judge, "fake", False):
            judge.next_text, judge.drop = text, {2: 4, 6: 0, 12: 9, 24: 7}.get(n_text, 0)
        heard = judge.transcribe(resample(dec["wav"], FS_OUT, FS_ASR))
        a = T.align_words(judge.words(text), judge.words(heard["text"]), open_end=not row["ended"])
        row["heard"] = heard
        row["wer"] = {**a, "wer": T.wer(a) and round(T.wer(a), 4)}
    return row


def run_one(lm, judge, q: dict, setting: dict, folder: Path | None, name: str, system: str | None = None,
            max_new: int | None = None, show: bool = True, think: bool = False) -> dict:
    pr = Printer() if show else None
    if show:
        sys.stdout.write("  ")
    if setting.get("chain"):
        events = lm.ask_chain(q["x"], q["fs"], on_event=pr, think=think)
    else:
        n_text = None if setting["n_text"] == lm.cfg_n_text else int(setting["n_text"])
        n_audio = None if setting["n_audio"] == lm.cfg_n_audio else int(setting["n_audio"])
        events = lm.ask(q["x"], q["fs"], system=system, n_text=n_text, n_audio=n_audio, max_new=max_new, on_event=pr)
    if show:
        print()
    dec = lm.decode_timed(T.frames_of(events, lm.eos), int(CFG["model"]["chunk_frames"]))
    row = analyse(events, dec, lm, judge, int(setting["n_text"]), int(setting["n_audio"]), cap=max_new)
    if setting.get("chain"):                             # 第二趟念的是整句整句的字，WER 的參考也用那一份
        row["chain"] = True
        row["tts_stray_text_tokens"] = int(getattr(lm, "tts_stray_text", 0))
        row["ended"] = bool(events) and events[-1]["k"] == "A" and events[-1]["c"][0] == lm.eos      # TTS 那一趟自己念完了
        if "heard" in row:
            said = T.last_sentence(T.whole_sentences(row["text"])) if think else T.whole_sentences(row["text"])
            row["spoken_text"] = said
            a = T.align_words(judge.words(said), judge.words(row["heard"]["text"]), open_end=not row["ended"])
            row["wer"] = {**a, "wer": T.wer(a) and round(T.wer(a), 4)}
    row.update(q=q["id"], setting=setting["key"], label=setting["label"], events=events, name=name)
    if folder is not None and len(dec["wav"]):
        C.save_wav(folder / f"{name}.wav", dec["wav"], FS_OUT)
        row["wav"] = f"{name}.wav"
    return row


def print_timeline(row: dict) -> None:
    tl, s = row["timeline"], row["steps"]
    under = "never" if tl["underrun"] is None else f"after {tl['underrun']:.2f} s of audio"
    print(f"    first text token {C.BOLD}{ms(tl['first_text'])}{C.RESET}   first speech frame {C.BOLD}{ms(tl['first_frame'])}{C.RESET}   "
          f"first sound ({tl['chunk_frames']} frames decoded) {C.BOLD}{ms(tl['first_sound'])}{C.RESET}   playback catches up with generation: {C.BOLD}{under}{C.RESET}")
    extra = ""
    if "wer" in row:
        w = row["wer"]
        cut = "" if row["ended"] else f"; cut off by the step limit, {w.get('n_unspoken', 0)} written words not yet spoken are left out"
        extra = f"   WER vs its own text {C.BOLD}{pct(w['wer'])}{C.RESET} ({w['errors']}/{w['n_ref']}{cut})"
    print(f"    {tl['n_text']} text tokens, {tl['n_frames']} frames = {tl['audio_s']:.1f} s of audio   per step: text {ms(s['D_text'])}, frame {ms(s['D_frame'])}"
          f" (real time needs ≤ 80 ms)   RTF {tl['rtf'] and round(tl['rtf'], 2)}   blank text tokens {pct(row['blank_fraction'])}{extra}")


def summarise(rows: list[dict]) -> list[dict]:
    """每個設定一列：時間取各題的中位數，WER 幾題合計。"""
    out = []
    for key in SETTINGS:
        rs = [r for r in rows if r["setting"] == key]
        if not rs:
            continue
        med = lambda f: (lambda v: round(float(st.median(v)), 4) if v else None)([f(r) for r in rs if f(r) is not None])  # noqa: E731
        pooled = T.pooled([r["wer"] for r in rs if "wer" in r]) if any("wer" in r for r in rs) else None
        out.append({"setting": key, "label": rs[0]["label"], "n_text": rs[0]["n_text"], "n_audio": rs[0]["n_audio"], "n_questions": len(rs),
                    "chain": bool(SETTINGS[key].get("chain")), "n_text_tokens": med(lambda r: r["timeline"]["n_text"]),
                    "first_text": med(lambda r: r["timeline"]["first_text"]), "first_frame": med(lambda r: r["timeline"]["first_frame"]),
                    "first_sound": med(lambda r: r["timeline"]["first_sound"]), "rtf": med(lambda r: r["timeline"]["rtf"]),
                    "D_text": med(lambda r: r["steps"]["D_text"]), "D_frame": med(lambda r: r["steps"]["D_frame"]),
                    # R_txt（每秒聲音配幾個文字 token）只取自己停下來的回答：被步數上限截斷的那種，字寫在前面、聲音還沒跟上，會高估
                    "R_txt": med(lambda r: r["lead"]["R_txt"] if r["ended"] else None), "min_lead": med(lambda r: r["lead"]["min_lead"] if r["ended"] else None),
                    "blank_fraction": med(lambda r: r["blank_fraction"]), "underruns": sum(r["timeline"]["underrun"] is not None for r in rs),
                    "cut_off": sum(not r["ended"] for r in rs),
                    "audio_s": med(lambda r: r["timeline"]["audio_s"]),
                    "wer": pooled and {**pooled, "wer": pooled["wer"] and round(pooled["wer"], 4)}})
    return out


def text_rate_fallback() -> float | None:
    """6 : 12 那一列沒有一個回答是自己停下來的時候，R_txt 退回投影片用的那個數（W6 Demo 2 的英語那一句）。"""
    try:
        return float(json.loads((ASSETS / "w11-data.json").read_text(encoding="utf-8"))["token_rate"]["text_tokens_per_s_english"])
    except Exception:
        return None


def conditions_table(summary: list[dict]) -> dict:
    """兩個條件。D_text、D_frame、R_txt 都取預設那一列（6 : 12）量到的——那是模型見過的排法。"""
    base = next((s for s in summary if s["setting"] == "6:12"), None)
    if not base or not base["D_text"] or not base["D_frame"]:
        return {}
    r_txt, src = base["R_txt"], "measured on the 6 : 12 answers"
    if not r_txt:
        r_txt, src = text_rate_fallback(), "W6 Demo 2 (no 6 : 12 answer finished within the step limit)"
    if not r_txt:
        return {}
    rows = []
    for s in summary:
        if s.get("chain"):
            continue
        c = T.conditions(s["n_text"], s["n_audio"], F_R, r_txt, base["D_text"], base["D_frame"])
        rows.append({"setting": s["setting"], **{k: (round(v, 4) if isinstance(v, float) else v) for k, v in c.items()}})
    return {"R_txt": round(r_txt, 3), "R_txt_source": src, "D_text": base["D_text"], "D_frame": base["D_frame"], "rows": rows}


# ── cascade（W1 Demo 2 的管線）───────────────────────────────────
def run_cascade(x: np.ndarray, fs: int, log=print) -> dict:
    c = CFG["cascade"]
    spec = importlib.util.spec_from_file_location("w01_d2_demo", P(c["dir"]) / "demo.py")
    w1 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(w1)
    cfg = w1.load_config()
    cfg["asr"]["language"] = c["language"]
    cfg["llm"]["system_prompt"], cfg["llm"]["max_tokens"] = c["system_prompt"], int(c["max_tokens"])
    cfg["tts"]["lang_code"], cfg["tts"]["voice"] = c["lang_code"], c["voice"]
    asr, llm, tts = w1.load_backends(cfg, False)
    x16 = resample(x, fs, FS_ASR).astype(np.float64)
    runs = []
    for i in range(int(c["repeats"]) + 1):               # 第一遍是暖機，不算
        r = w1.run_cascade(x16, FS_ASR, cfg, asr, llm, tts)
        if i:
            runs.append(r)
        log(f"    cascade {i}: asr {r['asr_tail_ms']} ms · llm first token {r['llm_first_token_ms']} ms · tts first packet {r['tts_first_packet_ms']} ms")
    med = lambda k: int(st.median([r[k] for r in runs]))  # noqa: E731
    out = {"asr_tail_ms": med("asr_tail_ms"), "llm_first_token_ms": med("llm_first_token_ms"), "tts_first_packet_ms": med("tts_first_packet_ms"),
           "repeats": len(runs), "transcript": runs[-1]["transcript"], "reply": runs[-1]["reply"],
           "backends": {"asr": f"whisper.cpp {cfg['asr']['whispercpp_model']}", "llm": cfg["llm"]["mlx_lm_repo"], "tts": cfg["tts"]["mlx_audio_repo"]},
           "source": "measured here, same question"}
    out["first_sound_ms"] = out["asr_tail_ms"] + out["llm_first_token_ms"] + out["tts_first_packet_ms"]
    del asr, llm, tts
    gc.collect()
    return out


def cascade_from_w1() -> dict | None:
    """現場的 cascade 跑不起來時的退路：W1 彩排的那一組（另一句話、華語），照實標明。"""
    try:
        d = json.loads((ASSETS / "w11-data.json").read_text(encoding="utf-8"))["w1_demo2_budget_ms"]
        return {"asr_tail_ms": d["asr_tail"], "llm_first_token_ms": d["llm_first_token"], "tts_first_packet_ms": d["tts_first_packet"],
                "first_sound_ms": d["middle_three"], "source": "W1 Demo 2's rehearsal (a different, Mandarin sentence) — not measured here"}
    except Exception:
        return None


# ── 全部算一遍 ───────────────────────────────────────────────────
def compute(folder: Path, lm, judge, n_questions: int | None = None, think: bool = True, cascade: bool = True, log=print) -> dict:
    folder.mkdir(parents=True, exist_ok=True)
    qs = [question(q["id"]) for q in CFG["questions"][: n_questions or None]]
    rec = {"date": C.now(), "mode": "fake" if lm.fake else "real", "config_digest": digest(), "versions": versions(),
           "models": {"lfm": lm.info(), "whisper": judge.info()}, "frame_rate": F_R, "chunk_frames": int(CFG["model"]["chunk_frames"]),
           "sampling": {k: CFG["model"][k] for k in ("text_temperature", "audio_temperature", "audio_top_k", "seed", "max_new_tokens")},
           "questions": [{"id": q["id"], "text": q["text"], "seconds": round(q["seconds"], 2), "own_voice": q["own"]} for q in qs], "runs": []}
    for q in qs:
        for key in SETTINGS:
            C.banner(f"{q['id']} · {SETTINGS[key]['label']}", C.GREY)
            row = run_one(lm, judge, q, SETTINGS[key], folder, f"{q['id']}_{key.replace(':', '-')}")
            print_timeline(row)
            rec["runs"].append(row)
    rec["summary"] = summarise(rec["runs"])
    rec["conditions"] = conditions_table(rec["summary"])
    if think:
        rec["think"] = compute_think(folder, lm, judge)
    if cascade:
        q1 = qs[0]
        try:
            if lm.fake:
                raise RuntimeError("fake")
            rec["cascade"] = run_cascade(q1["x"], q1["fs"], log)
        except Exception as e:
            rec["cascade"] = cascade_from_w1() or {}
            rec["cascade"]["error"] = f"{type(e).__name__}: {e}"[:300]
    (folder / "rehearsal.json").write_text(json.dumps(rec, ensure_ascii=False, indent=1), encoding="utf-8")
    return rec


def compute_think(folder: Path | None, lm, judge, which: tuple = ("plain", "reason"), show: bool = True) -> dict:
    """後半。同一道算術題，兩種做法：
      plain   預設的對話（6 : 12）：邊寫邊念，馬上開口；推理如果有，是念出來的。
      reason  先想再說：第一趟只寫字（一步一步推理，不念），第二趟只把最後一句念出來。
    每一種記：第一個聲音之前寫了幾個文字 token（U_th）、第一個聲音等了多久、答案對不對（字裡的最後一個數字；聲音裡的由 Whisper 聽）。"""
    th = CFG["think"]
    q = question(th["id"])
    out = {"question": th["text"], "expected": th["expected"]}
    for name in which:
        if show:
            C.banner(f"{th['id']} · {'answer as you go (6 : 12)' if name == 'plain' else 'reason in text first, then say only the answer (two passes)'}", C.GREY)
        if name == "plain":
            row = run_one(lm, judge, q, SETTINGS["6:12"], folder, f"{th['id']}_plain", max_new=int(th["max_new_tokens"]), show=show)
        else:
            row = run_one(lm, judge, q, SETTINGS["all"], folder, f"{th['id']}_reason", show=show, think=True)
        ans = T.answer_is(row["text"], int(th["expected"]))
        t_ans, n_before, acc = None, None, ""
        total = len(T.numbers_in(row["text"]))
        for i, e in enumerate(row["events"]):                # 答案寫完的時刻：最後一個數字的最後一個 token
            if e["k"] != "T" or t_ans is not None:
                continue
            acc += e["s"]
            nums = T.numbers_in(T.SPECIAL.sub("", acc))
            if total and len(nums) == total:
                t_ans, n_before = e["t"], sum(1 for v in row["events"][: i + 1] if v["k"] == "T")
        heard_ans = T.answer_is(row["heard"]["text"], int(th["expected"])) if "heard" in row else None
        r_txt, fs_ = row["lead"]["R_txt"], row["timeline"]["first_sound"]
        if name == "plain":                                  # 邊寫邊念：答案被念到的時刻用均勻語速估
            spoken_at = fs_ + n_before / r_txt if (t_ans is not None and r_txt and fs_ is not None) else None
        else:                                                # 只念最後一句：答案在那一句的結尾
            spoken_at = fs_ + row["timeline"]["audio_s"] if fs_ is not None else None
        d_text = row["steps"]["D_text"]
        u_th = row["tokens_before_first_frame"]
        row.update(answer=ans, heard_answer=heard_ans, final_number_written_at=t_ans, text_tokens_to_final_number=n_before,
                   answer_spoken_at_est=spoken_at and round(spoken_at, 2), U_th=u_th,
                   predicted_silence=d_text and round(u_th * d_text, 4))
        if show:
            print_timeline(row)
        out[name] = row
    return out


# ── 匯出 ─────────────────────────────────────────────────────────
def summary_for_metrics(rec: dict) -> dict:
    th = rec.get("think") or {}

    def half(row):
        if not row:
            return None
        return {"text": row["text"], "heard": row.get("heard", {}).get("text"), "U_th": row["U_th"], "n_text_tokens": row["timeline"]["n_text"],
                "audio_s": row["timeline"]["audio_s"], "first_sound_s": row["timeline"]["first_sound"], "predicted_silence_s": row["predicted_silence"],
                "final_number_written_at_s": row["final_number_written_at"], "answer_spoken_at_est_s": row["answer_spoken_at_est"],
                "last_number_in_text": row["answer"]["last"], "correct_in_text": row["answer"]["correct"],
                "correct_in_speech": row["heard_answer"] and row["heard_answer"]["correct"], "D_text_s": row["steps"]["D_text"], "D_frame_s": row["steps"]["D_frame"]}

    return {"date": rec["date"], "mode": rec["mode"], "models": rec["models"], "versions": {k: rec["versions"].get(k) for k in ("mlx", "mlx-audio", "transformers", "torch", "platform")},
            "config_digest": rec["config_digest"], "frame_rate": rec["frame_rate"], "chunk_frames": rec["chunk_frames"], "sampling": rec["sampling"],
            "questions": rec["questions"], "settings": rec["summary"], "conditions": rec.get("conditions"),
            "per_question": [{"q": r["q"], "setting": r["setting"], "text": r["text"], "heard": r.get("heard", {}).get("text"),
                              "first_sound_s": r["timeline"]["first_sound"], "wer": r.get("wer", {}).get("wer"), "blank_fraction": r["blank_fraction"]} for r in rec["runs"]],
            "think": th and {"question": th["question"], "expected": th["expected"], "plain": half(th.get("plain")), "reason": half(th.get("reason"))},
            "cascade": rec.get("cascade"),
            "_note": "times count from the moment the recorded question is handed to the model (no endpoint cell); first sound = first chunk_frames frames generated and decoded; "
                     "WER = Whisper small on the generated speech against the model's own text, pooled over the questions; only 6:12 was seen in training"}


def cmd_export(args=None, rec: dict | None = None) -> None:
    rec = rec or load_rec(RUNS / "rehearsal")
    if rec["mode"] != "real":
        raise SystemExit("彩排紀錄不是真的模型跑的（fake），不寫 metrics")
    d = json.loads(METRICS.read_text(encoding="utf-8")) if METRICS.exists() else {}
    d["demo3"] = summary_for_metrics(rec)
    METRICS.write_text(json.dumps(d, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    C.note(f"寫了 {METRICS.name} → demo3")


def load_rec(folder: Path) -> dict:
    p = folder / "rehearsal.json"
    if not p.exists():
        raise SystemExit(f"沒有 {p}：先跑 ./present.sh rehearse")
    return json.loads(p.read_text(encoding="utf-8"))


# ── 畫面 ─────────────────────────────────────────────────────────
def show_table(rec: dict) -> None:
    C.banner(f"who leads, and by how much — {len(rec['questions'])} question(s) pooled · measured {rec['date'][:10]}"
             + ("  [FAKE]" if rec["mode"] != "real" else ""))
    print(f"  {'setting':<46}{'first text':>11}{'first frame':>12}{'first sound':>12}{'RTF':>7}{'WER':>8}{'blank':>8}")
    for s in rec["summary"]:
        w = s["wer"]
        rtf = "%.2f" % s["rtf"] if s["rtf"] else "—"
        print(f"  {s['label']:<46}{ms(s['first_text']):>11}{ms(s['first_frame']):>12}{C.BOLD}{ms(s['first_sound']):>12}{C.RESET}"
              f"{rtf:>7}{C.BOLD}{pct(w and w['wer']):>8}{C.RESET}{pct(s['blank_fraction']):>8}")
    c = rec.get("cascade")
    if c and c.get("first_sound_ms") is not None:
        print(f"  {'the cascade of W1 Demo 2':<46}{'—':>11}{'—':>12}{C.BOLD}{str(c['first_sound_ms']) + ' ms':>12}{C.RESET}{'—':>7}{'—':>8}{'—':>8}   {C.GREY}{c['source']}{C.RESET}")
    C.note("  WER: Whisper small on the generated speech, against the model's own text. blank: text tokens that are nothing but space (a space glued to the next word, as before a number, is not counted).")
    C.note("  Only 6 : 12 was seen in training; every other row is out of distribution.")
    cut = [f"{s['setting']} ({s['cut_off']}/{s['n_questions']})" for s in rec["summary"] if s.get("cut_off")]
    if cut:
        C.note(f"  Hit the step limit before finishing: {', '.join(cut)} — there, written words that were not yet spoken are left out of the WER.")


def show_conditions(rec: dict) -> None:
    C.banner("the two conditions, with the numbers measured on this machine")
    cd = rec.get("conditions")
    if not cd:
        print("  （沒有 6 : 12 那一列的量測）")
        return
    print(f"  measured at 6 : 12:  one text step {C.BOLD}{ms(cd['D_text'])}{C.RESET}   one frame step {C.BOLD}{ms(cd['D_frame'])}{C.RESET}   f_r = {F_R:g} Hz")
    print(f"  text tokens per second of speech R_txt = {C.BOLD}{cd['R_txt']:.2f}{C.RESET}  {C.GREY}({cd['R_txt_source']}){C.RESET}")
    print(f"\n  {'ratio':<9}{'(i) n_txt/n_aud':>17}{'≥ R_txt/f_r':>14}{'':>5}{'(ii) one round takes':>23}{'≤ it plays for':>16}{'':>5}{'slack':>9}{'mean step':>11}{'bound':>9}")
    for c in cd["rows"]:
        ok = lambda b: f"{C.GREEN}ok{C.RESET}  " if b else f"{C.RED}FAIL{C.RESET}"  # noqa: E731
        print(f"  {c['setting']:<9}{c['i_lhs']:>17.3f}{c['i_rhs']:>14.3f}  {ok(c['i_ok'])} {ms(c['round_gen_s']):>22}{ms(c['round_audio_s']):>16}  {ok(c['ii_ok'])} "
              f"{ms(c['slack_s']):>8}{ms(c['mean_step_s']):>11}{ms(c['step_bound_s']):>9}")
    C.note("  'bound' is the slide's D_token limit with N_step = 1; the frame step here includes the depth model's 8 small steps.")


def show_cascade(rec: dict) -> None:
    C.banner("the same question, two budgets (the endpoint cell is the same setting on both sides and is not measured here)")
    c, base = rec.get("cascade"), next((s for s in rec["summary"] if s["setting"] == "6:12"), None)
    if not c or c.get("first_sound_ms") is None:
        print("  （沒有 cascade 的數字）")
        return
    print(f"  {'cascade (W1 Demo 2)':<34}{'':>10}    {'one model (LFM2.5-Audio, 6 : 12)':<34}")
    left = [("ASR tail", c["asr_tail_ms"]), ("LLM first token", c["llm_first_token_ms"]), ("TTS first packet", c["tts_first_packet_ms"]), ("sum", c["first_sound_ms"])]
    right = [("read the question, first text token", base and base["first_text"]), ("… to the first speech frame", base and base["first_frame"]),
             (f"… to the first sound ({rec['chunk_frames']} frames)", base and base["first_sound"]), ("", None)]
    for (a, av), (b, bv) in zip(left, right):
        print(f"  {a:<34}{str(av) + ' ms':>10}    {b:<40}{(ms(bv) if bv is not None else ''):>10}")
    C.note(f"  cascade: {c['source']}" + (f"  [{c['error']}]" if c.get("error") else ""))


def show_think(th: dict) -> None:
    C.banner(f"the same question, two ways to answer — the right answer is {th['expected']}")
    print(f"  {'':<44}{'text tokens before the first sound':>36}{'first sound':>13}{'predicted U_th·D_text':>23}{'answer heard at':>17}{'text':>7}{'speech':>8}")
    yn = lambda b: "—" if b is None else ("right" if b else "wrong")  # noqa: E731
    for name, lab in (("plain", "answer as you go (6 : 12)"), ("reason", "reason in text first, say only the answer")):
        r = th.get(name)
        if not r:
            continue
        print(f"  {lab:<44}{r['U_th']:>36}{C.BOLD}{ms(r['timeline']['first_sound']):>13}{C.RESET}{ms(r['predicted_silence']):>23}{ms(r['answer_spoken_at_est']):>17}"
              f"{yn(r['answer']['correct']):>7}{yn(r['heard_answer'] and r['heard_answer']['correct']):>8}")
    C.note("  The second row is two passes of the same model: a text-only pass (nothing is spoken), then its TTS mode on the last sentence.")
    C.note("  'predicted' leaves out reading the question and decoding the first chunk. 'answer heard at' is an estimate (even speaking rate in row 1; end of the sentence in row 2).")


def show_words(row: dict) -> None:
    def para(label, text, color=""):
        body = textwrap.fill(text or "(nothing)", WIDTH - 18, subsequent_indent=" " * 18)
        print(f"  {C.GREY}{label:<16}{C.RESET}{color}{body}{C.RESET}")
    para("wrote", row["text"], MAGENTA)
    if row.get("spoken_text") and row["spoken_text"] != row["text"]:
        para("sent to TTS", row["spoken_text"], MAGENTA)
    if "heard" in row:
        para("Whisper heard", row["heard"]["text"], CYAN)
        w = row["wer"]
        print(f"  {C.GREY}{'':<16}{C.RESET}WER {pct(w['wer'])}  ({w['S']} substituted, {w['D']} in the text but not heard, {w['I']} heard but not in the text; {w['n_ref']} words compared"
              + (f"; {w['n_unspoken']} written but not yet spoken when the step limit cut it off" if w.get("n_unspoken") else "") + ")")


MENU = """  p  play the question      1  all the text first, then 6 : 12        2  sweep the ratio: 2 : 12, 6 : 12, 12 : 12, 24 : 12
  t  the table (rehearsal)  3  the two conditions, measured           4  the cascade, same question (rehearsal; $ runs it now)
  5  the arithmetic question: 1st press answer-as-you-go, 2nd press reason-in-text-first       6  the second table
  w  last run: what it wrote vs what Whisper heard        l  play the last run again        ?  this menu        q  quit"""


class Session:
    def __init__(self, rec: dict | None, folder: Path, mode: str, lm=None, judge=None):
        self.rec, self.folder, self.mode, self.lm, self.judge = rec, folder, mode, lm, judge
        self.live = RUNS / ("fake" if mode == "fake" else "live")
        self.last, self.rows, self.think_n, self.think = None, {}, 0, {}

    def recorded(self, q: str, key: str) -> dict | None:
        return next((r for r in (self.rec or {}).get("runs", []) if r["q"] == q and r["setting"] == key), None)

    def one(self, qid: str, key: str) -> None:
        C.banner(f"{qid} · {SETTINGS[key]['label']}" + ("  [replay]" if self.mode == "replay" else "  [FAKE]" if self.mode == "fake" else ""), C.GREY)
        if self.mode == "replay":
            row = self.recorded(qid, key)
            if not row:
                print("  （彩排紀錄裡沒有這一格）")
                return
            sys.stdout.write("  ")
            pr = Printer(pace=True)
            for e in row["events"]:
                pr(e)
            print()
            wav = self.folder / row.get("wav", "")
        else:
            row = run_one(self.lm, self.judge, question(qid), SETTINGS[key], self.live, f"{qid}_{key.replace(':', '-')}")
            wav = self.live / row.get("wav", "")
        print_timeline(row)
        self.rows[key], self.last = row, (row, wav)
        if wav.is_file():
            C.play(wav)

    def think_step(self) -> None:
        name = ("plain", "reason")[self.think_n % 2]
        self.think_n += 1
        if self.mode == "replay":
            row = ((self.rec or {}).get("think") or {}).get(name)
            if not row:
                print("  （彩排紀錄裡沒有後半）")
                return
            C.banner(f"{CFG['think']['id']} · {'answer as you go (6 : 12)' if name == 'plain' else 'reason in text first, then say only the answer'}  [replay]", C.GREY)
            sys.stdout.write("  ")
            pr = Printer(pace=True)
            for e in row["events"]:
                pr(e)
            print()
            print_timeline(row)
            wav = self.folder / row.get("wav", "")
            self.think = self.rec["think"]
        else:
            got = compute_think(self.live, self.lm, self.judge, which=(name,))
            row, wav = got[name], self.live / got[name].get("wav", "")
            self.think.update(got)
        self.last = (row, wav)
        if wav.is_file():
            C.play(wav)

    def key(self, k: str) -> bool:
        q1 = CFG["questions"][0]["id"]
        if k == "q":
            return False
        if k == "?":
            print(MENU)
        elif k == "p":
            q = question(q1)
            print(f"  {C.BOLD}{q['text']}{C.RESET}  {C.GREY}({q['seconds']:.1f} s, {'own voice' if q['own'] else 'synthetic voice'}){C.RESET}")
            C.play(q["wav"])
        elif k == "1":
            for key in CFG["live"]["key1"]:
                self.one(q1, key)
        elif k == "2":
            for key in CFG["live"]["key2"]:
                self.one(q1, key)
            rows = [self.rows[k2] for k2 in CFG["live"]["key2"] if k2 in self.rows]
            print(f"\n  {'this question, just now':<46}{'first sound':>12}{'WER':>8}{'blank':>8}")
            for r in rows:
                print(f"  {r['label']:<46}{C.BOLD}{ms(r['timeline']['first_sound']):>12}{pct(r.get('wer', {}).get('wer')):>8}{C.RESET}{pct(r['blank_fraction']):>8}")
        elif k == "t":
            show_table(self.rec) if self.rec else print("  （沒有彩排紀錄：先 ./present.sh rehearse）")
        elif k == "3":
            show_conditions(self.rec) if self.rec else print("  （沒有彩排紀錄）")
        elif k == "4":
            show_cascade(self.rec) if self.rec else print("  （沒有彩排紀錄）")
        elif k == "$":
            if self.mode != "show":
                print("  （只有 show 模式現場跑 cascade）")
            else:
                q = question(q1)
                try:
                    c = run_cascade(q["x"], q["fs"])
                    show_cascade({**(self.rec or {"summary": [], "chunk_frames": CFG["model"]["chunk_frames"]}), "cascade": c})
                except Exception as e:
                    print(f"  {C.RED}cascade 跑不起來：{type(e).__name__}: {e}{C.RESET}")
        elif k == "5":
            self.think_step()
        elif k == "6":
            th = self.think if ("plain" in self.think and "reason" in self.think) else (self.rec or {}).get("think")
            show_think(th) if th else print("  （還沒有兩次的結果：按兩次 5，或先彩排）")
        elif k == "w":
            show_words(self.last[0]) if self.last else print("  （還沒有跑過任何一段）")
        elif k == "l":
            if self.last and self.last[1].is_file():
                C.play(self.last[1])
        return True

    def loop(self) -> None:
        tag = {"show": "", "replay": "  [replay — no model loaded]", "fake": "  [FAKE — no model, numbers mean nothing]"}[self.mode]
        C.banner("W11 Demo 3 · how far ahead is the text?" + tag)
        print(f"  {MAGENTA}{C.BOLD}purple{C.RESET} = a text token      {CYAN}▮{C.RESET} = one speech frame (8 codes, 80 ms of audio)      "
              f"{C.DIM}0.00s{C.RESET} = seconds since the question was handed over")
        print(MENU)
        while True:
            sys.stdout.write(f"\n{C.GREY}key ▸ {C.RESET}")
            sys.stdout.flush()
            try:
                if not self.key(C.getkey()):
                    break
            except KeyboardInterrupt:
                print("  （中斷）")
            except Exception as e:                           # 課堂上不要因為一個鍵掛掉整個視窗
                print(f"  {C.RED}{type(e).__name__}: {e}{C.RESET}")


# ── 子命令 ───────────────────────────────────────────────────────
def synth_questions(lm, force: bool = False) -> None:
    AUDIO.mkdir(parents=True, exist_ok=True)
    for q in list(CFG["questions"]) + [CFG["think"]]:
        path = AUDIO / f"{q['id']}.wav"
        if path.exists() and not force:
            continue
        t0 = time.perf_counter()
        wav = lm.tts(q["text"])
        C.save_wav(path, wav, FS_OUT)
        C.note(f"  {path.name}  {len(wav) / FS_OUT:.1f} s  （{time.perf_counter() - t0:.1f} s）  {q['text']}")


def cmd_fetch(args) -> None:
    from huggingface_hub import snapshot_download
    repo = CFG["model"]["repo"]
    lock = read_lock(LOCK)
    path = snapshot_download(repo, revision=lock.get(repo), allow_patterns=["*.json", "*.safetensors", "tokenizer*", "LICENSE*", "*.md"])
    lock[repo] = Path(path).name
    LOCK.write_text("".join(f"{k} {v}\n" for k, v in sorted(lock.items())), encoding="utf-8")
    C.note(f"{repo} @ {lock[repo]}")
    for r in (CFG["whisper"]["repo"],):
        try:
            C.note(f"{r} 在快取裡：{Path(local_snapshot(r)).name}")
        except FileNotFoundError as e:
            print(f"{C.RED}{e}（先跑 ../w07_d1_hallucination/present.sh fetch）{C.RESET}")
    # cascade 那一列（W1 Demo 2 的管線）在這裡用英語的 Kokoro voice；mlx-audio 第一次用到才下載 voice 檔，
    # 離線模式下會缺（2026-11-26 的彩排：voices/af_heart.safetensors）。先抓進快取。
    voice = CFG["cascade"].get("voice")
    if voice:
        try:
            p = Path(snapshot_download("prince-canuma/Kokoro-82M", allow_patterns=[f"voices/{voice}.safetensors"])) / "voices" / f"{voice}.safetensors"
            C.note(f"Kokoro voice {voice}：{'在快取裡' if p.exists() else '沒抓到'}")
        except Exception as e:
            print(f"{C.RED}Kokoro voice {voice} 沒抓到：{e!r}（cascade 那一列會退回 W1 彩排的數字）{C.RESET}")
    lm = LFM()
    synth_questions(lm, force=args.force)


def cmd_tts(args) -> None:
    synth_questions(LFM(), force=args.force)


def cmd_record(args) -> None:
    q = dict(QUESTIONS.get(args.qid) or (CFG["think"] if args.qid == CFG["think"]["id"] else {}))
    if not q:
        raise SystemExit(f"不認得的題號 {args.qid}：{', '.join(all_question_ids())}")
    print(f"  念這一句：{C.BOLD}{q['text']}{C.RESET}")
    input("  按 Enter 開始錄（念完等它結束）…")
    x = C.record(args.seconds, FS_OUT)
    keep = np.flatnonzero(np.abs(x) > 0.02 * np.max(np.abs(x) + 1e-9))
    if len(keep):
        x = x[max(keep[0] - FS_OUT // 10, 0): keep[-1] + FS_OUT // 5]
    OWN.mkdir(parents=True, exist_ok=True)
    C.save_wav(OWN / f"{args.qid}.wav", x, FS_OUT, peak=float(np.max(np.abs(x)) or 1.0))
    C.note(f"寫了 {OWN / (args.qid + '.wav')}（{len(x) / FS_OUT:.1f} s；不進版控。刪掉這個檔就回到合成的聲音）")


def cmd_rehearse(args) -> None:
    folder = RUNS / ("preview" if args.preview else "rehearsal")
    if args.max_new:
        if not args.preview:
            raise SystemExit("--max-new 只給 --preview 用（正式的彩排照設定檔）")
        CFG["model"]["max_new_tokens"] = CFG["think"]["max_new_tokens"] = int(args.max_new)
        CFG["chain"]["max_text_tokens"] = CFG["chain"]["max_frames"] = CFG["think"]["silent_max_text_tokens"] = int(args.max_new)
    lm = LFM(args.repo)
    judge = Judge()
    rec = compute(folder, lm, judge, n_questions=args.questions, think=not args.no_think, cascade=not args.no_cascade)
    show_table(rec)
    show_conditions(rec)
    if rec.get("think"):
        show_think(rec["think"])
    if rec.get("cascade"):
        show_cascade(rec)
    if not args.preview:
        cmd_export(rec=rec)


def cmd_show(args, mode: str) -> None:
    folder = RUNS / "rehearsal"
    rec = json.loads((folder / "rehearsal.json").read_text(encoding="utf-8")) if (folder / "rehearsal.json").exists() else None
    if mode == "replay":
        if rec is None:
            raise SystemExit("沒有彩排紀錄：先跑 ./present.sh rehearse")
        Session(rec, folder, "replay").loop()
    elif mode == "fake":
        global AUDIO, OWN
        lm, judge = FakeLFM(), FakeJudge()
        fk = RUNS / "fake"
        AUDIO, OWN = fk / "audio", fk / "own"            # 演練用自己的一份假問題，不碰 audio/
        synth_questions(lm, force=True)
        rec = compute(fk, lm, judge)
        Session(rec, fk, "fake", lm, judge).loop()
    else:
        C.note("載入模型……")
        lm, judge = LFM(args.repo), Judge()
        C.note(f"  {lm.repo}（{lm.load_s:.1f} s）")
        Session(rec, folder, "show", lm, judge).loop()


def cmd_check(args) -> None:
    out = {"date": C.now(), "versions": versions(), "steps": {}}

    def step(name: str, fn):
        C.banner(name, C.GREY)
        t0 = time.perf_counter()
        try:
            r = fn()
            out["steps"][name] = {"ok": True, "seconds": round(time.perf_counter() - t0, 2), **(r or {})}
            print(f"  {C.GREEN}ok{C.RESET}  {json.dumps(r, ensure_ascii=False)[:600] if r else ''}")
        except Exception as e:
            out["steps"][name] = {"ok": False, "error": f"{type(e).__name__}: {e}"[:500]}
            print(f"  {C.RED}FAIL{C.RESET}  {type(e).__name__}: {e}")

    box = {}

    def load():
        box["lm"] = LFM(args.repo)
        lm = box["lm"]
        assert lm.eos == EOS, f"聲音的結束碼是 {lm.eos}，不是 {EOS}"
        assert lm.codebooks == int(CFG["model"]["codebooks"])
        return lm.info()

    def layout():
        lm, q = box["lm"], question(CFG["questions"][0]["id"])
        ev = lm.ask(q["x"], q["fs"], max_new=int(CFG["check"]["probe_steps"]))
        box["ev"] = ev
        sch = T.schedule(ev)
        assert sch[0] == ("T", lm.cfg_n_text) and sch[1] == ("A", lm.cfg_n_audio), f"預設的排法不是 {lm.cfg_n_text} : {lm.cfg_n_audio}：{sch[:4]}"
        s = T.step_times(ev)
        return {"schedule": sch[:6], "first_token_s": ev[0]["t"], "D_text_ms": round(1000 * s["D_text"], 1), "D_frame_ms": round(1000 * s["D_frame"], 1),
                "real_time_needs_frame_ms": 1000 / F_R, "faster_than_real_time": s["D_frame"] < 1 / F_R, "text": T.text_of(ev)}

    def ratio():
        lm, q = box["lm"], question(CFG["questions"][0]["id"])
        ev = lm.ask(q["x"], q["fs"], n_text=2, n_audio=12, max_new=30)
        sch = T.schedule(ev)
        assert sch[0] == ("T", 2) and sch[1] == ("A", 12), f"比例的參數沒有生效：{sch[:4]}"
        return {"schedule_2_12": sch[:4]}

    def decode():
        lm = box["lm"]
        frames = T.frames_of(box["ev"], lm.eos)
        d = lm.decode_timed(frames, int(CFG["model"]["chunk_frames"]))
        return {"frames": len(frames), "samples_per_frame": len(d["wav"]) / max(len(frames), 1), "decode_first_ms": round(1000 * d["decode_first_s"], 1),
                "decode_all_ms": round(1000 * d["decode_all_s"], 1), "prefix_max_abs_diff": d["prefix_max_abs_diff"],
                "note": "prefix_max_abs_diff: decoding only the first chunk vs the same samples from decoding everything (0 = the decoder is causal)"}

    def sequential():
        lm, q = box["lm"], question(CFG["questions"][0]["id"])
        ev = lm.ask(q["x"], q["fs"], max_new=40, sequential=True)
        return {"schedule": T.schedule(ev)[:6], "blank_fraction": round(T.blank_fraction(ev), 3), "text": T.text_of(ev)[:200],
                "note": "generate_sequential on a spoken question with the chat system prompt; the model card reserves this mode for ASR and TTS"}

    def tts_mode():
        lm = box["lm"]
        t0 = time.perf_counter()
        wav = lm.tts("The quick brown fox jumps over the lazy dog.")
        dt = time.perf_counter() - t0
        box["tts_wav"] = wav
        assert len(wav) > FS_OUT // 2, "TTS 沒有出聲音"
        return {"seconds_of_audio": round(len(wav) / FS_OUT, 2), "took_s": round(dt, 2), "text_tokens_written_before_or_between_frames": lm.tts_stray_text,
                "note": "in TTS mode the first token should be the switch to audio; a non-zero count means the text half wandered"}

    def whisper():
        box["judge"] = Judge()
        lm, j = box["lm"], box["judge"]
        out = {**j.info(), "questions": {}}
        for qid in all_question_ids():                   # 合成的問題念得對不對：念錯的那一題改 [tts] seed 之後 ./present.sh tts --force
            if not question_wav(qid).exists():
                out["questions"][qid] = {"missing": True}
                print(f"    {qid}: {C.RED}沒有聲音檔{C.RESET}")
                continue
            q = question(qid)
            heard = j.transcribe(resample(q["x"], q["fs"], FS_ASR))
            a = T.align_words(j.words(q["text"]), j.words(heard["text"]))
            out["questions"][qid] = {"heard_as": heard["text"], "word_errors": a["errors"], "n_words": a["n_ref"], "own_voice": q["own"]}
            print(f"    {qid}: {a['errors']}/{a['n_ref']} word errors   {C.GREY}{heard['text']}{C.RESET}")
        if "tts_wav" in box:
            out["tts_probe_heard_as"] = j.transcribe(resample(box["tts_wav"], FS_OUT, FS_ASR))["text"]
        return out

    def cascade():
        q = question(CFG["questions"][0]["id"])
        return run_cascade(q["x"], q["fs"])

    step("load LFM2.5-Audio", load)
    if "lm" in box:
        step("default schedule and speed", layout)
        step("the ratio is an inference-time parameter", ratio)
        if "ev" in box:
            step("detokenizer: chunk vs whole", decode)
        step("sequential mode in a chat", sequential)
        step("TTS mode", tts_mode)
        step("Whisper on the question", whisper)
    if not args.no_cascade:
        step("cascade (W1 Demo 2 backends, English)", cascade)
    RUNS.mkdir(exist_ok=True)
    (RUNS / "check.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    bad = [k for k, v in out["steps"].items() if not v["ok"]]
    print(f"\n{'全部通過' if not bad else C.RED + '沒過的：' + '、'.join(bad) + C.RESET}  → runs/check.json")


def cmd_selftest(args) -> None:
    n = {"ok": 0, "bad": 0}

    def t(name: str, cond: bool, extra: str = "") -> None:
        n["ok" if cond else "bad"] += 1
        print(f"  {C.GREEN + 'ok  ' if cond else C.RED + 'FAIL'}{C.RESET} {name}  {C.GREY}{extra}{C.RESET}")

    def ev(pattern: str, dt_text=0.01, dt_frame=0.03, t0=0.2):
        out, tt, i = [], t0, 0
        for ch in pattern:
            if ch == "T":
                tt += dt_text
                out.append({"k": "T", "t": round(tt, 4), "id": i, "s": f" w{i}"})
                i += 1
            elif ch == "A":
                tt += dt_frame
                out.append({"k": "A", "t": round(tt, 4), "c": [1] * 8})
            elif ch == "E":
                tt += dt_frame
                out.append({"k": "A", "t": round(tt, 4), "c": [EOS] * 8})
            elif ch == "_":
                tt += dt_text
                out.append({"k": "T", "t": round(tt, 4), "id": 0, "s": " "})
        return out

    C.banner("trace.py")
    e = ev("T" * 6 + "A" * 12 + "T" * 6 + "A" * 12 + "E")
    t("schedule 讀得出 6 : 12", T.schedule(e)[:3] == [("T", 6), ("A", 12), ("T", 6)])
    t("結束碼的 frame 不算聲音", len(T.frames_of(e, EOS)) == 24)
    t("expected_schedule：字寫完之後只剩 frame", T.expected_schedule(6, 12, 12, 40) == [("T", 6), ("A", 12), ("T", 6), ("A", 28)])
    s = T.step_times(e)
    t("每一步的時間：文字 10 ms、frame 30 ms", abs(s["D_text"] - 0.01) < 1e-6 and abs(s["D_frame"] - 0.03) < 1e-6, str(s))
    tl = T.timeline(e, 12.5, 6, 0.05, 0.05, EOS)
    t("first text = 第一個文字 token 的時刻", abs(tl["first_text"] - 0.21) < 1e-6)
    t("first frame = 六個字之後的第一格", abs(tl["first_frame"] - (0.2 + 0.06 + 0.03)) < 1e-6)
    t("first sound = 第六格算完 + 解第一塊", abs(tl["first_sound"] - (0.2 + 0.06 + 0.18 + 0.05)) < 1e-6, f"{tl['first_sound']:.3f}")
    t("生成比播放快時不會斷", tl["underrun"] is None and tl["min_slack"] > 0, f"min slack {tl['min_slack']:.3f}")
    slow = T.timeline(ev("T" * 6 + "A" * 24, dt_frame=0.2), 12.5, 6, 0.05, 0.05, EOS)
    t("每格 200 ms（慢過即時的 80 ms）時第二塊就斷", slow["underrun"] is not None and abs(slow["underrun"] - 0.48) < 1e-9 and slow["rtf"] > 1, str(slow["underrun"]))
    t("audio_s = frame 數 ÷ 12.5", abs(tl["audio_s"] - 24 / 12.5) < 1e-9)
    ld = T.lead_curve(ev("T" * 6 + "A" * 12 + "T" * 6 + "A" * 12), 12.5, EOS)
    t("R_txt = 文字 token 數 ÷ 聲音秒數", abs(ld["R_txt"] - 12 / (24 / 12.5)) < 1e-9, f"{ld['R_txt']:.3f}")
    t("6 : 12 剛好等於自己的 R_txt 時，領先量不會掉到 0 以下", ld["min_lead"] > 0, f"{ld['min_lead']:.2f}")
    ld2 = T.lead_curve(ev("T" * 2 + "A" * 12 + "T" * 10 + "A" * 12), 12.5, EOS)
    t("先只給 2 個字時領先量掉到負的", ld2["min_lead"] < 0, f"{ld2['min_lead']:.2f} at {ld2['at_s']:.2f} s")
    t("blank：只有空白的文字 token 的比例", abs(T.blank_fraction(ev("TT__")) - 0.5) < 1e-9)
    num = [{"k": "T", "t": 0, "id": 0, "s": s_} for s_ in (" with", " ", "48", " passengers", " ", " ")]
    t("blank：數字前面那個空白 token 不算", abs(T.blank_fraction(num) - 2 / 6) < 1e-9)
    c = T.conditions(6, 12, 12.5, 3.42, 0.012, 0.030)
    t("(i) 6/12 ≥ 3.42/12.5", c["i_ok"] and abs(c["i_rhs"] - 0.2736) < 1e-9)
    t("(ii) 一輪 6×12 + 12×30 = 432 ms ≤ 960 ms，slack 528 ms", c["ii_ok"] and abs(c["round_gen_s"] - 0.432) < 1e-9 and abs(c["slack_s"] - 0.528) < 1e-9)
    t("投影片的上限：960 ÷ 18 = 53.3 ms", abs(c["step_bound_s"] - 0.96 / 18) < 1e-12, f"{1000 * c['step_bound_s']:.1f} ms")
    t("(i) 2 : 12 不成立", not T.conditions(2, 12, 12.5, 3.42, 0.012, 0.030)["i_ok"])
    a = T.align_words("the sky is blue".split(), "the sky blue today".split())
    t("WER：四個詞錯兩個 → 2/4", a["errors"] == 2 and a["S"] + a["D"] + a["I"] == 2 and T.wer(a) == 0.5)
    oe = T.align_words("a b c d e f".split(), "a b x".split(), open_end=True)
    t("被截斷時：還沒念到的字不算漏念", oe["errors"] == 1 and oe["n_ref"] == 3 and oe["n_unspoken"] == 3 and T.align_words("a b c d e f".split(), "a b x".split())["errors"] == 4)
    t("pooled：錯的詞相加 ÷ 參考詞相加", T.pooled([a, T.align_words(["a"] * 6, ["a"] * 6)])["wer"] == 0.2)
    t("norm_simple", T.norm_simple("It's 36, isn’t it?") == ["it's", "36", "isn't", "it"])
    t("numbers_in：字與數字", T.numbers_in("Forty-eight minus nineteen is 29; plus seven: thirty six.") == [48, 19, 29, 7, 36])
    t("answer_is 看最後一個數字", T.answer_is("48 minus 19 is 29, plus 7 is 36.", 36)["correct"] and not T.answer_is("36? no, 29.", 36)["correct"])
    t("text_of 拿掉標記", T.text_of([{"k": "T", "t": 0, "id": 1, "s": "Hi"}, {"k": "T", "t": 0, "id": 130, "s": "<|text_end|>"}]) == "Hi")

    C.banner("設定檔")
    t("設定裡有 6 : 12 與文字全部先寫的那一個", "6:12" in SETTINGS and bool(SETTINGS.get("all", {}).get("chain")))
    t("鍵 1、鍵 2 用到的設定都存在", all(k in SETTINGS for k in CFG["live"]["key1"] + CFG["live"]["key2"]))
    t("system prompt 照 model card", CFG["model"]["system"] == "Respond with interleaved text and audio." and CFG["tts"]["system"].startswith("Perform TTS."))
    t("算術題的正解", 48 - 19 + 7 == CFG["think"]["expected"] and T.numbers_in(CFG["think"]["text"]) == [48, 19, 7])
    t("frame rate 12.5、8 層", F_R == 12.5 and CFG["model"]["codebooks"] == 8)
    t("present.sh 可以執行", (HERE / "present.sh").stat().st_mode & 0o111 != 0)

    C.banner("fake 整條流程")
    import shutil
    import tempfile
    tmp = Path(tempfile.mkdtemp(prefix="w11d3_"))
    global AUDIO, OWN
    keep = (AUDIO, OWN)
    AUDIO, OWN = tmp / "audio", tmp / "own"
    try:
        lm, judge = FakeLFM(), FakeJudge()
        synth_questions(lm)
        t("四題的聲音都合成了", all((AUDIO / f"{i}.wav").exists() for i in all_question_ids()))
        import io
        from contextlib import redirect_stdout
        buf = io.StringIO()
        with redirect_stdout(buf):
            rec = compute(tmp / "run", lm, judge)
            show_table(rec), show_conditions(rec), show_think(rec["think"]), show_cascade(rec)
            sess = Session(rec, tmp / "run", "replay")
            sess.one("q1", "6:12"), sess.think_step(), sess.think_step(), sess.key("6"), sess.key("w"), sess.key("t"), sess.key("3"), sess.key("4")
        t("每題每個設定各一列", len(rec["runs"]) == len(CFG["questions"]) * len(SETTINGS), str(len(rec["runs"])))
        r612 = next(r for r in rec["runs"] if r["setting"] == "6:12")
        t("6 : 12 的事件照 6 : 12 排", r612["schedule"][:2] == [("T", 6), ("A", 12)], str(r612["schedule"][:3]))
        rall = next(r for r in rec["runs"] if r["setting"] == "all")
        t("文字全部先寫：第一個 frame 之前就是全部的文字", rall["tokens_before_first_frame"] == rall["timeline"]["n_text"] and rall.get("chain"))
        t("文字全部先寫的第一個聲音比 6 : 12 晚", rall["timeline"]["first_sound"] > r612["timeline"]["first_sound"])
        s = {x["setting"]: x for x in rec["summary"]}
        t("2 : 12 的第一個聲音最早", s["2:12"]["first_sound"] < s["6:12"]["first_sound"] < s["24:12"]["first_sound"])
        t("summary 的 WER 是合計", s["2:12"]["wer"]["n_ref"] == sum(r["wer"]["n_ref"] for r in rec["runs"] if r["setting"] == "2:12"))
        tp, tr = rec["think"]["plain"], rec["think"]["reason"]
        t("後半：兩次都抓到答案；先想再說的那次 U_th 是全部的文字、第一個聲音比較晚", tp["answer"]["correct"] and tr["answer"]["correct"]
          and tr["U_th"] == tr["timeline"]["n_text"] and tp["U_th"] <= 6 and tr["timeline"]["first_sound"] > tp["timeline"]["first_sound"],
          f"U_th {tp['U_th']} → {tr['U_th']}")
        t("後半：推理的 token 有標記、預測的沉默 = U_th × D_text", all(e.get("r") for e in tr["events"] if e["k"] == "T")
          and abs(tr["predicted_silence"] - tr["U_th"] * tr["steps"]["D_text"]) < 1e-3)
        t("cascade 跑不起來時退回 W1 的數字並標明", "error" in rec["cascade"])
        m = summary_for_metrics(rec)
        t("metrics 的摘要可以寫成 JSON", len(json.dumps(m)) > 500 and len(m["settings"]) == len(SETTINGS))
        t("畫面印得出來", "who leads" in buf.getvalue() and "two conditions" in buf.getvalue() and "two ways to answer" in buf.getvalue())
    finally:
        AUDIO, OWN = keep
        shutil.rmtree(tmp, ignore_errors=True)

    rp = RUNS / "rehearsal" / "rehearsal.json"
    if rp.exists() and METRICS.exists():
        C.banner("彩排紀錄對 w11-metrics.json")
        rec = json.loads(rp.read_text(encoding="utf-8"))
        d3 = json.loads(METRICS.read_text(encoding="utf-8")).get("demo3")
        t("metrics 的 demo3 與彩排紀錄同一次", bool(d3) and d3["date"] == rec["date"])
        t("彩排用的設定檔沒有再改過", rec["config_digest"] == digest(), "改過設定就重跑 rehearse")
    print(f"\n{n['ok']} 項通過" + (f"，{C.RED}{n['bad']} 項沒過{C.RESET}" if n["bad"] else ""))
    if n["bad"]:
        raise SystemExit(1)


def main() -> None:
    ap = argparse.ArgumentParser(description="W11 demo 3")
    sub = ap.add_subparsers(dest="cmd")
    for name in ("show", "replay", "fake", "selftest", "export"):
        p = sub.add_parser(name)
        p.add_argument("--repo", default=None)
    p = sub.add_parser("rehearse")
    p.add_argument("--preview", action="store_true")
    p.add_argument("--questions", type=int, default=None)
    p.add_argument("--no-think", action="store_true")
    p.add_argument("--max-new", type=int, default=None)
    p.add_argument("--no-cascade", action="store_true")
    p.add_argument("--repo", default=None)
    for name in ("fetch", "tts"):
        p = sub.add_parser(name)
        p.add_argument("--force", action="store_true")
    p = sub.add_parser("record")
    p.add_argument("qid")
    p.add_argument("--seconds", type=float, default=9.0)
    p = sub.add_parser("check")
    p.add_argument("--no-cascade", action="store_true")
    p.add_argument("--repo", default=None)
    args = ap.parse_args()
    cmd = args.cmd or "show"
    if cmd in ("show", "replay", "fake"):
        if not args.cmd:
            args.repo = None
        cmd_show(args, cmd)
    else:
        {"rehearse": cmd_rehearse, "fetch": cmd_fetch, "tts": cmd_tts, "record": cmd_record, "check": cmd_check,
         "selftest": cmd_selftest, "export": cmd_export}[cmd](args)


if __name__ == "__main__":
    main()

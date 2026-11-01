# -*- coding: utf-8 -*-
# 此檔由 bin/publish_demos.py 從 private repo 的 <course>/demos/ 複製而來，請勿在公開 repo 直接編輯。
"""W7 demo 1：幻覺——證據歸零時先驗長什麼樣（同一段聲音、兩個模型，差別只在輸出端有沒有先驗）。

  show       課堂：載入 Whisper small 與 wav2vec2-base-960h，終端機單鍵操作（下面的鍵）。離開時存 runs/live/<時間>/（不進版控）
  rehearse   課前：不互動，照課堂順序全部跑一遍，存 runs/rehearsal/（進版控 = 現場的退路）
  replay     現場模型出狀況時：從 runs/rehearsal/ 讀回，同樣的鍵、同樣的畫面，不碰 torch
  check      課前體檢：版本、兩個模型在不在快取、<|nospeech|> 的 id、encoder 歸零的 hook 有沒有生效、MPS 對 CPU（溫度 0 的轉錄要相同）
             → runs/check.json（進版控）
  compare    §6 第 8 項：同一段音訊在 transformers（這個 demo 的路）、pywhispercpp（whisper.cpp Metal）、mlx-whisper 三條路的
             時間與輸出 → runs/compare.json。缺哪條就跳過哪條並印原因；不是課堂上的鍵
  fetch      課前一週：下載兩個模型（commit 先讀 W4 Demo 3、W3 Demo 3 的 model.lock，寫進這裡的 model.lock；要網路）；
             --mlx 另抓 mlx-community/whisper-small-mlx 給 compare 用
  make       產生音訊（確定性）：audio/silence.wav、audio/noise_<snr>.wav、audio/long.wav（W3 那句 + 長靜音）。audio/ 不進版控
  record     錄一段自己的非語音（音樂、咳嗽、敲桌）→ audio/own.wav。不進版控、不上站（public_demos_exclude）
  fake       沒有模型也能演練流程（罐頭輸出，畫面標明 FAKE，不是給學生看的）
  selftest   不需要模型：壓縮比、fallback 判定、噪音的 SNR、CTC 摺疊、平均 log 機率、fake 整條流程
  export     runs/rehearsal/rehearsal.json 的摘要 → slides/assets/w07/w07-metrics.json（鍵 demo1；投影片數字進版控的出處）

畫面上的鍵（順序照投影片 p27 講稿的【操作】）：
  1  十秒靜音 → Whisper small：轉錄、逐 token log 機率、no_speech_prob、壓縮比；fallback 照目前的開關
  2  十秒白噪音，三個 SNR 各一次（同上）
  3  W3 那句 + 長靜音：固定 30 s 視窗，第二個視窗全是靜音；prompt 槽裡帶不帶第一個視窗的文字照 c 的狀態
  4  自己錄的非語音（audio/own.wav；沒有就說沒有）
  w  同一批檔案 → wav2vec2-base-960h greedy：T、blank 佔的比例、非 blank 的 frame 數、摺疊後的字串
  z  encoder 輸出歸零 → decoder 自由生成：beam 5 的 5 條、再抽樣 5 條        Z  同上，但換成 W3 那句 encoder 輸出的平均向量
  f  切換 fallback 開／關（開了之後 1、2、3、4 都會印每一次嘗試與觸發的規則）
  c  切換 condition_on_prev（3 用）        l  切換語言 en／zh（1、2、4 用：先驗換了，證據還是平的）
  s  W3 那句本身過兩個模型（對照：有語音時兩邊都對）       h  重印選單        q  離開

數字的定義：
  log 機率   = 生成之後 teacher-forced 一次前向算出的 log P(ℓ_u | ℓ_{<u}, X)（含 <|endoftext|>），不是 generate 過程經過 suppress 之後的分數
  平均       = Σ log 機率 ÷ (token 數 + 1)，與 openai/whisper 的 avg_logprob 同一個定義（+1 是 <|endoftext|>）
  no_speech  = decoder 吃 <|startoftranscript|> 那一格時 <|nospeech|> 的機率（openai/whisper decoding.py 的 no_speech_probs）
  壓縮比     = len(text) ÷ len(zlib.compress(text))（openai/whisper utils.compression_ratio）；重複迴圈會把它推高
  CTC blank  = pad_token_id（wav2vec2 的 <pad> 就是 blank）；「blank 佔的比例」= argmax 是 blank 的 frame ÷ T
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import tomllib
import zlib
from contextlib import contextmanager
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import common as C  # noqa: E402

CFG = tomllib.loads((HERE / "demo_config.toml").read_text(encoding="utf-8"))
FS = int(CFG["audio"]["fs"])
DIG = int(CFG["display"]["digits"])
MODEL_LOCK = HERE / "model.lock"
AUDIO = HERE / "audio"
METRICS = (HERE.parent.parent / "slides" / "assets" / "w07" / "w07-metrics.json").resolve()
SIBLING_LOCKS = {CFG["whisper"]["repo"]: HERE.parent / "w04_d3_attention_heads" / "model.lock",
                 CFG["ctc"]["repo"]: HERE.parent / "w03_d3_blank_heatmap" / "model.lock"}


def _p(rel: str) -> Path:
    return (HERE / rel).resolve()


# ── 純函數（selftest 測這些）──────────────────────────────────────────────
def compression_ratio(text: str) -> float:
    """openai/whisper utils.compression_ratio：UTF-8 位元組長度 ÷ zlib 壓縮後長度。空字串回 0。"""
    b = text.encode("utf-8")
    return len(b) / len(zlib.compress(b)) if b else 0.0


def avg_logprob(logprobs: list[float]) -> float:
    """openai/whisper：sum_logprobs / (len(tokens) + 1)。這裡 logprobs 已含 <|endoftext|>，所以除以 len。"""
    return float(sum(logprobs) / len(logprobs)) if logprobs else 0.0


def fallback_verdict(avg_lp: float, cr: float, no_speech: float, w: dict) -> dict:
    """Whisper §4.5 / transcribe.py decode_with_fallback 的規則，一次嘗試的判定。
    needs_fallback：壓縮比 > 2.4 或 平均 log 機率 < −1 → 升溫重試。
    is_silence：no_speech_prob > 0.6 且 平均 log 機率 < −1（兩個一起；原文：單看 <|nospeech|> 不夠）。
    原文的順序：先看 no_speech（should_skip），再看要不要 fallback；這裡兩個都算出來、都印。"""
    reasons = []
    if cr > w["compression_ratio_threshold"]:
        reasons.append(f"compression ratio {cr:.2f} > {w['compression_ratio_threshold']}")
    if avg_lp < w["logprob_threshold"]:
        reasons.append(f"avg logprob {avg_lp:.2f} < {w['logprob_threshold']}")
    is_silence = (no_speech > w["no_speech_threshold"]) and (avg_lp < w["logprob_threshold"])
    return {"needs_fallback": bool(reasons), "reasons": reasons, "is_silence": bool(is_silence)}


def white_noise(n: int, rms: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(n)
    return x / np.sqrt(np.mean(x ** 2)) * rms


def noise_for_snr(speech_rms: float, snr_db: float, n: int, seed: int) -> np.ndarray:
    """SNR 對 W3 那句的 RMS：noise_rms = speech_rms / 10^(snr/20)。"""
    return white_noise(n, speech_rms / 10 ** (snr_db / 20), seed)


def rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(x ** 2))) if len(x) else 0.0


def ctc_collapse(ids: list[int], blank: int) -> list[int]:
    """先摺疊重複、再刪 blank（W3 的順序）。"""
    out, prev = [], None
    for i in ids:
        if i != prev and i != blank:
            out.append(i)
        prev = i
    return out


def strip_prefix(seq: list[int], sot: int, eot: int, n_prefix: int) -> list[int]:
    """generate 回來的序列：找到 <|startoftranscript|>、跳過 n_prefix 個前綴 token、切到 <|endoftext|>。prompt 有沒有被剝掉都適用。"""
    if sot in seq:
        seq = seq[seq.index(sot) + n_prefix:]
    else:                                   # 沒看到 sot：當成已經剝掉前綴，只去掉開頭的特殊 token
        while seq and seq[0] >= eot:
            seq = seq[1:]
    if eot in seq:
        seq = seq[: seq.index(eot)]
    return [int(t) for t in seq if int(t) < eot]


def n_words(text: str) -> int:
    return len(text.split())


# ── 音訊 ─────────────────────────────────────────────────────────────────
CLIP_ORDER = ["silence", "noise_0", "noise_20", "noise_40", "long", "own"]


def make_audio() -> dict:
    a = CFG["audio"]
    w3 = C.load_wav(_p(a["w3_wav"]), FS)
    srms = rms(w3)
    AUDIO.mkdir(exist_ok=True)
    clips = {"silence": np.zeros(int(a["silence_s"] * FS))}
    for k, snr in enumerate(a["noise_snr_db"]):
        clips[f"noise_{int(snr)}"] = noise_for_snr(srms, snr, int(a["noise_s"] * FS), int(a["noise_seed"]) + k)
    clips["long"] = np.concatenate([w3, np.zeros(int(a["tail_silence_s"] * FS))])
    for name, x in clips.items():
        C.save_wav(AUDIO / f"{name}.wav", x, FS)
    return {"speech_rms": srms, "w3_s": len(w3) / FS, **{k: len(v) / FS for k, v in clips.items()}}


def load_clips() -> dict[str, np.ndarray]:
    if not (AUDIO / "silence.wav").exists():
        sys.exit("audio/ 還沒有音訊，先跑 ./present.sh make")
    clips = {}
    for name in CLIP_ORDER:
        f = AUDIO / f"{name}.wav"
        if f.exists():
            clips[name] = C.load_wav(f, FS)
    clips["w3"] = C.load_wav(_p(CFG["audio"]["w3_wav"]), FS)
    return clips


def clip_label(name: str) -> str:
    a = CFG["audio"]
    fixed = {"silence": f"{a['silence_s']:g} s 純靜音（全零）", "long": f"W3 那句 + {a['tail_silence_s']:g} s 靜音",
             "own": "自己錄的非語音（audio/own.wav）", "w3": "W3 那句（1272-128104-0000，有語音的對照）"}
    if name in fixed:
        return fixed[name]
    return f"{a['noise_s']:g} s 白噪音，SNR {name.split('_')[1]} dB 對 W3 那句"


# ── 模型 ─────────────────────────────────────────────────────────────────
def read_lock() -> dict[str, str]:
    """model.lock：每行 `repo sha`。沒有就退到兄弟 demo 的 lock（W4 d3、W3 d3 都是單行 sha）。"""
    lock = {}
    if MODEL_LOCK.exists():
        for line in MODEL_LOCK.read_text(encoding="utf-8").splitlines():
            parts = line.split()
            if len(parts) == 2:
                lock[parts[0]] = parts[1]
    for repo, sib in SIBLING_LOCKS.items():
        if repo not in lock and sib.exists():
            sha = sib.read_text(encoding="utf-8").strip()
            if sha:
                lock[repo] = sha
    return lock


def model_dir(repo: str, offline: bool = True) -> str:
    from huggingface_hub import snapshot_download
    return snapshot_download(repo, revision=read_lock().get(repo), local_files_only=offline,
                             allow_patterns=["*.json", "*.txt", "*.safetensors"])


def quiet_transformers() -> None:
    """課堂上不要進度條與 LOAD REPORT（wav2vec2 的 masked_spec_embed MISSING 是已知的、推論用不到）。"""
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


class Whisper:
    """transformers 的 WhisperForConditionalGeneration。generate 給文字、teacher-forced 前向給 log 機率、hook 給歸零的 encoder。"""
    name = "whisper"

    def __init__(self, device: str | None = None):
        import torch
        from transformers import WhisperForConditionalGeneration, WhisperProcessor
        quiet_transformers()
        self.torch = torch
        w = CFG["whisper"]
        path = model_dir(w["repo"])
        self.proc = WhisperProcessor.from_pretrained(path)
        self.tok = self.proc.tokenizer
        self.model = WhisperForConditionalGeneration.from_pretrained(path).eval()
        self.device = pick_device(device or w["device"])
        self.model.to(self.device)
        cfg, gen = self.model.config, self.model.generation_config
        self.n_params = sum(p.numel() for p in self.model.parameters())
        self.sot = int(gen.decoder_start_token_id)
        self.eot = int(gen.eos_token_id if gen.eos_token_id is not None else cfg.eos_token_id)
        self.notimestamps = int(gen.no_timestamps_token_id)
        self.lang_to_id, self.task_to_id = gen.lang_to_id, gen.task_to_id
        self.nospeech = None
        for t in ("<|nospeech|>", "<|nocaptions|>"):
            i = self.tok.convert_tokens_to_ids(t)
            if i is not None and i != self.tok.unk_token_id:
                self.nospeech, self.nospeech_token = int(i), t
                break
        self.task = w["task"]
        self.language = w["language"]
        self._hook = None
        self._mean_vec = None

    # 前綴、prompt
    def prefix(self, language: str | None = None) -> list[int]:
        lang = language or self.language
        return [self.sot, int(self.lang_to_id[f"<|{lang}|>"]), int(self.task_to_id[self.task]), self.notimestamps]

    def prompt_ids(self, text: str | None) -> list[int]:
        if not text:
            return []
        return [int(t) for t in np.asarray(self.proc.get_prompt_ids(text, return_tensors="np")).tolist()]

    def features(self, x: np.ndarray):
        return self.proc(x.astype(np.float32), sampling_rate=FS, return_tensors="pt", return_attention_mask=True)

    # 生成
    def generate(self, x: np.ndarray, temperature: float = 0.0, language: str | None = None, prompt: str | None = None,
                 num_beams: int | None = None, num_return: int = 1, seed: int | None = None) -> list[list[int]]:
        torch = self.torch
        w = CFG["whisper"]
        inp = self.features(x)
        kw = dict(language=language or self.language, task=self.task, return_timestamps=False,
                  max_new_tokens=int(w["max_new_tokens"]), return_dict_in_generate=True)
        if temperature > 0:
            if seed is not None:
                torch.manual_seed(seed)
            kw.update(do_sample=True, temperature=float(temperature), top_k=0, top_p=1.0, num_beams=1, num_return_sequences=num_return)
        else:
            nb = int(num_beams if num_beams is not None else w["num_beams"])
            kw.update(do_sample=False, num_beams=nb, num_return_sequences=min(num_return, nb))
        pid = self.prompt_ids(prompt)
        if pid:
            kw["prompt_ids"] = torch.tensor(pid, dtype=torch.long, device=self.device)
        try:
            with torch.no_grad():
                out = self.model.generate(inp.input_features.to(self.device), attention_mask=inp.attention_mask.to(self.device), **kw)
        except AttributeError as e:
            if self.device.type == "mps" and "layers" in str(e):
                raise RuntimeError("transformers 5.17.0 的 Whisper generate 在 MPS 上會掛（EncoderDecoderCache.layers）；"
                                   "demo_config.toml [whisper] device 改成 cpu") from e
            raise
        seqs = out.sequences if hasattr(out, "sequences") else out["sequences"] if isinstance(out, dict) else out
        n_prefix = len(self.prefix(language))
        return [strip_prefix([int(t) for t in s.tolist()], self.sot, self.eot, n_prefix) for s in seqs]

    def teacher_logprobs(self, x: np.ndarray, ids: list[int], language: str | None = None, prompt: str | None = None) -> list[float]:
        """log P(ℓ_u | ℓ_{<u}, X) for 每個文字 token 與 <|endoftext|>，一次前向。"""
        torch = self.torch
        pre = self.prompt_ids(prompt) + self.prefix(language)
        full = pre + list(ids) + [self.eot]
        inp = self.features(x)
        dec = torch.tensor([full], dtype=torch.long, device=self.device)
        with torch.no_grad():
            out = self.model(input_features=inp.input_features.to(self.device), decoder_input_ids=dec)
        logp = torch.log_softmax(out.logits[0].float(), dim=-1).cpu().numpy()
        return [float(logp[j, full[j + 1]]) for j in range(len(pre) - 1, len(full) - 1)]

    def no_speech_prob(self, x: np.ndarray, language: str | None = None, prompt: str | None = None) -> float | None:
        """<|startoftranscript|> 那一格的 softmax 裡 <|nospeech|> 的機率（openai decoding.py：logits[:, sot_index]）。"""
        if self.nospeech is None:
            return None
        torch = self.torch
        pid = self.prompt_ids(prompt)
        full = pid + self.prefix(language)
        inp = self.features(x)
        dec = torch.tensor([full], dtype=torch.long, device=self.device)
        with torch.no_grad():
            out = self.model(input_features=inp.input_features.to(self.device), decoder_input_ids=dec)
        probs = torch.softmax(out.logits[0, len(pid)].float(), dim=-1)
        return float(probs[self.nospeech])

    def pieces(self, ids: list[int]) -> list[str]:
        return [self.tok.decode([i]) for i in ids]

    def text(self, ids: list[int]) -> str:
        return self.tok.decode(ids, skip_special_tokens=True)

    # encoder 歸零
    def mean_vector(self, x_speech: np.ndarray):
        torch = self.torch
        if self._mean_vec is None:
            inp = self.features(x_speech)
            n_audio = len(x_speech) // (FS // 100) // 2
            with torch.no_grad():
                enc = self.model.model.encoder(inp.input_features.to(self.device)).last_hidden_state
            self._mean_vec = enc[0, :n_audio].mean(0)
        return self._mean_vec

    @contextmanager
    def encoder_replaced(self, mode: str, x_speech: np.ndarray | None = None):
        torch = self.torch
        vec = self.mean_vector(x_speech) if mode == "mean" else None

        def hook(module, args, output):
            h = output[0] if isinstance(output, tuple) else output.last_hidden_state
            z = torch.zeros_like(h) if vec is None else vec.to(h.dtype).expand_as(h).clone()
            if isinstance(output, tuple):
                return (z,) + tuple(output[1:])
            output.last_hidden_state = z
            return output
        handle = self.model.model.encoder.register_forward_hook(hook)
        try:
            yield
        finally:
            handle.remove()

    def encoder_norm(self, x: np.ndarray) -> float:
        """check 用：hook 之下 encoder 的輸出是不是真的全零（拿一次前向的 last_hidden_state 的範數）。"""
        torch = self.torch
        inp = self.features(x)
        with torch.no_grad():
            enc = self.model.model.encoder(inp.input_features.to(self.device)).last_hidden_state
        return float(enc.float().norm())


class CTC:
    """transformers 的 Wav2Vec2ForCTC（W3 Demo 3 的模型），只做 greedy。"""
    name = "wav2vec2"

    def __init__(self, device: str | None = None):
        import torch
        from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
        quiet_transformers()
        self.torch = torch
        path = model_dir(CFG["ctc"]["repo"])
        self.proc = Wav2Vec2Processor.from_pretrained(path)
        self.model = Wav2Vec2ForCTC.from_pretrained(path).eval()
        self.device = pick_device(device or CFG["ctc"]["device"])
        self.model.to(self.device)
        self.blank = int(self.model.config.pad_token_id)
        self.n_params = sum(p.numel() for p in self.model.parameters())

    def greedy(self, x: np.ndarray) -> dict:
        torch = self.torch
        inp = self.proc(x.astype(np.float32), sampling_rate=FS, return_tensors="pt")
        with torch.no_grad():
            logits = self.model(inp.input_values.to(self.device)).logits[0].float().cpu().numpy()
        am = logits.argmax(-1).tolist()
        T = len(am)
        collapsed = ctc_collapse(am, self.blank)
        text = self.proc.tokenizer.decode(collapsed, group_tokens=False)   # 已經摺疊過，不要讓 tokenizer 再摺一次（LL 會變 L）
        return {"T": T, "blank_frames": int(sum(1 for i in am if i == self.blank)), "nonblank_frames": int(sum(1 for i in am if i != self.blank)),
                "blank_fraction": float(np.mean([i == self.blank for i in am])) if T else 0.0, "n_symbols": len(collapsed), "text": text}


# ── fake 後端（沒有模型時演練流程）────────────────────────────────────────────
class FakeWhisper:
    name = "fake"
    nospeech_token = "<|nospeech|>"
    device = "fake"
    n_params = 0

    _canned = {"silence": "Thank you for watching.", "noise_0": "I'm going to go ahead and get started.",
               "noise_20": "Thank you.", "noise_40": "Thank you for watching!", "own": "(music)", "w3": "Mr. Quilter is the apostle of the middle classes, and we are glad to welcome his gospel.",
               "long_w0": "Mr. Quilter is the apostle of the middle classes, and we are glad to welcome his gospel.", "long_w1": "Thank you for watching."}

    def __init__(self, *_):
        self.language = CFG["whisper"]["language"]
        self.task = CFG["whisper"]["task"]
        self._cur = "silence"
        self._vocab: list[str] = []          # fake 的 id 就是這張表的索引，pieces／text 從這裡查回去

    def _toks(self, text: str) -> list[str]:
        return [(" " + w if i else w) for i, w in enumerate(text.split())]

    def _ids(self, text: str) -> list[int]:
        base = len(self._vocab)
        toks = self._toks(text)
        self._vocab += toks
        return list(range(base, base + len(toks)))

    def set_clip(self, name: str) -> None:
        self._cur = name

    def generate(self, x, temperature=0.0, language=None, prompt=None, num_beams=None, num_return=1, seed=None):
        text = self._canned.get(self._cur, "Thank you.")
        if language == "zh":
            text = "謝謝大家收看。"
        if temperature > 0:
            text = text.rstrip(".!") + " today."
        return [self._ids(text) for _ in range(max(1, num_return))]

    def pieces(self, ids):
        return [self._vocab[i] for i in ids]

    def text(self, ids):
        return "".join(self.pieces(ids))

    def teacher_logprobs(self, x, ids, language=None, prompt=None):
        rng = np.random.default_rng(len(ids))
        return [float(v) for v in -rng.uniform(0.05, 1.5, size=len(ids) + 1)]

    def no_speech_prob(self, x, language=None, prompt=None):
        return 0.72 if self._cur in ("silence", "long_w1") else 0.31

    @contextmanager
    def encoder_replaced(self, mode, x_speech=None):
        old = self._cur
        self._cur = "silence"
        try:
            yield
        finally:
            self._cur = old


class FakeCTC:
    name = "fake"
    device = "fake"
    n_params = 0
    blank = 0

    def greedy(self, x):
        T = len(x) // 320
        nb = 0 if np.max(np.abs(x)) < 1e-6 else int(0.02 * T)
        return {"T": T, "blank_frames": T - nb, "nonblank_frames": nb, "blank_fraction": (T - nb) / T if T else 0.0, "n_symbols": nb // 2,
                "text": "" if nb == 0 else "E A"}


# ── 輸出紀錄 ────────────────────────────────────────────────────────────────
class Out:
    def __init__(self):
        self.lines: list[str] = []

    def __call__(self, s: str = "") -> None:
        print(s)
        self.lines.append(s)

    def banner(self, s: str, color: str = C.BLUE) -> None:
        C.banner(s, color)
        self.lines += ["", "─" * 72, s, "─" * 72]

    def text(self) -> str:
        return "\n".join(self.lines) + "\n"


# ── Demo ────────────────────────────────────────────────────────────────────
class Demo:
    def __init__(self, out: Out, clips: dict[str, np.ndarray], whisper, ctc, stored: dict | None = None, tag: str = ""):
        self.out, self.clips, self.wh, self.ctc = out, clips, whisper, ctc
        self.res: dict = dict(stored or {})
        self.replay = stored is not None
        self.tag = tag
        w = CFG["whisper"]
        self.fallback = bool(w["fallback_default"])
        self.cond_prev = bool(w["condition_on_prev_default"])
        self.language = w["language"]

    # 一次嘗試
    def attempt(self, x: np.ndarray, temperature: float, language: str, prompt: str | None, k: int) -> dict:
        w = CFG["whisper"]
        t0 = time.perf_counter()
        ids = self.wh.generate(x, temperature=temperature, language=language, prompt=prompt, seed=int(w["seed"]) + k)[0]
        dt = time.perf_counter() - t0
        lps = self.wh.teacher_logprobs(x, ids, language=language, prompt=prompt)
        text = self.wh.text(ids)
        ns = self.wh.no_speech_prob(x, language=language, prompt=prompt)
        cr = compression_ratio(text)
        alp = avg_logprob(lps)
        return {"temperature": temperature, "ids": ids, "pieces": self.wh.pieces(ids), "logprobs": lps[:-1], "eot_logprob": lps[-1], "text": text,
                "n_tokens": len(ids), "n_words": n_words(text), "avg_logprob": alp, "max_logprob": max(lps[:-1]) if ids else None,
                "n_confident": int(sum(1 for v in lps[:-1] if v > CFG["display"]["confident_logprob"])),
                "no_speech_prob": ns, "compression_ratio": cr, "verdict": fallback_verdict(alp, cr, ns if ns is not None else 0.0, w), "seconds": dt}

    def transcribe(self, x: np.ndarray, language: str, prompt: str | None, fallback: bool) -> dict:
        """一個 30 s 以內的視窗：fallback 關 = 只試溫度 0；開 = §4.5 的階梯，不合格就升溫。回所有嘗試與最後採用的那一次。"""
        w = CFG["whisper"]
        temps = list(w["temperatures"]) if fallback else [0.0]
        attempts = []
        for k, t in enumerate(temps):
            a = self.attempt(x, t, language, prompt, k)
            attempts.append(a)
            if not a["verdict"]["needs_fallback"]:
                break
        final = attempts[-1]
        return {"attempts": attempts, "final_temperature": final["temperature"], "final": final, "fallback": fallback, "language": language,
                "prompt": prompt, "exhausted": fallback and final["verdict"]["needs_fallback"]}

    def transcribe_long(self, x: np.ndarray, language: str, cond_prev: bool, fallback: bool) -> dict:
        w = CFG["whisper"]
        n = int(w["window_s"] * FS)
        windows, prompt = [], None
        for i in range(0, len(x), n):
            seg = x[i: i + n]
            r = self.transcribe(seg, language, prompt if cond_prev else None, fallback)
            r.update({"start_s": i / FS, "end_s": min(i + n, len(x)) / FS, "prompt_used": prompt if cond_prev else None})
            windows.append(r)
            # openai transcribe.py：溫度 > prompt_reset_temperature 就把 prompt 斷掉；否則把這個視窗的文字接到 prompt
            if r["final_temperature"] > w["prompt_reset_temperature"]:
                prompt = None
            else:
                prompt = ((prompt or "") + " " + r["final"]["text"]).strip() or None
        return {"windows": windows, "condition_on_prev": cond_prev, "fallback": fallback, "language": language,
                "text": " ".join(wn["final"]["text"].strip() for wn in windows)}

    # 畫面
    def _print_attempt(self, a: dict, head: str) -> None:
        o, d = self.out, CFG["display"]
        v = a["verdict"]
        o(f"  {C.BOLD}{head}{C.RESET}  temperature {a['temperature']:.1f}  ({a['seconds']:.2f} s)")
        o(f"  轉錄：{C.RED}{a['text']!r}{C.RESET}")
        mx = f"{a['max_logprob']:.{DIG}f}" if a["max_logprob"] is not None else "—"
        o(f"  {a['n_words']} 個字、{a['n_tokens']} 個 token；平均 log 機率 {a['avg_logprob']:.{DIG}f}、最高 {mx}、"
          f"高於 {d['confident_logprob']} 的有 {a['n_confident']} 個；<|endoftext|> {a['eot_logprob']:.{DIG}f}")
        ns = a["no_speech_prob"]
        ns_s = f"{ns:.{DIG}f}" if ns is not None else "—"
        verdict = ("不合格 → " + "；".join(v["reasons"])) if v["needs_fallback"] else "合格"
        if v["is_silence"]:
            verdict += "；no_speech 門檻 + log 機率門檻都過 → 當靜音"
        o(f"  no_speech_prob {ns_s}   壓縮比 {a['compression_ratio']:.2f}   §4.5 判定：{verdict}")
        m = int(d["max_tokens_each"])
        pairs = list(zip(a["pieces"], a["logprobs"]))
        shown = pairs if len(pairs) <= 2 * m else pairs[:m] + [("…", None)] + pairs[-m:]
        o("  逐 token： " + "  ".join(f"{p!r}({lp:.2f})" if lp is not None else "…" for p, lp in shown))

    def _print_transcribe(self, r: dict, head: str) -> None:
        o = self.out
        for k, a in enumerate(r["attempts"]):
            self._print_attempt(a, f"{head} · 嘗試 {k + 1}/{len(r['attempts'])}" if r["fallback"] else head)
        if r["fallback"]:
            o(f"  → 採用溫度 {r['final_temperature']:.1f} 的輸出" + ("（階梯用完仍不合格，照 openai 的做法採最後一次）" if r["exhausted"] else ""))

    def _key(self, *parts) -> str:
        return ":".join(str(p) for p in parts)

    def _get(self, key: str, compute):
        if key in self.res:
            return self.res[key]
        if self.replay:
            self.out(f"  {C.AMBER}彩排紀錄裡沒有 {key}{C.RESET}")
            return None
        self.res[key] = compute()
        return self.res[key]

    def _settings(self) -> str:
        return f"[fallback {'開' if self.fallback else '關'} · condition_on_prev {'開' if self.cond_prev else '關'} · 語言 {self.language}]"

    def panel_clip(self, name: str) -> None:
        o = self.out
        if name not in self.clips and not self.replay:
            o(f"  沒有 audio/{name}.wav" + ("：先 ./present.sh record" if name == "own" else "：先 ./present.sh make"))
            return
        o.banner(f"{name}：{clip_label(name)} → Whisper small  {self._settings()}")
        key = self._key("whisper", name, self.language, "fb" if self.fallback else "nofb")
        if hasattr(self.wh, "set_clip"):
            self.wh.set_clip(name)
        r = self._get(key, lambda: self.transcribe(self.clips[name], self.language, None, self.fallback))
        if r:
            self._print_transcribe(r, name)

    def panel_noise(self) -> None:
        for snr in CFG["audio"]["noise_snr_db"]:
            self.panel_clip(f"noise_{int(snr)}")

    def panel_long(self) -> None:
        o = self.out
        if "long" not in self.clips and not self.replay:
            o("  沒有 audio/long.wav：先 ./present.sh make"); return
        o.banner(f"long：{clip_label('long')} → Whisper small，固定 {CFG['whisper']['window_s']:g} s 視窗  {self._settings()}")
        key = self._key("long", self.language, "cond" if self.cond_prev else "nocond", "fb" if self.fallback else "nofb")

        def compute():
            if hasattr(self.wh, "set_clip"):
                self.wh.set_clip("long_w0")
            return self.transcribe_long(self.clips["long"], self.language, self.cond_prev, self.fallback)
        r = self._get(key, compute)
        if not r:
            return
        for i, wn in enumerate(r["windows"]):
            if hasattr(self.wh, "set_clip"):
                self.wh.set_clip(f"long_w{i}")
            o(f"\n  視窗 {i + 1}：{wn['start_s']:.1f}–{wn['end_s']:.1f} s" + (f"，prompt 槽 = {wn['prompt_used']!r}" if wn["prompt_used"] else "，prompt 槽空"))
            self._print_transcribe(wn, f"視窗 {i + 1}")
        o(f"\n  整段：{C.RED}{r['text']!r}{C.RESET}")

    def panel_ctc(self) -> None:
        o = self.out
        o.banner("同一批檔案 → wav2vec2-base-960h（CTC）greedy：逐 frame argmax → 摺疊 → 刪 blank")
        o(f"  {'clip':<10s} {'T':>5s} {'blank':>7s} {'非blank':>7s} {'符號':>5s}  摺疊後")
        for name in CLIP_ORDER + ["w3"]:
            if name not in self.clips and not self.replay:
                continue
            r = self._get(self._key("ctc", name), lambda n=name: self.ctc.greedy(self.clips[n]))
            if r:
                o(f"  {name:<10s} {r['T']:>5d} {r['blank_fraction']:>6.1%} {r['nonblank_frames']:>7d} {r['n_symbols']:>5d}  {C.RED}{r['text']!r}{C.RESET}")
        o("  （blank 贏不是「偵測到靜音」，是「這一格沒有符號可發」——CTC 沒有靜音這個概念；W3 p48）")

    def panel_zeroed(self, mode: str) -> None:
        o, z = self.out, CFG["zeroed"]
        o.banner(f"encoder 輸出換成{'全零' if mode == 'zeros' else 'W3 那句 encoder 輸出的平均向量'} → decoder 自由生成（P_ILM 的樣本）  語言 {self.language}")

        def compute():
            x = self.clips.get("silence", np.zeros(FS))
            with self.wh.encoder_replaced(mode, self.clips.get("w3")):
                beams = self.wh.generate(x, 0.0, self.language, None, num_beams=int(z["num_beams"]), num_return=int(z["num_return"]))
                samples = [self.wh.generate(x, float(z["sample_temperature"]), self.language, None, seed=int(CFG["whisper"]["seed"]) + 100 + k)[0]
                           for k in range(int(z["n_samples"]))]
                out = {"mode": mode, "beams": [{"text": self.wh.text(b), "n_tokens": len(b), "logprobs": self.wh.teacher_logprobs(x, b, self.language)} for b in beams],
                       "samples": [{"text": self.wh.text(s), "n_tokens": len(s)} for s in samples]}
                for b in out["beams"]:
                    b["avg_logprob"] = avg_logprob(b["logprobs"])
                    del b["logprobs"]
                return out
        r = self._get(self._key("zeroed", mode, self.language), compute)
        if not r:
            return
        o(f"  beam {z['num_beams']}，前 {len(r['beams'])} 條（平均 log 機率是 encoder 歸零之下 teacher-forced 的）：")
        for k, b in enumerate(r["beams"]):
            o(f"   {k + 1}. {C.RED}{b['text']!r}{C.RESET}   ({b['n_tokens']} token，avg {b['avg_logprob']:.{DIG}f})")
        o(f"  溫度 {z['sample_temperature']} 抽樣 {len(r['samples'])} 條：")
        for k, s in enumerate(r["samples"]):
            o(f"   {k + 1}. {C.RED}{s['text']!r}{C.RESET}")
        o("  （這是 ILME 的近似：Meng et al. 2021 對 AED 是拿掉 context 向量，不是字面上把 encoder 輸出設零）")

    def panel_w3(self) -> None:
        o = self.out
        o.banner(f"對照：W3 那句（有語音）→ 兩個模型  {self._settings()}")
        if hasattr(self.wh, "set_clip"):
            self.wh.set_clip("w3")
        r = self._get(self._key("whisper", "w3", self.language, "fb" if self.fallback else "nofb"),
                      lambda: self.transcribe(self.clips["w3"], self.language, None, self.fallback))
        if r:
            self._print_transcribe(r, "w3")
        c = self._get(self._key("ctc", "w3"), lambda: self.ctc.greedy(self.clips["w3"]))
        if c:
            o(f"  CTC greedy：T {c['T']}，blank {c['blank_fraction']:.1%}，{C.RED}{c['text']!r}{C.RESET}")
        ref = _p(CFG["audio"]["w3_text"]).read_text(encoding="utf-8").strip() if _p(CFG["audio"]["w3_text"]).exists() else ""
        if ref:
            o(f"  參考：{ref!r}")

    def menu(self) -> None:
        o = self.out
        o(f"\n  {self._settings()}")
        o("  1 靜音   2 白噪音×3   3 W3 那句+長靜音（兩個視窗）   4 自己錄的非語音   w CTC 對照（全部檔案）")
        o("  z encoder 歸零   Z encoder 換平均向量   f 切 fallback   c 切 condition_on_prev   l 切語言 en/zh   s W3 那句對照   h 選單   q 離開")

    def dispatch(self, k: str) -> bool:
        if k == "1": self.panel_clip("silence")
        elif k == "2": self.panel_noise()
        elif k == "3": self.panel_long()
        elif k == "4": self.panel_clip("own")
        elif k == "w": self.panel_ctc()
        elif k == "z": self.panel_zeroed("zeros")
        elif k == "Z": self.panel_zeroed("mean")
        elif k == "f": self.fallback = not self.fallback; self.out(f"  fallback {'開' if self.fallback else '關'}")
        elif k == "c": self.cond_prev = not self.cond_prev; self.out(f"  condition_on_prev {'開' if self.cond_prev else '關'}")
        elif k == "l":
            self.language = CFG["whisper"]["language_alt"] if self.language == CFG["whisper"]["language"] else CFG["whisper"]["language"]
            self.out(f"  語言 {self.language}")
        elif k == "s": self.panel_w3()
        elif k == "h": self.menu()
        elif k == "q": return False
        return True

    def run_all(self) -> None:
        """彩排順序 = 課堂順序：溫度 0 的全部 → CTC → 歸零 → fallback 開再跑一次 → 語言換 zh 的靜音與噪音。"""
        self.fallback, self.cond_prev, self.language = False, True, CFG["whisper"]["language"]
        self.panel_w3()
        self.panel_clip("silence"); self.panel_noise(); self.panel_long()
        self.cond_prev = False; self.panel_long(); self.cond_prev = True
        self.panel_clip("own")
        self.panel_ctc()
        self.panel_zeroed("zeros"); self.panel_zeroed("mean")
        self.fallback = True
        self.panel_clip("silence"); self.panel_noise(); self.panel_long(); self.panel_clip("own")
        self.fallback = False
        self.language = CFG["whisper"]["language_alt"]
        self.panel_clip("silence"); self.panel_clip("noise_0")
        self.language = CFG["whisper"]["language"]

    def versions(self) -> dict:
        v = C.versions()
        for pkg in ("torch", "transformers", "huggingface_hub", "tokenizers"):
            try:
                from importlib import metadata
                v[pkg] = metadata.version(pkg)
            except Exception:
                v[pkg] = None
        v["models"] = read_lock()
        return v

    def save(self, folder: Path) -> None:
        folder.mkdir(parents=True, exist_ok=True)
        log = {"time": C.now(), "tag": self.tag, "backend": self.wh.name, "device": {"whisper": str(getattr(self.wh, "device", "")), "ctc": str(getattr(self.ctc, "device", ""))},
               "versions": self.versions(), "config": CFG, "clips": {k: len(v) / FS for k, v in self.clips.items()},
               "nospeech_token": getattr(self.wh, "nospeech_token", None), "results": self.res}
        (folder / "rehearsal.json").write_text(json.dumps(log, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
        (folder / "rehearsal.txt").write_text(self.out.text(), encoding="utf-8")
        C.note(f"  存在 {folder}/（rehearsal.json、rehearsal.txt）")


# ── 建構 ─────────────────────────────────────────────────────────────────────
def build(cmd: str, out: Out) -> Demo:
    if cmd == "replay":
        src = HERE / "runs" / "rehearsal" / "rehearsal.json"
        if not src.exists():
            sys.exit("沒有 runs/rehearsal/rehearsal.json，先跑 ./present.sh rehearse")
        log = json.loads(src.read_text(encoding="utf-8"))
        out.banner(f"REPLAY：{log['time']} 的彩排（backend {log['backend']}）", C.AMBER)
        return Demo(out, {}, FakeWhisper(), FakeCTC(), stored=log["results"], tag="replay")
    clips = load_clips()
    if cmd == "fake":
        out.banner("FAKE：罐頭輸出，只為了走流程，不是模型的輸出", C.RED)
        return Demo(out, clips, FakeWhisper(), FakeCTC(), tag="fake")
    t0 = time.perf_counter()
    wh = Whisper()
    ctc = CTC()
    C.note(f"  Whisper small {wh.n_params / 1e6:.0f} M（{wh.device}）、wav2vec2-base-960h {ctc.n_params / 1e6:.0f} M（{ctc.device}），"
           f"載入 {time.perf_counter() - t0:.1f} s；<|nospeech|> = {wh.nospeech}（{wh.nospeech_token if wh.nospeech is not None else '找不到！'}）")
    return Demo(out, clips, wh, ctc, tag="rehearsal" if cmd == "rehearse" else "")


# ── check ───────────────────────────────────────────────────────────────────
def check() -> None:
    C.banner("課前體檢", C.AMBER)
    res: dict = {"time": C.now()}
    d = Demo(Out(), {}, FakeWhisper(), FakeCTC())
    v = d.versions()
    for k in ("python", "platform", "numpy", "scipy", "torch", "transformers", "huggingface_hub", "models"):
        print(f"  {k:<16s} {v.get(k)}")
    res["versions"] = v
    try:
        import torch
    except ImportError:
        print("  沒有 torch：先在 w03_d3_blank_heatmap 跑 ./setup.sh"); return
    mps = bool(torch.backends.mps.is_available())
    print(f"  {'mps':<16s} available={mps}")
    res["mps_available"] = mps
    for repo in (CFG["whisper"]["repo"], CFG["ctc"]["repo"]):
        try:
            print(f"  {'model':<16s} {repo} → {model_dir(repo)}")
        except Exception as e:
            print(f"  {'model':<16s} {repo} 不在快取：{e}\n  先跑 ./present.sh fetch"); return
    clips = load_clips()
    x_sil, x_w3 = clips["silence"], clips["w3"]
    res["clips_s"] = {k: len(x) / FS for k, x in clips.items()}

    # 1. CPU：nospeech id、靜音的轉錄、log 機率、no_speech_prob
    wc = Whisper("cpu")
    res["nospeech"] = {"id": wc.nospeech, "token": getattr(wc, "nospeech_token", None)}
    print(f"  <|nospeech|> id {wc.nospeech}（{getattr(wc, 'nospeech_token', None)}）；prefix {wc.prefix()}，eot {wc.eot}")
    t0 = time.perf_counter(); ids_c = wc.generate(x_sil)[0]; t_c = time.perf_counter() - t0
    lp_c = wc.teacher_logprobs(x_sil, ids_c)
    ns_c = wc.no_speech_prob(x_sil)
    print(f"  cpu 靜音：{wc.text(ids_c)!r}  {len(ids_c)} token，avg {avg_logprob(lp_c):.3f}，no_speech {ns_c:.3f}，generate {t_c:.2f} s")
    res["cpu_silence"] = {"text": wc.text(ids_c), "ids": ids_c, "avg_logprob": avg_logprob(lp_c), "no_speech_prob": ns_c, "generate_s": t_c}
    ids_w3 = wc.generate(x_w3)[0]
    print(f"  cpu W3 那句：{wc.text(ids_w3)!r}")
    res["cpu_w3"] = {"text": wc.text(ids_w3)}
    # 2. hook：encoder 範數歸零
    n_before = wc.encoder_norm(x_sil)
    with wc.encoder_replaced("zeros"):
        n_after = wc.encoder_norm(x_sil)
        beams = wc.generate(x_sil, 0.0, num_beams=int(CFG["zeroed"]["num_beams"]), num_return=2)
    print(f"  encoder 範數：hook 前 {n_before:.1f}、hook 後 {n_after:.1f}；歸零之下 beam 前兩條：{[wc.text(b) for b in beams]}")
    res["zeroed_hook"] = {"norm_before": n_before, "norm_after": n_after, "beams": [wc.text(b) for b in beams]}
    # 3. prompt：帶 prompt 的 generate 與 teacher-forced 都跑得過
    try:
        ids_p = wc.generate(x_sil, prompt="Thank you for watching.")[0]
        lp_p = wc.teacher_logprobs(x_sil, ids_p, prompt="Thank you for watching.")
        print(f"  帶 prompt 的靜音：{wc.text(ids_p)!r}，avg {avg_logprob(lp_p):.3f}")
        res["prompt_path"] = {"text": wc.text(ids_p), "avg_logprob": avg_logprob(lp_p)}
    except Exception as e:
        res["prompt_path"] = {"error": f"{type(e).__name__}: {e}"}
        print(f"  帶 prompt 失敗：{type(e).__name__}: {e}")
    # 4. 抽樣可重現
    s1 = wc.generate(x_sil, 0.8, seed=1)[0]; s2 = wc.generate(x_sil, 0.8, seed=1)[0]
    res["sampling_reproducible_cpu"] = s1 == s2
    print(f"  溫度 0.8 同 seed 兩次相同：{s1 == s2}  {wc.text(s1)!r}")
    # 5. MPS 對 CPU
    if mps:
        try:
            wm = Whisper("mps")
            t0 = time.perf_counter(); ids_m = wm.generate(x_sil)[0]; t_m = time.perf_counter() - t0
            lp_m = wm.teacher_logprobs(x_sil, ids_m)
            res["mps_silence"] = {"text": wm.text(ids_m), "same_as_cpu": ids_m == ids_c, "avg_logprob": avg_logprob(lp_m),
                                  "maxabs_logprob_vs_cpu": float(np.max(np.abs(np.array(lp_m) - np.array(lp_c)))) if ids_m == ids_c else None,
                                  "no_speech_prob": wm.no_speech_prob(x_sil), "generate_s": t_m}
            print(f"  mps 靜音：{wm.text(ids_m)!r}  與 cpu 相同={ids_m == ids_c}，log 機率最大差 {res['mps_silence']['maxabs_logprob_vs_cpu']}，generate {t_m:.2f} s")
            del wm
        except Exception as e:
            res["mps_silence"] = {"error": f"{type(e).__name__}: {e}"}
            print(f"  mps 失敗：{type(e).__name__}: {e}（已知：5.17.0 的 generate 在 MPS 上掛，所以 Whisper 放 CPU）")
    # 6. CTC
    cc = CTC("cpu")
    g = cc.greedy(x_sil)
    print(f"  wav2vec2 cpu 靜音：T {g['T']}，blank {g['blank_fraction']:.1%}，{g['text']!r}；W3 那句：{cc.greedy(x_w3)['text']!r}")
    res["ctc_cpu"] = {"silence": g, "w3": cc.greedy(x_w3)}
    if mps:
        try:
            cm = CTC("mps"); gm = cm.greedy(x_sil)
            res["ctc_mps"] = {"silence": gm, "same_text_as_cpu": gm["text"] == g["text"]}
            print(f"  wav2vec2 mps 靜音：blank {gm['blank_fraction']:.1%}，與 cpu 同字串={gm['text'] == g['text']}")
        except Exception as e:
            res["ctc_mps"] = {"error": f"{type(e).__name__}: {e}"}
    # 7. §6 第 8 項的三條路裝了沒（compare 才真的跑）
    avail = {}
    for mod in ("pywhispercpp", "mlx_whisper"):
        try:
            from importlib import metadata
            __import__(mod); avail[mod] = metadata.version(mod.replace("_", "-"))
        except Exception as e:
            avail[mod] = f"無（{type(e).__name__}）"
    res["other_backends"] = avail
    print(f"  其他後端：{avail}（./present.sh compare 跑三方對比）")
    (HERE / "runs").mkdir(exist_ok=True)
    (HERE / "runs" / "check.json").write_text(json.dumps(res, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    C.note("  結果存在 runs/check.json（進版控）")


# ── compare（§6 第 8 項）────────────────────────────────────────────────────
def compare() -> None:
    C.banner("§6 第 8 項：transformers／pywhispercpp／mlx-whisper 三方對比（small，W3 那句與 10 s 噪音）", C.AMBER)
    clips = load_clips()
    tests = {"w3": clips["w3"], "noise_0": clips["noise_0"]}
    res: dict = {"time": C.now(), "backends": {}}

    def run(name, fn):
        r = {}
        try:
            for cn, x in tests.items():
                ts = []
                for _ in range(2):
                    t0 = time.perf_counter(); text = fn(x); ts.append(time.perf_counter() - t0)
                r[cn] = {"text": text, "first_s": ts[0], "warm_s": ts[1]}
                print(f"  {name:<14s} {cn:<8s} 第一次 {ts[0]:.2f} s、第二次 {ts[1]:.2f} s  {text!r}")
        except Exception as e:
            r["error"] = f"{type(e).__name__}: {e}"
            print(f"  {name:<14s} 跳過：{type(e).__name__}: {e}")
        res["backends"][name] = r

    wh = Whisper()
    run(f"transformers/{wh.device}", lambda x: wh.text(wh.generate(x)[0]))

    def wcpp(x):
        from pywhispercpp.model import Model
        if not hasattr(wcpp, "m"):
            wcpp.m = Model("small", n_threads=4, print_progress=False)
        return "".join(s.text for s in wcpp.m.transcribe(x.astype(np.float32), language="en")).strip()
    run("pywhispercpp", wcpp)

    def mlxw(x):
        import mlx_whisper
        return mlx_whisper.transcribe(x.astype(np.float32), path_or_hf_repo="mlx-community/whisper-small-mlx", language="en")["text"].strip()
    run("mlx_whisper", mlxw)
    (HERE / "runs").mkdir(exist_ok=True)
    (HERE / "runs" / "compare.json").write_text(json.dumps(res, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    C.note("  結果存在 runs/compare.json；三條路的溫度、beam、前處理不完全一樣，時間只看量級")


# ── fetch／make／record／export ─────────────────────────────────────────────
def fetch(mlx: bool = False) -> None:
    from huggingface_hub import HfApi, snapshot_download
    C.banner("下載模型", C.AMBER)
    lock = read_lock()
    lines = []
    for repo in (CFG["whisper"]["repo"], CFG["ctc"]["repo"]):
        sha = lock.get(repo) or HfApi().model_info(repo).sha
        path = snapshot_download(repo, revision=sha, allow_patterns=["*.json", "*.txt", "*.safetensors"])
        lines.append(f"{repo} {sha}")
        print(f"  {repo} @ {sha[:12]} → {path}")
    MODEL_LOCK.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("  commit 寫進 model.lock（進版控；與 W4 Demo 3、W3 Demo 3 的同一個 commit）")
    if mlx:
        path = snapshot_download("mlx-community/whisper-small-mlx")
        print(f"  mlx-community/whisper-small-mlx → {path}（只給 compare 用）")
    print("\n下一步：./present.sh make → ./present.sh check")


def record() -> None:
    a = CFG["audio"]
    C.banner(f"錄 {a['own_s']:g} 秒自己的非語音（音樂、咳嗽、敲桌——不要說話）", C.AMBER)
    x = C.record(float(a["own_s"]), FS)
    AUDIO.mkdir(exist_ok=True)
    C.save_wav(AUDIO / "own.wav", x, FS)
    print(f"  → audio/own.wav（RMS {rms(x):.4f}；不進版控、不上站）。想聽：afplay audio/own.wav")


def export() -> None:
    src = HERE / "runs" / "rehearsal" / "rehearsal.json"
    if not src.exists():
        sys.exit("沒有 runs/rehearsal/rehearsal.json，先跑 ./present.sh rehearse")
    log = json.loads(src.read_text(encoding="utf-8"))
    if log.get("backend") != "whisper":
        sys.exit(f"rehearsal.json 的 backend 是 {log.get('backend')}，不是真模型的紀錄，不匯出")
    R = log["results"]

    def short(r: dict | None) -> dict | None:
        if not r:
            return None
        f = r["final"]
        return {"text": f["text"], "n_words": f["n_words"], "n_tokens": f["n_tokens"], "avg_logprob": f["avg_logprob"], "max_logprob": f["max_logprob"],
                "n_confident": f["n_confident"], "no_speech_prob": f["no_speech_prob"], "compression_ratio": f["compression_ratio"],
                "temperature": f["temperature"], "n_attempts": len(r["attempts"]),
                "attempts": [{"temperature": a["temperature"], "text": a["text"], "avg_logprob": a["avg_logprob"], "compression_ratio": a["compression_ratio"],
                              "no_speech_prob": a["no_speech_prob"], "verdict": a["verdict"]} for a in r["attempts"]]}
    lang = CFG["whisper"]["language"]
    data = json.loads(METRICS.read_text(encoding="utf-8")) if METRICS.exists() else {}
    data["demo1"] = {
        "source": "demos/w07_d1_hallucination/runs/rehearsal/rehearsal.json", "time": log["time"], "backend": log["backend"], "device": log["device"],
        "models": log["versions"].get("models"), "transformers": log["versions"].get("transformers"), "torch": log["versions"].get("torch"),
        "clips_s": log["clips"], "noise_snr_db": CFG["audio"]["noise_snr_db"], "tail_silence_s": CFG["audio"]["tail_silence_s"],
        "thresholds": {k: CFG["whisper"][k] for k in ("compression_ratio_threshold", "logprob_threshold", "no_speech_threshold", "prompt_reset_temperature", "temperatures")},
        "whisper_t0": {n: short(R.get(f"whisper:{n}:{lang}:nofb")) for n in CLIP_ORDER + ["w3"] if n != "long"},
        "whisper_fallback": {n: short(R.get(f"whisper:{n}:{lang}:fb")) for n in CLIP_ORDER if n != "long"},
        "whisper_zh": {n: short(R.get(f"whisper:{n}:{CFG['whisper']['language_alt']}:nofb")) for n in ("silence", "noise_0")},
        "long": {k: ({"text": R[k]["text"], "windows": [{"start_s": w["start_s"], "end_s": w["end_s"], "prompt_used": w["prompt_used"], **short(w)} for w in R[k]["windows"]]}
                     if k in R else None) for k in (f"long:{lang}:cond:nofb", f"long:{lang}:nocond:nofb", f"long:{lang}:cond:fb")},
        "ctc": {n: R.get(f"ctc:{n}") for n in CLIP_ORDER + ["w3"]},
        "zeroed": {m: R.get(f"zeroed:{m}:{lang}") for m in ("zeros", "mean")},
    }
    METRICS.parent.mkdir(parents=True, exist_ok=True)
    METRICS.write_text(json.dumps(data, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    s = data["demo1"]["whisper_t0"].get("silence") or {}
    print(f"→ {METRICS}（demo1）：靜音 {s.get('text')!r}（{s.get('n_words')} 字、avg {s.get('avg_logprob')}、no_speech {s.get('no_speech_prob')}）")


# ── selftest ───────────────────────────────────────────────────────────────
def selftest() -> None:
    C.banner("selftest（不需要模型）", C.AMBER)
    fails = []

    def chk(name: str, cond: bool, extra: str = "") -> None:
        print(f"  {'ok ' if cond else 'FAIL'} {name}{('  ' + extra) if extra else ''}")
        if not cond:
            fails.append(name)
    w = CFG["whisper"]
    chk("壓縮比：空字串 0", compression_ratio("") == 0.0)
    rep = "Thank you. " * 40
    chk("壓縮比：重複 40 次 > 2.4", compression_ratio(rep) > w["compression_ratio_threshold"], f"{compression_ratio(rep):.2f}")
    one = "Mr. Quilter is the apostle of the middle classes, and we are glad to welcome his gospel."
    chk("壓縮比：一句正常的話 < 2.4", compression_ratio(one) < w["compression_ratio_threshold"], f"{compression_ratio(one):.2f}")
    v = fallback_verdict(-0.3, 1.1, 0.2, w)
    chk("判定：合格、不是靜音", not v["needs_fallback"] and not v["is_silence"])
    v = fallback_verdict(-1.5, 1.1, 0.9, w)
    chk("判定：log 機率不合格 + no_speech 高 → 靜音", v["needs_fallback"] and v["is_silence"], "；".join(v["reasons"]))
    v = fallback_verdict(-0.2, 1.1, 0.9, w)
    chk("判定：no_speech 高但 log 機率合格 → 不當靜音（原文：單看 nospeech 不夠）", not v["is_silence"])
    v = fallback_verdict(-0.2, 3.0, 0.1, w)
    chk("判定：壓縮比不合格", v["needs_fallback"] and "compression" in v["reasons"][0])
    chk("平均 log 機率：含 eot 的除法", abs(avg_logprob([-1.0, -2.0, -3.0]) + 2.0) < 1e-12)
    n = noise_for_snr(0.1, 20.0, 16000, 1)
    chk("噪音 SNR 20 dB → RMS 是語音的 1/10", abs(rms(n) - 0.01) < 1e-9, f"{rms(n):.5f}")
    chk("噪音：同 seed 可重現", np.array_equal(noise_for_snr(0.1, 0.0, 100, 7), noise_for_snr(0.1, 0.0, 100, 7)))
    chk("CTC 摺疊：先摺疊再刪 blank", ctc_collapse([0, 1, 1, 0, 1, 2, 2, 0], 0) == [1, 1, 2])
    chk("CTC 摺疊：全 blank → 空", ctc_collapse([0, 0, 0], 0) == [])
    chk("strip_prefix：有 prompt、有 sot", strip_prefix([50361, 7, 8, 50258, 50259, 50359, 50363, 11, 12, 50257], 50258, 50257, 4) == [11, 12])
    chk("strip_prefix：前綴已被剝掉", strip_prefix([50259, 50359, 50363, 11, 12, 50257, 50257], 50258, 50257, 4) == [11, 12])
    chk("strip_prefix：沒有 eot", strip_prefix([50258, 50259, 50359, 50363, 11, 12], 50258, 50257, 4) == [11, 12])
    # fake 整條流程（沒有 audio/ 也要能跑：用合成 clip）
    out = Out()
    clips = {"silence": np.zeros(FS * 2), "noise_0": white_noise(FS * 2, 0.05, 1), "noise_20": white_noise(FS * 2, 0.005, 2), "noise_40": white_noise(FS * 2, 0.0005, 3),
             "long": np.concatenate([white_noise(FS * 2, 0.05, 4), np.zeros(int(FS * CFG["whisper"]["window_s"]))]), "w3": white_noise(FS, 0.05, 5)}
    d = Demo(out, clips, FakeWhisper(), FakeCTC(), tag="selftest")
    try:
        d.run_all()
        for k in "123w4zZfcls":
            d.dispatch(k)
        chk("fake 整條流程跑完", True, f"{len(d.res)} 個結果")
        chk("長段落：兩個視窗", len(d.res["long:en:cond:nofb"]["windows"]) == 2)
        chk("長段落：第二個視窗的 prompt 槽是第一個視窗的文字", d.res["long:en:cond:nofb"]["windows"][1]["prompt_used"] == d.res["long:en:cond:nofb"]["windows"][0]["final"]["text"])
        chk("長段落：不帶前文時 prompt 槽空", d.res["long:en:nocond:nofb"]["windows"][1]["prompt_used"] is None)
        chk("fallback 開：靜音有多次嘗試或第一次就合格", len(d.res["whisper:silence:en:fb"]["attempts"]) >= 1)
        chk("replay 從 results 重建", Demo(Out(), {}, FakeWhisper(), FakeCTC(), stored=d.res).dispatch("1"))
        tmp = HERE / "runs" / "_selftest"
        d.save(tmp)
        j = json.loads((tmp / "rehearsal.json").read_text(encoding="utf-8"))
        chk("rehearsal.json 可讀", j["backend"] == "fake" and "results" in j)
        import shutil
        shutil.rmtree(tmp)
    except Exception as e:
        import traceback
        traceback.print_exc()
        chk("fake 整條流程", False, f"{type(e).__name__}: {e}")
    print(f"\n  {'全部通過' if not fails else '失敗：' + ', '.join(fails)}")
    if fails:
        sys.exit(1)


# ── main ───────────────────────────────────────────────────────────────────
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["show", "rehearse", "replay", "check", "compare", "fetch", "make", "record", "fake", "selftest", "export"])
    ap.add_argument("--mlx", action="store_true", help="fetch 時順便抓 mlx-community/whisper-small-mlx（給 compare）")
    args = ap.parse_args()
    if args.cmd == "check": check(); return
    if args.cmd == "compare": compare(); return
    if args.cmd == "fetch": fetch(args.mlx); return
    if args.cmd == "make":
        info = make_audio()
        print("audio/：" + "、".join(f"{k} {v:.3f}" if isinstance(v, float) else f"{k} {v}" for k, v in info.items()) + "（秒；speech_rms 是 W3 那句的 RMS）")
        return
    if args.cmd == "record": record(); return
    if args.cmd == "selftest": selftest(); return
    if args.cmd == "export": export(); return

    out = Out()
    d = build(args.cmd, out)
    if args.cmd == "rehearse":
        d.run_all()
        d.save(HERE / "runs" / "rehearsal")
        print("\n下一步：./present.sh 自己按一遍 1 2 3 c 3 4 w z f 1 2；把 rehearsal.json 的數字填進 p27 的表與 .gen 區塊、p24／p28 圖裡的例句；"
              "./present.sh export 寫進 w07-metrics.json。")
        return
    out.banner("W7 Demo 1：幻覺——證據歸零時先驗長什麼樣" + (f"   [{d.tag}]" if d.tag else ""))
    d.menu()
    while True:
        k = C.getkey()
        if not d.dispatch(k):
            break
    if args.cmd == "show":
        d.save(HERE / "runs" / "live" / C.now().replace(":", "-"))


if __name__ == "__main__":
    main()

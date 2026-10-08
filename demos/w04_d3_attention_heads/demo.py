# -*- coding: utf-8 -*-
# 此檔由 bin/publish_demos.py 從 private repo 的 <course>/demos/ 複製而來，請勿在公開 repo 直接編輯。
"""W4 demo 3：Whisper 的 cross-attention heads——attention ≠ alignment（本週的核心破解）。

  show       課堂：載入 Whisper small（eager attention）、轉錄那一句、teacher-forced 拿 144 個 head 的 cross-attention，開一個空視窗；
             鍵在視窗裡按（下面的鍵）。離開時存 runs/live/<時間>/（不進版控）
  rehearse   課前：不互動，照課堂順序全部跑一遍，存 runs/rehearsal/（進版控 = 現場的退路）
  replay     現場模型出狀況時：從 runs/rehearsal/ 讀回權重與文字，同一個視窗、同樣的鍵，不需要 torch
  check      課前體檢：套件版本、模型在不在快取、eager 回傳權重／sdpa 不回傳、base85 的 _ALIGNMENT_HEADS 對 HF 的 alignment_heads、
             自己的 DTW 對 transformers 的 return_token_timestamps、MPS 對 CPU → runs/check.json（進版控）
  fetch      課前一週：下載模型（鎖 commit 進 model.lock；要網路）。音訊重用 W3 Demo 3 的，不用再抓
  fake       沒有模型也能演練流程：合成的 144 個 head（畫面標明 FAKE，不是給學生看的）
  selftest   不需要模型：base85 解碼、DTW、中值濾波、分類判準、併字、字錯數
  export     runs/rehearsal/rehearsal.json 的摘要 → slides/assets/w04/w04-metrics.json（鍵 demo3；量測數字進版控的出處）
  recompute  改了判準／後處理之後：從 runs/rehearsal/ 已存的權重重新分類、重算 DTW、重畫圖、重寫 rehearsal.json——不碰模型

畫面上的鍵（順序照投影片 p45 講稿的【操作】）：
  1  播那句音訊；終端機印參考文字、轉錄、字錯數（原樣與正規化後）
  2  目前這一層 12 個 head 的熱圖（U 個文字 token × 音訊範圍內的 frame），每格標 band／flat／stare／other 與 padding 比例
  3  下一層（12 層循環）           0  全部 144 個 head 的縮圖，一格一個 head，邊框顏色 = 類別；標題是四類的數目（填 p45 的表）
  4  把 OpenAI _ALIGNMENT_HEADS（= HF alignment_heads）的 10 個框出來（在 2／0 的畫面上）
  5  那 10 個 head 照 timing.py 的做法（標準化 → 中值濾波 → 平均 → DTW）得到的矩陣與路徑；字的時間戳印在終端機
  a  在 5 的畫面上疊 W3 Demo 3 的 forced alignment（每個字的尖峰跨度）——座位表 vs. 權重（p46 那張圖的真實版）
  h  重印選單        q  離開

數字的定義：U = 轉錄出來的文字 token 數（不含 <|startoftranscript|><|en|><|transcribe|><|notimestamps|> 前綴與 <|endoftext|>）；
frame = encoder 一格 20 ms，30 s padded 輸入共 1500 格，「音訊範圍」= 音訊長度 ÷ 20 ms（與 transformers 的 num_frames // 2 同一個數）。
head 的四類判準在 demo_config.toml [classify]，README 有說明；課堂上要念出來。
"""
from __future__ import annotations

import argparse
import base64
import gzip
import io
import json
import math
import re
import shutil
import sys
import threading
import time
import tomllib
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import common as C  # noqa: E402

CFG = tomllib.loads((HERE / "demo_config.toml").read_text(encoding="utf-8"))
FS = CFG["audio"]["fs"]
FRAME_S = CFG["model"]["frame_ms"] / 1000.0
N_ENC = int(CFG["model"]["encoder_frames"])
DIG = CFG["display"]["digits"]
MODEL_LOCK = HERE / "model.lock"
METRICS = (HERE.parent.parent / "slides" / "assets" / "w04" / "w04-metrics.json").resolve()
CLASS_COLOR = {"band": "#2a9d8f", "flat": "#9a9a9a", "stare": "#e76f51", "other": "#457b9d"}
ALIGN_COLOR = "#d1345b"


def _p(rel: str) -> Path:
    return (HERE / rel).resolve()


# ── OpenAI 的 _ALIGNMENT_HEADS：base85 + gzip 的布林陣列 ─────────────────
def decode_alignment_heads(b85: str, n_layer: int, n_head: int) -> list[list[int]]:
    """照 openai/whisper `whisper/__init__.py` 的解法：b85decode → gzip.decompress → bool 陣列 (n_layer, n_head)。回 [[layer, head], …] 排序後。"""
    arr = np.frombuffer(gzip.decompress(base64.b85decode(b85.encode())), dtype=bool).copy().reshape(n_layer, n_head)
    return sorted([int(l), int(h)] for l, h in zip(*np.nonzero(arr)))


# ── 字錯數（詞層級的 Levenshtein）────────────────────────────────────────
def word_errors(ref: str, hyp: str) -> dict:
    r, h = ref.split(), hyp.split()
    n, m = len(r), len(h)
    d = np.zeros((n + 1, m + 1), dtype=np.int64)
    d[:, 0] = np.arange(n + 1)
    d[0, :] = np.arange(m + 1)
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            d[i, j] = min(d[i - 1, j] + 1, d[i, j - 1] + 1, d[i - 1, j - 1] + (r[i - 1] != h[j - 1]))
    # 回溯算 S／D／I
    i, j, S, D, I = n, m, 0, 0, 0
    while i > 0 or j > 0:
        if i > 0 and j > 0 and d[i, j] == d[i - 1, j - 1] + (r[i - 1] != h[j - 1]):
            S += int(r[i - 1] != h[j - 1]); i -= 1; j -= 1
        elif i > 0 and d[i, j] == d[i - 1, j] + 1:
            D += 1; i -= 1
        else:
            I += 1; j -= 1
    return {"errors": int(d[n, m]), "S": S, "D": D, "I": I, "N_ref": n, "N_hyp": m}


def simple_normalize(text: str) -> str:
    """沒有 tokenizer 時的退路（fake／replay）：小寫、去標點、幾個 Whisper 常見的縮寫。真的跑時用 tokenizer.normalize（EnglishTextNormalizer）。"""
    t = text.lower()
    t = re.sub(r"[^a-z0-9' ]+", " ", t)
    t = re.sub(r"\bmr\b", "mister", t)
    t = re.sub(r"\bmrs\b", "missus", t)
    return " ".join(t.split())


# ── token → 字 ──────────────────────────────────────────────────────────
def group_words(pieces: list[str]) -> list[list[int]]:
    """Whisper 的 byte-level BPE：一個 token 以空白開頭就是新字的開頭。回每個字包含哪些 token 的 index（只吃文字 token）。
    第一個 token 沒有空白也算一個字的開頭；只含標點的 token 併進前一個字。"""
    words: list[list[int]] = []
    for i, p in enumerate(pieces):
        starts = p.startswith(" ") or not words
        if starts and words and not re.search(r"\w", p):
            starts = False
        if starts:
            words.append([i])
        else:
            words[-1].append(i)
    return words


# ── timing.py 的三個零件：中值濾波、DTW、標準化 ─────────────────────────────
def median_filter(x: np.ndarray, width: int) -> np.ndarray:
    """沿最後一軸、寬 width（奇數）、兩端 reflect——與 transformers 的 _median_filter 同一個定義。"""
    if width <= 0 or width % 2 != 1:
        raise ValueError("width 要是奇數")
    pad = width // 2
    if x.shape[-1] <= pad:
        return x
    xp = np.pad(x, [(0, 0)] * (x.ndim - 1) + [(pad, pad)], mode="reflect")
    win = np.lib.stride_tricks.sliding_window_view(xp, width, axis=-1)
    return np.sort(win, axis=-1)[..., pad]


def dtw(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """openai/whisper timing.py 的 dtw（transformers 的 _dynamic_time_warping 逐行相同）：
    三種步（對角、往下、往右），cost 累加 matrix[i-1, j-1]；回 (text_indices, time_indices)。matrix 是 cost（傳 −權重）。"""
    N, M = matrix.shape
    cost = np.full((N + 1, M + 1), np.inf, dtype=np.float64)
    trace = -np.ones((N + 1, M + 1), dtype=np.int8)
    cost[0, 0] = 0.0
    for j in range(1, M + 1):
        for i in range(1, N + 1):
            c0, c1, c2 = cost[i - 1, j - 1], cost[i - 1, j], cost[i, j - 1]
            if c0 < c1 and c0 < c2:
                c, t = c0, 0
            elif c1 < c0 and c1 < c2:
                c, t = c1, 1
            else:
                c, t = c2, 2
            cost[i, j] = matrix[i - 1, j - 1] + c
            trace[i, j] = t
    i, j = N, M
    trace[0, :] = 2
    trace[:, 0] = 1
    ti, tj = [], []
    while i > 0 or j > 0:
        ti.append(i - 1)
        tj.append(j - 1)
        if trace[i, j] == 0:
            i -= 1; j -= 1
        elif trace[i, j] == 1:
            i -= 1
        else:
            j -= 1
    return np.array(ti[::-1]), np.array(tj[::-1])


def timing_matrix(weights: np.ndarray, n_audio: int, width: int) -> np.ndarray:
    """weights (H, L_tok, ≥n_audio) → 只留 n_audio 格 → 每格在 token 軸標準化 → 沿時間中值濾波 → head 平均。回 (L_tok, n_audio)。"""
    w = weights[..., :n_audio].astype(np.float64)
    mean = w.mean(axis=-2, keepdims=True)
    std = w.std(axis=-2, keepdims=True)           # unbiased=False，與 torch.std(unbiased=False) 同
    w = (w - mean) / np.where(std == 0, 1.0, std)
    w = median_filter(w, width)
    return w.mean(axis=0)


def token_times(matrix_text: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """對 −matrix 做 DTW；每個 token 的時間 = 路徑上第一次到那一列的 frame × 20 ms。回 (times_s, text_idx, time_idx)。"""
    ti, tj = dtw(-matrix_text)
    jumps = np.pad(np.diff(ti), (1, 0), constant_values=1).astype(bool)
    return tj[jumps] * FRAME_S, ti, tj


# ── head 的四類判準 ────────────────────────────────────────────────────
def spearman(a: np.ndarray, b: np.ndarray) -> float:
    from scipy.stats import spearmanr
    if len(a) < 3 or len(np.unique(b)) < 2:
        return 0.0
    r = spearmanr(a, b).statistic
    return 0.0 if (r is None or np.isnan(r)) else float(r)


def classify_head(w_text: np.ndarray, pad_mass: np.ndarray) -> dict:
    """w_text (U, n_audio)：文字 token 的列、音訊範圍內的 frame；pad_mass (U,)：每列落在 padding 的權重比例。"""
    c = CFG["classify"]
    U, n = w_text.shape
    tot = w_text.sum(axis=1)
    tot = np.where(tot <= 0, 1e-12, tot)
    tstar = w_text.argmax(axis=1)
    k = int(c["window_frames"])
    conc = np.array([w_text[u, max(0, tstar[u] - k): tstar[u] + k + 1].sum() / tot[u] for u in range(U)])
    rho = spearman(np.arange(U), tstar)
    med_conc = float(np.median(conc))
    med_pad = float(np.median(pad_mass))
    distinct = int(len(np.unique(tstar)))
    # 形狀只看音訊範圍內的權重；落在 padding 的比例另外記成 pad_heavy，不混進類別
    # （2026-10-06 Mac 彩排後改：第一版把 pad ≥ 0.5 併進 stare，結果三個 OpenAI alignment head 在音訊範圍內是乾淨的帶、卻被標成 stare）
    if rho >= c["rho_min"] and med_conc >= c["conc_min"]:
        cls = "band"
    elif med_conc < c["conc_flat"]:
        cls = "flat"
    elif distinct <= max(int(c["stare_distinct"]), c["stare_frac"] * U):
        cls = "stare"
    else:
        cls = "other"
    return {"class": cls, "rho": round(rho, 3), "conc": round(med_conc, 3), "pad": round(med_pad, 3),
            "pad_heavy": bool(med_pad >= c["pad_max"]), "distinct": distinct, "tstar": tstar.tolist()}


# ── W3 Demo 3 的 forced alignment（座位表）─────────────────────────────────
def load_alignment(path: Path) -> list[dict] | None:
    if not path.exists():
        return None
    d = json.loads(path.read_text(encoding="utf-8"))
    al = d["sections"].get("align")
    if not al or "words" not in al:
        return None
    frame = float(d["config"]["model"]["frame_ms"]) / 1000.0
    return [{"word": w["word"], "start": float(w["t_first_spike"]), "end": round(float(w["t_last_spike"]) + frame, 3)} for w in al["words"]]


# ── 模型 ─────────────────────────────────────────────────────────────
def model_dir(offline: bool = True) -> str:
    from huggingface_hub import snapshot_download
    rev = MODEL_LOCK.read_text(encoding="utf-8").strip() if MODEL_LOCK.exists() else None
    return snapshot_download(CFG["model"]["repo"], revision=rev, local_files_only=offline,
                             allow_patterns=["*.json", "*.txt", "*.safetensors"])


class Real:
    """transformers 的 WhisperForConditionalGeneration，attention 一律 eager（sdpa 不回傳權重）。權重用 numpy 交出去。"""
    name = "whisper"

    def __init__(self, device: str, attn: str = "eager"):
        import torch
        from transformers import WhisperForConditionalGeneration, WhisperProcessor
        self.torch = torch
        path = model_dir()
        self.proc = WhisperProcessor.from_pretrained(path)
        self.model = WhisperForConditionalGeneration.from_pretrained(path, attn_implementation=attn).eval()
        if device == "auto":
            device = "mps" if torch.backends.mps.is_available() else "cpu"
        self.device = torch.device(device)
        self.model.to(self.device)
        self.attn = attn
        cfg, gen = self.model.config, self.model.generation_config
        self.n_layer, self.n_head = int(cfg.decoder_layers), int(cfg.decoder_attention_heads)
        self.n_params = sum(p.numel() for p in self.model.parameters())
        self.eot = int(gen.eos_token_id if gen.eos_token_id is not None else cfg.eos_token_id)
        lang, task = CFG["model"]["language"], CFG["model"]["task"]
        self.prompt = [int(gen.decoder_start_token_id), int(gen.lang_to_id[f"<|{lang}|>"]), int(gen.task_to_id[task]), int(gen.no_timestamps_token_id)]
        self.alignment_heads = [list(map(int, p)) for p in getattr(gen, "alignment_heads", [])]
        self.median_filter_width = int(getattr(cfg, "median_filter_width", CFG["timestamps"]["median_filter_width"]))

    def features(self, x: np.ndarray):
        return self.proc(x.astype(np.float32), sampling_rate=FS, return_tensors="pt", return_attention_mask=True)

    def transcribe(self, x: np.ndarray) -> list[int]:
        """greedy、強制 en／transcribe、不產時間戳 token。回文字 token 的 id（去前綴、到 <|endoftext|> 為止）。"""
        torch = self.torch
        inp = self.features(x)
        with torch.no_grad():
            seq = self.model.generate(inp.input_features.to(self.device), attention_mask=inp.attention_mask.to(self.device),
                                      language=CFG["model"]["language"], task=CFG["model"]["task"], return_timestamps=False,
                                      do_sample=False, num_beams=1, max_new_tokens=int(CFG["model"]["max_new_tokens"]))
        ids = [int(t) for t in seq[0].tolist()]
        while ids and ids[0] >= self.eot:          # 去掉前綴的特殊 token（<|startoftranscript|> 等 id 都 ≥ eot）
            ids = ids[1:]
        if self.eot in ids:
            ids = ids[: ids.index(self.eot)]
        return ids

    def cross_attention(self, x: np.ndarray, text_ids: list[int]) -> np.ndarray:
        """teacher-forced 一次前向：decoder 吃 prompt + text + eot，回 cross-attention (n_layer, n_head, L_tok, 1500) float32。
        eager 之下 outputs.cross_attentions 是 n_layer 個 (1, n_head, L_tok, 1500)；sdpa 之下會是 None（check 會證明）。"""
        torch = self.torch
        inp = self.features(x)
        dec = torch.tensor([self.prompt + list(text_ids) + [self.eot]], dtype=torch.long, device=self.device)
        with torch.no_grad():
            out = self.model(input_features=inp.input_features.to(self.device), decoder_input_ids=dec, output_attentions=True)
        if out.cross_attentions is None or out.cross_attentions[0] is None:
            raise RuntimeError(f"cross_attentions 是 None（attn_implementation={self.attn}）——要 eager")
        return torch.stack([a[0] for a in out.cross_attentions]).float().cpu().numpy()

    def hf_token_timestamps(self, x: np.ndarray) -> tuple[list[int], np.ndarray]:
        """transformers 自己的 return_token_timestamps（同一套 DTW，吃 generate 過程的 cross-attention）。check 用來對自己的 DTW。"""
        torch = self.torch
        inp = self.features(x)
        with torch.no_grad():
            out = self.model.generate(inp.input_features.to(self.device), attention_mask=inp.attention_mask.to(self.device),
                                      language=CFG["model"]["language"], task=CFG["model"]["task"], return_timestamps=False,
                                      do_sample=False, num_beams=1, max_new_tokens=int(CFG["model"]["max_new_tokens"]),
                                      return_token_timestamps=True)
        seq = [int(t) for t in out["sequences"][0].tolist()]
        ts = out["token_timestamps"][0].float().cpu().numpy()
        return seq, ts

    def pieces(self, ids: list[int]) -> list[str]:
        return [self.proc.tokenizer.decode([i]) for i in ids]

    def normalize(self, text: str) -> str:
        try:
            return self.proc.tokenizer.normalize(text)
        except Exception:
            return simple_normalize(text)


# ── fake：合成 144 個 head ──────────────────────────────────────────────
FAKE_WORDS = "MISTER QUILTER IS THE APOSTLE OF THE MIDDLE CLASSES AND WE ARE GLAD TO WELCOME HIS GOSPEL".split()


def fake_bundle(duration_s: float, align: list[dict] | None) -> dict:
    """合成：token = 參考句切成的 BPE 樣子（幾個字拆成兩片），每個 head 依 (layer, head) 決定一種樣子。alignment heads 做成 band。"""
    rng = np.random.default_rng(20261006)
    n_layer, n_head = 12, 12
    pieces: list[str] = []
    for w in FAKE_WORDS:
        wl = w.capitalize() if w != "MISTER" else "Mr."
        if len(wl) > 6:
            pieces += [" " + wl[:3], wl[3:]]
        else:
            pieces.append(" " + wl)
    pieces[0] = pieces[0].lstrip()
    pieces[-1] = pieces[-1] + "."
    U = len(pieces)
    n_audio = int(duration_s * FS) // (FS // 100) // 2     # 與 build() 的算法相同（mel 10 ms → encoder 20 ms）
    # 每個 token 的「真」中心：有 W3 的對齊就照字的跨度等分，否則線性
    if align and len(align) == len(FAKE_WORDS):
        centers = []
        for wi, toks in enumerate(group_words(pieces)):
            a = align[wi]
            for k, _ in enumerate(toks):
                centers.append((a["start"] + (k + 0.5) / len(toks) * (a["end"] - a["start"])) / FRAME_S)
        centers = np.array(centers)
    else:
        centers = np.linspace(0.08 * n_audio, 0.92 * n_audio, U)
    hf_heads = [tuple(p) for p in CFG["timestamps"]["hf_alignment_heads"]]
    kinds = ["flat", "stare", "other", "band", "flat", "stare", "pad", "flat", "other", "flat", "stare", "flat"]
    cross = np.zeros((n_layer, n_head, U + 5, N_ENC), dtype=np.float32)    # 4 個前綴列 + U + eot
    frames = np.arange(N_ENC)
    for l in range(n_layer):
        for h in range(n_head):
            kind = "band" if (l, h) in hf_heads else kinds[(l * 7 + h) % len(kinds)]
            W = np.zeros((U + 5, N_ENC))
            for r in range(U + 5):
                u = r - 4
                if kind == "band":
                    mu = centers[min(max(u, 0), U - 1)] + rng.normal(0, 1.5)
                    row = np.exp(-0.5 * ((frames - mu) / (3 + 2 * rng.random())) ** 2)
                elif kind == "flat":
                    row = np.exp(rng.normal(0, 0.3, N_ENC)) * (frames < n_audio) + 0.02
                elif kind == "stare":
                    row = np.exp(-0.5 * ((frames - (0.3 * n_audio + 3 * (h % 3))) / 2.0) ** 2) + 0.3 * np.exp(-0.5 * ((frames - 0.7 * n_audio) / 2.0) ** 2)
                elif kind == "pad":
                    row = np.exp(-0.5 * ((frames - (n_audio + 40)) / 6.0) ** 2) + 0.05 * np.exp(-0.5 * ((frames - centers[min(max(u, 0), U - 1)]) / 5.0) ** 2)
                else:   # other：集中但亂跳
                    mu = centers[rng.integers(0, U)]
                    row = np.exp(-0.5 * ((frames - mu) / 3.0) ** 2)
                row = row + 1e-6
                W[r] = row / row.sum()
            cross[l, h] = W
    text = " ".join(FAKE_WORDS)
    hyp = "".join(pieces)
    return {"pieces": pieces, "prefix_len": 4, "cross": cross, "n_audio": n_audio, "transcript": hyp, "reference": text,
            "alignment_heads": [list(p) for p in hf_heads], "n_layer": n_layer, "n_head": n_head,
            "normalize": simple_normalize, "median_filter_width": CFG["timestamps"]["median_filter_width"]}


# ── 輸出紀錄 ───────────────────────────────────────────────────────────
class Out:
    def __init__(self):
        self.buf = io.StringIO()

    def __call__(self, s: str = "") -> None:
        print(s)
        self.buf.write(re.sub(r"\033\[[0-9;]*m", "", s) + "\n")

    def banner(self, s: str, color: str = C.BLUE) -> None:
        C.banner(s, color)
        self.buf.write(f"\n{'─' * 72}\n{s}\n{'─' * 72}\n")

    def text(self) -> str:
        return self.buf.getvalue()


# ── 主體 ───────────────────────────────────────────────────────────────
class Demo:
    def __init__(self, out: Out, pieces: list[str], prefix_len: int, cross_audio: np.ndarray, pad_mass: np.ndarray, n_audio: int,
                 transcript: str, reference: str, alignment_heads: list[list[int]], backend: str, tag: str = "",
                 wav_path: Path | None = None, duration_s: float | None = None, normalize=simple_normalize,
                 median_filter_width: int | None = None, align: list[dict] | None = None):
        self.out, self.pieces, self.prefix_len = out, pieces, prefix_len
        self.cross = cross_audio.astype(np.float32)        # (n_layer, n_head, L_tok, n_audio)——只存音訊範圍
        self.pad_mass = pad_mass.astype(np.float32)        # (n_layer, n_head, L_tok)
        self.n_audio = int(n_audio)
        self.transcript, self.reference = transcript, reference
        self.alignment_heads = [list(map(int, p)) for p in alignment_heads]
        self.backend, self.tag, self.wav_path = backend, tag, wav_path
        self.duration_s = duration_s if duration_s is not None else self.n_audio * FRAME_S
        self.normalize = normalize
        self.mfw = int(median_filter_width or CFG["timestamps"]["median_filter_width"])
        self.align = align
        self.n_layer, self.n_head, self.L_tok, _ = self.cross.shape
        self.U = len(pieces)
        self.text_rows = slice(self.prefix_len, self.prefix_len + self.U)
        assert self.L_tok == self.prefix_len + self.U + 1, (self.L_tok, self.prefix_len, self.U)
        self.layer = int(CFG["display"]["start_layer"]) % self.n_layer
        self.mode = None
        self.boxed = False
        self.seating = False
        self.classes: dict | None = None
        self.timing: dict | None = None
        self.fig = None
        self.playing = False
        self.log: dict = {"time": C.now(), "backend": backend, "tag": tag, "config": CFG, "versions": self.versions(),
                          "transcript": transcript, "reference": reference, "pieces": pieces, "U": self.U, "n_audio": self.n_audio,
                          "duration_s": round(self.duration_s, 3), "alignment_heads": self.alignment_heads, "sections": {}}

    @staticmethod
    def versions() -> dict:
        v = C.versions()
        from importlib import metadata
        for pkg in ("torch", "transformers", "huggingface_hub", "soundfile", "tokenizers"):
            try:
                v[pkg] = metadata.version(pkg)
            except metadata.PackageNotFoundError:
                v[pkg] = None
        if MODEL_LOCK.exists():
            v["model_commit"] = MODEL_LOCK.read_text(encoding="utf-8").strip()
        return v

    # -- 分類（一次算完 144 個）--
    def classify_all(self) -> dict:
        if self.classes is not None:
            return self.classes
        heads = {}
        counts = {"band": 0, "flat": 0, "stare": 0, "other": 0}
        for l in range(self.n_layer):
            for h in range(self.n_head):
                r = classify_head(self.cross[l, h, self.text_rows, :], self.pad_mass[l, h, self.text_rows])
                heads[f"{l},{h}"] = r
                counts[r["class"]] += 1
        ah = {f"{l},{h}" for l, h in self.alignment_heads}
        ah_classes = {k: heads[k]["class"] for k in sorted(ah, key=lambda s: tuple(map(int, s.split(","))))}
        band_not_ah = [k for k, r in heads.items() if r["class"] == "band" and k not in ah]
        pad_heavy = sum(1 for r in heads.values() if r["pad_heavy"])
        pad_all = float(np.median(self.pad_mass[:, :, self.text_rows]))
        self.classes = {"counts": counts, "total": self.n_layer * self.n_head, "heads": heads, "alignment_heads_class": ah_classes,
                        "alignment_heads_band": sum(1 for v in ah_classes.values() if v == "band"),
                        "alignment_heads_pad_heavy": sum(1 for k in ah if heads[k]["pad_heavy"]),
                        "band_not_in_alignment_heads": band_not_ah,
                        "pad_heavy_heads": pad_heavy, "pad_mass_median_all": round(pad_all, 3),
                        "band_pad_heavy": sum(1 for r in heads.values() if r["class"] == "band" and r["pad_heavy"]),
                        "criteria": CFG["classify"]}
        self.log["sections"]["classes"] = {k: v for k, v in self.classes.items() if k != "heads"} | \
            {"heads": {k: {kk: vv for kk, vv in r.items() if kk != "tstar"} for k, r in heads.items()}}
        return self.classes

    # -- 1 播放與文字 --
    def play(self) -> None:
        if self.wav_path is not None and self.wav_path.exists() and not self.playing:
            self.playing = True
            self.out(f"播放 {self.wav_path.name}（{self.duration_s:.2f} s）……")

            def run():
                C.play(self.wav_path)
                self.playing = False
            threading.Thread(target=run, daemon=True).start()
        elif self.wav_path is None or not self.wav_path.exists():
            self.out(f"{C.AMBER}沒有音檔可播（{self.backend}）。{C.RESET}")
        self.transcript_panel()

    def transcript_panel(self) -> dict:
        o = self.out
        ref_n, hyp_n = self.normalize(self.reference), self.normalize(self.transcript)
        raw = word_errors(self.reference.upper(), re.sub(r"[^\w' ]+", " ", self.transcript.upper()))
        norm = word_errors(ref_n, hyp_n)
        o.banner("1  轉錄 vs. 參考")
        o(f"  參考      {self.reference}")
        o(f"  Whisper   {self.transcript.strip()}")
        o(f"  正規化後  {hyp_n}")
        o(f"  字錯數    原樣（大寫、去標點）{raw['errors']}（S {raw['S']} D {raw['D']} I {raw['I']}，參考 {raw['N_ref']} 字）"
          f"   正規化後 {norm['errors']}（S {norm['S']} D {norm['D']} I {norm['I']}）")
        o(f"  token     U = {self.U} 個文字 token：{' | '.join(repr(p) for p in self.pieces)}")
        o(f"  frame     音訊 {self.duration_s:.3f} s = {self.n_audio} 個 encoder frame（20 ms）；之後到 1500 是 padding")
        r = {"word_errors_raw": raw, "word_errors_normalized": norm, "transcript_normalized": hyp_n, "reference_normalized": ref_n}
        self.log["sections"]["transcript"] = r
        if self.fig is not None:
            self.summary.set_text(f"Whisper: {self.transcript.strip()}    word errors: raw {raw['errors']} · normalized {norm['errors']}")
            self.fig.canvas.draw_idle()
        return r

    # -- 圖 --
    def make_fig(self):
        import matplotlib.pyplot as plt
        self.plt = plt
        d = CFG["display"]
        self.fig = plt.figure(figsize=(d["fig_w"], d["fig_h"]))
        try:
            self.fig.canvas.manager.set_window_title("W4 Demo 3 — attention heads")
        except Exception:
            pass
        self.fig.canvas.mpl_connect("key_press_event", self.on_key)
        self.hint = self.fig.text(0.5, 0.5, "1  play + transcript        2  heads of one layer        3  next layer        0  all 144 heads\n\n(ask the question first)",
                                  ha="center", va="center", fontsize=20, color="0.35")
        self.tagtxt = self.fig.text(0.99, 0.005, self.tag, ha="right", va="bottom", fontsize=11, color=ALIGN_COLOR, fontweight="bold")
        self.summary = self.fig.text(0.01, 0.985, "", ha="left", va="top", fontsize=11.5, fontweight="bold", family="monospace")
        self.foot = self.fig.text(0.01, 0.005, "", ha="left", va="bottom", fontsize=10.5, family="monospace")
        return self.fig

    def _clear_axes(self) -> None:
        for ax in list(self.fig.axes):
            ax.remove()
        self.hint.set_visible(False)
        self.fig.suptitle("")

    def _frame_box(self, ax, color: str, lw: float) -> None:
        for sp in ax.spines.values():
            sp.set_edgecolor(color)
            sp.set_linewidth(lw)

    def _title_head(self, l: int, h: int, r: dict, small: bool = False) -> str:
        if small:
            return f"L{l}H{h}"
        return f"L{l} H{h} · {r['class']} · ρ {r['rho']:+.2f} · conc {r['conc']:.2f} · pad {100 * r['pad']:.0f}%"

    def layer_grid(self) -> None:
        if self.fig is None:
            return
        cl = self.classify_all()
        self._clear_axes()
        self.mode = "layer"
        l = self.layer
        rows, cols = 3, 4
        ext = [0, self.n_audio * FRAME_S, self.U - 0.5, -0.5]
        for h in range(self.n_head):
            ax = self.fig.add_subplot(rows, cols, h + 1)
            W = self.cross[l, h, self.text_rows, :]
            ax.imshow(W / np.maximum(W.max(axis=1, keepdims=True), 1e-12), aspect="auto", cmap=CFG["display"]["cmap"], extent=ext, interpolation="nearest")
            r = cl["heads"][f"{l},{h}"]
            ax.set_title(self._title_head(l, h, r), fontsize=8.5, color=CLASS_COLOR[r["class"]], fontweight="bold")
            self._frame_box(ax, CLASS_COLOR[r["class"]], 2.0)
            if h % cols == 0:
                ax.set_yticks(range(self.U))
                ax.set_yticklabels([p.strip() for p in self.pieces], fontsize=5.5)
            else:
                ax.set_yticks([])
            if h // cols == rows - 1:
                ax.set_xlabel("time (s)", fontsize=8)
                ax.tick_params(axis="x", labelsize=7)
            else:
                ax.set_xticks([])
            if self.boxed and [l, h] in self.alignment_heads:
                self._mark_alignment(ax)
        cnt = sum(1 for h in range(self.n_head) if cl["heads"][f"{l},{h}"]["class"] == "band")
        nah = sum(1 for h in range(self.n_head) if [l, h] in self.alignment_heads)
        self.fig.suptitle(f"decoder layer {l} of {self.n_layer} — {self.n_head} cross-attention heads · rows: {self.U} text tokens · "
                          f"columns: {self.n_audio} encoder frames (20 ms) · each row scaled to its own max",
                          fontsize=11, y=0.955)
        self.foot.set_text(f"this layer: {cnt} band  ·  OpenAI alignment heads in this layer: {nah}" + (" (boxed)" if self.boxed else "") +
                           "    border: teal band / grey flat / orange stare / blue other    3 → next layer, 0 → all layers")
        self.fig.subplots_adjust(left=0.07, right=0.99, top=0.9, bottom=0.07, hspace=0.42, wspace=0.08)
        self.fig.canvas.draw_idle()
        self.out(f"層 {l}：" + "  ".join(f"H{h} {cl['heads'][f'{l},{h}']['class']}" for h in range(self.n_head)))

    def next_layer(self) -> None:
        self.layer = (self.layer + 1) % self.n_layer
        self.layer_grid()

    def _mark_alignment(self, ax) -> None:
        self._frame_box(ax, ALIGN_COLOR, 4.0)
        ax.text(0.02, 0.96, "★ OpenAI", transform=ax.transAxes, fontsize=8, color=ALIGN_COLOR, fontweight="bold", va="top",
                bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.85))

    def overview(self) -> None:
        if self.fig is None:
            return
        cl = self.classify_all()
        self._clear_axes()
        self.mode = "overview"
        for l in range(self.n_layer):
            for h in range(self.n_head):
                ax = self.fig.add_subplot(self.n_layer, self.n_head, l * self.n_head + h + 1)
                W = self.cross[l, h, self.text_rows, :]
                ax.imshow(W / np.maximum(W.max(axis=1, keepdims=True), 1e-12), aspect="auto", cmap=CFG["display"]["cmap"], interpolation="nearest")
                ax.set_xticks([]); ax.set_yticks([])
                r = cl["heads"][f"{l},{h}"]
                self._frame_box(ax, CLASS_COLOR[r["class"]], 2.2)
                if h == 0:
                    ax.set_ylabel(f"L{l}", fontsize=8, rotation=0, labelpad=12, va="center")
                if l == 0:
                    ax.set_title(f"H{h}", fontsize=8)
                if r["pad_heavy"]:
                    ax.text(0.97, 0.93, "pad", transform=ax.transAxes, fontsize=6, color="0.75", ha="right", va="top")
                if self.boxed and [l, h] in self.alignment_heads:
                    self._frame_box(ax, ALIGN_COLOR, 3.5)
                    ax.text(0.5, 0.5, "★", transform=ax.transAxes, fontsize=14, color=ALIGN_COLOR, ha="center", va="center", fontweight="bold")
        c, n = cl["counts"], cl["total"]
        self.fig.suptitle(f"all {n} cross-attention heads · band {c['band']}  ·  flat {c['flat']}  ·  stare {c['stare']}  ·  other {c['other']}"
                          f"  ·  {cl['pad_heavy_heads']} put > half their mass on the padding"
                          + (f"  ·  ★ OpenAI's {len(self.alignment_heads)} alignment heads, {cl['alignment_heads_band']} of them band" if self.boxed else ""),
                          fontsize=12, y=0.958)
        self.foot.set_text(f"border = shape inside the audio (teal band / grey flat / orange stare / blue other) · 'pad' = ≥ {CFG['classify']['pad_max']:.0%} of the row's mass "
                           f"on the 30 s padding · band: ρ ≥ {CFG['classify']['rho_min']}, conc ≥ {CFG['classify']['conc_min']} within ±{CFG['classify']['window_frames']} frames")
        self.fig.subplots_adjust(left=0.04, right=0.995, top=0.915, bottom=0.045, hspace=0.25, wspace=0.08)
        self.fig.canvas.draw_idle()
        self.out(f"全部 {n} 個 head：band {c['band']}、flat {c['flat']}、stare {c['stare']}、other {c['other']}；"
                 f"OpenAI 的 {len(self.alignment_heads)} 個 alignment heads 裡 {cl['alignment_heads_band']} 個是 band；"
                 f"band 但不在 OpenAI 名單的 {len(cl['band_not_in_alignment_heads'])} 個：{', '.join(cl['band_not_in_alignment_heads']) or '—'}")
        self.out(f"padding：{cl['pad_heavy_heads']} / {n} 個 head 把超過一半的權重放在音訊結尾之後的 padding（全部 head 的中位數 {100 * cl['pad_mass_median_all']:.0f}%）；"
                 f"band 裡有 {cl['band_pad_heavy']} 個、OpenAI 的 {len(self.alignment_heads)} 個裡有 {cl['alignment_heads_pad_heavy']} 個——音訊範圍內是帶、整列看是在看 padding")

    def box_alignment(self) -> None:
        self.boxed = True
        cl = self.classify_all()
        self.out(f"OpenAI _ALIGNMENT_HEADS（small）= {self.alignment_heads}；各自的類別："
                 + "  ".join(f"L{k.replace(',', 'H')} {v}" for k, v in cl["alignment_heads_class"].items()))
        if self.mode == "layer":
            self.layer_grid()
        elif self.mode == "overview":
            self.overview()
        else:
            self.overview()

    # -- 5 DTW 與時間戳 --
    def compute_timing(self) -> dict:
        if self.timing is not None:
            return self.timing
        idx = [(l, h) for l, h in self.alignment_heads]
        weights = np.stack([self.cross[l, h] for l, h in idx])                        # (H, L_tok, n_audio)
        M = timing_matrix(weights, self.n_audio, self.mfw)                               # (L_tok, n_audio)
        M_text = M[self.prefix_len:-1]                                                   # 去前綴、去最後一列（eot）
        times, ti, tj = token_times(M_text)
        words = group_words(self.pieces)
        starts = [float(times[w[0]]) for w in words]
        ends = starts[1:] + [float(min((tj[-1] + 1) * FRAME_S, self.n_audio * FRAME_S))]
        wl = [{"i": i, "word": "".join(self.pieces[k] for k in w).strip(), "start": round(s, 3), "end": round(e, 3)}
              for i, (w, s, e) in enumerate(zip(words, starts, ends))]
        comp = None
        if self.align and len(self.align) == len(wl):
            diffs = [round(w["start"] - a["start"], 3) for w, a in zip(wl, self.align)]
            comp = {"n_words": len(wl), "start_minus_w3_first_spike_s": diffs, "median_abs_s": round(float(np.median(np.abs(diffs))), 3),
                    "max_abs_s": round(float(np.max(np.abs(diffs))), 3)}
        self.timing = {"matrix": M_text, "path": (ti, tj), "token_times_s": [round(float(t), 3) for t in times], "words": wl, "vs_w3": comp}
        self.log["sections"]["timing"] = {k: v for k, v in self.timing.items() if k not in ("matrix", "path")}
        return self.timing

    def dtw_panel(self) -> None:
        tm = self.compute_timing()
        o = self.out
        o.banner("5  alignment heads → timing.py 的做法 → DTW → 字的時間戳")
        o(f"  用的 head：{self.alignment_heads}（OpenAI _ALIGNMENT_HEADS = HF alignment_heads）")
        o(f"  標準化（token 軸）→ 中值濾波 {self.mfw} → {len(self.alignment_heads)} 個 head 平均 → 對 −matrix 做 DTW（{self.U} × {self.n_audio}）")
        hdr = f"  {'word':<12s}{'start':>8s}{'end':>8s}"
        if tm["vs_w3"]:
            hdr += f"{'W3 1st spike':>14s}{'Δstart':>9s}"
        o(hdr)
        for i, w in enumerate(tm["words"]):
            line = f"  {w['word']:<12s}{w['start']:>8.3f}{w['end']:>8.3f}"
            if tm["vs_w3"]:
                line += f"{self.align[i]['start']:>14.3f}{tm['vs_w3']['start_minus_w3_first_spike_s'][i]:>+9.3f}"
            o(line)
        if tm["vs_w3"]:
            o(f"  字的起點對 W3 forced alignment 第一個尖峰：|Δ| 中位數 {tm['vs_w3']['median_abs_s']:.3f} s、最大 {tm['vs_w3']['max_abs_s']:.3f} s"
              f"（W3 的是 CTC 尖峰不是音段邊界，Demo 2 的提醒同樣適用）")
        elif self.align:
            o(f"  W3 的對齊有 {len(self.align)} 個字、Whisper 切出 {len(tm['words'])} 個字，不逐字比；按 a 可以疊圖")
        if self.fig is None:
            return
        self._clear_axes()
        self.mode = "dtw"
        ax = self.fig.add_axes([0.11, 0.09, 0.86, 0.70])
        M = tm["matrix"]
        ext = [0, self.n_audio * FRAME_S, self.U - 0.5, -0.5]
        ax.imshow(M, aspect="auto", cmap=CFG["display"]["cmap"], extent=ext, interpolation="nearest")
        ti, tj = tm["path"]
        ax.plot((tj + 0.5) * FRAME_S, ti, color="white", lw=2.0, drawstyle="steps-post")
        for w in tm["words"]:
            ax.axvline(w["start"], color="#8ecae6", lw=1.2, ls="--")
            ax.text(w["start"] + 0.01, -0.4, w["word"], color="#8ecae6", fontsize=7.5, rotation=90, va="top", ha="left")
        ax.set_yticks(range(self.U))
        ax.set_yticklabels([p.strip() for p in self.pieces], fontsize=7)
        ax.set_xlabel("time (s) — white: DTW path; dashed: word start = first token's jump time")
        ax.set_title(f"mean of {len(self.alignment_heads)} alignment heads after per-frame standardisation and median filter ({self.mfw}) — the matrix the DTW runs on",
                     fontsize=10.5)
        self.ax_dtw = ax
        self.ax_seat = self.fig.add_axes([0.11, 0.83, 0.86, 0.08], sharex=ax)
        self.ax_seat.set_yticks([]); self.ax_seat.set_xticks([])
        self.ax_seat.set_ylim(0, 1)
        for sp in self.ax_seat.spines.values():
            sp.set_visible(False)
        self.ax_seat.text(0.0, 0.5, "a → W3 forced alignment (seating chart)", fontsize=9, color="0.5", va="center", transform=self.ax_seat.transAxes)
        self.foot.set_text("timestamps are a post-processing artefact: hand-picked heads + smoothing + a monotonic DP — not a model output")
        if self.seating:
            self.overlay_seating(redraw=False)
        self.fig.canvas.draw_idle()

    def overlay_seating(self, redraw: bool = True) -> None:
        if not self.align:
            self.out(f"{C.AMBER}沒有 W3 Demo 3 的 forced alignment（{CFG['audio']['alignment']}），沒有座位表可疊。{C.RESET}")
            return
        self.seating = True
        if self.fig is None:
            return
        if self.mode != "dtw":
            self.dtw_panel()
            return
        ax = self.ax_seat
        for t in list(ax.texts):
            t.remove()
        for i, a in enumerate(self.align):
            ax.axvspan(a["start"], a["end"], ymin=0.15, ymax=0.85, color="0.55", alpha=0.75)
            ax.text((a["start"] + a["end"]) / 2, 0.5, a["word"], ha="center", va="center", fontsize=6.5, color="white", fontweight="bold")
            self.ax_dtw.axvspan(a["start"], a["end"], color="white", alpha=0.08, lw=0)
        ax.text(0.0, 1.05, "seating chart: W3 Demo 3 CTC forced alignment (first → last spike of each word)", fontsize=8.5, color="0.4",
                transform=ax.transAxes, va="bottom")
        self.foot.set_text("grey bars: the alignment (each word owns a contiguous span) — below: the weights (soft, and not the same object)")
        if redraw:
            self.fig.canvas.draw_idle()
        self.out("  疊上 W3 的 forced alignment：灰條是座位表（每個字一段連續的 frame），底下是權重——兩個不同的物件。")

    # -- 選單、鍵、流程 --
    def menu(self) -> None:
        o = self.out
        o(f"\n{C.BOLD}p45{C.RESET}  1 播音＋轉錄   2 這一層的 12 個 head   3 下一層   0 全部 144 個   4 框出 OpenAI 的 alignment heads   5 DTW 與時間戳")
        o(f"{C.BOLD}p46{C.RESET}  a 疊 W3 的座位表        h 選單   q 離開")

    def dispatch(self, k: str) -> bool:
        if k == "1":
            self.play()
        elif k == "2":
            self.layer_grid()
        elif k == "3":
            self.next_layer()
        elif k == "0":
            self.overview()
        elif k == "4":
            self.box_alignment()
        elif k == "5":
            self.dtw_panel()
        elif k == "a":
            self.overlay_seating()
        elif k == "h":
            self.menu()
        elif k == "q":
            return False
        return True

    def on_key(self, ev) -> None:
        if not self.dispatch(ev.key):
            self.plt.close(self.fig)

    def run_all(self, save_dir: Path | None = None) -> None:
        """課堂順序全部跑一遍（rehearse 用；不播音訊）。有 save_dir 就把每個畫面存成 png。"""
        self.transcript_panel()
        self.classify_all()
        for i, l in enumerate(range(self.n_layer)):
            self.layer = l
            self.layer_grid()
            if save_dir is not None and self.fig is not None:
                self.fig.savefig(save_dir / f"layer{l:02d}.png", dpi=72)
        self.layer = int(CFG["display"]["start_layer"]) % self.n_layer
        self.overview()
        if save_dir is not None and self.fig is not None:
            self.fig.savefig(save_dir / "overview.png", dpi=110)
        self.box_alignment()
        if save_dir is not None and self.fig is not None:
            self.fig.savefig(save_dir / "overview-boxed.png", dpi=110)
        self.dtw_panel()
        self.overlay_seating()
        if save_dir is not None and self.fig is not None:
            self.fig.savefig(save_dir / "dtw.png", dpi=110)

    def save(self, folder: Path) -> None:
        folder.mkdir(parents=True, exist_ok=True)
        stem = "rehearsal" if folder.name == "rehearsal" else "session"
        np.save(folder / "cross_audio.npy", self.cross.astype(np.float16))         # 144 × L_tok × n_audio，float16（約 2 MB）
        np.save(folder / "pad_mass.npy", self.pad_mass.astype(np.float32))
        meta = {"pieces": self.pieces, "prefix_len": self.prefix_len, "n_audio": self.n_audio, "transcript": self.transcript,
                "reference": self.reference, "alignment_heads": self.alignment_heads, "duration_s": self.duration_s,
                "median_filter_width": self.mfw}
        (folder / "bundle.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
        if self.wav_path and self.wav_path.exists() and not (folder / "utterance.wav").exists():
            shutil.copy(self.wav_path, folder / "utterance.wav")
        (folder / f"{stem}.json").write_text(json.dumps(self.log, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
        (folder / f"{stem}.txt").write_text(self.out.text(), encoding="utf-8")
        if self.fig is not None:
            self.fig.savefig(folder / "last-screen.png", dpi=110)
        C.note(f"  存到 {folder}/（cross_audio.npy、pad_mass.npy、bundle.json、utterance.wav、{stem}.json、{stem}.txt、png）")


# ── 載入：真的、fake、replay ─────────────────────────────────────────────
def load_audio() -> tuple[np.ndarray, str, Path]:
    wav, txt = _p(CFG["audio"]["wav"]), _p(CFG["audio"]["text"])
    if not wav.exists() or not txt.exists():
        sys.exit(f"找不到 {wav} 或 {txt}。這一句重用 W3 Demo 3 的彩排紀錄；先在 w03_d3_blank_heatmap 跑 fetch＋rehearse。")
    return C.load_wav(wav, FS), txt.read_text(encoding="utf-8").strip(), wav


def split_bundle(cross_full: np.ndarray, n_audio: int) -> tuple[np.ndarray, np.ndarray]:
    """(L, H, L_tok, 1500) → 音訊範圍內的權重 (L, H, L_tok, n_audio) 與每列落在 padding 的比例 (L, H, L_tok)。"""
    tot = cross_full.sum(axis=-1)
    pad = cross_full[..., n_audio:].sum(axis=-1) / np.where(tot <= 0, 1.0, tot)
    return cross_full[..., :n_audio].copy(), pad


def build(cmd: str, out: Out) -> Demo:
    align = load_alignment(_p(CFG["audio"]["alignment"]))
    if cmd == "fake":
        wav = _p(CFG["audio"]["wav"])
        dur = len(C.load_wav(wav, FS)) / FS if wav.exists() else 5.855
        b = fake_bundle(dur, align)
        cross, pad = split_bundle(b["cross"], b["n_audio"])
        return Demo(out, b["pieces"], b["prefix_len"], cross, pad, b["n_audio"], b["transcript"], b["reference"], b["alignment_heads"],
                    "fake", "FAKE — synthetic attention, not a model", wav_path=wav if wav.exists() else None, duration_s=dur, align=align)
    if cmd == "replay":
        folder = HERE / "runs" / "rehearsal"
        if not (folder / "cross_audio.npy").exists():
            sys.exit(f"沒有彩排紀錄 {folder}/cross_audio.npy。先跑 ./present.sh rehearse。")
        cross = np.load(folder / "cross_audio.npy").astype(np.float32)
        pad = np.load(folder / "pad_mass.npy")
        m = json.loads((folder / "bundle.json").read_text(encoding="utf-8"))
        wav = folder / "utterance.wav"
        return Demo(out, m["pieces"], m["prefix_len"], cross, pad, m["n_audio"], m["transcript"], m["reference"], m["alignment_heads"],
                    "replay", "REPLAY — pre-computed in rehearsal, not live", wav_path=wav if wav.exists() else None,
                    duration_s=m["duration_s"], median_filter_width=m.get("median_filter_width"), align=align)
    x, text, wav = load_audio()
    C.note(f"載入 {CFG['model']['repo']}（{CFG['model']['device']}，attention = eager）……")
    t0 = time.perf_counter()
    be = Real(CFG["model"]["device"], CFG["model"]["attn_implementation"])
    C.note(f"  模型就緒（{time.perf_counter() - t0:.1f} s，{be.n_params / 1e6:.0f} M 參數，{be.n_layer} 層 × {be.n_head} 個 head）")
    t0 = time.perf_counter()
    ids = be.transcribe(x)
    t_gen = time.perf_counter() - t0
    t0 = time.perf_counter()
    cross_full = be.cross_attention(x, ids)
    t_fwd = time.perf_counter() - t0
    n_audio = len(x) // (FS // 100) // 2            # mel 10 ms 一格 → encoder 20 ms 一格；= transformers 的 num_frames // 2
    cross, pad = split_bundle(cross_full, n_audio)
    pieces = be.pieces(ids)
    C.note(f"  轉錄 {1000 * t_gen:.0f} ms、teacher-forced 前向 {1000 * t_fwd:.0f} ms；cross-attention {cross_full.shape} → 只留 {n_audio} 格")
    if be.alignment_heads != [list(p) for p in CFG["timestamps"]["hf_alignment_heads"]]:
        C.note(f"{C.AMBER}注意：模型 generation_config 的 alignment_heads 與 demo_config.toml 抄的不同：{be.alignment_heads}。用模型的。{C.RESET}")
    d = Demo(out, pieces, len(be.prompt), cross, pad, n_audio, be.proc.tokenizer.decode(ids), text, be.alignment_heads, "whisper",
             wav_path=wav, duration_s=len(x) / FS, normalize=be.normalize, median_filter_width=be.median_filter_width, align=align)
    d.log["model"] = {"repo": CFG["model"]["repo"], "n_params": be.n_params, "device": str(be.device), "attn_implementation": be.attn,
                      "n_layer": be.n_layer, "n_head": be.n_head, "prompt_ids": be.prompt, "eot": be.eot,
                      "seconds_generate": round(t_gen, 3), "seconds_forward": round(t_fwd, 3)}
    return d


# ── check ───────────────────────────────────────────────────────────────
def check() -> None:
    C.banner("課前體檢", C.AMBER)
    v = Demo.versions()
    for k in ("python", "platform", "numpy", "scipy", "matplotlib", "torch", "transformers", "huggingface_hub", "tokenizers", "model_commit"):
        print(f"  {k:<16s} {v.get(k)}")
    res: dict = {"time": C.now(), "versions": v}
    try:
        import torch
    except ImportError:
        print("  沒有 torch：先在 w03_d3_blank_heatmap 跑 ./setup.sh（torch／transformers 進共用 .venv）")
        return
    mps = bool(torch.backends.mps.is_available())
    print(f"  {'mps':<16s} built={torch.backends.mps.is_built()} available={mps}")
    res["mps_available"] = mps
    try:
        path = model_dir()
        print(f"  {'model':<16s} {path}")
        res["model_path"] = str(path)
    except Exception as e:
        print(f"  {'model':<16s} 不在快取：{e}\n  先跑 ./present.sh fetch")
        return
    x, text, _ = load_audio()
    n_audio = len(x) // (FS // 100) // 2
    res["n_audio"] = n_audio

    # 1. eager：轉錄、teacher-forced、權重形狀；alignment heads 對 base85
    be = Real("cpu", "eager")
    res["model"] = {"n_params": be.n_params, "n_layer": be.n_layer, "n_head": be.n_head, "prompt_ids": be.prompt, "eot": be.eot,
                    "alignment_heads_hf": be.alignment_heads, "median_filter_width": be.median_filter_width}
    dec = decode_alignment_heads(CFG["timestamps"]["openai_alignment_heads_b85"], be.n_layer, be.n_head)
    res["alignment_heads_openai_b85"] = dec
    res["openai_b85_equals_hf"] = dec == be.alignment_heads
    print(f"  alignment heads：HF generation_config {be.alignment_heads}")
    print(f"                   OpenAI base85 解碼 {dec}  → {'相同' if res['openai_b85_equals_hf'] else '不同！'}")
    ids = be.transcribe(x)
    hyp = be.proc.tokenizer.decode(ids)
    we = word_errors(be.normalize(text), be.normalize(hyp))
    print(f"  轉錄（cpu）：「{hyp.strip()}」  U = {len(ids)}  正規化字錯 {we['errors']}")
    res["transcript"] = {"text": hyp, "U": len(ids), "word_errors_normalized": we, "ids": ids}
    ts = []
    for _ in range(2):
        t0 = time.perf_counter(); cross = be.cross_attention(x, ids); ts.append(time.perf_counter() - t0)
    res["eager_cpu"] = {"cross_shape": list(cross.shape), "forward_ms": round(1000 * min(ts), 1),
                        "rows_sum_to_one_maxdev": float(np.max(np.abs(cross.sum(-1) - 1.0)))}
    print(f"  eager cpu：cross-attention {cross.shape}，前向 {res['eager_cpu']['forward_ms']:.0f} ms，每列和對 1 最大偏差 {res['eager_cpu']['rows_sum_to_one_maxdev']:.1e}")

    # 2. 自己的 DTW 對 transformers 的 return_token_timestamps
    try:
        seq, hf_ts = be.hf_token_timestamps(x)
        _, pad = split_bundle(cross, n_audio)
        M = timing_matrix(np.stack([cross[l, h] for l, h in be.alignment_heads]), n_audio, be.median_filter_width)
        mine, _, _ = token_times(M[len(be.prompt):-1])
        text_pos = [i for i, t in enumerate(seq) if t < be.eot]
        hf_text = hf_ts[text_pos][: len(mine)] if len(text_pos) else np.zeros(0)
        same_tokens = [seq[i] for i in text_pos] == ids
        diff = float(np.max(np.abs(hf_text - mine))) if len(hf_text) == len(mine) and len(mine) else None
        res["dtw_vs_transformers"] = {"same_tokens": same_tokens, "hf_token_times": hf_text.round(3).tolist(), "mine": mine.round(3).tolist(),
                                      "max_abs_diff_s": diff}
        print(f"  自己的 DTW 對 transformers return_token_timestamps：token 相同={same_tokens}，時間最大差 {diff if diff is None else f'{diff:.3f} s'}")
    except Exception as e:
        res["dtw_vs_transformers"] = {"error": f"{type(e).__name__}: {e}"}
        print(f"  transformers 的 return_token_timestamps 失敗：{type(e).__name__}: {e}")

    # 3. sdpa：output_attentions 是不是 None
    try:
        be_s = Real("cpu", "sdpa")
        inp = be_s.features(x)
        decin = torch.tensor([be_s.prompt + ids + [be_s.eot]])
        with torch.no_grad():
            out = be_s.model(input_features=inp.input_features, decoder_input_ids=decin, output_attentions=True)
        none = out.cross_attentions is None or all(a is None for a in out.cross_attentions)
        res["sdpa_returns_attentions"] = not none
        print(f"  sdpa 之下 output_attentions=True：cross_attentions {'全是 None（所以一定要 eager）' if none else '有回傳'}")
        del be_s
    except Exception as e:
        res["sdpa_returns_attentions"] = {"error": f"{type(e).__name__}: {e}"}
        print(f"  sdpa 測試失敗：{type(e).__name__}: {e}")

    # 4. mps 對 cpu
    if mps:
        try:
            be_m = Real("mps", "eager")
            ids_m = be_m.transcribe(x)
            tsm = []
            for _ in range(2):
                t0 = time.perf_counter(); cross_m = be_m.cross_attention(x, ids); tsm.append(time.perf_counter() - t0)
            res["eager_mps"] = {"forward_ms": round(1000 * min(tsm), 1), "same_transcript": ids_m == ids,
                                "maxabs_vs_cpu": float(np.max(np.abs(cross_m - cross)))}
            print(f"  eager mps：前向 {res['eager_mps']['forward_ms']:.0f} ms，轉錄相同={ids_m == ids}，權重對 cpu 最大差 {res['eager_mps']['maxabs_vs_cpu']:.1e}")
        except Exception as e:
            res["eager_mps"] = {"error": f"{type(e).__name__}: {e}"}
            print(f"  eager mps 失敗：{type(e).__name__}: {e}")
    (HERE / "runs").mkdir(exist_ok=True)
    (HERE / "runs" / "check.json").write_text(json.dumps(res, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    C.note("  結果存在 runs/check.json（進版控：eager／sdpa、base85 對 HF、DTW 對 transformers、MPS 對 CPU 的實測紀錄）")


# ── fetch ───────────────────────────────────────────────────────────────
def fetch() -> None:
    from huggingface_hub import HfApi, snapshot_download
    C.banner("下載模型", C.AMBER)
    repo = CFG["model"]["repo"]
    sha = MODEL_LOCK.read_text(encoding="utf-8").strip() if MODEL_LOCK.exists() else HfApi().model_info(repo).sha
    path = snapshot_download(repo, revision=sha, allow_patterns=["*.json", "*.txt", "*.safetensors"])
    MODEL_LOCK.write_text(sha + "\n", encoding="utf-8")
    print(f"  {repo} @ {sha[:12]} → {path}\n  commit 寫進 model.lock（進版控）")
    wav = _p(CFG["audio"]["wav"])
    print(f"  音訊重用 W3 Demo 3 的 {wav}{'（在）' if wav.exists() else '（不在！先在 w03_d3_blank_heatmap 跑 fetch＋rehearse）'}")
    print("\n下一步：./present.sh check")


# ── export ──────────────────────────────────────────────────────────────
def export() -> None:
    src = HERE / "runs" / "rehearsal" / "rehearsal.json"
    if not src.exists():
        sys.exit("沒有 runs/rehearsal/rehearsal.json，先跑 ./present.sh rehearse")
    log = json.loads(src.read_text(encoding="utf-8"))
    if log.get("backend") != "whisper":
        sys.exit(f"rehearsal.json 的 backend 是 {log.get('backend')}，不是真模型的紀錄，不匯出")
    cl, tr, tm = log["sections"]["classes"], log["sections"]["transcript"], log["sections"].get("timing", {})
    data = json.loads(METRICS.read_text(encoding="utf-8")) if METRICS.exists() else {}
    data["demo3"] = {
        "source": "demos/w04_d3_attention_heads/runs/rehearsal/rehearsal.json", "time": log["time"], "backend": log["backend"],
        "model": log.get("model", {}).get("repo"), "model_commit": log["versions"].get("model_commit"),
        "transformers": log["versions"].get("transformers"), "torch": log["versions"].get("torch"),
        "sentence": log["reference"], "transcript": log["transcript"].strip(),
        "word_errors_raw": tr["word_errors_raw"]["errors"], "word_errors_normalized": tr["word_errors_normalized"]["errors"],
        "U_text_tokens": log["U"], "n_audio_frames": log["n_audio"],
        "heads_total": cl["total"], "heads": cl["counts"], "criteria": cl["criteria"],
        "alignment_heads": log["alignment_heads"], "alignment_heads_band": cl["alignment_heads_band"],
        "alignment_heads_class": cl["alignment_heads_class"], "band_not_in_alignment_heads": cl["band_not_in_alignment_heads"],
        "pad_heavy_heads": cl.get("pad_heavy_heads"), "pad_mass_median_all": cl.get("pad_mass_median_all"),
        "band_pad_heavy": cl.get("band_pad_heavy"), "alignment_heads_pad_heavy": cl.get("alignment_heads_pad_heavy"),
        "word_timestamps": tm.get("words"), "vs_w3_forced_alignment": tm.get("vs_w3"),
    }
    METRICS.parent.mkdir(parents=True, exist_ok=True)
    METRICS.write_text(json.dumps(data, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    c = cl["counts"]
    print(f"→ {METRICS}（demo3）：band {c['band']} / {cl['total']}、flat {c['flat']}、stare {c['stare']}、other {c['other']}；"
          f"字錯 原樣 {tr['word_errors_raw']['errors']}／正規化 {tr['word_errors_normalized']['errors']}；alignment heads {len(log['alignment_heads'])} 個、{cl['alignment_heads_band']} 個是 band")


# ── recompute ───────────────────────────────────────────────────────────
def recompute() -> None:
    """改了判準或後處理之後：從 runs/rehearsal/ 已存的權重重新分類、重算 DTW、重畫圖、重寫 rehearsal.json／.txt——不碰模型。
    model 區塊與版本沿用舊的 rehearsal.json，另記 recomputed 時間。"""
    import matplotlib
    matplotlib.use("Agg")
    folder = HERE / "runs" / "rehearsal"
    src = folder / "rehearsal.json"
    if not src.exists():
        sys.exit("沒有 runs/rehearsal/rehearsal.json，先跑 ./present.sh rehearse")
    old = json.loads(src.read_text(encoding="utf-8"))
    out = Out()
    d = build("replay", out)
    d.backend, d.tag = old.get("backend", "whisper"), "rehearsal"
    d.log["backend"], d.log["tag"] = d.backend, d.tag
    d.log["time"], d.log["versions"] = old["time"], old["versions"]
    if "model" in old:
        d.log["model"] = old["model"]
    d.log["recomputed"] = {"time": C.now(), "versions": Demo.versions()}
    d.make_fig()
    d.run_all(folder)
    d.save(folder)
    c = d.classes["counts"]
    print(f"\n重算完成：band {c['band']}、flat {c['flat']}、stare {c['stare']}、other {c['other']}；pad-heavy {d.classes['pad_heavy_heads']}；"
          f"alignment heads 裡 band {d.classes['alignment_heads_band']}。接著 ./present.sh export。")


# ── selftest ────────────────────────────────────────────────────────────
def selftest() -> None:
    C.banner("selftest（不需要模型）", C.AMBER)
    ok = True

    def chk(name: str, cond: bool, extra: str = "") -> None:
        nonlocal ok
        ok &= bool(cond)
        print(f"  [{'ok' if cond else 'FAIL'}] {name}{('  ' + extra) if extra else ''}")

    ts = CFG["timestamps"]
    dec = decode_alignment_heads(ts["openai_alignment_heads_b85"], 12, 12)
    chk("OpenAI small 的 base85 解碼 == HF alignment_heads", dec == [list(p) for p in ts["hf_alignment_heads"]], str(dec))
    dec_en = decode_alignment_heads(ts["openai_alignment_heads_b85_small_en"], 12, 12)
    chk("small.en 的那串解得開（對照用）", len(dec_en) == 19, f"{len(dec_en)} 個 head")

    # 中值濾波對 scipy
    from scipy.ndimage import median_filter as sp_med
    rng = np.random.default_rng(0)
    a = rng.normal(size=(3, 5, 40))
    mine = median_filter(a, 7)
    # torch 的 F.pad(mode="reflect") 與 np.pad(mode="reflect") 都不重複邊界點 = scipy 的 "mirror"（scipy 的 "reflect" 會重複邊界點）
    ref = sp_med(a, size=(1, 1, 7), mode="mirror")
    chk("中值濾波（寬 7、reflect 不重複邊界點）對 scipy.ndimage mode=mirror 逐位", np.allclose(mine, ref))

    # DTW：對角線最強 → 路徑貼對角線，token 時間遞增
    U, n = 8, 60
    M = np.zeros((U, n))
    centers = np.linspace(5, 55, U)
    for u in range(U):
        M[u] = np.exp(-0.5 * ((np.arange(n) - centers[u]) / 2.0) ** 2)
    times, ti, tj = token_times(M)
    chk("DTW 路徑：text 與 time 都單調不減、起點 (0,0)、終點 (U−1, n−1)",
        bool(np.all(np.diff(ti) >= 0) and np.all(np.diff(tj) >= 0) and ti[0] == 0 and tj[0] == 0 and ti[-1] == U - 1 and tj[-1] == n - 1))
    chk("DTW 的 token 時間遞增、落在高斯中心附近（±3 格）", bool(np.all(np.diff(times) > 0) and np.all(np.abs(times / FRAME_S - (centers - 2)) <= 3.5)),
        str(np.round(times / FRAME_S).astype(int).tolist()))

    # 分類判準
    n_audio, U = 300, 20
    frames = np.arange(N_ENC)
    pad_frames = N_ENC - n_audio
    band = np.stack([np.exp(-0.5 * ((frames - (15 * u + 10)) / 3) ** 2) for u in range(U)]); band /= band.sum(1, keepdims=True)
    flat = np.ones((U, N_ENC)) * (frames < n_audio); flat /= flat.sum(1, keepdims=True)
    stare = np.stack([np.exp(-0.5 * ((frames - 100) / 2) ** 2) for _ in range(U)]); stare /= stare.sum(1, keepdims=True)
    padh = np.stack([np.exp(-0.5 * ((frames - (n_audio + 50)) / 5) ** 2) for _ in range(U)]); padh /= padh.sum(1, keepdims=True)
    rng = np.random.default_rng(1)
    other = np.stack([np.exp(-0.5 * ((frames - rng.integers(10, 290)) / 3) ** 2) for _ in range(U)]); other /= other.sum(1, keepdims=True)
    for name, W, want in [("band", band, "band"), ("flat", flat, "flat"), ("stare", stare, "stare"), ("other", other, "other")]:
        W4 = W[None, None]
        ca, pm = split_bundle(W4, n_audio)
        r = classify_head(ca[0, 0], pm[0, 0])
        chk(f"判準：合成的 {name} → {want}", r["class"] == want and not r["pad_heavy"], f"got {r['class']} (ρ {r['rho']}, conc {r['conc']}, pad {r['pad']}, distinct {r['distinct']})")
    ca, pm = split_bundle(padh[None, None], n_audio)
    r = classify_head(ca[0, 0], pm[0, 0])
    chk("判準：盯著 padding 的 head → pad_heavy，形狀另外算", r["pad_heavy"] and r["pad"] > 0.9, f"got {r['class']}, pad {r['pad']}")

    # 併字與字錯
    pieces = ["Mr", ".", " Qu", "ilter", " is", " the", " apostle", ","]
    g = group_words(pieces)
    chk("併字：以空白開頭的 token 開新字、標點併前字", g == [[0, 1], [2, 3], [4], [5], [6, 7]], str(g))
    we = word_errors("MISTER QUILTER IS THE APOSTLE", "MR QUILTER IS THE APOSTLE")
    chk("字錯數：MISTER vs MR 原樣算 1 個替換", we["errors"] == 1 and we["S"] == 1, str(we))
    we2 = word_errors(simple_normalize("MISTER QUILTER IS THE APOSTLE"), simple_normalize("Mr. Quilter is the apostle,"))
    chk("字錯數：正規化後 0", we2["errors"] == 0, str(we2))
    we3 = word_errors("a b c d", "a c d e")
    chk("字錯數：刪 1 插 1", we3["errors"] == 2 and we3["D"] == 1 and we3["I"] == 1, str(we3))

    # fake 的整條流程（不開視窗）
    import matplotlib
    matplotlib.use("Agg")
    out = Out()
    d = build("fake", out)
    d.make_fig()
    d.run_all()
    cl = d.classes
    chk("fake：alignment heads 全部分到 band", cl["alignment_heads_band"] == len(d.alignment_heads), str(cl["alignment_heads_class"]))
    chk("fake：四類都出現", all(v > 0 for v in cl["counts"].values()), str(cl["counts"]))
    chk("fake：DTW 切出的字數 = 參考句的字數", len(d.timing["words"]) == len(FAKE_WORDS), str(len(d.timing["words"])))
    print(f"\n{'全部通過' if ok else '有項目失敗'}")
    sys.exit(0 if ok else 1)


# ── main ────────────────────────────────────────────────────────────────
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["show", "rehearse", "replay", "check", "fetch", "fake", "selftest", "export", "recompute"])
    ap.add_argument("--selftest", metavar="DIR", help="不開視窗：照課堂順序全部跑一遍、把每個畫面存進 DIR、印數字（測試用）")
    args = ap.parse_args()

    if args.cmd == "check":
        check(); return
    if args.cmd == "fetch":
        fetch(); return
    if args.cmd == "selftest":
        selftest(); return
    if args.cmd == "export":
        export(); return
    if args.cmd == "recompute":
        recompute(); return

    if args.selftest or args.cmd == "rehearse":
        import matplotlib
        matplotlib.use("Agg")

    out = Out()
    d = build(args.cmd, out)
    d.make_fig()

    if args.selftest:
        folder = Path(args.selftest)
        folder.mkdir(parents=True, exist_ok=True)
        d.run_all(folder)
        c = d.classes["counts"]
        print(f"selftest ok: U={d.U} n_audio={d.n_audio} band={c['band']} flat={c['flat']} stare={c['stare']} other={c['other']} "
              f"ah_band={d.classes['alignment_heads_band']} words={len(d.timing['words'])} → {folder}/")
        return

    if args.cmd == "rehearse":
        d.tag = "rehearsal"
        folder = HERE / "runs" / "rehearsal"
        folder.mkdir(parents=True, exist_ok=True)
        d.run_all(folder)
        d.save(folder)
        print("\n下一步：./present.sh 自己按一遍 1 2 3 0 4 5 a；把 rehearsal.json → sections.classes.counts 與 alignment_heads_band、"
              "sections.transcript 的字錯數填進 p45 的表；./present.sh export 寫進 w04-metrics.json。")
        return

    out.banner("W4 Demo 3：attention heads" + (f"   [{d.tag}]" if d.tag else ""))
    d.menu()
    C.note("  鍵在視窗裡按（視窗要有焦點）。終端機這裡是紀錄。")
    try:
        d.plt.show()
    finally:
        if d.log["sections"] and args.cmd != "replay":
            d.save(HERE / "runs" / "live" / C.now().replace(":", ""))


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
# 此檔由 bin/publish_demos.py 從 private repo 的 <course>/demos/ 複製而來，請勿在公開 repo 直接編輯。
"""W3 demo 3：blank 後驗熱圖 + ctc_loss 的 inf／zero_infinity——本週的核心失敗案例。

  show       課堂：載入模型、算一句 LibriSpeech 的 logits，開一個空視窗；鍵在視窗裡按（下面的鍵）。離開時存 runs/live/<時間>/
  rehearse   課前：不互動，照課堂順序全部跑一遍，存 runs/rehearsal/（進版控 = 現場的退路）
  replay     現場模型出狀況時：從 runs/rehearsal/ 讀回 logits 與數字，同一個視窗、同樣的鍵，不需要 torch
  check      課前體檢：套件版本、模型在不在快取、MPS 上 ctc_loss 的前向／反向對 CPU、forced_align 還在不在（§6 待實測的那幾項）
  fetch      課前一週：下載模型（鎖 commit 進 model.lock）與那一句 LibriSpeech（要網路）
  fake       沒有模型也能演練流程：合成的 peaky 後驗（畫面標明 FAKE，不是給學生看的）

畫面上的鍵（順序照投影片講稿的【操作】，p37 與 p42）：
  1  播那句音訊                       2  熱圖：31 個非 blank 符號的後驗、argmax 一列、blank 一列放大
  3  四個數字：T；U、r、U + r；blank 是 argmax 的 frame；有聲段裡 blank 是 argmax 的 frame（填投影片 p37 的表）
  4  正確文字配 logits 的 ctc_loss（有限；同時印 numpy 在 ℓ' 上做 log 域 forward 的值——Demo 2 那條遞推，同一個數）
  5  文字接長到 U + r = T（仍有限）與 U + r = T + 1：loss = inf、梯度 = nan（zero_infinity=False）
  6  同一組、zero_infinity=True：loss = 0、梯度 = 0
  7  加分：ℓ' 上的 Viterbi（forward 的 Σ 換 max）→ 每個字元的尖峰、每個詞的尖峰跨度；有人工邊界就印偏了多少
  a  熱圖上疊 Viterbi 的尖峰位置       h  重印選單        q  離開

數字的定義：T = 模型 conv 前端輸出的 frame 數（20 ms 一格）；U = 標籤序列長度，**含詞界 |**（模型的符號表裡空白是一個符號，
$\\mathcal{V}$ 先於 loss）；r = 相鄰重複數；blank = config.pad_token_id。loss 用 reduction="sum"、batch 一句，所以就是 −ln P(ℓ|X)。
"""
from __future__ import annotations

import argparse
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
import dsp          # noqa: E402

CFG = tomllib.loads((HERE / "demo_config.toml").read_text(encoding="utf-8"))
FS = CFG["audio"]["fs"]
DIG = CFG["display"]["digits"]
MODEL_LOCK = HERE / "model.lock"
MEDIA = HERE.parent / "_media"

# wav2vec2-base-960h 的 vocab.json 順序（真的跑時從 tokenizer 讀；fake 與 replay 沒有 tokenizer 時用這份；check 會比對）
FAKE_VOCAB = ["<pad>", "<s>", "</s>", "<unk>", "|", "E", "T", "A", "O", "N", "I", "H", "S", "R", "D", "L", "U",
              "M", "W", "C", "F", "G", "Y", "P", "B", "V", "K", "'", "X", "J", "Q", "Z"]
FAKE_TEXT = "THIS IS A FAKE SENTENCE WITH NO SOUND IN IT"


# ── 純 numpy 的 CTC：ℓ'、forward（log 域）、Viterbi ─────────────────────
def extend(ids: list[int], blank: int) -> list[int]:
    """ℓ' = (∅, ℓ_1, ∅, …, ℓ_U, ∅)，S = 2U + 1。"""
    lp = [blank]
    for c in ids:
        lp += [c, blank]
    return lp


def repeats(ids: list[int]) -> int:
    return sum(1 for u in range(len(ids) - 1) if ids[u] == ids[u + 1])


def _allow3(lp: list[int], s: int, blank: int) -> bool:
    return s >= 2 and lp[s] != blank and lp[s] != lp[s - 2]


def log_forward(logp: np.ndarray, ids: list[int], blank: int) -> float:
    """log P(ℓ|X)。logp 的形狀是 (T, K)。與 Demo 2 的遞推逐字相同，只是在 log 域（np.logaddexp）。"""
    lp = extend(ids, blank)
    T, S = logp.shape[0], len(lp)
    NEG = -np.inf
    a = np.full(S, NEG)
    a[0] = logp[0, lp[0]]
    if S > 1:
        a[1] = logp[0, lp[1]]
    for t in range(1, T):
        b = np.full(S, NEG)
        for s in range(S):
            v = a[s]
            if s >= 1:
                v = np.logaddexp(v, a[s - 1])
            if _allow3(lp, s, blank):
                v = np.logaddexp(v, a[s - 2])
            b[s] = logp[t, lp[s]] + v
        a = b
    return float(np.logaddexp(a[S - 1], a[S - 2])) if S > 1 else float(a[0])


def viterbi(logp: np.ndarray, ids: list[int], blank: int) -> np.ndarray:
    """同一個格點、Σ 換成 max、記 backpointer。回傳每個 frame 停在 ℓ' 的哪個 state（長度 T；沒有合法路徑時回傳空陣列）。"""
    lp = extend(ids, blank)
    T, S = logp.shape[0], len(lp)
    NEG = -np.inf
    d = np.full((T, S), NEG)
    bp = np.zeros((T, S), dtype=np.int64)
    d[0, 0] = logp[0, lp[0]]
    if S > 1:
        d[0, 1] = logp[0, lp[1]]
    for t in range(1, T):
        for s in range(S):
            cands = [(d[t - 1, s], s)]
            if s >= 1:
                cands.append((d[t - 1, s - 1], s - 1))
            if _allow3(lp, s, blank):
                cands.append((d[t - 1, s - 2], s - 2))
            v, k = max(cands)
            d[t, s] = logp[t, lp[s]] + v
            bp[t, s] = k
    ends = [S - 1, S - 2] if S > 1 else [0]
    s = max(ends, key=lambda k: d[T - 1, k])
    if d[T - 1, s] == NEG:
        return np.zeros(0, dtype=np.int64)
    path = np.zeros(T, dtype=np.int64)
    for t in range(T - 1, -1, -1):
        path[t] = s
        s = bp[t, s]
    return path


def collapse(argmax: np.ndarray, blank: int) -> list[int]:
    out, prev = [], None
    for k in argmax.tolist():
        if k != prev and k != blank:
            out.append(k)
        prev = k
    return out


# ── 文字 ↔ id ──────────────────────────────────────────────────
class Vocab:
    """符號表。真的跑時由 tokenizer 給；fake／replay 用 FAKE_VOCAB。"""

    def __init__(self, tokens: list[str], blank: int, word_delim: str = "|"):
        self.tokens, self.blank, self.delim = tokens, blank, word_delim
        self.index = {t: i for i, t in enumerate(tokens)}
        self.unk = self.index.get("<unk>", blank)

    def encode(self, text: str) -> list[int]:
        ids = []
        for ch in text.upper().strip():
            ids.append(self.index[self.delim] if ch == " " else self.index.get(ch, self.unk))
        return ids

    def decode(self, ids) -> str:
        return "".join(" " if self.tokens[i] == self.delim else self.tokens[i] for i in ids)

    def sym(self, i: int) -> str:
        t = self.tokens[i]
        return "∅" if i == self.blank else ("␣" if t == self.delim else t)


def letters_only(text: str) -> int:
    return len(re.sub(r"[\s\W_]", "", text, flags=re.UNICODE))


def lengthen(vocab: Vocab, text: str, T: int) -> tuple[list[int], list[int], str]:
    """同一句重複接在後面、逐字元加到 U + r = T 與 U + r = T + 1。回傳 (ids_at_T, ids_at_T1, 用到的文字)。"""
    sep = CFG["too_long"]["separator"]
    src = text.strip()
    stream = src
    ids: list[int] = []
    at_T = None
    pos = 0
    while True:
        if pos >= len(stream):
            stream += sep + src
        ch = stream[pos]
        pos += 1
        ids = vocab.encode(stream[:pos])
        ur = len(ids) + repeats(ids)
        if ur == T and at_T is None:
            at_T = list(ids)
        if ur > T:
            return (at_T if at_T is not None else ids[:-1]), ids, stream[:pos]


# ── 後端 ───────────────────────────────────────────────────────
def model_dir(offline: bool = True) -> str:
    from huggingface_hub import snapshot_download
    rev = MODEL_LOCK.read_text(encoding="utf-8").strip() if MODEL_LOCK.exists() else None
    return snapshot_download(CFG["model"]["repo"], revision=rev, local_files_only=offline,
                             allow_patterns=["*.json", "*.txt", "*.safetensors"])


class Real:
    """transformers 的 Wav2Vec2ForCTC。logits (T, K) 用 numpy 交出去，torch 只在 ctc_loss 與推論時碰。"""
    name = "wav2vec2"

    def __init__(self, device: str):
        import torch
        from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
        self.torch = torch
        path = model_dir()
        self.proc = Wav2Vec2Processor.from_pretrained(path)
        self.model = Wav2Vec2ForCTC.from_pretrained(path).eval()
        if device == "auto":
            device = "mps" if torch.backends.mps.is_available() else "cpu"
        self.device = torch.device(device)
        self.model.to(self.device)
        tok = self.proc.tokenizer
        tokens = tok.convert_ids_to_tokens(list(range(len(tok))))
        self.vocab = Vocab(tokens, self.model.config.pad_token_id, tok.word_delimiter_token)
        self.n_params = sum(p.numel() for p in self.model.parameters())

    def logits(self, x: np.ndarray) -> np.ndarray:
        torch = self.torch
        inp = self.proc(x.astype(np.float32), sampling_rate=FS, return_tensors="pt")
        with torch.no_grad():
            out = self.model(inp.input_values.to(self.device)).logits[0]
        return out.float().cpu().numpy()

    def encode(self, text: str) -> list[int]:
        """用 tokenizer 本人編碼（空白 → |），與 Vocab.encode 應相同；check 會比對。"""
        return list(self.proc.tokenizer(text.upper().strip()).input_ids)


def torch_ctc(logits: np.ndarray, ids: list[int], blank: int, zero_infinity: bool, device: str = "cpu") -> dict:
    """torch.nn.functional.ctc_loss，batch 一句、reduction='sum' → loss = −ln P(ℓ|X)。順便 backward 看梯度。"""
    import torch
    import torch.nn.functional as F
    dev = torch.device(device)
    lg = torch.tensor(logits, dtype=torch.float32, device=dev, requires_grad=True)
    logp = torch.log_softmax(lg, dim=-1).unsqueeze(1)                     # (T, N=1, K)
    tgt = torch.tensor([ids], dtype=torch.long, device=dev)
    T, U = logits.shape[0], len(ids)
    t0 = time.perf_counter()
    loss = F.ctc_loss(logp, tgt, torch.tensor([T]), torch.tensor([U]), blank=blank,
                      reduction="sum", zero_infinity=zero_infinity)
    loss.backward()
    if dev.type == "mps":
        torch.mps.synchronize()
    dt = time.perf_counter() - t0
    g = lg.grad.detach().float().cpu().numpy()
    return {"loss": float(loss.item()), "grad_has_nan": bool(np.isnan(g).any()),
            "grad_all_zero": bool(np.all(g == 0)), "grad_absmax": float(np.nanmax(np.abs(g))) if not np.isnan(g).all() else float("nan"),
            "seconds": dt, "device": dev.type, "zero_infinity": zero_infinity}


def fake_ctc(logp: np.ndarray, ids: list[int], blank: int, zero_infinity: bool) -> dict:
    """沒有 torch 時：loss 用 numpy 的 log 域 forward；梯度只能模擬 inf → nan、zero_infinity → 0。"""
    lp = log_forward(logp, ids, blank)
    loss = -lp
    if math.isinf(loss) and zero_infinity:
        return {"loss": 0.0, "grad_has_nan": False, "grad_all_zero": True, "grad_absmax": 0.0, "seconds": 0.0, "device": "numpy", "zero_infinity": True}
    return {"loss": loss, "grad_has_nan": math.isinf(loss), "grad_all_zero": False,
            "grad_absmax": float("nan"), "seconds": 0.0, "device": "numpy", "zero_infinity": zero_infinity}


def fake_utterance() -> tuple[np.ndarray, str, np.ndarray, Vocab]:
    """合成：每個字元一小段母音、詞之間 80 ms 靜音、前後 300 ms 靜音；後驗是 peaky 的（每個字元一個尖峰，其餘 blank）。"""
    rng = np.random.default_rng(3)
    vocab = Vocab(FAKE_VOCAB, 0)
    text = FAKE_TEXT
    hop = int(FS * CFG["model"]["frame_ms"] / 1000)
    sil = np.zeros(int(0.3 * FS))
    pieces, spans = [sil], []                       # spans: (start_sample, end_sample) per label id
    for w, word in enumerate(text.split(" ")):
        if w:
            gap = np.zeros(int(0.08 * FS))
            spans.append((len(np.concatenate(pieces)), len(np.concatenate(pieces)) + len(gap), vocab.index["|"]))
            pieces.append(gap)
        for ch in word:
            dur = 0.07 + 0.05 * rng.random()
            f1, f2 = 300 + 500 * rng.random(), 1000 + 1200 * rng.random()
            seg = dsp.synth_vowel(FS, dur, 110 + 20 * rng.standard_normal(), (f1, f2, 2600)) * 0.4
            start = len(np.concatenate(pieces))
            spans.append((start, start + len(seg), vocab.index[ch]))
            pieces.append(seg)
    pieces.append(sil)
    x = np.concatenate(pieces)
    T = (len(x) - int(FS * CFG["model"]["receptive_ms"] / 1000)) // hop + 1
    K = len(FAKE_VOCAB)
    p = np.full((T, K), 0.002)
    p[:, vocab.blank] = 0.93
    for a, b, k in spans:
        ta, tb = a // hop, max(a // hop + 1, b // hop)
        t = int(ta + (tb - ta) * (0.15 + 0.7 * rng.random()))     # 尖峰落在音段裡的隨機位置（不在邊界）
        t = min(max(t, 0), T - 1)
        p[t, :] = 0.004
        p[t, k] = 0.82
        p[t, vocab.blank] = 0.10
        for dt_ in (-1, 1):                                          # 旁邊一格 blank 稍弱
            if 0 <= t + dt_ < T:
                p[t + dt_, k] = 0.12
                p[t + dt_, vocab.blank] = 0.80
    p /= p.sum(axis=1, keepdims=True)
    return x, text, np.log(p), vocab


# ── 有聲段、frame 對時間 ───────────────────────────────────────
def speech_mask(x: np.ndarray, T: int) -> np.ndarray:
    hop = int(FS * CFG["model"]["frame_ms"] / 1000)
    win = int(FS * CFG["speech"]["win_ms"] / 1000)
    e = np.array([np.mean(x[t * hop: t * hop + win] ** 2) + 1e-12 for t in range(T)])
    db = 10 * np.log10(e)
    return db > db.max() - CFG["speech"]["threshold_db"]


def frame_time(t) -> np.ndarray:
    """frame t 的中心時間（秒）：stride 20 ms、感受野 25 ms。"""
    return (np.asarray(t) * CFG["model"]["frame_ms"] + CFG["model"]["receptive_ms"] / 2) / 1000


# ── 終端機輸出（同時存成純文字，rehearse 用）──────────────────
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


# ── 主體 ───────────────────────────────────────────────────────
class Demo:
    def __init__(self, out: Out, x: np.ndarray, text: str, logits: np.ndarray, vocab: Vocab,
                 backend: str, tag: str = "", recorded: dict | None = None, wav_path: Path | None = None):
        self.out, self.x, self.text, self.logits, self.vocab, self.backend, self.tag = out, x, text, logits, vocab, backend, tag
        self.recorded = recorded or {}            # replay：彩排時的 4／5／6／7 結果
        self.wav_path = wav_path
        self.T, self.K = logits.shape
        m = logits.max(axis=1, keepdims=True)
        self.logp = logits - m - np.log(np.exp(logits - m).sum(axis=1, keepdims=True))
        self.prob = np.exp(self.logp)
        self.ids = vocab.encode(text)
        self.argmax = self.prob.argmax(axis=1)
        self.speech = speech_mask(x, self.T)
        self.path: np.ndarray | None = None
        self.log: dict = {"time": C.now(), "backend": backend, "tag": tag, "text": text, "config": CFG,
                          "versions": self.versions(), "sections": {}}
        self.fig = None
        self.playing = False

    # -- 版本 --
    @staticmethod
    def versions() -> dict:
        v = C.versions()
        from importlib import metadata
        for pkg in ("torch", "torchaudio", "transformers", "huggingface_hub", "soundfile"):
            try:
                v[pkg] = metadata.version(pkg)
            except metadata.PackageNotFoundError:
                v[pkg] = None
        if MODEL_LOCK.exists():
            v["model_commit"] = MODEL_LOCK.read_text(encoding="utf-8").strip()
        return v

    # -- 1 播放 --
    def play(self) -> None:
        if self.wav_path is None or not self.wav_path.exists():
            self.out(f"{C.AMBER}沒有音檔可播（{self.backend}）。{C.RESET}")
            return
        if self.playing:
            return
        self.playing = True
        self.out(f"播放 {self.wav_path.name}（{len(self.x) / FS:.2f} s）……")

        def run():
            C.play(self.wav_path)
            self.playing = False
        threading.Thread(target=run, daemon=True).start()

    # -- 2 熱圖 --
    def rows(self) -> list[int]:
        """非 blank 的符號列，由上而下：詞界 |、A–Z、'、其餘特殊符號。"""
        toks = self.vocab.tokens
        letters = sorted(i for i, t in enumerate(toks) if len(t) == 1 and t.isalpha())
        delim = [i for i, t in enumerate(toks) if t == self.vocab.delim]
        apos = [i for i, t in enumerate(toks) if t == "'"]
        rest = [i for i in range(self.K) if i not in letters and i not in delim and i not in apos and i != self.vocab.blank]
        return delim + letters + apos + rest

    def make_fig(self):
        import matplotlib
        import matplotlib.pyplot as plt
        self.plt = plt
        d = CFG["display"]
        self.fig = plt.figure(figsize=(d["fig_w"], d["fig_h"]))
        try:
            self.fig.canvas.manager.set_window_title("W3 Demo 3 — blank heat-map")
        except Exception:
            pass
        self.fig.canvas.mpl_connect("key_press_event", self.on_key)
        self.axes = None
        self.hint = self.fig.text(0.5, 0.5, "1  play        2  heat-map        3  numbers\n\n(ask the question first)",
                                  ha="center", va="center", fontsize=22, color="0.35")
        self.tagtxt = self.fig.text(0.99, 0.985, self.tag, ha="right", va="top", fontsize=11, color="#d1345b", fontweight="bold")
        self.summary = self.fig.text(0.01, 0.985, "", ha="left", va="top", fontsize=12, fontweight="bold", family="monospace")
        self.losstxt = self.fig.text(0.01, 0.005, "", ha="left", va="bottom", fontsize=12.5, family="monospace")
        self.spike_artists = []
        return self.fig

    def heatmap(self) -> None:
        if self.fig is None:
            self.make_fig()
        if self.axes is not None:
            return
        self.hint.set_visible(False)
        plt = self.plt
        gs = self.fig.add_gridspec(3, 1, height_ratios=[6.0, 0.8, 2.2], hspace=0.10, left=0.06, right=0.985, top=0.90, bottom=0.20)
        ax0, ax1, ax2 = [self.fig.add_subplot(gs[i, 0]) for i in range(3)]
        self.axes = (ax0, ax1, ax2)
        rows = self.rows()
        dur = frame_time(self.T - 1) + CFG["model"]["frame_ms"] / 2000
        t0 = frame_time(0) - CFG["model"]["frame_ms"] / 2000
        ax0.imshow(self.prob[:, rows].T, aspect="auto", origin="upper", cmap=CFG["display"]["cmap"], vmin=0, vmax=1,
                   extent=[t0, dur, len(rows) - 0.5, -0.5], interpolation="nearest")
        ax0.set_yticks(range(len(rows)))
        ax0.set_yticklabels([self.vocab.sym(i) for i in rows], fontsize=8.5, family="monospace")
        ax0.set_title(f"posterior $y^t_k$ — one column per frame ({CFG['model']['frame_ms']:g} ms), blank row removed and enlarged below",
                      loc="left", fontsize=12.5, fontweight="bold")
        ax0.tick_params(labelbottom=False)
        # argmax 一列
        ax1.set_xlim(t0, dur)
        ax1.set_ylim(0, 1)
        ax1.set_yticks([])
        ax1.set_ylabel("argmax", fontsize=9, rotation=0, ha="right", va="center")
        for t in range(self.T):
            k = self.argmax[t]
            if k != self.vocab.blank:
                ax1.text(frame_time(t), 0.5, self.vocab.sym(k), ha="center", va="center", fontsize=9, family="monospace", color="#1f4e79")
        ax1.tick_params(labelbottom=False)
        # blank 一列放大
        tt = frame_time(np.arange(self.T))
        pb = self.prob[:, self.vocab.blank]
        ax2.fill_between(tt, 0, pb, step="mid", color="#d1345b", alpha=0.35, lw=0)
        ax2.step(tt, pb, where="mid", color="#d1345b", lw=1.2)
        ax2.axhline(0.5, color="0.4", lw=0.8, ls=":")
        ax2.set_ylim(0, 1)
        ax2.set_xlim(t0, dur)
        ax2.set_ylabel("P(∅ | frame)", fontsize=10)
        ax2.set_xlabel("time (s)", fontsize=10)
        # 無聲段灰底（兩個 panel 都畫）
        for ax in (ax0, ax2):
            t = 0
            while t < self.T:
                if not self.speech[t]:
                    u = t
                    while u < self.T and not self.speech[u]:
                        u += 1
                    ax.axvspan(frame_time(t) - CFG["model"]["frame_ms"] / 2000, frame_time(u - 1) + CFG["model"]["frame_ms"] / 2000,
                               color="0.6" if ax is ax2 else "0.75", alpha=0.35 if ax is ax2 else 0.0, lw=0, zorder=0)
                    if ax is ax0:
                        ax.axvspan(frame_time(t) - CFG["model"]["frame_ms"] / 2000, frame_time(u - 1) + CFG["model"]["frame_ms"] / 2000,
                                   facecolor="none", edgecolor="white", hatch="///", lw=0, alpha=0.25, zorder=3)
                    t = u
                else:
                    t += 1
        ax2.set_title("grey = below the energy threshold (no speech)", loc="right", fontsize=8.5, color="0.35")
        self.fig.canvas.draw_idle()

    # -- 3 四個數字 --
    def numbers(self) -> dict:
        b = self.vocab.blank
        blank_arg = self.argmax == b
        n_speech = int(self.speech.sum())
        U, r = len(self.ids), repeats(self.ids)
        greedy = self.vocab.decode(collapse(self.argmax, b))
        m = {"T": int(self.T), "frame_ms": CFG["model"]["frame_ms"], "seconds": round(len(self.x) / FS, 3),
             "U_letters": letters_only(self.text), "U": U, "r": r, "U_plus_r": U + r, "frames_per_label": round(self.T / U, 2),
             "blank_argmax": int(blank_arg.sum()), "blank_argmax_frac": round(float(blank_arg.mean()), 4),
             "speech_frames": n_speech, "blank_argmax_in_speech": int((blank_arg & self.speech).sum()),
             "blank_argmax_in_speech_frac": round(float((blank_arg & self.speech).sum() / max(n_speech, 1)), 4),
             "greedy": greedy, "greedy_matches_reference": greedy.strip() == self.text.upper().strip(),
             "min_blank_prob_in_speech": round(float(self.prob[self.speech, b].min()), 4) if n_speech else None,
             "median_blank_prob_in_speech": round(float(np.median(self.prob[self.speech, b])), 4) if n_speech else None}
        o = self.out
        o.banner("四個數字（填 p37 的表）", C.AMBER)
        o(f"  句子：「{self.text}」")
        o(f"  greedy decode：「{greedy}」   {'= 參考文字' if m['greedy_matches_reference'] else '≠ 參考文字'}")
        o(f"  frames T                          {m['T']:>6d}    （{m['seconds']:.2f} s，每 {m['frame_ms']:g} ms 一格）")
        o(f"  labels U（含詞界 |）, r, U + r     {U:>6d}  {r:>3d}  {U + r:>4d}    （只數字母是 {m['U_letters']}；T / U = {m['frames_per_label']}）")
        o(f"  frames where ∅ is the argmax      {m['blank_argmax']:>6d}    = {100 * m['blank_argmax_frac']:.1f} % of T")
        o(f"  … inside speech ({n_speech} speech frames) {m['blank_argmax_in_speech']:>6d}    = {100 * m['blank_argmax_in_speech_frac']:.1f} % of speech frames")
        o(f"  有聲段裡 P(∅) 的中位數 {m['median_blank_prob_in_speech']}、最小值 {m['min_blank_prob_in_speech']}")
        C.note("  有聲段 = 能量高於峰值以下 35 dB 的 frame（demo_config.toml [speech]）；U 含詞界，因為模型的符號表裡空白是一個符號。")
        self.log["sections"]["numbers"] = m
        if self.fig is not None:
            self.summary.set_text(f"T = {m['T']}   U = {U} (+ r = {r} → {U + r})   ∅ argmax: {m['blank_argmax']}/{m['T']} = {100 * m['blank_argmax_frac']:.0f} %   "
                                  f"inside speech: {m['blank_argmax_in_speech']}/{n_speech} = {100 * m['blank_argmax_in_speech_frac']:.0f} %")
            self.fig.canvas.draw_idle()
        return m

    # -- 4／5／6 ctc_loss --
    def ctc(self, ids: list[int], zero_infinity: bool) -> dict:
        key = f"ctc_{len(ids)}_{int(zero_infinity)}"
        if self.backend == "replay" and key in self.recorded:
            return dict(self.recorded[key], replayed=True)
        if self.backend in ("fake", "replay"):
            r = fake_ctc(self.logp, ids, self.vocab.blank, zero_infinity)
        else:
            r = torch_ctc(self.logits, ids, self.vocab.blank, zero_infinity, CFG["model"]["device"])
        self.recorded[key] = r
        return r

    def fmt_loss(self, v: float) -> str:
        return "inf" if math.isinf(v) else f"{v:.{DIG}f}"

    def grad_str(self, r: dict) -> str:
        if r["device"] == "numpy":
            return "nan（模擬）" if r["grad_has_nan"] else ("0（模擬）" if r["grad_all_zero"] else "—（fake：沒有 torch）")
        if r["grad_has_nan"]:
            return "nan"
        if r["grad_all_zero"]:
            return "0（全部）"
        return f"有限，max |∂L/∂logit| = {r['grad_absmax']:.{DIG}f}"

    def loss_panel(self) -> None:
        if self.fig is None:
            return
        rows = self.log["sections"]
        lines = []
        if "loss_correct" in rows:
            lines.append(f"correct transcript          U+r = {rows['loss_correct']['U_plus_r']:>4d} ≤ T   loss = {self.fmt_loss(rows['loss_correct']['torch']['loss']):>10s}   gradient finite")
        if "loss_too_long" in rows:
            a, b = rows["loss_too_long"]["at_T"], rows["loss_too_long"]["at_T_plus_1"]
            lines.append(f"longer, U+r = T             U+r = {a['U_plus_r']:>4d} = T   loss = {self.fmt_loss(a['loss']):>10s}   gradient finite")
            lines.append(f"longer, zero_infinity=False U+r = {b['U_plus_r']:>4d} > T   loss = {self.fmt_loss(b['loss']):>10s}   gradient nan")
        if "loss_zero_infinity" in rows:
            z = rows["loss_zero_infinity"]
            lines.append(f"longer, zero_infinity=True  U+r = {z['U_plus_r']:>4d} > T   loss = {self.fmt_loss(z['loss']):>10s}   gradient 0")
        self.losstxt.set_text("\n".join(lines))
        self.fig.canvas.draw_idle()

    def loss_correct(self) -> None:
        o = self.out
        o.banner("4  正確文字的 ctc_loss（batch 一句、reduction='sum' → −ln P(ℓ|X)）", C.BLUE)
        r = self.ctc(self.ids, False)
        lp = log_forward(self.logp, self.ids, self.vocab.blank)
        U, rr = len(self.ids), repeats(self.ids)
        o(f"  U + r = {U + rr} ≤ T = {self.T}")
        o(f"  torch.nn.functional.ctc_loss        = {self.fmt_loss(r['loss'])}    （{r['device']}，{1000 * r['seconds']:.1f} ms 含 backward）")
        o(f"  numpy 在 ℓ' 上 log 域 forward 的 −ln P = {-lp:.{DIG}f}    差 {abs(r['loss'] + lp):.2e}   ← Demo 2 那條遞推，同一個數")
        o(f"  每個 frame 平均 {r['loss'] / self.T:.{DIG}f} nats（〈Read it as a cross-entropy〉那張的讀法）")
        o(f"  梯度：{self.grad_str(r)}")
        self.log["sections"]["loss_correct"] = {"U_plus_r": U + rr, "T": self.T, "torch": r, "numpy_neg_logP": -lp, "per_frame": r["loss"] / self.T}
        self.loss_panel()

    def loss_too_long(self) -> None:
        o = self.out
        o.banner("5  文字接長到 U + r > T：zero_infinity=False", C.RED)
        ids_T, ids_T1, used = lengthen(self.vocab, self.text, self.T)
        n = CFG["display"]["max_label_text"]
        o(f"  接長的文字（{len(used)} 個字元）：「{used[:n]}{'…' if len(used) > n else ''}」")
        a = self.ctc(ids_T, False)
        b = self.ctc(ids_T1, False)
        ua, ub = len(ids_T) + repeats(ids_T), len(ids_T1) + repeats(ids_T1)
        o(f"  U + r = {ua} {'=' if ua == self.T else '<'} T = {self.T}     loss = {self.fmt_loss(a['loss'])}    梯度：{self.grad_str(a)}    （還有合法路徑，還有限）")
        o(f"  U + r = {ub} > T = {self.T}     loss = {self.fmt_loss(b['loss'])}    梯度：{self.grad_str(b)}    ← 沒有任何合法路徑：P = 0")
        C.note("  條件是精確的：差一個符號就從有限跳到 inf。預設 reduction='mean' 時一句 inf 會把整批的平均變成 inf、梯度整批 nan。")
        self.log["sections"]["loss_too_long"] = {"text_used": used, "at_T": dict(a, U_plus_r=ua), "at_T_plus_1": dict(b, U_plus_r=ub)}
        self.loss_panel()

    def loss_zero_infinity(self) -> None:
        o = self.out
        o.banner("6  同一組、zero_infinity=True", C.RED)
        _, ids_T1, _ = lengthen(self.vocab, self.text, self.T)
        z = self.ctc(ids_T1, True)
        ub = len(ids_T1) + repeats(ids_T1)
        o(f"  U + r = {ub} > T = {self.T}     loss = {self.fmt_loss(z['loss'])}    梯度：{self.grad_str(z)}")
        C.note("  這句話從此對訓練沒有貢獻，而且沒有任何訊息告訴你。止血帶，不是治療。")
        self.log["sections"]["loss_zero_infinity"] = dict(z, U_plus_r=ub)
        self.loss_panel()

    # -- 7 forced alignment --
    def align(self) -> dict:
        o = self.out
        o.banner("7  加分：ℓ' 上的 Viterbi（Σ → max）——每個字元的尖峰、每個詞的尖峰跨度", C.BLUE)
        if self.backend == "replay" and "align" in self.recorded:
            res = self.recorded["align"]
            self.path = np.asarray(res["path"], dtype=np.int64)
        else:
            t0 = time.perf_counter()
            self.path = viterbi(self.logp, self.ids, self.vocab.blank)
            res = {"seconds_numpy": time.perf_counter() - t0}
            if self.path.size == 0:
                o("  沒有合法路徑（T < U + r）。")
                return res
            res["path"] = self.path.tolist()
            res["torchaudio_forced_align"] = self.check_torchaudio_align()
            self.recorded["align"] = res
        path = self.path
        # 每個 label u 的 state 是 2u+1；尖峰 = 停在該 state 的 frame（可能不只一格）
        runs = []
        for u, k in enumerate(self.ids):
            fr = np.flatnonzero(path == 2 * u + 1)
            runs.append((u, k, int(fr[0]), int(fr[-1])))
        run_len = [b - a + 1 for _, _, a, b in runs]
        res["labels"] = [{"u": u, "token": self.vocab.sym(k), "first_frame": a, "last_frame": b, "t_first": round(float(frame_time(a)), 3)} for u, k, a, b in runs]
        res["run_length_mean"] = round(float(np.mean(run_len)), 3)
        res["run_length_is_1_frac"] = round(float(np.mean([l == 1 for l in run_len])), 3)
        # 詞：從第一個字元的尖峰到最後一個字元的尖峰
        words, cur = [], []
        for u, k, a, b in runs:
            if self.vocab.tokens[k] == self.vocab.delim:
                if cur:
                    words.append(cur)
                cur = []
            else:
                cur.append((u, k, a, b))
        if cur:
            words.append(cur)
        res["words"] = []
        o(f"  {len(self.ids)} 個標籤，尖峰平均寬 {res['run_length_mean']} 格，{100 * res['run_length_is_1_frac']:.0f} % 只有一格（peaky）")
        o(f"  {'word':<12s}{'first spike':>12s}{'last spike':>12s}{'span':>8s}   spikes (s)")
        for w in words:
            word = "".join(self.vocab.tokens[k] for _, k, _, _ in w)
            ta, tb = float(frame_time(w[0][2])), float(frame_time(w[-1][3]))
            spikes = " ".join(f"{self.vocab.tokens[k]}@{float(frame_time(a)):.2f}" for _, k, a, _ in w)
            o(f"  {word:<12s}{ta:>11.2f}s{tb:>11.2f}s{tb - ta:>7.2f}s   {spikes}")
            res["words"].append({"word": word, "t_first_spike": round(ta, 3), "t_last_spike": round(tb, 3)})
        manual = CFG["forced_align"]["manual"]
        if manual:
            o(f"\n  對人工標的詞邊界（demo_config.toml [forced_align].manual）：")
            o(f"  {'word':<12s}{'manual':>16s}{'spikes':>16s}{'Δstart':>9s}{'Δend':>9s}")
            res["manual"] = []
            for m in manual:
                hit = [w for w in res["words"] if w["word"] == m["word"].upper()]
                if not hit:
                    o(f"  {m['word']:<12s}   （句子裡沒有這個詞）")
                    continue
                w = hit[0]
                ds, de = w["t_first_spike"] - m["start"], w["t_last_spike"] - m["end"]
                o(f"  {w['word']:<12s}{m['start']:>7.2f}–{m['end']:<7.2f}{w['t_first_spike']:>7.2f}–{w['t_last_spike']:<7.2f}{ds:>+8.2f}s{de:>+8.2f}s")
                res["manual"].append({"word": w["word"], "start": m["start"], "end": m["end"], "d_start": round(ds, 3), "d_end": round(de, 3)})
            C.note("  Δstart > 0：第一個尖峰比詞的起點晚；Δend < 0：最後一個尖峰比詞的終點早。尖峰在詞裡面，但不在邊界上。")
        else:
            C.note("  沒有人工邊界可比。要比就把 Praat 讀到的一兩個詞的起訖填進 demo_config.toml [forced_align].manual。")
        ta = res.get("torchaudio_forced_align")
        if ta:
            o(f"  torchaudio.functional.forced_align：{ta['status']}" + (f"，與 numpy Viterbi 逐格{'相同' if ta.get('same_path') else '不同'}" if "same_path" in ta else ""))
        self.log["sections"]["align"] = {k: v for k, v in res.items() if k != "path"}
        return res

    def check_torchaudio_align(self) -> dict:
        if self.backend in ("fake", "replay"):
            return {"status": "（fake／replay：沒試）"}
        try:
            import torch
            import torchaudio
            from torchaudio.functional import forced_align
        except Exception as e:
            return {"status": f"不在（{type(e).__name__}: {e}）"}
        try:
            lp = torch.tensor(self.logp, dtype=torch.float32).unsqueeze(0)
            tgt = torch.tensor([self.ids], dtype=torch.long)
            labels, scores = forced_align(lp, tgt, blank=self.vocab.blank)
            lab = labels[0].numpy()
            # torchaudio 回的是每個 frame 的 token id；numpy 的 path 是 state。比較「每個 frame 的 token」
            lpr = extend(self.ids, self.vocab.blank)
            mine = np.array([lpr[s] for s in self.path])
            same = bool(np.array_equal(lab, mine))
            return {"status": f"還在（torchaudio {torchaudio.__version__}）", "same_path": same,
                    "frames_differ": int((lab != mine).sum())}
        except Exception as e:
            return {"status": f"呼叫失敗（{type(e).__name__}: {e}）"}

    def overlay_spikes(self) -> None:
        if self.fig is None or self.axes is None:
            return
        if self.path is None or self.path.size == 0:
            self.align()
        if self.path is None or self.path.size == 0:
            return
        if self.spike_artists:
            for a in self.spike_artists:
                a.remove()
            self.spike_artists = []
            self.fig.canvas.draw_idle()
            return
        ax0 = self.axes[0]
        rows = self.rows()
        pos = {k: i for i, k in enumerate(rows)}
        for u, k in enumerate(self.ids):
            fr = np.flatnonzero(self.path == 2 * u + 1)
            for t in fr:
                self.spike_artists.append(ax0.plot(frame_time(t), pos[k], marker="s", ms=5, mfc="none", mec="#3ddc97", mew=1.3)[0])
        self.spike_artists.append(ax0.text(0.995, 0.02, "green = Viterbi path (forced alignment)", transform=ax0.transAxes,
                                           ha="right", va="bottom", fontsize=9, color="#3ddc97"))
        self.fig.canvas.draw_idle()

    # -- 選單、鍵 --
    def menu(self) -> None:
        o = self.out
        o(f"\n{C.BOLD}p37{C.RESET}  1 播音訊   2 熱圖   3 四個數字        {C.BOLD}p42{C.RESET}  4 正確文字 loss   5 接長 → inf   6 zero_infinity")
        o(f"{C.BOLD}加分{C.RESET}  7 Viterbi 尖峰   a 疊到熱圖上         h 選單   q 離開")

    def dispatch(self, k: str) -> bool:
        if k == "1":
            self.play()
        elif k == "2":
            self.heatmap()
        elif k == "3":
            self.numbers()
        elif k == "4":
            self.loss_correct()
        elif k == "5":
            self.loss_too_long()
        elif k == "6":
            self.loss_zero_infinity()
        elif k == "7":
            self.align()
        elif k == "a":
            self.overlay_spikes()
        elif k == "h":
            self.menu()
        elif k == "q":
            return False
        return True

    def on_key(self, ev) -> None:
        if not self.dispatch(ev.key):
            self.plt.close(self.fig)

    def run_all(self) -> None:
        """課堂順序全部跑一遍（rehearse 用；不播音訊）。"""
        self.heatmap()
        self.numbers()
        self.loss_correct()
        self.loss_too_long()
        self.loss_zero_infinity()
        self.align()
        self.overlay_spikes()

    def save(self, folder: Path) -> None:
        folder.mkdir(parents=True, exist_ok=True)
        stem = "rehearsal" if folder.name == "rehearsal" else "session"
        np.save(folder / "logits.npy", self.logits.astype(np.float32))
        if self.wav_path and self.wav_path.exists() and not (folder / "utterance.wav").exists():
            shutil.copy(self.wav_path, folder / "utterance.wav")
        (folder / "utterance.txt").write_text(self.text + "\n", encoding="utf-8")
        (folder / "vocab.json").write_text(json.dumps({"tokens": self.vocab.tokens, "blank": self.vocab.blank, "delim": self.vocab.delim},
                                                      ensure_ascii=False), encoding="utf-8")
        self.log["recorded"] = self.recorded
        (folder / f"{stem}.json").write_text(json.dumps(self.log, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
        (folder / f"{stem}.txt").write_text(self.out.text(), encoding="utf-8")
        if self.fig is not None:
            self.fig.savefig(folder / "heatmap.png", dpi=130)
        C.note(f"  存到 {folder}/（logits.npy、utterance.wav／.txt、vocab.json、heatmap.png、{stem}.json、{stem}.txt）")


# ── 載入：真的、fake、replay ───────────────────────────────────
def load_audio(wav: Path) -> tuple[np.ndarray, str]:
    if not wav.exists():
        sys.exit(f"找不到 {wav}。先 ./present.sh fetch（或 --wav 指到自己的 wav，文字放同名 .txt）。")
    txt = wav.with_suffix(".txt")
    if not txt.exists():
        sys.exit(f"找不到 {txt}（那句話的文字，一行）。")
    return C.load_wav(wav, FS), txt.read_text(encoding="utf-8").strip()


def build(cmd: str, wav_arg: str | None, out: Out) -> Demo:
    if cmd == "fake":
        x, text, logp, vocab = fake_utterance()
        wav = HERE / "audio" / "fake.wav"
        C.save_wav(wav, x, FS, float(np.max(np.abs(x))) + 1e-9)
        (wav.with_suffix(".txt")).write_text(text + "\n", encoding="utf-8")
        return Demo(out, x, text, logp, vocab, "fake", "FAKE — synthetic posteriors, not a model", wav_path=wav)
    if cmd == "replay":
        folder = HERE / "runs" / "rehearsal"
        if not (folder / "logits.npy").exists():
            sys.exit(f"沒有彩排紀錄 {folder}/logits.npy。先跑 ./present.sh rehearse。")
        logits = np.load(folder / "logits.npy")
        v = json.loads((folder / "vocab.json").read_text(encoding="utf-8"))
        vocab = Vocab(v["tokens"], v["blank"], v["delim"])
        wav = folder / "utterance.wav"
        x, text = load_audio(wav) if wav.exists() else (np.zeros(int(FS * logits.shape[0] * CFG["model"]["frame_ms"] / 1000)),
                                                       (folder / "utterance.txt").read_text(encoding="utf-8").strip())
        rec = json.loads((folder / "rehearsal.json").read_text(encoding="utf-8")).get("recorded", {})
        return Demo(out, x, text, logits, vocab, "replay", "REPLAY — pre-computed in rehearsal, not live", recorded=rec, wav_path=wav if wav.exists() else None)
    wav = Path(wav_arg) if wav_arg else HERE / CFG["audio"]["wav"]
    x, text = load_audio(wav)
    C.note(f"載入 {CFG['model']['repo']}（{CFG['model']['device']}）……")
    be = Real(CFG["model"]["device"])
    t0 = time.perf_counter()
    logits = be.logits(x)
    C.note(f"logits ready（{1000 * (time.perf_counter() - t0):.0f} ms，{be.n_params / 1e6:.1f} M 參數，符號表 {len(be.vocab.tokens)}，blank = id {be.vocab.blank}）")
    d = Demo(out, x, text, logits, be.vocab, "wav2vec2", wav_path=wav)
    d.log["model"] = {"repo": CFG["model"]["repo"], "n_params": be.n_params, "device": str(be.device), "K": len(be.vocab.tokens), "blank": be.vocab.blank}
    if be.encode(text) != d.ids:
        C.note(f"{C.AMBER}注意：tokenizer 的編碼與 Vocab.encode 不同（tokenizer {len(be.encode(text))} 個 id、自己的 {len(d.ids)} 個）。用 tokenizer 的。{C.RESET}")
        d.ids = be.encode(text)
    return d


# ── check ──────────────────────────────────────────────────────
def check() -> None:
    from importlib import metadata
    C.banner("課前體檢", C.AMBER)
    v = Demo.versions()
    for k in ("python", "platform", "numpy", "matplotlib", "torch", "torchaudio", "transformers", "huggingface_hub", "soundfile", "model_commit"):
        print(f"  {k:<16s} {v.get(k)}")
    res = {"time": C.now(), "versions": v}
    try:
        import torch
    except ImportError:
        print("  沒有 torch：先跑 ./setup.sh")
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
    wav = HERE / CFG["audio"]["wav"]
    if not wav.exists():
        print(f"  {'audio':<16s} 找不到 {wav}，先跑 ./present.sh fetch")
        return
    x, text = load_audio(wav)
    out = Out()
    d = None
    for dev in ["cpu"] + (["mps"] if mps else []):
        try:
            be = Real(dev)
            be.logits(x)                                   # 暖機
            ts = []
            for _ in range(3):
                t0 = time.perf_counter()
                lg = be.logits(x)
                ts.append(time.perf_counter() - t0)
            print(f"  推論 {dev:<4s} {1000 * min(ts):.0f} ms（3 次取最快）  T = {lg.shape[0]}  K = {lg.shape[1]}")
            res[f"infer_{dev}"] = {"ms": round(1000 * min(ts), 1), "T": int(lg.shape[0]), "K": int(lg.shape[1])}
            if dev == "cpu":
                d = Demo(out, x, text, lg, be.vocab, "wav2vec2", "check")
                res["n_params"] = be.n_params
                res["vocab_matches_FAKE_VOCAB"] = be.vocab.tokens == FAKE_VOCAB
                res["tokenizer_matches_Vocab_encode"] = be.encode(text) == d.ids
                L = len(x)
                T_pred = (L - int(FS * CFG["model"]["receptive_ms"] / 1000)) // int(FS * CFG["model"]["frame_ms"] / 1000) + 1
                res["T_predicted_from_stride"] = int(T_pred)
                print(f"  參數 {be.n_params / 1e6:.2f} M；符號表與 FAKE_VOCAB {'相同' if res['vocab_matches_FAKE_VOCAB'] else '不同'}；"
                      f"tokenizer 編碼與自己的 {'相同' if res['tokenizer_matches_Vocab_encode'] else '不同'}；"
                      f"T 由 stride 預測 {T_pred}、實際 {lg.shape[0]}")
                cpu_logits = lg
            else:
                res["infer_mps_vs_cpu_maxabs"] = float(np.max(np.abs(lg - cpu_logits)))
                print(f"  mps 與 cpu 的 logits 最大差 {res['infer_mps_vs_cpu_maxabs']:.2e}")
        except Exception as e:
            print(f"  推論 {dev}：失敗 {type(e).__name__}: {e}")
            res[f"infer_{dev}"] = {"error": f"{type(e).__name__}: {e}"}
    if d is None:
        return
    ids_T, ids_T1, _ = lengthen(d.vocab, text, d.T)
    cases = [("correct", d.ids, False), ("U+r=T", ids_T, False), ("U+r=T+1", ids_T1, False), ("U+r=T+1 zero_inf", ids_T1, True)]
    res["ctc"] = {}
    for dev in ["cpu"] + (["mps"] if mps else []):
        for name, ids, zi in cases:
            try:
                r = torch_ctc(d.logits, ids, d.vocab.blank, zi, dev)
                print(f"  ctc_loss {dev:<4s} {name:<18s} loss = {d.fmt_loss(r['loss']):>12s}  grad nan={r['grad_has_nan']} zero={r['grad_all_zero']}  {1000 * r['seconds']:.1f} ms")
                res["ctc"][f"{dev}/{name}"] = r
            except Exception as e:
                print(f"  ctc_loss {dev:<4s} {name:<18s} 失敗 {type(e).__name__}: {e}")
                res["ctc"][f"{dev}/{name}"] = {"error": f"{type(e).__name__}: {e}"}
    if mps and "cpu/correct" in res["ctc"] and "mps/correct" in res["ctc"] and "loss" in res["ctc"]["mps/correct"]:
        print(f"  ctc_loss mps 對 cpu：loss 差 {abs(res['ctc']['cpu/correct']['loss'] - res['ctc']['mps/correct']['loss']):.2e}")
    lp = log_forward(d.logp, d.ids, d.vocab.blank)
    print(f"  numpy log 域 forward 的 −ln P = {-lp:.{DIG}f}（對 cpu/correct 差 {abs(res['ctc']['cpu/correct']['loss'] + lp):.2e}）")
    res["numpy_neg_logP"] = -lp
    d.path = viterbi(d.logp, d.ids, d.vocab.blank)
    res["torchaudio_forced_align"] = d.check_torchaudio_align()
    print(f"  torchaudio forced_align：{res['torchaudio_forced_align']['status']}"
          + (f"，與 numpy Viterbi 逐格{'相同' if res['torchaudio_forced_align'].get('same_path') else '不同'}" if "same_path" in res["torchaudio_forced_align"] else ""))
    (HERE / "runs").mkdir(exist_ok=True)
    (HERE / "runs" / "check.json").write_text(json.dumps(res, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    C.note("  結果存在 runs/check.json（進版控：這是 §6「MPS 上的 ctc_loss」與「forced_align 還在不在」的實測紀錄）")


# ── fetch ──────────────────────────────────────────────────────
def fetch() -> None:
    import tarfile
    import urllib.request
    from huggingface_hub import HfApi, snapshot_download
    C.banner("下載模型與那一句 LibriSpeech", C.AMBER)
    repo = CFG["model"]["repo"]
    sha = MODEL_LOCK.read_text(encoding="utf-8").strip() if MODEL_LOCK.exists() else HfApi().model_info(repo).sha
    path = snapshot_download(repo, revision=sha, allow_patterns=["*.json", "*.txt", "*.safetensors"])
    MODEL_LOCK.write_text(sha + "\n", encoding="utf-8")
    print(f"  {repo} @ {sha[:12]} → {path}\n  commit 寫進 model.lock（進版控）")

    utt = CFG["audio"]["utterance"]
    spk, chap, _ = utt.split("-")
    tgz = MEDIA / Path(CFG["audio"]["librispeech_url"]).name
    MEDIA.mkdir(exist_ok=True)
    if not tgz.exists():
        print(f"  下載 {CFG['audio']['librispeech_url']}（約 337 MB）→ {tgz} ……")
        urllib.request.urlretrieve(CFG["audio"]["librispeech_url"], tgz)
    want_flac = f"LibriSpeech/dev-clean/{spk}/{chap}/{utt}.flac"
    want_txt = f"LibriSpeech/dev-clean/{spk}/{chap}/{spk}-{chap}.trans.txt"
    flac_bytes = txt_bytes = None
    with tarfile.open(tgz, "r|gz") as tf:                  # 串流讀，不整包解開
        for m in tf:
            if m.name == want_flac:
                flac_bytes = tf.extractfile(m).read()
            elif m.name == want_txt:
                txt_bytes = tf.extractfile(m).read()
            if flac_bytes and txt_bytes:
                break
    if not (flac_bytes and txt_bytes):
        sys.exit(f"tar 裡找不到 {want_flac} 或 {want_txt}")
    line = next(l for l in txt_bytes.decode().splitlines() if l.startswith(utt + " "))
    text = line.split(" ", 1)[1].strip()
    import soundfile as sf
    x, fs_in = sf.read(io.BytesIO(flac_bytes), dtype="float64")
    x = dsp.resample(dsp.to_mono_float(x), fs_in, FS)
    wav = HERE / CFG["audio"]["wav"]
    C.save_wav(wav, x, FS)
    wav.with_suffix(".txt").write_text(text + "\n", encoding="utf-8")
    print(f"  {utt}：{len(x) / FS:.2f} s，「{text}」\n  → {wav}、{wav.with_suffix('.txt')}（進版控；CC BY 4.0，見 media-sources/w03-librispeech.md）")
    print("\n下一步：./present.sh check")


# ── main ───────────────────────────────────────────────────────
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["show", "rehearse", "replay", "check", "fetch", "fake"])
    ap.add_argument("--wav", help="換一句：wav 路徑（16 kHz 以外會重取樣），文字放同名 .txt")
    ap.add_argument("--selftest", metavar="PNG", help="不開視窗：照課堂順序全部跑一遍、存圖到 PNG、印數字（測試用）")
    args = ap.parse_args()

    if args.cmd == "check":
        check()
        return
    if args.cmd == "fetch":
        fetch()
        return

    if args.selftest or args.cmd == "rehearse":
        import matplotlib
        matplotlib.use("Agg")

    out = Out()
    d = build(args.cmd, args.wav, out)
    d.make_fig()

    if args.selftest:
        d.run_all()
        d.fig.savefig(args.selftest, dpi=110)
        m = d.log["sections"]["numbers"]
        print(f"selftest ok: T={m['T']} U+r={m['U_plus_r']} blank_argmax={m['blank_argmax']} in_speech={m['blank_argmax_in_speech']}/{m['speech_frames']}")
        return

    if args.cmd == "rehearse":
        d.tag = "rehearsal"
        d.run_all()
        d.save(HERE / "runs" / "rehearsal")
        print("\n下一步：./present.sh 自己按一遍 1 2 3 4 5 6 7；把 rehearsal.json → sections.numbers 的四個數字填進 p37 的表、"
              "loss_correct／loss_too_long／loss_zero_infinity 填進 p42 講稿。")
        return

    out.banner(f"W3 Demo 3：blank heat-map" + (f"   [{d.tag}]" if d.tag else ""))
    d.menu()
    C.note("  鍵在視窗裡按（視窗要有焦點）。終端機這裡是紀錄。")
    try:
        d.plt.show()
    finally:
        if d.log["sections"] and args.cmd != "replay":
            d.save(HERE / "runs" / "live" / C.now().replace(":", ""))


if __name__ == "__main__":
    main()

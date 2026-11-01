# -*- coding: utf-8 -*-
# 此檔由 bin/publish_demos.py 從 private repo 的 <course>/demos/ 複製而來，請勿在公開 repo 直接編輯。
"""ARPA 格式的詞級 n-gram LM（最高三元、Katz backoff），純 numpy；加上接在 CTC prefix beam search 上的 LMBonus。

    lm = ArpaLM.parse("3-gram.pruned.1e-7.arpa.gz")      # 第一次：讀文字檔（幾十秒）
    lm.save("lm.npz"); lm = ArpaLM.load("lm.npz")         # 之後：幾秒
    lm.logp("THE", ("<s>",))                               # ln P(THE | <s>)，不在表裡就 backoff

ARPA 裡的機率是 log10；這裡對外一律換成自然對數，與 CTC 的 log softmax 同一個底。
字典裡沒有的詞（OOV）當 <unk>，**再加 oov_penalty + oov_per_char × 字元數（ln；預設都是 0，demo 的 toml 給 -5 與 -3）**。
2026-11-01 彩排抓到的：LibriSpeech 的 ARPA 給 <UNK> 的機率比多數真實詞的三元機率還高（ln P(<UNK> | THE) ≈ -6，一個普通詞 -5 到 -12），
λ 一大 beam 就把整句的空白吃掉、拼成一個「詞」當 <unk> 繳一次稅——右邊回升變成「刪空白」而不是「LM 喜歡的字」；固定一筆罰（-10）
擋不住 27 個詞黏成一個（省下 λ·140 的 LM 稅只多繳 λ·10），所以罰要隨長度長：每個字元 -3 ≈ ln(1/27)，等於給 OOV 一個均勻的拼字模型。
pyctcdecode 的 unk_score_offset 是前者；後者是這裡自己加的。**修剪時半個詞也先當 OOV 估分**（LMBonus.partial_as_oov），否則黏詞的假設在搜尋
途中不用付錢、先把付了錢的對手擠出 beam（第三輪彩排才看懂的事，見 _partial_est）。
查不到的 n-gram 照 ARPA 的定義退回：
    P(w | h1 h2) = tri[h1 h2 w]                      有這條
                 = bo(h1 h2) · P(w | h2)             沒有；bo(h1 h2) 是 bigram 那一行最後一欄，沒有那行就是 1
    P(w | h2)    = bi[h2 w]  或  bo(h2) · P(w)
    P(w)         = uni[w]    或  uni[<unk>]           字典裡沒有的詞一律當 <unk>

n-gram 存成排序過的 int64 鍵（詞 id 各佔 18 bit）配 float32 的 log10，查表用 searchsorted；一個 34 MB 的 .arpa.gz 大約幾百 MB 記憶體。

LMBonus 與 ctc_prefix_beam.HotwordBonus 同一個介面（start／step／finish／score）：prefix 每接到一個詞界，剛走完的那個詞
乘上 λ·ln P(w | 前兩個詞) 再加 η（每個詞一次的插入獎勵）；句尾再乘 λ·ln P(</s> | …)。走在一個詞中間的字元不算分——
詞級 LM 只在空白處乘（Hannun et al. 2014 §3.1 的做法）。
"""
from __future__ import annotations

import gzip
import math
import re
from pathlib import Path

import numpy as np

LN10 = math.log(10.0)
BITS = 18
BASE = 1 << BITS


def _open_text(path: Path):
    p = Path(path)
    if p.suffix == ".gz":
        return gzip.open(p, "rt", encoding="utf-8", errors="replace")
    return open(p, "rt", encoding="utf-8", errors="replace")


class ArpaLM:
    def __init__(self, words: list[str], uni_lp: np.ndarray, uni_bo: np.ndarray, bi_keys: np.ndarray, bi_lp: np.ndarray, bi_bo: np.ndarray,
                 tri_keys: np.ndarray, tri_lp: np.ndarray, unk: str | None, bos: str = "<s>", eos: str = "</s>", source: str = ""):
        self.words = list(words)
        self.vocab = {w: i for i, w in enumerate(self.words)}
        self.uni_lp, self.uni_bo = uni_lp.astype(np.float32), uni_bo.astype(np.float32)
        self.bi_keys, self.bi_lp, self.bi_bo = bi_keys, bi_lp.astype(np.float32), bi_bo.astype(np.float32)
        self.tri_keys, self.tri_lp = tri_keys, tri_lp.astype(np.float32)
        self.unk = unk
        self.unk_id = self.vocab[unk] if unk else None
        self.bos_id, self.eos_id = self.vocab.get(bos), self.vocab.get(eos)
        self.source = source
        self.oov_penalty = 0.0                              # ln，加在每個 OOV 詞的 log 機率上；load_lm 從 toml 設
        self.oov_per_char = 0.0                             # ln，每個字元再加一次（OOV 的拼字模型）
        self.n_lookups = 0
        self._bi_cache: dict[int, float] = {}
        self._tri_cache: dict[int, float] = {}

    # ── 建立 ───────────────────────────────────────────────────────────────
    @classmethod
    def parse(cls, path: Path | str, progress=None) -> "ArpaLM":
        """讀 ARPA（可 .gz）。只收到三元；更高階的區段略過。"""
        counts: dict[int, int] = {}
        words: list[str] = []
        vocab: dict[str, int] = {}
        uni_lp: list[float] = []
        uni_bo: list[float] = []
        bi: list[tuple[int, float, float]] = []
        tri: list[tuple[int, float]] = []
        section = 0
        n_done = 0
        with _open_text(path) as f:
            for line in f:
                line = line.rstrip("\n")
                if not line:
                    continue
                if line.startswith("\\"):
                    m = re.match(r"\\(\d+)-grams:", line)
                    section = int(m.group(1)) if m else (0 if line.startswith("\\data") else -1)
                    if line.startswith("\\end"):
                        break
                    continue
                if section == 0:
                    m = re.match(r"ngram (\d+)=(\d+)", line)
                    if m:
                        counts[int(m.group(1))] = int(m.group(2))
                    continue
                if section < 1 or section > 3:
                    continue
                parts = line.split("\t") if "\t" in line else line.split()
                if len(parts) < 2:
                    continue
                lp = float(parts[0])
                toks = parts[1].split() if "\t" in line else parts[1:1 + section]
                bo = float(parts[2]) if len(parts) > 2 and section < 3 else 0.0
                if section == 1:
                    w = toks[0]
                    if w not in vocab:
                        vocab[w] = len(words)
                        words.append(w)
                        uni_lp.append(lp)
                        uni_bo.append(bo)
                    else:                                   # 重複的 1-gram（不該有）：以後者為準
                        uni_lp[vocab[w]], uni_bo[vocab[w]] = lp, bo
                else:
                    ids = []
                    ok = True
                    for w in toks:
                        i = vocab.get(w)
                        if i is None:                       # 高階 n-gram 用到沒有 1-gram 的詞：跳過（ARPA 不該這樣，防呆）
                            ok = False
                            break
                        ids.append(i)
                    if not ok or len(ids) != section:
                        continue
                    if section == 2:
                        bi.append((ids[0] * BASE + ids[1], lp, bo))
                    else:
                        tri.append(((ids[0] * BASE + ids[1]) * BASE + ids[2], lp))
                n_done += 1
                if progress and n_done % 1_000_000 == 0:
                    progress(n_done)
        if len(words) >= BASE:
            raise ValueError(f"字典 {len(words)} 個詞超過 {BASE}，鍵的位數不夠")
        bi_keys = np.array([k for k, _, _ in bi], dtype=np.int64)
        order = np.argsort(bi_keys, kind="stable")
        bi_keys = bi_keys[order]
        bi_lp = np.array([lp for _, lp, _ in bi], dtype=np.float32)[order]
        bi_bo = np.array([bo for _, _, bo in bi], dtype=np.float32)[order]
        tri_keys = np.array([k for k, _ in tri], dtype=np.int64)
        order = np.argsort(tri_keys, kind="stable")
        tri_keys = tri_keys[order]
        tri_lp = np.array([lp for _, lp in tri], dtype=np.float32)[order]
        unk = next((w for w in ("<unk>", "<UNK>") if w in vocab), None)
        lm = cls(words, np.array(uni_lp), np.array(uni_bo), bi_keys, bi_lp, bi_bo, tri_keys, tri_lp, unk, source=str(path))
        lm.counts = counts
        return lm

    def save(self, path: Path | str) -> None:
        np.savez(path, words=np.array(self.words, dtype=object), uni_lp=self.uni_lp, uni_bo=self.uni_bo, bi_keys=self.bi_keys, bi_lp=self.bi_lp,
                 bi_bo=self.bi_bo, tri_keys=self.tri_keys, tri_lp=self.tri_lp, unk=np.array(self.unk or "", dtype=object), source=np.array(self.source, dtype=object))

    @classmethod
    def load(cls, path: Path | str) -> "ArpaLM":
        z = np.load(path, allow_pickle=True)
        unk = str(z["unk"]) or None
        return cls([str(w) for w in z["words"]], z["uni_lp"], z["uni_bo"], z["bi_keys"], z["bi_lp"], z["bi_bo"], z["tri_keys"], z["tri_lp"], unk, source=str(z["source"]))

    # ── 查表 ───────────────────────────────────────────────────────────────
    @property
    def n_unigrams(self) -> int:
        return len(self.words)

    @property
    def n_bigrams(self) -> int:
        return int(self.bi_keys.shape[0])

    @property
    def n_trigrams(self) -> int:
        return int(self.tri_keys.shape[0])

    def word_id(self, w: str) -> int:
        i = self.vocab.get(w)
        if i is None:
            if self.unk_id is None:
                return -1
            return self.unk_id
        return i

    def in_vocab(self, w: str) -> bool:
        return w in self.vocab

    @staticmethod
    def _find(keys: np.ndarray, k: int) -> int:
        i = int(np.searchsorted(keys, k))
        return i if i < keys.shape[0] and int(keys[i]) == k else -1

    def _uni10(self, w: int) -> float:
        if w < 0:
            return -7.0                                      # 沒有 <unk> 的 LM 才會到這裡：地板值（toml 可改成別的策略）
        return float(self.uni_lp[w])

    def _bi10(self, h: int, w: int) -> float:
        """log10 P(w | h)，含 backoff。"""
        key = h * BASE + w
        c = self._bi_cache.get(key)
        if c is not None:
            return c
        i = self._find(self.bi_keys, key) if h >= 0 and w >= 0 else -1
        if i >= 0:
            v = float(self.bi_lp[i])
        else:
            v = (float(self.uni_bo[h]) if h >= 0 else 0.0) + self._uni10(w)
        if len(self._bi_cache) < 2_000_000:
            self._bi_cache[key] = v
        return v

    def _tri10(self, h1: int, h2: int, w: int) -> float:
        key = (h1 * BASE + h2) * BASE + w
        c = self._tri_cache.get(key)
        if c is not None:
            return c
        i = self._find(self.tri_keys, key) if h1 >= 0 and h2 >= 0 and w >= 0 else -1
        if i >= 0:
            v = float(self.tri_lp[i])
        else:
            j = self._find(self.bi_keys, h1 * BASE + h2) if h1 >= 0 and h2 >= 0 else -1
            bo = float(self.bi_bo[j]) if j >= 0 else 0.0
            v = bo + self._bi10(h2, w)
        if len(self._tri_cache) < 2_000_000:
            self._tri_cache[key] = v
        return v

    def logp10_ids(self, w: int, hist: tuple[int, ...]) -> float:
        self.n_lookups += 1
        if len(hist) >= 2:
            return self._tri10(hist[-2], hist[-1], w)
        if len(hist) == 1:
            return self._bi10(hist[0], w)
        return self._uni10(w)

    def logp_ids(self, w: int, hist: tuple[int, ...]) -> float:
        """ln P，純查表（不含 OOV 罰）。"""
        return self.logp10_ids(w, hist) * LN10

    def oov_cost(self, word: str) -> float:
        """OOV 的罰（ln，≤ 0）：固定一筆 + 每字元一筆；字典裡的詞是 0。"""
        if self.in_vocab(word):
            return 0.0
        return self.oov_penalty + self.oov_per_char * len(word)

    def logp_word(self, word: str, hist_ids: tuple[int, ...]) -> float:
        """ln P(word | hist) 含 OOV 罰；hist 是 id。LMBonus 用這個。"""
        return self.logp_ids(self.word_id(word), hist_ids) + self.oov_cost(word)

    def logp(self, word: str, hist: tuple[str, ...] = ()) -> float:
        """ln P(word | hist) 含 OOV 罰。hist 是前面的詞（最多用最後兩個）；句首放 "<s>"。"""
        return self.logp_word(word, tuple(self.word_id(h) for h in hist[-2:]))

    def sentence_logp(self, words: list[str], eos: bool = True) -> tuple[float, list[float]]:
        """整句 ln P，含 <s> 起頭、（可選）</s> 結尾；回傳 (總和, 逐詞)。"""
        hist: tuple[int, ...] = (self.bos_id,) if self.bos_id is not None else ()
        per = []
        for w in words:
            i = self.word_id(w)
            per.append(self.logp_word(w, hist))
            hist = (hist + (i,))[-2:]
        if eos and self.eos_id is not None:
            per.append(self.logp_ids(self.eos_id, hist))
        return float(sum(per)), per

    def perplexity(self, sentences: list[list[str]]) -> float:
        tot, n = 0.0, 0
        for s in sentences:
            lp, per = self.sentence_logp(s)
            tot += lp
            n += len(per)
        return math.exp(-tot / n) if n else float("nan")


# ── 接在 prefix beam search 上 ─────────────────────────────────────────────
class LMBonus:
    """與 ctc_prefix_beam.HotwordBonus 同介面。state = (h1, h2, partial, lm_ln, n_words, n_oov)：
    h1、h2 是最後兩個詞的 id（句首 (<s>, <s>) 的前一格用 -2 當「沒有」）、partial 是正在拼的字元、lm_ln 是已走完的詞的 Σ ln P、n_words 是詞數。
    score = λ·lm_ln + η·n_words。"""

    def __init__(self, lm: ArpaLM | None, lam: float, eta: float, id2ch: dict[int, str], delim: int, partial_as_oov: bool = True):
        self.lm, self.lam, self.eta, self.id2ch, self.delim = lm, float(lam), float(eta), id2ch, int(delim)
        self.partial_as_oov = bool(partial_as_oov)
        self.bos = lm.bos_id if lm is not None and lm.bos_id is not None else -2

    def _partial_est(self, partial) -> float:
        """還沒接到詞界的半個詞，修剪時先當 OOV 估分（固定罰 + 每字元罰），接到詞界再換成真的 ln P。
        不估的話，不加空白的假設在搜尋途中一毛錢都不用付、付了錢的假設先被修剪掉——2026-11-01 第三輪彩排 λ = 4 時整句黏成一個 <unk>，
        句尾算總分明明輸 1,700 ln，可是 beam 裡已經沒有別的候選了。pyctcdecode 的 partial-token score 是同一件事。"""
        if not partial or self.lm is None or not self.partial_as_oov:
            return 0.0
        return self.lm.oov_penalty + self.lm.oov_per_char * len(partial)

    def start(self):
        return (-2, self.bos, (), 0.0, 0, 0)

    def _complete(self, state):
        h1, h2, partial, lm_ln, n, n_oov = state
        if not partial:
            return state
        word = "".join(partial)
        if self.lm is None:
            return (h2, -2, (), lm_ln, n + 1, n_oov)
        wid = self.lm.word_id(word)
        hist = tuple(h for h in (h1, h2) if h != -2)
        lp = self.lm.logp_word(word, hist)
        return (h2, wid, (), lm_ln + lp, n + 1, n_oov + (0 if self.lm.in_vocab(word) else 1))

    def step(self, state, c: int):
        if c == self.delim:
            return self._complete(state)
        h1, h2, partial, lm_ln, n, n_oov = state
        return (h1, h2, partial + (self.id2ch.get(c, ""),), lm_ln, n, n_oov)

    def finish(self, state) -> float:
        s = self._complete(state)
        h1, h2, partial, lm_ln, n, n_oov = s
        if self.lm is not None and self.lm.eos_id is not None and n > 0:
            hist = tuple(h for h in (h1, h2) if h != -2)
            lm_ln += self.lm.logp_ids(self.lm.eos_id, hist)
        return self.lam * lm_ln + self.eta * n

    def score(self, state) -> float:
        return self.lam * (state[3] + self._partial_est(state[2])) + self.eta * state[4]

    def parts(self, state) -> dict:
        """句尾：把分數拆成三項（給畫面印）。"""
        s = self._complete(state)
        h1, h2, partial, lm_ln, n, n_oov = s
        eos = 0.0
        if self.lm is not None and self.lm.eos_id is not None and n > 0:
            hist = tuple(h for h in (h1, h2) if h != -2)
            eos = self.lm.logp_ids(self.lm.eos_id, hist)
        return {"lm_ln": lm_ln + eos, "n_words": n, "n_oov": n_oov, "lam": self.lam, "eta": self.eta}

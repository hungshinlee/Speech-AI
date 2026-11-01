# -*- coding: utf-8 -*-
# 此檔由 bin/publish_demos.py 從 private repo 的 <course>/demos/ 複製而來，請勿在公開 repo 直接編輯。
"""CTC prefix beam search（Hannun et al. 2014, Algorithm 1）配一張名單的 hotword bonus，純 numpy、字元級。

    beams = prefix_beam_search(logprobs, blank, beam=32, bonus=HotwordBonus(trie, delta, delim))

logprobs 是 (T, V) 的 log softmax。每個 prefix 維持兩個機率：p_b（以 blank 結尾的路徑）與 p_nb（以非 blank 結尾的路徑），
log 域相加。修剪時的分數 = log(p_b + p_nb) + bonus(prefix)，bonus 由 HotwordBonus 依 prefix 增量算：

  * prefix 正走在名單某個片語的路上：每個字元 +delta（Zhao et al. 2019 把權重推到每個 subword 上，才不會在修剪時先被丟掉）
  * 岔開（下一個字元不在 trie 裡）：把這個片語累積的加分扣回（failure arc，Zhao §2.2）
  * 片語走完並接到詞界（或句尾）：加分留下

名單的片語只從詞界開始比對；多詞片語（"MARY GRANT"）的詞界符號也在 trie 裡。
這個檔只依賴 numpy，W7 Demo 3 的 λ 掃描會再用它（詞級 LM 在詞界處乘上去，接在同一個迴圈裡）。
"""
from __future__ import annotations

import math
from collections import defaultdict

import numpy as np

NEG = -math.inf


def lse(a: float, b: float) -> float:
    if a == NEG:
        return b
    if b == NEG:
        return a
    m = max(a, b)
    return m + math.log1p(math.exp(-abs(a - b)))


class Trie:
    """字元級 trie。節點是 int，children[node] = {symbol: node}，terminal 是走完一個片語的節點集合。"""

    def __init__(self, phrases: list[tuple[int, ...]]):
        self.children: list[dict[int, int]] = [{}]
        self.terminal: set[int] = set()
        self.depth: list[int] = [0]
        for ph in phrases:
            node = 0
            for s in ph:
                nxt = self.children[node].get(s)
                if nxt is None:
                    nxt = len(self.children)
                    self.children.append({})
                    self.depth.append(self.depth[node] + 1)
                    self.children[node][s] = nxt
                node = nxt
            if node != 0:
                self.terminal.add(node)

    @property
    def n_nodes(self) -> int:
        return len(self.children)


# 名單狀態：(node, partial)。node = -1 表示「在一個不在名單上的詞裡」，到下一個詞界才回到 root。
ROOT, OFF = 0, -1


class HotwordBonus:
    def __init__(self, trie: Trie | None, delta: float, delim: int):
        self.trie, self.delta, self.delim = trie, float(delta), int(delim)

    def start(self) -> tuple[int, float, float]:
        """(node, partial, committed)：partial 是走在路上還沒確定的加分，committed 是已走完片語的加分。"""
        return (ROOT, 0.0, 0.0)

    def step(self, state: tuple[int, float, float], c: int) -> tuple[int, float, float]:
        if self.trie is None or self.delta == 0.0:
            return state
        node, partial, committed = state
        ch = self.trie.children
        if c == self.delim:
            if node not in (ROOT, OFF) and self.delim in ch[node]:       # 多詞片語的中間那個空白
                return (ch[node][self.delim], partial + self.delta, committed)
            if node not in (ROOT, OFF) and node in self.trie.terminal:   # 片語走完、接到詞界：加分留下
                return (ROOT, 0.0, committed + partial)
            return (ROOT, 0.0, committed)                                 # 岔開（partial 扣回）或本來就不在路上
        if node == OFF:
            return state
        nxt = ch[node].get(c)
        if nxt is not None:
            return (nxt, partial + self.delta, committed)
        if node == ROOT:
            return (OFF, 0.0, committed)                                  # 這個詞不是名單上任何片語的開頭
        return (OFF, 0.0, committed)                                      # failure arc：退回累積的加分

    def finish(self, state: tuple[int, float, float]) -> float:
        """句尾：走完的片語留下，沒走完的扣回。"""
        node, partial, committed = state
        if self.trie is not None and node not in (ROOT, OFF) and node in self.trie.terminal:
            return committed + partial
        return committed

    def score(self, state: tuple[int, float, float]) -> float:
        return state[1] + state[2]


def prefix_beam_search(logprobs: np.ndarray, blank: int, beam: int = 32, bonus: HotwordBonus | None = None,
                       prune_logp: float = -12.0, n_best: int = 5) -> list[dict]:
    """回傳前 n_best 條：{"ids", "logp"（log(p_b+p_nb)）, "bonus", "score"（logp + bonus）}，依 score 排序。
    prune_logp：某 frame 上 log 機率低於它的符號不展開（Hannun 的演算法不修剪符號，這裡只是省時間；-12 ≈ 6e-6）。"""
    T, V = logprobs.shape
    bonus = bonus or HotwordBonus(None, 0.0, -1)
    beams: dict[tuple[int, ...], list[float]] = {(): [0.0, NEG]}          # prefix → [log p_b, log p_nb]
    states: dict[tuple[int, ...], tuple[int, float, float]] = {(): bonus.start()}
    for t in range(T):
        lp = logprobs[t]
        cands = [int(c) for c in np.nonzero(lp > prune_logp)[0]]
        if blank not in cands:
            cands.append(blank)
        nxt: dict[tuple[int, ...], list[float]] = defaultdict(lambda: [NEG, NEG])
        for prefix, (pb, pnb) in beams.items():
            total = lse(pb, pnb)
            cur = nxt[prefix]
            cur[0] = lse(cur[0], total + float(lp[blank]))                 # 讀 blank：prefix 不變、改以 blank 結尾
            last = prefix[-1] if prefix else None
            if last is not None:
                cur[1] = lse(cur[1], pnb + float(lp[last]))                # 重複最後一個字元、沒有 blank 隔開：prefix 不變
            for c in cands:
                if c == blank:
                    continue
                newp = prefix + (c,)
                n = nxt[newp]
                if c == last:
                    n[1] = lse(n[1], pb + float(lp[c]))                    # 同字元只能從 blank 結尾的路徑延伸
                else:
                    n[1] = lse(n[1], total + float(lp[c]))
                if newp not in states:
                    states[newp] = bonus.step(states[prefix], c)
        ranked = sorted(nxt.items(), key=lambda kv: lse(kv[1][0], kv[1][1]) + bonus.score(states[kv[0]]), reverse=True)
        beams = dict(ranked[:beam])
        keep = set(beams)
        states = {k: v for k, v in states.items() if k in keep}
    out = []
    for prefix, (pb, pnb) in beams.items():
        logp = lse(pb, pnb)
        b = bonus.finish(states[prefix])
        out.append({"ids": list(prefix), "logp": logp, "bonus": b, "score": logp + b})
    out.sort(key=lambda d: d["score"], reverse=True)
    return out[:n_best]


def greedy(logprobs: np.ndarray, blank: int) -> list[int]:
    am = logprobs.argmax(-1).tolist()
    out, prev = [], None
    for i in am:
        if i != prev and i != blank:
            out.append(int(i))
        prev = i
    return out

# -*- coding: utf-8 -*-
# 此檔由 bin/publish_demos.py 從 private repo 的 <course>/demos/ 複製而來，請勿在公開 repo 直接編輯。
"""W4 demo 1：RNN-T 的格點——路徑窮舉 vs. forward，與 max_symbols_per_step。

  show       課堂：單鍵選單（下面的鍵）。離開時把這次算過的東西存進 runs/live/<時間>/（不進版控）
  rehearse   課前：不互動，把所有段落照課堂順序跑一遍，存進 runs/rehearsal/（進版控 = 現場的退路）
  replay     現場 Python 出狀況時：印 runs/rehearsal/rehearsal.txt（純文字）
  selftest   對 slides/assets/w04/w04-data.json：路徑清單逐條、C(T+U−1, U)、CTC 對照、forward 均勻與隨機（seed 20261004）逐位；
             隨機 y 的窮舉 vs. forward vs. log 域；對角線和 = P；greedy decode 有上限／沒上限的行為

畫面上的鍵（順序照投影片講稿的【操作】，p29／p33／p31）：
  1  "ab" T = 2 的全部 RNN-T 路徑（3 條）     2  "aa" T = 1（1 條；CTC 無解）
  3  "ab" T = 3（6 條；CTC 5 條）              0  "aa" T = 2（3 條；CTC 0 條）
  4  隨機一組 y^{t,u}："ab"、T = 2，每個節點一個 softmax；三條路徑各自的乘積與總和（窮舉）
  5  同一組 y^{t,u} 逐格填 α(t, u)，最後 α(T, U)·∅(T, U)，與窮舉比；對角線 Σ αβ = P
  r  換一組隨機 y^{t,u}（seed 用時間；證明 4／5 不是預先湊好的）
  6  greedy decode，有 max_symbols_per_step：到上限就強制 ∅
  u  同一個 joint，拿掉上限：∅ 校準壞掉的那個 frame 停不下來（程式在 safety_limit 自己停）
  h  重印選單      q  離開

純標準庫（numpy 只在 common.py 裡被 import），跑 CPU。數字的定義：路徑 = 從 (1, 0) 到 (T, U) 的 T−1 次 R 與 U 次 E 的交錯再補一個 R；
α(t, u) = α(t−1, u) ∅(t−1, u) + α(t, u−1) y(t, u−1)，α(1, 0) = 1，P = α(T, U) ∅(T, U)——與投影片 p32 及大綱〈關鍵數學 4〉逐字相同
（Graves 2012 eq. 16–17）；β 照 eq. 18、對角線照 eq. 19。
"""
from __future__ import annotations

import argparse
import io
import itertools
import json
import math
import random
import re
import sys
import time
import tomllib
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import common as C  # noqa: E402

CFG = tomllib.loads((HERE / "demo_config.toml").read_text(encoding="utf-8"))
BLANK = "-"
DIG = CFG["display"]["digits"]
ASSETS = HERE.parent.parent / "slides" / "assets"
DATA_JSON = ASSETS / "w04" / "w04-data.json"
W3_JSON = ASSETS / "w03" / "w03-data.json"

# make_figs_w04.py 的 data_json() 抽隨機 logits 的順序：這兩個 tuple 不能改，改了 seed 20261004 就對不上 json。
JSON_LABELS = ("ab", "aa")
JSON_TS = (1, 2, 3)


# ── 符號 ────────────────────────────────────────────────────────
def sym(s: str) -> str:
    return "∅" if s == BLANK else s


def show_ctc(p: str) -> str:
    return "".join(sym(c) for c in p)


def repeats(label: str) -> int:
    return sum(1 for u in range(len(label) - 1) if label[u] == label[u + 1])


# ── RNN-T 路徑 ─────────────────────────────────────────────────
def rnnt_paths(T: int, U: int) -> list[str]:
    """從 (1, 0) 到 (T, U)：T−1 次 R 與 U 次 E 的交錯，最後補一個 R。順序 = itertools.combinations（與 make_figs_w04.py 相同）。"""
    out = []
    for pos in itertools.combinations(range(T - 1 + U), U):
        moves = ["R"] * (T - 1 + U)
        for p in pos:
            moves[p] = "E"
        out.append("".join(moves) + "R")
    return out


def nodes_of(moves: str) -> list[tuple[int, int]]:
    """路徑經過的節點序列，從 (1, 0) 起；最後一個 R 離開 (T, U)。"""
    t, u, seq = 1, 0, [(1, 0)]
    for mv in moves[:-1]:
        if mv == "E":
            u += 1
        else:
            t += 1
        seq.append((t, u))
    return seq


def emits_per_frame(moves: str, T: int) -> list[int]:
    cnt, t = [0] * (T + 1), 1
    for mv in moves:
        if mv == "E":
            cnt[t] += 1
        else:
            t += 1
    return cnt[1:]


def path_factors(moves: str, Y: dict) -> list[tuple[str, float]]:
    """每一步用到的那個數：("y(1,0)", 0.42) 或 ("∅(1,2)", 0.55)。Y[(t, u)] = (p_a, p_b, p_∅)，emit 取 ℓ_{u+1} 那格。"""
    label = Y["_label"]
    t, u, out = 1, 0, []
    for mv in moves:
        if mv == "E":
            out.append((f"y({t},{u})", Y[(t, u)]["emit"]))
            u += 1
        else:
            out.append((f"∅({t},{u})", Y[(t, u)]["blank"]))
            t += 1
    return out


def path_prob(moves: str, Y: dict) -> float:
    p = 1.0
    for _, v in path_factors(moves, Y):
        p *= v
    return p


# ── CTC 對照（W3 同一支 enumerate）─────────────────────────────
def ctc_collapse(path) -> str:
    out, prev = [], None
    for p in path:
        if p != prev and p != BLANK:
            out.append(p)
        prev = p
    return "".join(out)


def ctc_paths(label: str, T: int, vocab: str) -> list[str]:
    return ["".join(p) for p in itertools.product(vocab + BLANK, repeat=T) if ctc_collapse(p) == label]


def ctc_from_w3(label: str, T: int) -> list[str] | None:
    """w03-data.json 有這一格就讀它（p29 講稿說「從 W3 讀」），沒有回 None 由呼叫端現場算。"""
    if not W3_JSON.exists():
        return None
    d = json.loads(W3_JSON.read_text(encoding="utf-8"))
    try:
        return list(d["paths"][label]["by_T"][str(T)]["paths"])
    except KeyError:
        return None


# ── forward／backward（Graves 2012 eq. 16–19）───────────────────
def forward(T: int, U: int, Y: dict) -> tuple[list[list[float]], float]:
    a = [[0.0] * (U + 1) for _ in range(T + 1)]
    a[1][0] = 1.0
    for t in range(1, T + 1):
        for u in range(0, U + 1):
            if t == 1 and u == 0:
                continue
            v = 0.0
            if t > 1:
                v += a[t - 1][u] * Y[(t - 1, u)]["blank"]
            if u > 0:
                v += a[t][u - 1] * Y[(t, u - 1)]["emit"]
            a[t][u] = v
    return a, a[T][U] * Y[(T, U)]["blank"]


def backward(T: int, U: int, Y: dict) -> list[list[float]]:
    b = [[0.0] * (U + 2) for _ in range(T + 2)]
    b[T][U] = Y[(T, U)]["blank"]
    for t in range(T, 0, -1):
        for u in range(U, -1, -1):
            if t == T and u == U:
                continue
            v = 0.0
            if t < T:
                v += b[t + 1][u] * Y[(t, u)]["blank"]
            if u < U:
                v += b[t][u + 1] * Y[(t, u)]["emit"]
            b[t][u] = v
    return b


def forward_log(T: int, U: int, Y: dict) -> float:
    la = [[-math.inf] * (U + 1) for _ in range(T + 1)]
    la[1][0] = 0.0
    for t in range(1, T + 1):
        for u in range(0, U + 1):
            if t == 1 and u == 0:
                continue
            terms = []
            if t > 1:
                terms.append(la[t - 1][u] + math.log(Y[(t - 1, u)]["blank"]))
            if u > 0:
                terms.append(la[t][u - 1] + math.log(Y[(t, u - 1)]["emit"]))
            m = max(terms)
            la[t][u] = m + math.log(sum(math.exp(v - m) for v in terms)) if m > -math.inf else -math.inf
    return la[T][U] + math.log(Y[(T, U)]["blank"])


# ── 隨機 y^{t,u} ────────────────────────────────────────────────
def _softmax(z: list[float]) -> list[float]:
    m = max(z)
    e = [math.exp(v - m) for v in z]
    s = sum(e)
    return [v / s for v in e]


def _draw_table(rng: random.Random, label: str, T: int, vocab: str, std: float) -> dict:
    """每個節點 (t, u) 抽 K 個 logits（a, b, …, ∅ 的順序），softmax；emit 取 ℓ_{u+1} 那格（u = U 時沒有下一個符號，emit 設 0）。"""
    K = len(vocab) + 1
    U = len(label)
    Y = {"_label": label, "_T": T, "_U": U, "_vocab": vocab}
    for t in range(1, T + 1):
        for u in range(0, U + 1):
            p = _softmax([rng.gauss(0, std) for _ in range(K)])
            Y[(t, u)] = {"p": p, "blank": p[-1], "emit": p[vocab.index(label[u])] if u < U else 0.0}
    return Y


def random_y(label: str, T: int, seed: int, vocab: str, std: float) -> dict:
    """seed 等於 make_figs_w04.py 的 20261004 且 (label, T) 在 json 的清單裡時，照 json 的順序把前面的節點抽掉，數字才會逐位相同。"""
    rng = random.Random(seed)
    if seed == 20261004 and label in JSON_LABELS and T in JSON_TS and vocab == "ab":
        for lab in JSON_LABELS:
            for TT in JSON_TS:
                Y = _draw_table(rng, lab, TT, vocab, std)
                if lab == label and TT == T:
                    return Y
    return _draw_table(rng, label, T, vocab, std)


# ── 第三段：合成 joint 的 greedy decode ────────────────────────
class Joint:
    """合成的 joint，不是模型：z^{t,u} = f_t + g(上一個發射的符號)。
    f_t（encoder 那一側）：每個符號一個 N(0, σ²)；∅ 的 logit 放在「第二高的符號 + blank_margin」——所以一個 frame 通常只發得出最高的那個符號，
    發完之後 g 把它壓下去，∅ 就贏了。第 blank_starved_frame 個 frame 例外：∅ 的 logit = 最低的符號 − blank_penalty，怎樣都贏不了。
    g(prev)（prediction network 那一側）：剛發過的符號減 repeat_penalty（「不喜歡重複」），其餘 0；<s> 全 0。"""

    def __init__(self, dc: dict):
        self.vocab, self.T = dc["vocab"], dc["T"]
        self.syms = list(self.vocab) + [BLANK]
        self.starved, self.penalty, self.margin = dc["blank_starved_frame"], dc["blank_penalty"], dc["blank_margin"]
        rng = random.Random(dc["seed"])
        K = len(self.syms)
        self.f = []
        for t in range(self.T + 1):
            z = [rng.gauss(0, dc["logit_std"]) for _ in range(K - 1)]
            s = sorted(z)
            z.append(s[0] - self.penalty if t == self.starved else s[-2] + self.margin)
            self.f.append(z)
        self.g = {"<s>": [0.0] * K}
        for i, s_ in enumerate(self.vocab):
            g = [0.0] * K
            g[i] = -dc["repeat_penalty"]
            self.g[s_] = g

    def probs(self, t: int, prev: str) -> list[float]:
        return _softmax([a + b for a, b in zip(self.f[t], self.g[prev])])


def greedy(J: Joint, cap: int | None, safety: int) -> dict:
    """每個 frame：argmax；是符號就 emit、留在同一 frame；是 ∅ 就前進。cap = 每 frame 最多 emit 幾個；None = 沒上限。"""
    hyp, frames, prev, total, runaway = [], [], "<s>", 0, False
    for t in range(1, J.T + 1):
        steps, forced = [], False
        while True:
            p = J.probs(t, prev)
            k = max(range(len(p)), key=p.__getitem__)
            s = J.syms[k]
            if s == BLANK:
                steps.append((BLANK, p[k], False))
                break
            if cap is not None and sum(1 for st in steps if st[0] != BLANK) >= cap:
                steps.append((BLANK, p[-1], True))
                forced = True
                break
            steps.append((s, p[k], False))
            hyp.append(s)
            prev = s
            total += 1
            if total >= safety:
                runaway = True
                break
        frames.append({"t": t, "steps": steps, "forced_blank": forced,
                       "emits": sum(1 for st in steps if st[0] != BLANK)})
        if runaway:
            break
    return {"cap": cap, "hyp": "".join(hyp), "frames": frames, "total_emits": total, "runaway": runaway}


# ── 輸出（同時寫到終端機與 transcript）───────────────────────────
class Out:
    def __init__(self):
        self.buf = io.StringIO()

    def __call__(self, text: str = "") -> None:
        print(text)
        self.buf.write(re.sub(r"\033\[[0-9;]*m", "", text) + "\n")

    def banner(self, text: str, color: str = C.BLUE) -> None:
        C.banner(text, color)
        self.buf.write("\n" + "─" * 72 + "\n" + text + "\n" + "─" * 72 + "\n")

    def text(self) -> str:
        return self.buf.getvalue()


def fmt(v: float) -> str:
    return f"{v:.{DIG}f}"


def node(t: int, u: int) -> str:
    return f"({t},{u})"


# ── 各段落 ──────────────────────────────────────────────────────
class Demo:
    def __init__(self, out: Out, seed: int | None = None, tag: str = ""):
        self.out = out
        self.tag = tag
        self.paper = CFG["paper"]
        self.vocab = self.paper["vocab"]
        rc = CFG["random"]
        self.seed = rc["seed"] if seed is None else seed
        self.Y: dict | None = None
        self.enum_sum: float | None = None
        self.log: dict = {"date": C.now(), "tag": tag, "versions": C.versions(), "config": CFG, "sections": {}}

    # 第一段：列路徑 ---------------------------------------------------
    def list_paths(self, label: str, T: int) -> None:
        o = self.out
        U, r = len(label), repeats(label)
        o.banner(f'ℓ = "{label}"  U = {U}  r = {r}      T = {T}      格點 T × (U + 1) = {T} × {U + 1}')
        ps = rnnt_paths(T, U)
        o(f"RNN-T 路徑：{T - 1} 次 R（blank，往右）與 {U} 次 E（emit，往上）的任意交錯，最後補一個 R → "
          f"{C.BOLD}{len(ps)}{C.RESET} 條 = C(T+U−1, U) = C({T + U - 1}, {U}) = {math.comb(T + U - 1, U)}")
        for p in ps:
            names = "，".join(f"emit {label[u]}" if mv == "E" else "blank" for u, mv in
                              zip(itertools.accumulate((1 if m == "E" else 0 for m in p), initial=0), p))
            route = "→".join(node(*n) for n in nodes_of(p)) + "→■"
            epf = emits_per_frame(p, T)
            flag = f"   {C.AMBER}同一 frame 連發 {max(epf)} 個{C.RESET}" if max(epf) > 1 else ""
            o(f"  {C.BOLD}{' '.join(p)}{C.RESET}   {route}   [{names}]{flag}")
        cps = ctc_from_w3(label, T)
        src = "w03-data.json"
        if cps is None:
            cps, src = ctc_paths(label, T, self.vocab), "同一支 enumerate 現場算"
        o(f"\nCTC 對照（W3，{src}）：候選 K^T = {len(self.vocab) + 1}^{T}，B 摺成 \"{label}\" 的 {C.BOLD}{len(cps)}{C.RESET} 條"
          + ("  " + "、".join(show_ctc(p) for p in cps) if cps else
             f"   {C.RED}（空集合：T = {T} < U + r = {U + r}）{C.RESET}"))
        o(f"  RNN-T 沒有 T ≥ U + r 這條：emit 不花 frame，相鄰重複不需要 ∅ 隔開；T = 1 對任何 ℓ 都有 1 條。")
        self.log["sections"][f"paths_{label}_T{T}"] = {
            "label": label, "T": T, "U": U, "r": r, "rnnt_count": len(ps), "binom": math.comb(T + U - 1, U),
            "rnnt_paths": ps, "ctc_count": len(cps), "ctc_paths": cps, "ctc_source": src}

    # 第二段：隨機 y^{t,u}，窮舉 vs. forward ----------------------------
    def draw_y(self, seed: int | None = None) -> None:
        rc = CFG["random"]
        if seed is not None:
            self.seed = seed
        self.Y = random_y(rc["label"], rc["T"], self.seed, rc["vocab"], rc["logit_std"])
        self.enum_sum = None

    def print_y(self) -> None:
        o = self.out
        rc = CFG["random"]
        Y, label, T, U = self.Y, rc["label"], rc["T"], len(rc["label"])
        syms = list(rc["vocab"]) + [BLANK]
        o(f"seed = {self.seed}；每個節點 (t, u) 一個 softmax(N(0, {rc['logit_std']}²)) over {{{', '.join(sym(s) for s in syms)}}}，"
          f"{T * (U + 1)} 個節點" + ("（與 w04-data.json 同一組）" if self.seed == 20261004 else ""))
        o("   (t,u)   " + "".join(f"{'y_' + sym(s):>{DIG + 5}}" for s in syms) + "    這個節點的 emit 是")
        for t in range(1, T + 1):
            for u in range(0, U + 1):
                p = Y[(t, u)]["p"]
                nxt = f"y({t},{u}) = y_{label[u]} = {fmt(Y[(t, u)]['emit'])}" if u < U else "— （u = U，只能往右）"
                o(f"   {node(t, u):<7}" + "".join(f"{fmt(v):>{DIG + 5}}" for v in p) + f"    {nxt}")

    def enumerate_random(self) -> None:
        o = self.out
        rc = CFG["random"]
        if self.Y is None:
            self.draw_y()
        label, T, U = rc["label"], rc["T"], len(rc["label"])
        o.banner(f'窮舉：ℓ = "{label}"、T = {T}，隨機 y^{{t,u}}，每條路徑的機率' + (f"   [{self.tag}]" if self.tag else ""))
        self.print_y()
        ps = rnnt_paths(T, U)
        o(f"\n{len(ps)} 條路徑，每條 T + U = {T + U} 個因子（{T} 個 ∅、{U} 個 y）：")
        total, rows = 0.0, []
        for p in ps:
            fs = path_factors(p, self.Y)
            v = path_prob(p, self.Y)
            total += v
            o(f"  {C.BOLD}{' '.join(p)}{C.RESET}   " + " · ".join(f"{n}={fmt(x)}" for n, x in fs) + f" = {fmt(v)}")
            rows.append({"path": p, "factors": [[n, x] for n, x in fs], "prob": v})
        o(f"\n  {C.BOLD}Σ 窮舉 = {total:.{DIG + 6}f}{C.RESET}")
        self.enum_sum = total
        self.log["sections"]["enumeration_random"] = {
            "seed": self.seed, "label": label, "T": T,
            "y": {f"{t},{u}": self.Y[(t, u)]["p"] for t in range(1, T + 1) for u in range(0, U + 1)},
            "paths": rows, "sum": total}

    def forward_random(self, step: bool = True) -> None:
        o = self.out
        rc = CFG["random"]
        if self.Y is None:
            self.draw_y()
        label, T, U = rc["label"], rc["T"], len(rc["label"])
        a, P = forward(T, U, self.Y)
        b = backward(T, U, self.Y)
        o.banner(f'forward 表：同一組 y^{{t,u}}，ℓ = "{label}"，格點 {T} × {U + 1}' + ("   （任一鍵下一格，a 全部，q 中止）" if step else ""))
        o("  α(t,u) = α(t−1,u)·∅(t−1,u) + α(t,u−1)·y(t,u−1)      α(1,0) = 1      P = α(T,U)·∅(T,U)")
        o("  第一項：從左邊走 blank 進來；第二項：從下面 emit 進來。t = 1 沒有左邊，u = 0 沒有下面。\n")
        order = [(t, u) for t in range(1, T + 1) for u in range(0, U + 1)]
        shown: set[tuple[int, int]] = set()
        w = DIG + 4

        def table():
            o("    u\\t " + "".join(f"{'t=' + str(t):>{w}}" for t in range(1, T + 1)))
            for u in range(U, -1, -1):
                o(f"  {u:>3}   " + "".join(f"{fmt(a[t][u]) if (t, u) in shown else '·':>{w}}" for t in range(1, T + 1))
                  + (f"    ← u = U = {U}：到這一列就只差最後一個 ∅" if u == U else ""))

        def detail(t: int, u: int):
            if (t, u) == (1, 0):
                o(f"    α(1,0) = 1（起點）")
                return
            parts, vals = [], []
            if t > 1:
                parts.append(f"α({t - 1},{u})·∅({t - 1},{u})")
                vals.append(f"{fmt(a[t - 1][u])}·{fmt(self.Y[(t - 1, u)]['blank'])}")
            else:
                parts.append(f"{C.GREY}[t=1 無左邊]{C.RESET}")
            if u > 0:
                parts.append(f"α({t},{u - 1})·y({t},{u - 1})")
                vals.append(f"{fmt(a[t][u - 1])}·{fmt(self.Y[(t, u - 1)]['emit'])}")
            else:
                parts.append(f"{C.GREY}[u=0 無下面]{C.RESET}")
            o(f"    α{node(t, u)} = {' + '.join(parts)} = {' + '.join(vals)} = {fmt(a[t][u])}")

        for (t, u) in order:
            shown.add((t, u))
            table()
            detail(t, u)
            o("")
            if step and (t, u) != order[-1]:
                k = C.getkey()
                if k == "q":
                    o("（中止）")
                    return
                if k == "a":
                    step = False
        o(f"  {C.BOLD}P(ℓ|X) = α({T},{U})·∅({T},{U}) = {fmt(a[T][U])} · {fmt(self.Y[(T, U)]['blank'])} = {P:.{DIG + 6}f}{C.RESET}")
        if self.enum_sum is None:
            self.enum_sum = sum(path_prob(p, self.Y) for p in rnnt_paths(T, U))
        diff = abs(P - self.enum_sum)
        o(f"  Σ 窮舉          = {self.enum_sum:.{DIG + 6}f}")
        o(f"  |差|            = {diff:.2e}   " + (f"{C.GREEN}（逐位相同：格點這麼小，兩邊做的是同一組乘法）{C.RESET}" if diff == 0.0 else
                                           f"{C.GREEN}（浮點誤差）{C.RESET}" if diff < 1e-12 else f"{C.RED}（不是浮點誤差，程式有錯）{C.RESET}"))
        o(f"  格數 T·(U+1) = {T * (U + 1)}，兩項遞推；路徑 C({T + U - 1},{U}) = {math.comb(T + U - 1, U)} 條，每條 {T + U} 個因子")
        # 對角線（p34 用）
        diag = {}
        for n in range(1, T + U + 1):
            diag[n] = sum(a[t][u] * b[t][u] for t in range(1, T + 1) for u in range(0, U + 1) if t + u == n)
        o(f"\n  backward β(t,u) = β(t+1,u)·∅(t,u) + β(t,u+1)·y(t,u)，β({T},{U}) = ∅({T},{U})；每一條對角線 t + u = n 上 Σ α·β：")
        for n, v in diag.items():
            o(f"    n = {n}: {v:.{DIG + 6}f}" + ("   ← = P" if abs(v - P) < 1e-15 else f"   {C.RED}≠ P{C.RESET}"))
        self.log["sections"]["forward_random"] = {
            "seed": self.seed, "alpha": a, "beta": b, "P": P, "enumeration_sum": self.enum_sum, "abs_diff": diff,
            "diagonal_sums": {str(n): v for n, v in diag.items()}, "all_diagonals_equal_P": all(abs(v - P) < 1e-15 for v in diag.values())}

    # 第三段：greedy decode 與上限 ----------------------------------------
    def decode(self, capped: bool) -> None:
        o = self.out
        dc = CFG["decode"]
        J = Joint(dc)
        cap = dc["max_symbols_per_step"] if capped else None
        res = greedy(J, cap, dc["safety_limit"])
        o.banner(f"greedy decode，T = {dc['T']}，" + (f"max_symbols_per_step = {cap}" if capped else "沒有上限（u）"),
                 C.BLUE if capped else C.AMBER)
        o(f"  合成的 joint（不是模型）：z^{{t,u}} = f_t + g(上一個符號)。f_t 的 ∅ 只比第二高的符號多 {J.margin}；g 把剛發過的符號壓 {CFG['decode']['repeat_penalty']}。")
        o(f"  frame {J.starved} 例外：∅ 的 logit 比最低的符號還低 {J.penalty}——一個校準壞掉、不肯吐 blank 的 ∅(t,u)。")
        o(f"  規則：每個 frame 取 argmax；是符號就 emit、留在同一個 frame；是 ∅ 就前進。" + (f" 連發到 {cap} 個就強制 ∅。" if capped else ""))
        for fr in res["frames"]:
            parts = []
            for s, p, forced in fr["steps"]:
                if s == BLANK:
                    parts.append(f"{C.RED}[cap → 強制 ∅，∅ 其實只有 {p:.4f}]{C.RESET}" if forced else f"∅ {p:.2f}")
                else:
                    parts.append(f"{C.BOLD}{s}{C.RESET} {p:.2f}")
            if len(parts) > 8:   # 沒上限時那個 frame 有幾十個 emit，不要印成一行
                parts = parts[:6] + [f"{C.GREY}… 共 {fr['emits']} 個，{'/'.join(sorted(set(s for s, _, _ in fr['steps'] if s != BLANK)))} 一直交替，∅ 從沒贏過{C.RESET}"]
            tail = f"   emits: {fr['emits']}"
            if fr["t"] == J.starved:
                tail += f"   {C.AMBER}← ∅ 被餓死的 frame{C.RESET}"
            o(f"  t = {fr['t']}:  " + "  ".join(parts) + tail)
        if res["runaway"]:
            o(f"\n  {C.RED}停在 safety_limit = {dc['safety_limit']} 次 emit——這是程式自己的保險；沒有它，這個 frame 的迴圈不會結束。{C.RESET}")
            o(f"  模型沒有變、y^{{t,u}} 沒有變；變的只是解碼器少了一行 if。這就是「工程上限 vs. 模型性質」。")
        else:
            o(f"\n  {C.BOLD}hyp = \"{res['hyp']}\"{C.RESET}（{res['total_emits']} 個符號）"
              + (f"；frame {J.starved} 在第 {cap} 個 emit 被上限截斷" if any(f['forced_blank'] for f in res['frames']) else ""))
        if capped and self.Y is not None:
            # 把上限的意思接回第二段的格點：訓練的總和裡有多少質量來自「同一 frame 連發」的路徑
            rc = CFG["random"]
            T, U = rc["T"], len(rc["label"])
            ps = rnnt_paths(T, U)
            tot = sum(path_prob(p, self.Y) for p in ps)
            multi = [(p, path_prob(p, self.Y)) for p in ps if max(emits_per_frame(p, T)) > 1]
            share = sum(v for _, v in multi) / tot
            o(f"\n  對照第二段的格點（\"{rc['label']}\"、T = {T}）：{len(multi)} 條路徑在同一個 frame 連發 2 個——"
              + "、".join(" ".join(p) for p, _ in multi) + f"——佔 P 的 {share:.1%}。訓練時這些都在總和裡；cap 只動解碼。")
            self.log["sections"]["cap_vs_training_sum"] = {"multi_emit_paths": [p for p, _ in multi], "share_of_P": share}
        self.log["sections"]["decode_capped" if capped else "decode_uncapped"] = res

    # 選單 --------------------------------------------------------------
    def menu(self) -> None:
        o = self.out
        items = "   ".join(f"{it['key']} \"{it['label']}\" T={it['T']}" for it in self.paper["items"])
        o(f"\n{C.BOLD}第一段{C.RESET}  {items}")
        o(f"{C.BOLD}第二段{C.RESET}  4 隨機 y 窮舉   5 forward 表（逐格）   r 換一組 y")
        o(f"{C.BOLD}第三段{C.RESET}  6 greedy 有上限   u 拿掉上限        h 選單   q 離開")

    def dispatch(self, k: str) -> bool:
        for it in self.paper["items"]:
            if k == it["key"]:
                self.list_paths(it["label"], it["T"])
                return True
        if k == "4":
            self.enumerate_random()
        elif k == "5":
            self.forward_random(step=CFG["display"]["step_cells"] and sys.stdin.isatty())
        elif k == "r":
            self.draw_y(seed=int(time.time()) % 10_000_000)
            self.out(f"換了一組：seed = {self.seed}（按 4、5 重看；講稿上的數字是 seed 20261004 那組）")
        elif k == "6":
            self.decode(capped=True)
        elif k == "u":
            self.decode(capped=False)
        elif k == "h":
            self.menu()
        elif k == "q":
            return False
        return True

    def run_all(self) -> None:
        """課堂順序全部跑一遍（rehearse 用）。"""
        for it in self.paper["items"]:
            self.list_paths(it["label"], it["T"])
        self.enumerate_random()
        self.forward_random(step=False)
        self.decode(capped=True)
        self.decode(capped=False)


def interactive(d: Demo) -> None:
    d.out.banner("W4 Demo 1：RNN-T lattice — paths vs. forward, and the cap" + (f"   [{d.tag}]" if d.tag else ""))
    d.menu()
    while True:
        k = C.getkey()
        if not d.dispatch(k):
            break


def save(d: Demo, folder: Path) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    (folder / ("rehearsal.json" if folder.name == "rehearsal" else "session.json")).write_text(
        json.dumps(d.log, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    (folder / ("rehearsal.txt" if folder.name == "rehearsal" else "session.txt")).write_text(d.out.text(), encoding="utf-8")
    C.note(f"存到 {folder}/")


# ── selftest ────────────────────────────────────────────────────
def selftest() -> None:
    ok = True

    def check(name: str, cond: bool) -> None:
        nonlocal ok
        ok &= cond
        print(f"  [{'ok' if cond else 'FAIL'}] {name}")

    rc = CFG["random"]
    if DATA_JSON.exists():
        d = json.loads(DATA_JSON.read_text(encoding="utf-8"))
        for label, blk in d["paths"].items():
            U = len(label)
            check(f"{label}: U={blk['U']} r={blk['r']} ctc_min_T={blk['ctc_min_T']}",
                  blk["U"] == U and blk["r"] == repeats(label) and blk["ctc_min_T"] == U + repeats(label))
            for T, v in blk["by_T"].items():
                T = int(T)
                ps = rnnt_paths(T, U)
                check(f"{label} T={T}: RNN-T 路徑逐條同序（{len(ps)} 條）= C({T + U - 1},{U})",
                      ps == v["rnnt_paths"] and len(ps) == v["rnnt_count"] == math.comb(T + U - 1, U) == v["binom_T_plus_U_minus_1_choose_U"])
                cps = ctc_paths(label, T, "ab")
                w3 = ctc_from_w3(label, T)
                check(f"{label} T={T}: CTC 對照 {len(cps)} 條 = json" + ("，= w03-data.json" if w3 is not None else "（w03 無此格）"),
                      cps == v["ctc_paths"] and len(cps) == v["ctc_count"] and (w3 is None or w3 == cps))
        for label, blk in d["forward_check"].items():
            U = len(label)
            for T, v in blk.items():
                T = int(T)
                Yu = {"_label": label}
                for t in range(1, T + 1):
                    for u in range(0, U + 1):
                        Yu[(t, u)] = {"p": [1 / 3] * 3, "blank": 1 / 3, "emit": 1 / 3 if u < U else 0.0}
                _, P = forward(T, U, Yu)
                check(f"{label} T={T}: 均勻 forward {P:.15g} = json {v['uniform']['forward']:.15g}", P == v["uniform"]["forward"])
                Y = random_y(label, T, 20261004, "ab", rc["logit_std"])
                a, P = forward(T, U, Y)
                E = sum(path_prob(p, Y) for p in rnnt_paths(T, U))
                b = backward(T, U, Y)
                diag = [sum(a[t][u] * b[t][u] for t in range(1, T + 1) for u in range(0, U + 1) if t + u == n) for n in range(1, T + U + 1)]
                rv = v["random_seed_20261004"]
                same = P == rv["forward"] and E == rv["enumeration"]
                close = abs(P - rv["forward"]) < 1e-15 and abs(E - rv["enumeration"]) < 1e-15
                check(f"{label} T={T}: 隨機（seed 20261004）forward {P:.12g} / 窮舉 {E:.12g} vs json "
                      + ("逐位相同" if same else f"差 {abs(P - rv['forward']):.1e}（libm 最後一位；仍算過）" if close else "不同"), close)
                check(f"{label} T={T}: 對角線 Σαβ = P（{T + U} 條）", all(abs(x - P) < 1e-15 for x in diag) and rv["diagonal_sums_all_equal_P"])
    else:
        print(f"  （找不到 {DATA_JSON}，跳過與 w04-data.json 的比對）")

    bad = 0
    for label in ("ab", "aa", "aba", "abba", "bab"):
        U = len(label)
        for T in range(1, 6):
            for seed in (1, 2, 3):
                Y = random_y(label, T, seed, "ab", 1.5)
                ps = rnnt_paths(T, U)
                E = sum(path_prob(p, Y) for p in ps)
                _, P = forward(T, U, Y)
                LP = forward_log(T, U, Y)
                good = len(ps) == math.comb(T + U - 1, U) and abs(P - E) < 1e-12 * E and abs(math.exp(LP) - E) < 1e-12 * E
                good &= all(sum(1 for m in p if m == "E") == U and sum(1 for m in p if m == "R") == T and p[-1] == "R" for p in ps)
                if not good:
                    bad += 1
                    check(f"{label} T={T} seed={seed}: 窮舉 {E!r} forward {P!r} log {LP!r}", False)
    check("隨機 y：路徑數 = C(T+U−1,U)、每條 U 個 E 與 T 個 R 且以 R 結尾、窮舉 = forward = exp(log forward)，5 個標籤 × T = 1…5 × 3 個 seed", bad == 0)

    dc = CFG["decode"]
    J = Joint(dc)
    r1 = greedy(J, dc["max_symbols_per_step"], dc["safety_limit"])
    r2 = greedy(J, None, dc["safety_limit"])
    check(f"有上限：沒有 frame 超過 {dc['max_symbols_per_step']} 個 emit，frame {dc['blank_starved_frame']} 被強制 ∅，hyp = \"{r1['hyp']}\"",
          all(f["emits"] <= dc["max_symbols_per_step"] for f in r1["frames"]) and not r1["runaway"]
          and r1["frames"][dc["blank_starved_frame"] - 1]["forced_blank"])
    check(f"沒上限：在 frame {dc['blank_starved_frame']} 停不下來，碰到 safety_limit = {dc['safety_limit']}",
          r2["runaway"] and r2["frames"][-1]["t"] == dc["blank_starved_frame"])
    check("有上限時 frame 1、2 的輸出與沒上限時相同（上限只在被餓死的 frame 起作用）",
          [f["steps"] for f in r1["frames"][:dc["blank_starved_frame"] - 1]] == [f["steps"] for f in r2["frames"][:dc["blank_starved_frame"] - 1]])

    print("\nselftest:", "全部通過" if ok else "有失敗")
    sys.exit(0 if ok else 1)


# ── main ────────────────────────────────────────────────────────
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["show", "rehearse", "replay", "selftest"])
    args = ap.parse_args()

    if args.cmd == "selftest":
        selftest()
        return

    rehearsal = HERE / "runs" / "rehearsal"
    if args.cmd == "replay":
        txt = rehearsal / "rehearsal.txt"
        if not txt.exists():
            sys.exit(f"沒有彩排紀錄 {txt}。先跑 ./present.sh rehearse。")
        C.banner("REPLAY（課前彩排的輸出，不是現場算的）", C.AMBER)
        print(txt.read_text(encoding="utf-8"))
        return

    out = Out()
    if args.cmd == "rehearse":
        d = Demo(out, tag="rehearsal")
        d.run_all()
        save(d, rehearsal)
        print("\n下一步：./present.sh 上課；把 rehearsal.json 的 enumeration_random.sum、forward_random.P、abs_diff、"
              "decode_capped.hyp 與日期填進 p33／p31 講稿。")
        return

    d = Demo(out)
    try:
        interactive(d)
    finally:
        if d.log["sections"]:
            save(d, HERE / "runs" / "live" / C.now().replace(":", ""))


if __name__ == "__main__":
    main()

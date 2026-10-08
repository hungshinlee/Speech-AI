# -*- coding: utf-8 -*-
# 此檔由 bin/publish_demos.py 從 private repo 的 <course>/demos/ 複製而來，請勿在公開 repo 直接編輯。
"""W3 demo 2：CTC 路徑窮舉 vs. forward 表——同一個數，指數多條路徑加總，或 O(TS) 的動態規劃。

  show       課堂：單鍵選單（下面的鍵）。離開時把這次算過的東西存進 runs/live/<時間>/（不進版控）
  rehearse   課前：不互動，把所有段落照課堂順序跑一遍，存進 runs/rehearsal/（進版控 = 現場的退路）
  replay     現場 Python 出狀況時：印 runs/rehearsal/rehearsal.txt（純文字，不需要 numpy 以外的任何東西）
  selftest   對 slides/assets/w03/w03-data.json：路徑清單逐條、forward 值逐位；隨機 y 的窮舉 vs. forward；線性 vs. log

畫面上的鍵（順序照投影片講稿的【操作】）：
  1  "aa"  T = 3 的全部路徑      2  "aba" T = 3      3  "aa" T = 4（5 條）      0  "aa" T = 2（空集合）
  x  上一題的全部 K^T 條候選，各自摺成什麼（抓「a a ∅ 算進 aa」那種錯）
  4  隨機一組 y^t："ab"、T = 4，每條合法路徑的機率與總和（窮舉）
  5  同一組 y^t 逐格填 forward 表，最後 α_T(S) + α_T(S-1)，與窮舉比
  r  換一組隨機 y^t（seed 用時間；證明 4／5 不是預先湊好的）
  6  T = 500、符號表 32：線性域的 forward——α 全部歸零、P = 0、loss = inf
  l  同一份 T = 500：log 域——log P 有限，loss 是幾百，exp(log P) 在 float64 裡仍是 0
  h  重印選單      q  離開

純 numpy，跑 CPU。數字的定義：路徑 π ∈ V'^T，B 先摺疊相鄰重複再刪 blank，P(ℓ|X) = Σ_{π∈B⁻¹(ℓ)} Π_t y^t_{π_t}；
forward 的遞推、第三項的條件、邊界與終點都與投影片〈Forward on the CTC lattice〉逐字相同。
"""
from __future__ import annotations

import argparse
import io
import itertools
import json
import math
import re
import sys
import time
import tomllib
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import common as C  # noqa: E402

CFG = tomllib.loads((HERE / "demo_config.toml").read_text(encoding="utf-8"))
BLANK = "-"
DIG = CFG["display"]["digits"]
DATA_JSON = HERE.parent.parent / "slides" / "assets" / "w03" / "w03-data.json"


# ── 符號 ────────────────────────────────────────────────────────
def sym(s: str) -> str:
    return "∅" if s == BLANK else s


def show_path(p: str) -> str:
    return "".join(sym(c) for c in p)


def extend(label: str) -> list[str]:
    """ℓ' = (∅, ℓ_1, ∅, ℓ_2, …, ℓ_U, ∅)，長度 S = 2U + 1。"""
    lp = [BLANK]
    for c in label:
        lp += [c, BLANK]
    return lp


def repeats(label: str) -> int:
    """r = 相鄰重複的個數；最短合法路徑長 U + r。"""
    return sum(1 for u in range(len(label) - 1) if label[u] == label[u + 1])


# ── 路徑側：B 與窮舉 ────────────────────────────────────────────
def collapse(path) -> str:
    """B：先摺疊相鄰重複，再刪 blank。順序不能反（"a∅a" 要留兩個 a）。"""
    out, prev = [], None
    for p in path:
        if p != prev and p != BLANK:
            out.append(p)
        prev = p
    return "".join(out)


def candidates(vocab: str, T: int):
    """全部 K^T 條候選，順序 = itertools.product（與 make_figs_w03.py 相同）。"""
    return ("".join(p) for p in itertools.product(vocab + BLANK, repeat=T))


def enumerate_paths(label: str, T: int, vocab: str) -> list[str]:
    return [p for p in candidates(vocab, T) if collapse(p) == label]


def path_prob(p: str, y: np.ndarray, idx: dict) -> float:
    """Π_t y^t_{π_t}。y 的形狀是 (K, T)。"""
    v = 1.0
    for t, c in enumerate(p):
        v *= float(y[idx[c], t])
    return v


# ── 格點側：forward ─────────────────────────────────────────────
def allow3(lp: list[str], s: int) -> bool:
    """第三項的條件：ℓ'_s ≠ ∅ 且 ℓ'_s ≠ ℓ'_{s-2}（s 從 0 起算的內部索引）。"""
    return s >= 2 and lp[s] != BLANK and lp[s] != lp[s - 2]


def forward(label: str, y: np.ndarray, idx: dict) -> tuple[np.ndarray, float]:
    """線性域。回傳 α（形狀 (T, S)）與 P(ℓ|X) = α_T(S) + α_T(S-1)。"""
    lp = extend(label)
    S, T = len(lp), y.shape[1]
    a = np.zeros((T, S))
    a[0, 0] = y[idx[lp[0]], 0]
    a[0, 1] = y[idx[lp[1]], 0]
    for t in range(1, T):
        for s in range(S):
            v = a[t - 1, s]
            if s >= 1:
                v += a[t - 1, s - 1]
            if allow3(lp, s):
                v += a[t - 1, s - 2]
            a[t, s] = y[idx[lp[s]], t] * v
    return a, float(a[T - 1, S - 1] + a[T - 1, S - 2])


def forward_log(label: str, logy: np.ndarray, idx: dict) -> tuple[np.ndarray, float]:
    """log 域：乘變加、和變 log-sum-exp（np.logaddexp）。回傳 log α 與 log P。"""
    lp = extend(label)
    S, T = len(lp), logy.shape[1]
    a = np.full((T, S), -np.inf)
    a[0, 0] = logy[idx[lp[0]], 0]
    a[0, 1] = logy[idx[lp[1]], 0]
    for t in range(1, T):
        for s in range(S):
            v = a[t - 1, s]
            if s >= 1:
                v = np.logaddexp(v, a[t - 1, s - 1])
            if allow3(lp, s):
                v = np.logaddexp(v, a[t - 1, s - 2])
            a[t, s] = logy[idx[lp[s]], t] + v
    return a, float(np.logaddexp(a[T - 1, S - 1], a[T - 1, S - 2]))


# ── 隨機 y ──────────────────────────────────────────────────────
def random_y(symbols: list[str], T: int, seed: int, logit_std: float) -> tuple[np.ndarray, dict]:
    """每個 frame 一個 softmax(N(0, σ²)) 分布。形狀 (K, T)；idx 把符號對到列。"""
    rng = np.random.default_rng(seed)
    u = rng.normal(0.0, logit_std, size=(len(symbols), T))
    u -= u.max(axis=0, keepdims=True)
    y = np.exp(u)
    y /= y.sum(axis=0, keepdims=True)
    return y, {s: i for i, s in enumerate(symbols)}


def label_symbols(vocab: str) -> list[str]:
    return list(vocab) + [BLANK]


def big_symbols(K: int, label: str) -> list[str]:
    """K 個符號：label 用到的字母先放，其餘用 s1, s2, … 補，最後 blank。"""
    seen = []
    for c in label:
        if c not in seen:
            seen.append(c)
    fill = [f"s{i}" for i in range(1, K - len(seen))]
    return seen + fill + [BLANK]


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


# ── 各段落 ──────────────────────────────────────────────────────
class Demo:
    def __init__(self, out: Out, seed: int | None = None, tag: str = ""):
        self.out = out
        self.tag = tag
        self.paper = CFG["paper"]
        self.vocab = self.paper["vocab"]
        self.last_item: tuple[str, int] | None = None
        self.seed = CFG["random"]["seed"] if seed is None else seed
        self.log: dict = {"date": C.now(), "tag": tag, "versions": C.versions(), "config": CFG, "sections": {}}
        self.y = None

    # 第一段：列路徑 ---------------------------------------------------
    def list_paths(self, label: str, T: int) -> None:
        o = self.out
        r, S = repeats(label), 2 * len(label) + 1
        o.banner(f'ℓ = "{label}"  U = {len(label)}  r = {r}  →  最短合法路徑 U + r = {len(label) + r}      T = {T}')
        ps = enumerate_paths(label, T, self.vocab)
        K = len(self.vocab) + 1
        o(f"候選 K^T = {K}^{T} = {K ** T} 條；B 摺成 \"{label}\" 的：{C.BOLD}{len(ps)}{C.RESET} 條")
        if not ps:
            o(f"{C.RED}  （空集合）{C.RESET}  T = {T} < U + r = {len(label) + r}：兩個 {label[0]} 加一個隔開它們的 ∅ 塞不進 {T} 格 → P(ℓ|X) = 0，loss = ∞")
        for p in ps:
            o(f"  {C.BOLD}{show_path(p)}{C.RESET}")
        o(f"  S = 2U + 1 = {S}（這是 ℓ' 的長度，不是路徑的下限；下限是 U + r = {len(label) + r}）")
        self.last_item = (label, T)
        self.log["sections"][f"paths_{label}_T{T}"] = {"label": label, "T": T, "U": len(label), "r": r,
                                                       "count": len(ps), "paths": ps}

    def list_candidates(self) -> None:
        """上一題的全部 K^T 條候選，各自摺成什麼。抓常見錯。"""
        o = self.out
        if self.last_item is None:
            o("先按 0／1／2／3 選一題。")
            return
        label, T = self.last_item
        rows = [(p, collapse(p)) for p in candidates(self.vocab, T)]
        o.banner(f'全部 {len(rows)} 條候選，各自被 B 摺成什麼（目標 "{label}"）')
        width = 3
        cells = []
        for p, c in rows:
            mark = f"{C.GREEN}✓{C.RESET}" if c == label else " "
            cells.append(f"{show_path(p):<{T + 1}}→ {c or '（空）':<6}{mark}")
        for i in range(0, len(cells), width):
            o("  " + "   ".join(cells[i:i + width]))
        wrong = [p for p, c in rows if c != label and p.replace(BLANK, "") == label]
        if wrong:
            o(f"\n  常見錯：這些看起來像 \"{label}\"，但 B 先摺相鄰重複——" +
              "、".join(f"{show_path(p)} → {collapse(p)}" for p in wrong[:6]))

    # 第二段：隨機 y，窮舉 vs. forward ----------------------------------
    def draw_y(self, seed: int | None = None) -> None:
        rc = CFG["random"]
        if seed is not None:
            self.seed = seed
        syms = label_symbols(rc["vocab"])
        self.y, self.idx = random_y(syms, rc["T"], self.seed, rc["logit_std"])
        self.syms = syms
        self.enum_sum = None

    def print_y(self) -> None:
        o = self.out
        rc = CFG["random"]
        o(f"seed = {self.seed}；每個 frame 一個 softmax(N(0, {rc['logit_std']}²)) 分布，K = {len(self.syms)}，T = {rc['T']}")
        o("        " + "".join(f"{'t=' + str(t + 1):>{DIG + 4}}" for t in range(rc["T"])))
        for s in self.syms:
            o(f"  y_{sym(s):<3}  " + "".join(f"{fmt(self.y[self.idx[s], t]):>{DIG + 4}}" for t in range(rc["T"])))
        o("        " + "".join(f"{fmt(self.y[:, t].sum()):>{DIG + 4}}" for t in range(rc["T"])) + "   ← 每欄加起來是 1")

    def enumerate_random(self) -> None:
        o = self.out
        rc = CFG["random"]
        if self.y is None:
            self.draw_y()
        label, T = rc["label"], rc["T"]
        o.banner(f'窮舉：ℓ = "{label}"、T = {T}，隨機 y^t，每條合法路徑的機率' + (f"   [{self.tag}]" if self.tag else ""))
        self.print_y()
        ps = enumerate_paths(label, T, rc["vocab"])
        o(f"\n{len(self.syms) ** T} 條候選裡摺成 \"{label}\" 的 {len(ps)} 條：")
        total = 0.0
        rows = []
        for p in ps:
            v = path_prob(p, self.y, self.idx)
            total += v
            terms = " × ".join(fmt(self.y[self.idx[c], t]) for t, c in enumerate(p))
            o(f"  {C.BOLD}{show_path(p)}{C.RESET}   {terms} = {fmt(v)}")
            rows.append({"path": p, "prob": v})
        o(f"\n  {C.BOLD}Σ 窮舉 = {total:.{DIG + 6}f}{C.RESET}")
        self.enum_sum = total
        self.log["sections"]["enumeration_random"] = {"seed": self.seed, "label": label, "T": T,
                                                      "y": self.y.tolist(), "symbols": [sym(s) for s in self.syms],
                                                      "paths": rows, "sum": total}

    def forward_random(self, step: bool = True) -> None:
        o = self.out
        rc = CFG["random"]
        if self.y is None:
            self.draw_y()
        label, T = rc["label"], rc["T"]
        lp = extend(label)
        S = len(lp)
        a, P = forward(label, self.y, self.idx)
        o.banner(f'forward 表：同一組 y^t，ℓ\' = ({", ".join(sym(c) for c in lp)})，S = {S}，T = {T}' +
                 ("   （任一鍵下一欄，a 全部，q 中止）" if step else ""))
        o(f"  α_t(s) = y^t_{{ℓ'_s}} · ( α_{{t-1}}(s) + α_{{t-1}}(s-1) + [ℓ'_s ≠ ∅ ∧ ℓ'_s ≠ ℓ'_{{s-2}}] · α_{{t-1}}(s-2) )")
        o(f"  邊界：α_1(1) = y^1_∅，α_1(2) = y^1_{sym(lp[1])}，其餘 0。終點：P = α_T(S) + α_T(S-1)\n")
        shown = 0

        def table():
            o("   s  ℓ'_s " + "".join(f"{'t=' + str(t + 1):>{DIG + 4}}" for t in range(shown)))
            for s in range(S - 1, -1, -1):
                o(f"  {s + 1:>2}   {sym(lp[s]):<3}" + "".join(f"{fmt(a[t, s]):>{DIG + 4}}" for t in range(shown)))

        def detail(t: int):
            for s in range(S):
                ys = f"y^{t + 1}_{sym(lp[s])}"
                if t == 0:
                    o(f"    α_1({s + 1}) = {ys} = {fmt(a[0, s])}" if s < 2 else f"    α_1({s + 1}) = 0")
                    continue
                parts = [f"α_{t}({s + 1})"]
                vals = [a[t - 1, s]]
                if s >= 1:
                    parts.append(f"α_{t}({s})")
                    vals.append(a[t - 1, s - 1])
                if allow3(lp, s):
                    parts.append(f"α_{t}({s - 1})")
                    vals.append(a[t - 1, s - 2])
                elif s >= 2:
                    parts.append(f"{C.GREY}[×]{C.RESET}")
                o(f"    α_{t + 1}({s + 1}) = {ys} · ({' + '.join(parts)}) = {fmt(self.y[self.idx[lp[s]], t])} · ("
                  f"{' + '.join(fmt(v) for v in vals)}) = {fmt(a[t, s])}")

        while shown < T:
            shown += 1
            table()
            detail(shown - 1)
            o("")
            if step and shown < T:
                k = C.getkey()
                if k == "q":
                    o("（中止）")
                    return
                if k == "a":
                    step = False
        o(f"  {C.BOLD}P(ℓ|X) = α_{T}({S}) + α_{T}({S - 1}) = {fmt(a[T - 1, S - 1])} + {fmt(a[T - 1, S - 2])} = {P:.{DIG + 6}f}{C.RESET}")
        if self.enum_sum is None:
            self.enum_sum = sum(path_prob(p, self.y, self.idx) for p in enumerate_paths(label, T, rc["vocab"]))
        diff = abs(P - self.enum_sum)
        o(f"  Σ 窮舉      = {self.enum_sum:.{DIG + 6}f}")
        o(f"  |差|        = {diff:.2e}   {C.GREEN}（浮點誤差）{C.RESET}" if diff < 1e-12 else
          f"  |差|        = {diff:.2e}   {C.RED}（不是浮點誤差，程式有錯）{C.RESET}")
        o(f"  格數 T·S = {T * S}，候選 K^T = {len(self.syms) ** T}")
        self.log["sections"]["forward_random"] = {"seed": self.seed, "alpha": a.tolist(), "P": P,
                                                  "enumeration_sum": self.enum_sum, "abs_diff": diff}

    # 第三段：T = 500 --------------------------------------------------
    def long_run(self, log_domain: bool) -> None:
        o = self.out
        lc = CFG["long"]
        K, T, label = lc["K"], lc["T"], lc["label"]
        syms = big_symbols(K, label)
        y, idx = random_y(syms, T, lc["seed"], lc["logit_std"])
        lp = extend(label)
        S = len(lp)
        a, P = forward(label, y, idx)
        mx = a.max(axis=1)
        first_zero = next((t + 1 for t in range(T) if mx[t] == 0.0), None)
        first_sub = next((t + 1 for t in range(T) if 0 < mx[t] < sys.float_info.min), None)
        o.banner(f'T = {T}、K = |V\'| = {K}、ℓ = "{label}"（U = {len(label)}，r = {repeats(label)}，S = {S}）：'
                 + ("log 域" if log_domain else "線性域"))
        if not log_domain:
            o(f"  每個 frame 的 y^t 隨機（seed {lc['seed']}），平均每個符號 1/{K}；線性域逐格乘。")
            o(f"  {'t':>5}   {'max_s α_t(s)':>16}")
            for t in lc["checkpoints"]:
                v = mx[t - 1]
                flag = f"  {C.RED}← 0{C.RESET}" if v == 0.0 else (f"  {C.AMBER}← 次正規數（< 2.2e-308）{C.RESET}" if v < sys.float_info.min else "")
                o(f"  {t:>5}   {v:>16.3e}{flag}")
            o(f"\n  第一次掉進次正規數：t = {first_sub}；整欄全部變 0：t = {first_zero}")
            o(f"  {C.BOLD}P(ℓ|X) = α_T(S) + α_T(S-1) = {P!r}   →  loss = -ln P = {-math.log(P) if P > 0 else math.inf}{C.RESET}")
            o(f"  {C.RED}這個 ∞ 與 U + r 無關{C.RESET}：T = {T} ≫ U + r = {len(label) + repeats(label)}，合法路徑多得很；歸零的是 float64。")
            o(f"  對照：均勻 y、K = 3 時 3^-{T} = 10^{-T * math.log10(3):.1f}，還在 float64 範圍內——所以 T = {T} 要配夠大的符號表才看得到歸零。")
            self.log["sections"]["long_linear"] = {"K": K, "T": T, "label": label, "P": P,
                                                   "first_subnormal_t": first_sub, "first_zero_t": first_zero,
                                                   "max_alpha_at_checkpoints": {str(t): float(mx[t - 1]) for t in lc["checkpoints"]}}
            return
        la, logP = forward_log(label, np.log(y), idx)
        lmx = la.max(axis=1)
        o(f"  同一組 y^t，同一條遞推：乘 → 加，加 → logaddexp。")
        o(f"  {'t':>5}   {'max_s α_t(s)（線性）':>22}   {'max_s ln α_t(s)（log 域）':>26}")
        for t in lc["checkpoints"]:
            o(f"  {t:>5}   {mx[t - 1]:>22.3e}   {lmx[t - 1]:>26.3f}")
        o(f"\n  {C.BOLD}ln P(ℓ|X) = logaddexp(α_T(S), α_T(S-1)) = {logP:.3f}   →  loss = -ln P = {-logP:.3f}{C.RESET}")
        o(f"  同一個數換回線性域：exp({logP:.1f}) 在 float64 裡 = {math.exp(logP) if logP > -745 else 0.0!r}"
          f"（最小次正規數 5e-324 = e^{math.log(5e-324):.1f}）")
        o(f"  PyTorch 的 ctc_loss 內部就是 log 域；`inf` 出現時，數值是診斷流程的第三步，不是第一步。")
        self.log["sections"]["long_log"] = {"K": K, "T": T, "label": label, "logP": logP, "loss": -logP,
                                            "max_logalpha_at_checkpoints": {str(t): float(lmx[t - 1]) for t in lc["checkpoints"]}}

    # 選單 --------------------------------------------------------------
    def menu(self) -> None:
        o = self.out
        items = "   ".join(f"{it['key']} \"{it['label']}\" T={it['T']}" for it in self.paper["items"])
        o(f"\n{C.BOLD}第一段{C.RESET}  {items}   x 全部候選")
        o(f"{C.BOLD}第二段{C.RESET}  4 隨機 y 窮舉   5 forward 表   r 換一組 y")
        o(f"{C.BOLD}第三段{C.RESET}  6 T=500 線性域   l T=500 log 域        h 選單   q 離開")

    def dispatch(self, k: str) -> bool:
        for it in self.paper["items"]:
            if k == it["key"]:
                self.list_paths(it["label"], it["T"])
                return True
        if k == "x":
            self.list_candidates()
        elif k == "4":
            self.enumerate_random()
        elif k == "5":
            self.forward_random(step=CFG["display"]["step_columns"] and sys.stdin.isatty())
        elif k == "r":
            self.draw_y(seed=int(time.time()) % 10_000_000)
            self.out(f"換了一組：seed = {self.seed}（按 4、5 重看）")
        elif k == "6":
            self.long_run(log_domain=False)
        elif k == "l":
            self.long_run(log_domain=True)
        elif k == "h":
            self.menu()
        elif k == "q":
            return False
        return True

    def run_all(self) -> None:
        """課堂順序全部跑一遍（rehearse 用）。"""
        for it in self.paper["items"]:
            self.list_paths(it["label"], it["T"])
        self.last_item = (self.paper["items"][0]["label"], self.paper["items"][0]["T"])
        self.list_candidates()
        self.enumerate_random()
        self.forward_random(step=False)
        self.long_run(log_domain=False)
        self.long_run(log_domain=True)


def interactive(d: Demo) -> None:
    d.out.banner(f"W3 Demo 2：paths vs. forward" + (f"   [{d.tag}]" if d.tag else ""))
    d.menu()
    while True:
        k = C.getkey()
        if not d.dispatch(k):
            break


def save(d: Demo, folder: Path) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    (folder / ("rehearsal.json" if folder.name == "rehearsal" else "session.json")).write_text(
        json.dumps(d.log, ensure_ascii=False, indent=1), encoding="utf-8")
    (folder / ("rehearsal.txt" if folder.name == "rehearsal" else "session.txt")).write_text(d.out.text(), encoding="utf-8")
    C.note(f"存到 {folder}/")


# ── selftest ────────────────────────────────────────────────────
def selftest() -> None:
    ok = True

    def check(name: str, cond: bool) -> None:
        nonlocal ok
        ok &= cond
        print(f"  [{'ok' if cond else 'FAIL'}] {name}")

    if DATA_JSON.exists():
        d = json.loads(DATA_JSON.read_text(encoding="utf-8"))
        for label, blk in d["paths"].items():
            check(f"{label}: U={blk['U']} r={blk['r']} min_T={blk['min_T']}",
                  blk["U"] == len(label) and blk["r"] == repeats(label) and blk["min_T"] == len(label) + repeats(label))
            for T, v in blk["by_T"].items():
                ps = enumerate_paths(label, int(T), "ab")
                check(f"{label} T={T}: 路徑清單逐條相同（{len(ps)} 條）", ps == v["paths"] and len(ps) == v["count"])
        for label, blk in d["forward_uniform"].items():
            for T, v in blk.items():
                T = int(T)
                y = np.full((3, T), 1 / 3)
                _, P = forward(label, y, {"a": 0, "b": 1, BLANK: 2})
                check(f"{label} T={T}: forward（均勻 y）= {P:.15g} vs json {v['forward']:.15g}", abs(P - v["forward"]) < 1e-15)
    else:
        print(f"  （找不到 {DATA_JSON}，跳過與 w03-data.json 的比對）")

    rng_seeds = (1, 2, 3)
    for label, vocab in (("ab", "ab"), ("aa", "ab"), ("aba", "ab"), ("abba", "ab"), ("aab", "abc"), ("hello", "helo")):
        syms = label_symbols(vocab)
        for T in range(1, 7):
            for seed in rng_seeds:
                y, idx = random_y(syms, T, seed, 1.5)
                ps = enumerate_paths(label, T, vocab)
                E = sum(path_prob(p, y, idx) for p in ps)
                _, P = forward(label, y, idx)
                _, LP = forward_log(label, np.log(y), idx)
                tol = 1e-12 * max(1.0, E)
                good = abs(P - E) < tol and ((E == 0 and LP == -np.inf) or abs(math.exp(LP) - E) < tol)
                if not good:
                    check(f"{label} T={T} seed={seed}: 窮舉 {E!r} forward {P!r} log {LP!r}", False)
        zero_T = len(label) + repeats(label) - 1
        y, idx = random_y(syms, zero_T, 1, 1.5)
        check(f"{label}: T = U + r - 1 = {zero_T} → 窮舉空、forward 0", enumerate_paths(label, zero_T, vocab) == [] and forward(label, y, idx)[1] == 0.0)
    check("隨機 y：窮舉 = forward = exp(log forward)，6 個標籤 × T = 1…6 × 3 個 seed", ok)

    lc = CFG["long"]
    syms = big_symbols(lc["K"], lc["label"])
    y, idx = random_y(syms, lc["T"], lc["seed"], lc["logit_std"])
    _, P = forward(lc["label"], y, idx)
    _, LP = forward_log(lc["label"], np.log(y), idx)
    check(f"T = {lc['T']}、K = {lc['K']}：線性域 P = {P!r}（要是 0）", P == 0.0)
    check(f"T = {lc['T']}、K = {lc['K']}：log 域 ln P = {LP:.2f}（要有限）", math.isfinite(LP))
    y3, idx3 = random_y(label_symbols("ab"), lc["T"], lc["seed"], lc["logit_std"])
    _, P3 = forward("ab", y3, idx3)
    check(f"對照：K = 3、T = {lc['T']}、隨機 y 線性域 P = {P3:.3e}（次正規數：還沒歸零、精度只剩幾個 bit；均勻 y 是 3^-{lc['T']} = 10^{-lc['T'] * math.log10(3):.1f}）",
          0.0 < P3 < sys.float_info.min)
    y3, idx3 = random_y(label_symbols("ab"), 200, lc["seed"], lc["logit_std"])
    _, P3 = forward("ab", y3, idx3)
    _, LP3 = forward_log("ab", np.log(y3), idx3)
    check(f"K = 3、T = 200：log 域與線性域一致（P = {P3:.3e}，rel {abs(math.exp(LP3) - P3) / P3:.1e}）", abs(math.exp(LP3) - P3) / P3 < 1e-9)

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
        print("\n下一步：./present.sh 上課；把 rehearsal.json 的 enumeration_random.sum、forward_random.P、long_log.loss 填進 p29 講稿。")
        return

    d = Demo(out)
    try:
        interactive(d)
    finally:
        if d.log["sections"]:
            save(d, HERE / "runs" / "live" / C.now().replace(":", ""))


if __name__ == "__main__":
    main()

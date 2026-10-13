# -*- coding: utf-8 -*-
# 此檔由 bin/publish_demos.py 從 private repo 的 <course>/demos/ 複製而來，請勿在公開 repo 直接編輯。
"""W5 demo 3：InfoNCE 的天花板——已知互資訊的高斯對，N 從 2 掃到 1024。

  show       課堂：單鍵選單（下面的鍵）。離開時把這次算過的東西存進 runs/live/<時間>/（不進版控）
  rehearse   課前：不互動，照課堂順序 1 → 2 → 3 → b 全部跑一遍，存進 runs/rehearsal/（進版控 = 現場的退路；圖存成 PNG）
  replay     現場 Python 出狀況時：印 runs/rehearsal/rehearsal.txt（純文字），圖用 runs/rehearsal/*.png
  selftest   對 slides/assets/w05/w05-data.json → infonce_ceiling：同一個 seed 重跑整個掃描，十個 N 的估計值、L_N、標準差逐一比對
             （容忍 1e-9；順便報有幾個逐位相同）；再查不等式該有的性質：估計值 ≤ ln N、估計值 < 真值、隨 N 不減、變異隨 N 降

畫面上的鍵（順序照投影片 p13 講稿的【操作】）：
  1  真值與天花板：I = −(d/2) ln(1 − ρ²)，各 N 的 ln N；天花板在真值底下的 N 標出來；天花板過真值要多少 N
  2  掃 N：逐個 N 訓練 critic（Adam 固定步數），每訓完一個印 ln N − L_N、離天花板多遠、離真值多遠、耗時；圖上多一個點
  3  變異：固定幾個 N（預設 2、32、1024），200 個 held-out batch 上的估計值直方圖與標準差（用 2 訓好的 critic；沒按過 2 會先跑）
  b  純雙線性 critic（沒有 a‖y‖² 那一項）在幾個 N 的估計值，與 2 的對照——critic 為什麼要多一項
  r  換 seed（用時間；再按 2 證明曲線不是預先湊好的；講稿的數字是 seed 20261011 那組）
  h  重印選單      q  離開

純 numpy（matplotlib 只畫圖），跑 CPU。數字的定義：
  資料  每一維 y = ρx + sqrt(1 − ρ²) ε，x、ε ~ N(0, 1)，d 維獨立；真實互資訊 I = −(d/2) ln(1 − ρ²)（閉式解）。
  critic  f(x, y) = exp(xᵀWy + a‖y‖²)。高斯對的 log 密度比是 x、y 的二次式，x 的平方項在列內 softmax 裡消掉、‖y‖² 那一項消不掉，
          所以純雙線性的 critic 學不到最佳 critic；多一個純量 a 就夠了（鍵 b 看差多少）。
  loss    一個 batch 是 N 個 pair，S_ij = f(x_i, y_j)，對角線是正樣本、同一列其餘 N − 1 個是負樣本；
          L_N = −mean_i log softmax(S)_ii。估計值 = ln N − L_N（在 200 個 held-out batch 上平均）——投影片 p11 第五行：
          對任何 critic 都有 I ≥ ln N − L_N，而 L_N ≥ 0，所以估計值永遠 ≤ ln N。
訓練迴圈與 slides/scripts/make_figs_w05.py 的 infonce_sweep() 逐行相同（同一個 rng、同一個抽樣順序），
所以 seed 20261011 時與 w05-data.json 的十個數字對得上（selftest 就是在對這件事）。
"""
from __future__ import annotations

import argparse
import io
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
DIG = CFG["display"]["digits"]
DATA_JSON = HERE.parent.parent / "slides" / "assets" / "w05" / "w05-data.json"


# ── 資料與真值 ───────────────────────────────────────────────────
def true_mi(d: int, rho: float) -> float:
    """I(x; y) = −(d/2) ln(1 − ρ²)：每一維 h(y) − h(y | x) = ½ ln(1/(1 − ρ²))，d 維獨立相加。"""
    return -0.5 * d * math.log(1 - rho * rho)


def draw_pair(rng: np.random.Generator, N: int, d: int, rho: float) -> tuple[np.ndarray, np.ndarray]:
    """抽一個 batch：N 個 (x, y) pair。抽樣順序與 make_figs_w05.py 相同（先 X 再 ε），不能換。"""
    X = rng.standard_normal((N, d))
    Y = rho * X + math.sqrt(1 - rho * rho) * rng.standard_normal((N, d))
    return X, Y


# ── critic 與 InfoNCE ────────────────────────────────────────────
def scores(X: np.ndarray, Y: np.ndarray, W: np.ndarray, a: float) -> np.ndarray:
    """S_ij = x_iᵀ W y_j + a ‖y_j‖²（a = 0 就是純雙線性）。"""
    ny = (Y * Y).sum(axis=1)
    return X @ W @ Y.T + a * ny[None, :]


def infonce(S: np.ndarray) -> float:
    """L_N = −mean_i log softmax(S)_ii（列內 softmax：每一列一個正樣本、N − 1 個負樣本）。"""
    S = S - S.max(axis=1, keepdims=True)
    logP = S - np.log(np.exp(S).sum(axis=1, keepdims=True))
    return -float(np.mean(np.diag(logP)))


def train_critic(rng: np.random.Generator, N: int, d: int, rho: float, steps: int, lr: float,
                 learn_a: bool = True) -> tuple[np.ndarray, float]:
    """Adam 訓練 W（與 a）。與 make_figs_w05.py infonce_sweep() 的訓練迴圈逐行相同；learn_a=False 時 a 固定 0（鍵 b 用），
    抽樣順序不變。"""
    W = np.zeros((d, d))
    a = 0.0
    m = np.zeros_like(W); v = np.zeros_like(W); ma = 0.0; va = 0.0
    b1, b2, eps = 0.9, 0.999, 1e-8
    for it in range(1, steps + 1):
        X, Y = draw_pair(rng, N, d, rho)
        ny = (Y * Y).sum(axis=1)                     # ‖y_j‖²
        S = X @ W @ Y.T + a * ny[None, :]            # N×N scores
        S = S - S.max(axis=1, keepdims=True)
        P = np.exp(S)
        P /= P.sum(axis=1, keepdims=True)
        dS = (P - np.eye(N)) / N                     # ∂L/∂S，L = −mean(log P_ii)
        G = X.T @ dS @ Y
        Ga = float((dS * ny[None, :]).sum())
        m = b1 * m + (1 - b1) * G; v = b2 * v + (1 - b2) * G * G
        ma = b1 * ma + (1 - b1) * Ga; va = b2 * va + (1 - b2) * Ga * Ga
        W -= lr * (m / (1 - b1 ** it)) / (np.sqrt(v / (1 - b2 ** it)) + eps)
        if learn_a:
            a -= lr * (ma / (1 - b1 ** it)) / (math.sqrt(va / (1 - b2 ** it)) + eps)
    return W, a


def evaluate(rng: np.random.Generator, W: np.ndarray, a: float, N: int, d: int, rho: float, batches: int) -> list[float]:
    """held-out：batches 個新 batch 各算一個 L_N。"""
    Ls = []
    for _ in range(batches):
        X, Y = draw_pair(rng, N, d, rho)
        Ls.append(infonce(scores(X, Y, W, a)))
    return Ls


# ── 輸出（同時寫到終端機與 transcript）───────────────────────────
class Out:
    def __init__(self):
        self.buf = io.StringIO()

    def __call__(self, text: str = "") -> None:
        print(text, flush=True)
        self.buf.write(re.sub(r"\033\[[0-9;]*m", "", text) + "\n")

    def banner(self, text: str, color: str = C.BLUE) -> None:
        C.banner(text, color)
        self.buf.write("\n" + "─" * 72 + "\n" + text + "\n" + "─" * 72 + "\n")

    def text(self) -> str:
        return self.buf.getvalue()


def fmt(v: float) -> str:
    return f"{v:.{DIG}f}"


# ── 圖 ──────────────────────────────────────────────────────────
class Plots:
    """一張天花板圖（鍵 2 逐點加）、一張直方圖（鍵 3）。headless 時存 PNG。"""

    def __init__(self, enabled: bool, headless: bool, save_dir: Path | None):
        self.enabled = enabled
        self.headless = headless
        self.save_dir = save_dir
        self.plt = None
        self.ceiling = None
        if not enabled:
            return
        import matplotlib
        if headless:
            matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        self.plt = plt
        if not headless:
            plt.ion()

    def ceiling_axes(self, Ns: list[int], I_true: float, d: int, rho: float):
        plt = self.plt
        fig, ax = plt.subplots(num="W5 Demo 3 — the ceiling at ln N", figsize=(8.5, 4.6))
        ax.set_xscale("log", base=2)
        ax.set_xticks(Ns); ax.set_xticklabels([str(n) for n in Ns])
        ax.set_xlabel(r"$N = |\mathcal{Z}|$  (one positive + $N-1$ negatives)")
        ax.set_ylabel("nats")
        ax.set_ylim(0, math.log(Ns[-1]) * 1.05)
        ax.grid(True, alpha=0.3)
        ax.plot(Ns, [math.log(n) for n in Ns], "--", color="0.45", lw=1.6, label=r"$\ln N$  (the ceiling)")
        ax.axhline(I_true, color="#2a9d4a", lw=2, label=r"$I(\mathbf{z};\mathbf{c}) = %.2f$  ($d=%d,\ \rho=%.1f$)" % (I_true, d, rho))
        (self.est_line,) = ax.plot([], [], "o-", color="#d1495b", lw=2.2, ms=5, label=r"$\ln N - \mathcal{L}_N$  (trained critic)")
        self.est_err = None
        ax.legend(loc="upper left", fontsize=9)
        fig.tight_layout()
        self.ceiling = (fig, ax)
        self._flush()

    def add_point(self, rows: list[dict]) -> None:
        if not self.ceiling:
            return
        fig, ax = self.ceiling
        xs = [r["N"] for r in rows]; ys = [r["estimate"] for r in rows]; es = [r["estimate_std_over_batches"] for r in rows]
        self.est_line.set_data(xs, ys)
        if self.est_err is not None:
            self.est_err.remove()
        self.est_err = ax.errorbar(xs, ys, yerr=es, fmt="none", ecolor="#d1495b", elinewidth=1, capsize=3, alpha=0.8)
        self._flush()

    def add_bilinear(self, rows: list[dict]) -> None:
        if not self.ceiling:
            return
        fig, ax = self.ceiling
        ax.plot([r["N"] for r in rows], [r["estimate"] for r in rows], "s", color="#e0a100", ms=6,
                label=r"bilinear only, $a \equiv 0$")
        ax.legend(loc="upper left", fontsize=9)
        self._flush()

    def histograms(self, per_batch: dict[int, list[float]], I_true: float) -> None:
        plt = self.plt
        Ns = sorted(per_batch)
        fig, axes = plt.subplots(1, len(Ns), num="W5 Demo 3 — variance over held-out batches", figsize=(3.2 * len(Ns), 3.6), sharey=True)
        axes = np.atleast_1d(axes)
        lo = min(min(per_batch[N]) for N in Ns) - 0.2
        hi = max(I_true, max(max(per_batch[N]) for N in Ns)) + 0.4
        edges = np.linspace(lo, hi, 90)            # 三格同一個 x 軸、同一組 bin，寬窄才能比
        for ax, N in zip(axes, Ns):
            v = np.asarray(per_batch[N])
            ax.hist(v, bins=edges, color="#d1495b", alpha=0.75)
            if lo <= math.log(N) <= hi:
                ax.axvline(math.log(N), color="0.45", ls="--", lw=1.4)
            ax.axvline(I_true, color="#2a9d4a", lw=1.6)
            ax.set_xlim(lo, hi)
            ax.set_title(r"$N=%d$:  std %.3f,  $\ln N = %.2f$" % (N, float(np.std(v)), math.log(N)), fontsize=10)
            ax.set_xlabel(r"$\ln N - \mathcal{L}_N$ per batch")
        axes[0].set_ylabel("batches")
        fig.tight_layout()
        self._flush("variance.png", fig)

    def _flush(self, name: str = "ceiling.png", fig=None) -> None:
        if not self.enabled:
            return
        if self.headless:
            if self.save_dir:
                self.save_dir.mkdir(parents=True, exist_ok=True)
                (fig or self.ceiling[0]).savefig(self.save_dir / name, dpi=130)
        else:
            self.plt.pause(0.05)

    def save_all(self, folder: Path) -> None:
        if not self.enabled or self.headless:
            return
        folder.mkdir(parents=True, exist_ok=True)
        for num in self.plt.get_fignums():
            f = self.plt.figure(num)
            f.savefig(folder / (re.sub(r"[^a-z0-9]+", "_", f.get_label().lower()).strip("_") + ".png"), dpi=130)

    def pump(self) -> None:
        if self.enabled and not self.headless:
            self.plt.pause(0.01)


# ── 各段落 ──────────────────────────────────────────────────────
class Demo:
    def __init__(self, out: Out, plots: Plots, seed: int | None = None, tag: str = ""):
        self.out = out
        self.plots = plots
        self.tag = tag
        sw = CFG["sweep"]
        self.seed = sw["seed"] if seed is None else seed
        self.d, self.rho = sw["d"], sw["rho"]
        self.Ns = list(sw["Ns"])
        self.steps, self.lr, self.eval_batches = sw["steps"], sw["lr"], sw["eval_batches"]
        self.I_true = true_mi(self.d, self.rho)
        self.rows: list[dict] = []           # 鍵 2 的結果（照 Ns 順序）
        self.per_batch: dict[int, list[float]] = {}
        self.log: dict = {"date": C.now(), "tag": tag, "versions": C.versions(), "config": CFG, "seed": self.seed,
                          "I_true_nats": self.I_true, "sections": {}}

    # 1：真值與天花板 ---------------------------------------------------
    def truth_table(self) -> None:
        o = self.out
        o.banner("1  真值與天花板：I 有閉式解，ln N 是 InfoNCE 估計值的上限")
        o(f"資料：d = {self.d} 維高斯對，每一維 y = ρx + √(1−ρ²)·ε，ρ = {self.rho}")
        o(f"真值：I(x; y) = −(d/2)·ln(1 − ρ²) = −{self.d / 2:g}·ln({1 - self.rho ** 2:.2f}) = {C.BOLD}{self.I_true:.4f} nat{C.RESET}")
        N_star = math.exp(self.I_true)
        o(f"天花板過真值的條件：ln N > I  ⇔  N > e^I = {N_star:.1f}，也就是至少 {math.ceil(N_star)} 個樣本（{math.ceil(N_star) - 1} 個負樣本）")
        o()
        o(f"{'N':>6} {'負樣本':>6} {'ln N':>8}   天花板 vs 真值")
        rows = []
        for N in self.Ns:
            lnN = math.log(N)
            below = lnN < self.I_true
            mark = f"{C.RED}在真值底下 — 估計值碰不到 {self.I_true:.2f}，不管訓練多久{C.RESET}" if below else f"{C.GREEN}過了真值{C.RESET}"
            o(f"{N:>6} {N - 1:>6} {lnN:>8.{DIG}f}   {mark}")
            rows.append({"N": N, "ln_N": lnN, "ceiling_below_truth": below})
        o()
        o(f"{C.DIM}（p13 的問題：N = 8 時估計值碰得到真值嗎？ln 8 = {math.log(8):.3f} < {self.I_true:.3f}，答案是不可能——不是 critic 不夠好，是不等式不允許。）{C.RESET}")
        self.log["sections"]["truth"] = {"I_true_nats": self.I_true, "N_star": N_star, "rows": rows}

    # 2：掃 N ------------------------------------------------------------
    def sweep(self) -> None:
        o = self.out
        o.banner(f"2  掃 N：每個 N 訓一個 critic（Adam {self.steps} 步、lr {self.lr}），{self.eval_batches} 個 held-out batch 上算 ln N − L_N" +
                 (f"   [seed {self.seed}]" if self.seed != CFG["sweep"]["seed"] else ""))
        o(f"critic f(x, y) = exp(xᵀWy + a‖y‖²)，W 是 {self.d}×{self.d}，從 0 起訓；seed = {self.seed}")
        o()
        o(f"{'N':>6} {'ln N':>8} {'估計值':>8} {'L_N':>8}  {'離天花板':>8} {'離真值':>8} {'std':>7} {'秒':>6}")
        rng = np.random.default_rng(self.seed)
        self.rows, self.per_batch = [], {}
        if self.plots.enabled and self.plots.ceiling is None:
            self.plots.ceiling_axes(self.Ns, self.I_true, self.d, self.rho)
        t_all = time.time()
        for N in self.Ns:
            t0 = time.time()
            W, a = train_critic(rng, N, self.d, self.rho, self.steps, self.lr)
            Ls = evaluate(rng, W, a, N, self.d, self.rho, self.eval_batches)
            dt = time.time() - t0
            L_N = float(np.mean(Ls))
            est = math.log(N) - L_N
            ests = [math.log(N) - l for l in Ls]
            row = {"N": N, "ln_N": math.log(N), "L_N": L_N, "estimate": est,
                   "estimate_std_over_batches": float(np.std(ests)),
                   "estimate_le_ln_N": est <= math.log(N) + 1e-9,
                   "gap_to_ceiling": math.log(N) - est, "gap_to_truth": self.I_true - est,
                   "a": a, "W_fro": float(np.linalg.norm(W)), "seconds": dt}
            self.rows.append(row)
            self.per_batch[N] = ests
            hot = (N == 8)
            est_s = f"{C.BOLD}{est:>8.{DIG}f}{C.RESET}" if hot else f"{est:>8.{DIG}f}"
            o(f"{N:>6} {row['ln_N']:>8.{DIG}f} {est_s} {L_N:>8.{DIG}f}  "
              f"{row['gap_to_ceiling']:>8.{DIG}f} {row['gap_to_truth']:>8.{DIG}f} {row['estimate_std_over_batches']:>7.3f} {dt:>6.1f}"
              + (f"   {C.RED}← 天花板 {math.log(8):.2f} < 真值 {self.I_true:.2f}：被壓著{C.RESET}" if hot else ""))
            self.plots.add_point(self.rows)
        total = time.time() - t_all
        o()
        r = {row["N"]: row for row in self.rows}
        if 64 in r and 1024 in r:
            o(f"64 → 1024：負樣本 {C.BOLD}16 倍{C.RESET}，估計值只多 {r[1024]['estimate'] - r[64]['estimate']:.2f} nat")
        o("標準差（200 個 batch）：" + "、".join(f"N={x['N']} {x['estimate_std_over_batches']:.3f}" for x in self.rows if x["N"] in (2, 64, 1024)))
        o(f"每一個 N 的估計值都 ≤ ln N：{all(x['estimate_le_ln_N'] for x in self.rows)}；都 < 真值：{all(x['estimate'] < self.I_true for x in self.rows)}"
          f"（p11 第五行：任何 critic 都是下界，而我們的 critic 是有限步數訓出來的）")
        o(f"{C.DIM}全部 {total:.0f} 秒。{C.RESET}")
        self.log["sections"]["sweep"] = {"seed": self.seed, "rows": self.rows, "seconds_total": total,
                                         "per_batch_estimates": {str(k): v for k, v in self.per_batch.items()}}
        self.compare_json()

    def compare_json(self) -> None:
        """seed 與 json 相同時，順手報與 w05-data.json 的差（selftest 的簡版）。"""
        if not DATA_JSON.exists() or self.seed != CFG["sweep"]["seed"]:
            return
        j = json.loads(DATA_JSON.read_text(encoding="utf-8"))["infonce_ceiling"]
        ref = {r["N"]: r for r in j["sweep"]}
        diffs = [abs(ref[r["N"]]["estimate"] - r["estimate"]) for r in self.rows if r["N"] in ref]
        if diffs:
            same = sum(1 for d_ in diffs if d_ == 0.0)
            self.out(f"{C.DIM}對 w05-data.json：{len(diffs)} 個 N，最大差 {max(diffs):.1e}，逐位相同 {same} 個（差只來自 BLAS 的加總順序）。{C.RESET}")
            self.log["sections"]["sweep"]["vs_w05_data_json"] = {"max_abs_diff_estimate": max(diffs), "bitwise_equal": same, "n": len(diffs)}

    # 3：變異 --------------------------------------------------------------
    def variance(self) -> None:
        o = self.out
        if not self.rows:
            o(f"{C.AMBER}還沒掃過 N，先跑 2。{C.RESET}")
            self.sweep()
        Ns = [N for N in CFG["variance"]["Ns"] if N in self.per_batch]
        o.banner(f"3  變異：同一個 critic，{self.eval_batches} 個 held-out batch 各算一次 ln N − L_N")
        o(f"{'N':>6} {'平均':>8} {'std':>7}  {'最小':>8} {'最大':>8}   有幾個 batch 超過真值")
        res = []
        for N in Ns:
            v = np.asarray(self.per_batch[N])
            over = int((v > self.I_true).sum())
            o(f"{N:>6} {v.mean():>8.{DIG}f} {v.std():>7.3f}  {v.min():>8.{DIG}f} {v.max():>8.{DIG}f}   {over}/{len(v)}")
            res.append({"N": N, "mean": float(v.mean()), "std": float(v.std()), "min": float(v.min()), "max": float(v.max()),
                        "batches_over_truth": over})
        o()
        o("天花板過了真值之後，再加 N 買到的是這一欄變小——不是新的資訊。單一 batch 可以越過真值（它不是下界，平均才是），std 大的 N 更常見。")
        self.plots.histograms({N: self.per_batch[N] for N in Ns}, self.I_true)
        self.log["sections"]["variance"] = res

    # b：純雙線性 ----------------------------------------------------------
    def bilinear(self) -> None:
        o = self.out
        Ns = list(CFG["bilinear"]["Ns"])
        o.banner("b  純雙線性 critic（a ≡ 0）：少了 ‖y‖² 那一項，最佳 critic 不在模型族裡")
        o("高斯對的 log 密度比 ln p(y|x)/p(y) 是 x、y 的二次式：xᵀ(·)y 之外還有 ‖y‖² 一項（與 x 無關的 ‖x‖² 項在列內 softmax 裡消掉，‖y‖² 消不掉）。")
        a_star = -self.rho ** 2 / (2 * (1 - self.rho ** 2))
        o(f"最佳 critic 的那一項：a* = −ρ²/(2(1−ρ²)) = {a_star:+.3f}（鍵 2 學到的 a 在最後一欄，拿來對）")
        o(f"{'N':>6} {'ln N':>8} {'雙線性':>8} {'多一項 a':>9} {'差':>7}   (a 的學到值)")
        rng = np.random.default_rng(self.seed + 1)   # 另一條 rng：這一段不在 json 裡，不必與 2 共用抽樣
        full = {r["N"]: r for r in self.rows}
        res = []
        for N in Ns:
            W, a0 = train_critic(rng, N, self.d, self.rho, self.steps, self.lr, learn_a=False)
            Ls = evaluate(rng, W, a0, N, self.d, self.rho, self.eval_batches)
            est_b = math.log(N) - float(np.mean(Ls))
            if N in full:
                est_f, a_f = full[N]["estimate"], full[N]["a"]
            else:
                Wf, a_f = train_critic(rng, N, self.d, self.rho, self.steps, self.lr)
                est_f = math.log(N) - float(np.mean(evaluate(rng, Wf, a_f, N, self.d, self.rho, self.eval_batches)))
            o(f"{N:>6} {math.log(N):>8.{DIG}f} {est_b:>8.{DIG}f} {est_f:>9.{DIG}f} {est_f - est_b:>7.{DIG}f}   (a = {a_f:+.3f})")
            res.append({"N": N, "estimate_bilinear": est_b, "estimate_full": est_f, "a_full": a_f})
        o()
        o("兩個都是合法的下界（p11 第五行對任何 critic 成立），差的是**鬆緊**：critic 族裡沒有最佳 critic，下界就緊不起來。")
        self.plots.add_bilinear([{"N": r["N"], "estimate": r["estimate_bilinear"]} for r in res])
        self.log["sections"]["bilinear"] = {"a_star": a_star, "rows": res}

    # 選單 ---------------------------------------------------------------
    def menu(self) -> None:
        o = self.out
        o(f"\n{C.BOLD}鍵{C.RESET}  1 真值與天花板   2 掃 N（訓 critic，逐點畫）   3 變異（直方圖）")
        o(f"    b 純雙線性對照   r 換 seed   h 選單   q 離開")

    def dispatch(self, k: str) -> bool:
        if k == "1":
            self.truth_table()
        elif k == "2":
            self.sweep()
        elif k == "3":
            self.variance()
        elif k == "b":
            self.bilinear()
        elif k == "r":
            self.seed = int(time.time()) % 10_000_000
            self.rows, self.per_batch = [], {}
            if self.plots.ceiling:
                self.plots.ceiling = None
            self.out(f"換了 seed = {self.seed}（按 2 重掃；講稿上的數字是 seed {CFG['sweep']['seed']} 那組）")
        elif k == "h":
            self.menu()
        elif k == "q":
            return False
        return True

    def run_all(self) -> None:
        """課堂順序全部跑一遍（rehearse 用）。"""
        self.truth_table()
        self.sweep()
        self.variance()
        self.bilinear()


def interactive(d: Demo) -> None:
    d.out.banner("W5 Demo 3：InfoNCE 的天花板 — ln N 壓得住任何 critic" + (f"   [{d.tag}]" if d.tag else ""))
    d.menu()
    while True:
        d.plots.pump()
        k = C.getkey()
        if not d.dispatch(k):
            break


def save(d: Demo, folder: Path) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    (folder / ("rehearsal.json" if folder.name == "rehearsal" else "session.json")).write_text(
        json.dumps(d.log, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    (folder / ("rehearsal.txt" if folder.name == "rehearsal" else "session.txt")).write_text(d.out.text(), encoding="utf-8")
    d.plots.save_all(folder)
    C.note(f"存到 {folder}/")


# ── selftest ────────────────────────────────────────────────────
def selftest() -> None:
    fails = 0

    def check(name: str, cond: bool, detail: str = "") -> None:
        nonlocal fails
        fails += 0 if cond else 1
        print(f"  [{'ok' if cond else 'FAIL'}] {name}" + (f"  {detail}" if detail else ""))

    tol = CFG["selftest"]["tolerance"]
    C.banner("selftest：重跑整個掃描，對 w05-data.json → infonce_ceiling")
    if not DATA_JSON.exists():
        sys.exit(f"找不到 {DATA_JSON}。")
    j = json.loads(DATA_JSON.read_text(encoding="utf-8"))["infonce_ceiling"]
    sw = CFG["sweep"]
    check("config 的 seed／d／ρ 與 json 相同", (j["seed"], j["d"], j["rho"]) == (sw["seed"], sw["d"], sw["rho"]),
          f"json ({j['seed']}, {j['d']}, {j['rho']}) vs config ({sw['seed']}, {sw['d']}, {sw['rho']})")
    check("config 的 N 清單與 json 相同", [r["N"] for r in j["sweep"]] == list(sw["Ns"]))
    I_true = true_mi(sw["d"], sw["rho"])
    check("真值閉式解逐位相同", I_true == j["I_true_nats"], f"{I_true!r}")

    t0 = time.time()
    rng = np.random.default_rng(sw["seed"])
    rows = []
    for N in sw["Ns"]:
        W, a = train_critic(rng, N, sw["d"], sw["rho"], sw["steps"], sw["lr"])
        Ls = evaluate(rng, W, a, N, sw["d"], sw["rho"], sw["eval_batches"])
        L_N = float(np.mean(Ls))
        rows.append({"N": N, "L_N": L_N, "estimate": math.log(N) - L_N,
                     "estimate_std_over_batches": float(np.std([math.log(N) - l for l in Ls]))})
    print(f"  （掃描 {time.time() - t0:.0f} 秒）")
    same = 0
    worst = 0.0
    for mine, ref in zip(rows, j["sweep"]):
        for key in ("estimate", "L_N", "estimate_std_over_batches"):
            d_ = abs(mine[key] - ref[key])
            worst = max(worst, d_)
            same += d_ == 0.0
        check(f"N = {mine['N']:>4}：estimate {mine['estimate']:.6f} vs json {ref['estimate']:.6f}",
              abs(mine["estimate"] - ref["estimate"]) <= tol and abs(mine["L_N"] - ref["L_N"]) <= tol
              and abs(mine["estimate_std_over_batches"] - ref["estimate_std_over_batches"]) <= tol)
    print(f"  30 個數字裡逐位相同 {same} 個，最大差 {worst:.1e}（容忍 {tol:g}；差只會來自 BLAS 加總順序，不同機器可能不同）")

    ests = [r["estimate"] for r in rows]
    check("每個 N：估計值 ≤ ln N", all(r["estimate"] <= math.log(r["N"]) + 1e-9 for r in rows))
    check("每個 N：估計值 < 真值（有限步數的 critic 給的是下界）", all(e < I_true for e in ests))
    check("估計值隨 N 不減", all(b >= a_ for a_, b in zip(ests, ests[1:])))
    check("標準差 N = 2 → N = 1024 下降", rows[0]["estimate_std_over_batches"] > rows[-1]["estimate_std_over_batches"],
          f"{rows[0]['estimate_std_over_batches']:.3f} → {rows[-1]['estimate_std_over_batches']:.3f}")
    first_over = next((r["N"] for r in rows if math.log(r["N"]) > I_true), None)
    check("天花板在 N = 16 才過真值（ln 8 < I < ln 16）", first_over == 16, f"first N with ln N > I: {first_over}")

    Nb = CFG["selftest"]["bilinear_N"]
    rng = np.random.default_rng(sw["seed"] + 1)
    Wb, _ = train_critic(rng, Nb, sw["d"], sw["rho"], sw["steps"], sw["lr"], learn_a=False)
    est_b = math.log(Nb) - float(np.mean(evaluate(rng, Wb, 0.0, Nb, sw["d"], sw["rho"], sw["eval_batches"])))
    est_f = next(r["estimate"] for r in rows if r["N"] == Nb)
    check(f"N = {Nb}：純雙線性 critic 的下界比多一項的鬆", est_b < est_f, f"{est_b:.3f} < {est_f:.3f}")

    print()
    if fails:
        sys.exit(f"{fails} 項沒過。")
    print("全部通過。")


# ── main ────────────────────────────────────────────────────────
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["show", "rehearse", "replay", "selftest"])
    ap.add_argument("--no-plot", action="store_true", help="不開 matplotlib 視窗（只在終端機）")
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
        pngs = sorted(rehearsal.glob("*.png"))
        if pngs:
            C.note("圖：" + "、".join(str(p) for p in pngs) + "（open 開）")
        return

    out = Out()
    plot = CFG["display"]["plot"] and not args.no_plot
    if args.cmd == "rehearse":
        d = Demo(out, Plots(plot, headless=True, save_dir=rehearsal), tag="rehearsal")
        d.run_all()
        save(d, rehearsal)
        print("\n下一步：./present.sh 上課；把 rehearsal.json 的 sweep.rows（estimate、std、秒數）、variance、bilinear 與日期填進 p13 講稿。")
        return

    d = Demo(out, Plots(plot, headless=not sys.stdin.isatty(), save_dir=None))
    try:
        interactive(d)
    finally:
        if d.log["sections"]:
            save(d, HERE / "runs" / "live" / C.now().replace(":", ""))


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
# 此檔由 bin/publish_demos.py 從 private repo 的 <course>/demos/ 複製而來，請勿在公開 repo 直接編輯。
"""W6 demo 3：2-D VQ 的 dead code 與三種修法——十六個 code、四團資料、五種訓法。

  show       課堂：單鍵選單（下面的鍵）。離開時把這次算過的東西存進 runs/live/<時間>/（不進版控）
  rehearse   課前：不互動，照課堂順序 1 → 2 → 3 → 4 → 5 → i → t 全部跑一遍，存進 runs/rehearsal/（進版控 = 現場的退路；圖存成 PNG）
  replay     現場 Python 出狀況時：印 runs/rehearsal/rehearsal.txt（純文字），圖用 runs/rehearsal/panels.png
  selftest   對 slides/assets/w06/w06-data.json → dead_codes：同一個 seed 重跑五種設定，用到的 code 數、MSE、codebook、
             每個點的指派逐一比對；再查該有的性質（1 與 2 用到的一樣少、3 與 4 全活、FSQ 全活但 MSE 比 3 高、init 移進資料就不死）

畫面上的鍵（順序照投影片 p16 講稿的【操作】）：
  1  random init + gradient     codebook 以 N(0, I) 初始化在資料之外，只用 codebook loss 的梯度更新——只有被選到的 code 動
  2  random init + EMA          同樣的初始化，改 EMA 更新（c_v、m_v）——一樣只更新被選到的 code：EMA 不是 collapse 的解
  3  k-means init + EMA         codebook 以第一個 batch 的 k-means 中心初始化（SoundStream 的做法），再 EMA
  4  random init + EMA + reset  連續幾步沒被選到的 code 換成當前 batch 的一個點（Jukebox／SoundStream）——十四個紅叉在第 5 步一起被搬進資料裡
  5  FSQ Λ = (4, 4)             沒有 codebook：z 經固定仿射 → tanh → 十六個格點；沒有東西可以死，但格點等距、對四團資料是浪費
  i  init 移進資料裡            同一組設定、codebook 從 [3, 6]² 均勻抽，gradient 與 EMA 各跑一次——預期大多數都活（collapse 不是調出來的）
  t  投影片的表                 五列：用到幾個 code／MSE（p16 面板底下那行字）
  r  換 seed（用時間；再按 1–5 證明不是湊的；講稿的數字是 seed 20261017 那組）
  h  重印選單      q  離開

純 numpy（matplotlib 只畫圖），跑 CPU。數字的定義：
  資料      四團高斯，中心 (3.6,3.6) (5.4,3.6) (3.6,5.4) (5.4,5.4)、std 0.3，落在 [3, 6]² 裡；2000 點
  codebook  V = 16，鍵 1、2、4 都從同一組 E0 ~ N(0, I) 起步（離資料很遠）
  一步      抽 128 個點（不放回）→ 最近鄰指派 → 更新。gradient：e_v += lr · mean(z − e_v)（∂/∂e_v ‖sg[z] − e_v‖² 的方向，只有被選到的 v）；
            EMA：c_v ← μc_v + (1−μ)·n_v，m_v ← μm_v + (1−μ)·Σz，e_v = m_v / c_v（c_v > 0 的 v 才更新——沒被選到的 c_v 只是衰減，e_v 不動）
  用到幾個  訓完之後對全部 2000 點做最近鄰指派，有被指到的 code 數（投影片面板上的 "k / 16 codes used"）
  MSE       同一次指派的平方距離平均（投影片面板上的 MSE）；終端機另印最後 20 步 batch 的平均
  FSQ       每一維 tanh((z − 4.5)·0.9) → 四個等距格點 {−0.75, −0.25, 0.25, 0.75}（偶數 Λ 的半格偏移；格點不碰 ±1 才能 arctanh 映回資料平面算 MSE）
五種設定的迴圈與 make_figs_w06.py 的 dead_code_toy() 逐行相同（同一條 rng、同一個順序 1 → 2 → 3 → 4，所以程式一啟動就把四種算完、
每個鍵只是把算好的軌跡放出來——這樣不管課堂上按鍵的順序如何，數字都與 w06-data.json 對得上；selftest 就是在對這件事）。
"""
from __future__ import annotations

import argparse
import io
import json
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
DATA_JSON = HERE.parent.parent / "slides" / "assets" / "w06" / "w06-data.json"

MODES = [("1", "random_grad", "grad", "random init + gradient", "only chosen codes move"),
         ("2", "random_ema", "ema", "random init + EMA", "EMA is not the cure"),
         ("3", "kmeans_ema", "kmeans", "k-means init + EMA", "start inside the data"),
         ("4", "random_ema_reset", "reset", "random init + EMA + reset", "revive unused codes"),
         ("5", "fsq_4x4", "fsq", "FSQ, Λ = (4, 4)", "no codebook to kill")]
KEY2NAME = {k: n for k, n, *_ in MODES}

C_DATA, C_USED, C_DEAD, C_BOX = "#cbd5e1", "#6d4fc2", "#d1495b", "#2a9d4a"


# ── 資料與五種設定（與 make_figs_w06.py 的 dead_code_toy() 逐行相同；只多了軌跡紀錄）────
def assign(E: np.ndarray, zb: np.ndarray):
    d2 = ((zb[:, None, :] - E[None, :, :]) ** 2).sum(-1)
    return d2.argmin(1), d2.min(1)


def make_data(rng: np.random.Generator, t: dict):
    """照 dead_code_toy() 的順序抽：先 lab、再 Z、再 E0。不能換順序。"""
    centers = np.array(t["centers"], dtype=float)
    lab = rng.integers(0, 4, t["n_points"])
    Z = centers[lab] + t["std"] * rng.standard_normal((t["n_points"], 2))
    E0 = rng.standard_normal((t["V"], 2))  # N(0, I)，離資料 [3, 6]² 很遠
    return Z, E0


def run(mode: str, rng: np.random.Generator, Z: np.ndarray, E0: np.ndarray, t: dict, every: int) -> dict:
    """一種設定跑 steps 步。rng 的用法與 dead_code_toy() 的 run() 完全相同；snapshots 只是讀取，不碰 rng。"""
    V, n_points, steps, batch, lr, mu, reset_after = (t["V"], t["n_points"], t["steps"], t["batch"], t["lr"], t["mu"], t["reset_after"])
    E = E0.copy()
    if mode == "kmeans":
        zb = Z[rng.choice(n_points, batch, replace=False)]
        # 第一個 batch 上的 k-means（10 輪 Lloyd，從 batch 裡抽 V 個點起步）
        E = zb[rng.choice(batch, V, replace=False)].copy()
        for _ in range(10):
            k, _ = assign(E, zb)
            for v in range(V):
                if (k == v).any():
                    E[v] = zb[k == v].mean(0)
    E_init = E.copy()
    c = np.zeros(V)
    m = np.zeros((V, 2))
    unused = np.zeros(V, dtype=int)
    ever = np.zeros(V, dtype=bool)        # 到目前為止被選過嗎（動畫的紫／紅）
    mse_hist, used_hist, ever_hist = [], [], []
    snaps = []                            # (step, E 副本, ever 副本, 這步被選到的)
    resets = []                           # (step, 哪些 code 被重置)
    for s in range(steps):
        zb = Z[rng.choice(n_points, batch, replace=False)]
        k, d2 = assign(E, zb)
        mse_hist.append(d2.mean())
        cnt = np.bincount(k, minlength=V)
        ever |= cnt > 0
        if mode == "grad":
            for v in range(V):
                if cnt[v]:
                    E[v] += lr * (zb[k == v] - E[v]).mean(0)  # ∂/∂e_v ‖sg[z] − e_v‖² 的方向，只有被選到的 v
        else:  # EMA（kmeans／reset 也走 EMA）
            sums = np.zeros((V, 2))
            np.add.at(sums, k, zb)
            c = mu * c + (1 - mu) * cnt
            m = mu * m + (1 - mu) * sums
            upd = c > 1e-3
            E[upd] = m[upd] / c[upd][:, None]
            if mode == "reset":
                unused = np.where(cnt > 0, 0, unused + 1)
                dead = np.where(unused >= reset_after)[0]
                if len(dead):
                    E[dead] = zb[rng.choice(batch, len(dead), replace=False)]
                    c[dead] = 1.0
                    m[dead] = E[dead]
                    unused[dead] = 0
                    resets.append((s + 1, dead.tolist()))
        used_hist.append(int((cnt > 0).sum()))
        ever_hist.append(int(ever.sum()))
        if (s + 1) % every == 0 or s == 0 or s == steps - 1:
            snaps.append((s + 1, E.copy(), ever.copy(), (cnt > 0).copy()))
    k_all, d2_all = assign(E, Z)
    return {"codes_used_full_data": int(len(np.unique(k_all))),
            "codes_used_last_batch": used_hist[-1],
            "mse_full_data": float(d2_all.mean()),
            "mse_last20_mean": float(np.mean(mse_hist[-20:])),
            "mse_first_batch": float(mse_hist[0]),
            "codebook": E.round(4).tolist(),
            "assign_full": k_all.tolist(),
            "_E": E, "_E_init": E_init, "_k_all": k_all, "_snaps": snaps, "_resets": resets,
            "_used_hist": used_hist, "_ever_hist": ever_hist, "_mse_hist": mse_hist}


def run_fsq(Z: np.ndarray, t: dict) -> dict:
    """(5) FSQ：Λ = (4, 4)。與 dead_code_toy() 的 FSQ 段相同。"""
    L = t["fsq_levels"]
    k_aff = t["fsq_k_aff"]
    Zs = (Z - 4.5) * k_aff
    th = np.tanh(Zs)
    q = (np.round(L / 2 * th - 0.5) + 0.5) / (L / 2)  # ∈ {−0.75, −0.25, 0.25, 0.75}
    q = np.clip(q, -0.75, 0.75)
    lv = np.array([-0.75, -0.25, 0.25, 0.75])
    ix = np.abs(q[:, 0][:, None] - lv[None, :]).argmin(1)
    iy = np.abs(q[:, 1][:, None] - lv[None, :]).argmin(1)
    idx = ix * L + iy
    z_hat = np.arctanh(q) / k_aff + 4.5
    grid = [[float(np.arctanh(a) / k_aff + 4.5), float(np.arctanh(b) / k_aff + 4.5)] for a in lv for b in lv]
    return {"codes_used_full_data": int(len(np.unique(idx))), "levels": [L, L], "levels_in_tanh_space": lv.tolist(),
            "affine": "(z - 4.5) * %g" % k_aff,
            "mse_full_data_in_data_plane": float(((z_hat - Z) ** 2).sum(1).mean()),
            "assign_full": idx.tolist(), "grid_points_data_plane": grid,
            "_grid": np.array(grid), "_k_all": idx, "_z_hat": z_hat}


def compute_all(seed: int, every: int) -> dict:
    """程式一啟動就照 dead_code_toy() 的順序把四種設定算完（同一條 rng），加 FSQ。"""
    t = CFG["toy"]
    rng = np.random.default_rng(seed)
    Z, E0 = make_data(rng, t)
    res = {}
    for _, name, mode, *_ in MODES[:4]:
        res[name] = run(mode, rng, Z, E0, t, every)
    res["fsq_4x4"] = run_fsq(Z, t)
    return {"Z": Z, "E0": E0, "results": res}


def run_inside(seed: int, Z: np.ndarray, every: int) -> dict:
    """鍵 i：codebook 從 [3, 6]² 均勻抽，另一條 rng（seed + 1）；gradient 與 EMA 各跑一次。"""
    t = CFG["toy"]
    lo, hi = CFG["inside"]["low"], CFG["inside"]["high"]
    rng = np.random.default_rng(seed + 1)
    E0 = rng.uniform(lo, hi, (t["V"], 2))
    out = {"E0": E0}
    for mode in ("grad", "ema"):
        out[mode] = run(mode, rng, Z, E0, t, every)
    return out


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
    """一張 2×3 的圖：五個面板（與投影片 p16 同一排法）加鍵 i 的面板。每個鍵把自己的面板動畫跑一遍，最後一格留著。"""

    def __init__(self, enabled: bool, headless: bool, save_dir: Path | None):
        self.enabled = enabled
        self.headless = headless
        self.save_dir = save_dir
        self.plt = None
        self.fig = None
        self.axes = {}
        if not enabled:
            return
        import matplotlib
        if headless:
            matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        self.plt = plt
        if not headless:
            plt.ion()

    def ensure(self, Z: np.ndarray) -> None:
        if not self.enabled or self.fig is not None:
            return
        plt = self.plt
        fig, axs = plt.subplots(2, 3, num="W6 Demo 3 — sixteen codes, four clusters, five ways to train", figsize=(11, 7.4))
        keys = ["1", "2", "3", "4", "5", "i"]
        titles = {k: t for k, _, _, t, _ in MODES}
        titles["i"] = "init inside the data (key i)"
        for ax, k in zip(axs.ravel(), keys):
            ax.scatter(Z[::3, 0], Z[::3, 1], s=4, color=C_DATA, zorder=1)
            ax.add_patch(plt.Rectangle((3, 3), 3, 3, fill=False, ls="--", lw=1.0, ec=C_BOX))
            ax.set_xlim(-3, 7); ax.set_ylim(-3, 7); ax.set_aspect("equal")
            ax.set_xticks([-2, 0, 2, 4, 6]); ax.set_yticks([-2, 0, 2, 4, 6])
            ax.tick_params(labelsize=8)
            ax.set_title(f"{k}  {titles[k]}", fontsize=10, loc="left")
            ax.text(0.98, 0.03, "", transform=ax.transAxes, ha="right", va="bottom", fontsize=10, fontweight="bold")
            self.axes[k] = ax
        fig.suptitle("grey: data · purple: a code chosen so far · red ×: never chosen · dashed: the data's support", fontsize=9, color="0.4")
        fig.tight_layout()
        self.fig = fig
        self._flush()

    def animate(self, key: str, res: dict, V: int, label_final: str) -> None:
        """照 snapshots 逐格畫：被選過的 code 紫點、還沒被選過的紅叉。"""
        if not self.enabled:
            return
        ax = self.axes[key]
        art = []
        for s, E, ever, _ in res["_snaps"]:
            for a in art:
                a.remove()
            art = []
            if ever.any():
                art.append(ax.scatter(E[ever, 0], E[ever, 1], s=36, color=C_USED, zorder=3))
            if (~ever).any():
                art.append(ax.scatter(E[~ever, 0], E[~ever, 1], s=48, marker="x", color=C_DEAD, lw=1.6, zorder=3))
            ax.texts[-1].set_text(f"step {s}: {int(ever.sum())} / {V}")
            if not self.headless:
                self.plt.pause(CFG["display"]["frame_pause"])
        ax.texts[-1].set_text(label_final)
        self._flush()

    def draw_fsq(self, key: str, res: dict, V: int, label_final: str) -> None:
        if not self.enabled:
            return
        ax = self.axes[key]
        used = set(res["assign_full"])
        G = res["_grid"]
        on = np.array([i in used for i in range(len(G))])
        ax.scatter(G[on, 0], G[on, 1], s=36, color=C_USED, zorder=3)
        if (~on).any():
            ax.scatter(G[~on, 0], G[~on, 1], s=48, marker="x", color=C_DEAD, lw=1.6, zorder=3)
        ax.texts[-1].set_text(label_final)
        self._flush()

    def _flush(self, name: str = "panels.png") -> None:
        if not self.enabled:
            return
        if self.headless:
            if self.save_dir:
                self.save_dir.mkdir(parents=True, exist_ok=True)
                self.fig.savefig(self.save_dir / name, dpi=130)
        else:
            self.plt.pause(0.05)

    def save_all(self, folder: Path) -> None:
        if not self.enabled or self.headless or self.fig is None:
            return
        folder.mkdir(parents=True, exist_ok=True)
        self.fig.savefig(folder / "panels.png", dpi=130)

    def pump(self) -> None:
        if self.enabled and not self.headless:
            self.plt.pause(0.01)


# ── 各段落 ──────────────────────────────────────────────────────
class Demo:
    def __init__(self, out: Out, plots: Plots, seed: int | None = None, tag: str = ""):
        self.out = out
        self.plots = plots
        self.tag = tag
        self.t = CFG["toy"]
        self.V = self.t["V"]
        self.seed = self.t["seed"] if seed is None else seed
        self.every = CFG["display"]["frame_every"]
        self.state = compute_all(self.seed, self.every)
        self.inside = None
        self.done: dict[str, dict] = {}
        self.log: dict = {"date": C.now(), "tag": tag, "versions": C.versions(), "config": CFG, "seed": self.seed,
                          "sections": {}}

    def _header(self, key: str) -> None:
        o = self.out
        _, name, mode, title, sub = next(m for m in MODES if m[0] == key)
        o.banner(f"{key}  {title} — {sub}" + (f"   [seed {self.seed}]" if self.seed != self.t['seed'] else ""))
        t = self.t
        if mode == "grad":
            o(f"codebook E0 ~ N(0, I)，{t['V']} 個；每步抽 {t['batch']} 點、最近鄰指派、e_v += {t['lr']}·mean(z − e_v)——只有被選到的 v 有這一項（codebook loss 的梯度）")
        elif mode == "ema":
            o(f"同一組 E0；改成 EMA：c_v ← μc_v + (1−μ)n_v、m_v ← μm_v + (1−μ)Σz、e_v = m_v/c_v，μ = {t['mu']}——c_v 仍然只在被選到時增加")
        elif mode == "kmeans":
            o(f"E0 換成第一個 batch（{t['batch']} 點）上 10 輪 Lloyd 的 {t['V']} 個中心（SoundStream 的做法），之後同 2 的 EMA")
        elif mode == "reset":
            o(f"同 2 的 E0 與 EMA；連續 {t['reset_after']} 步沒被選到的 code 換成當前 batch 裡隨機一個點（Jukebox／SoundStream 的 dead-code reset）")
        else:
            o(f"沒有 codebook：每一維 tanh((z − 4.5)·{t['fsq_k_aff']}) → 四個等距格點 {{±0.25, ±0.75}}（偶數 Λ 的半格偏移）；隱含 codebook {t['fsq_levels']}×{t['fsq_levels']} = 16 個格點")

    def _trace(self, res: dict) -> list[dict]:
        o = self.out
        o()
        o(f"{'step':>5} {'這步用到':>8} {'到此為止用過':>12} {'batch MSE':>10}")
        rows = []
        for s in CFG["display"]["trace_steps"]:
            if s > self.t["steps"]:
                continue
            u, e, m = res["_used_hist"][s - 1], res["_ever_hist"][s - 1], res["_mse_hist"][s - 1]
            o(f"{s:>5} {u:>8} {e:>12} {m:>10.{DIG}f}")
            rows.append({"step": s, "used_this_batch": u, "ever_used": e, "mse_batch": float(m)})
        return rows

    def _final(self, key: str, res: dict, mse_key: str) -> str:
        o = self.out
        n = res["codes_used_full_data"]
        mse = res[mse_key]
        col = C.GREEN if n >= 12 else C.RED
        o()
        o(f"訓完對全部 {self.t['n_points']} 點指派：{col}{C.BOLD}{n} / {self.V} codes used{C.RESET}，MSE {C.BOLD}{fmt(mse)}{C.RESET}"
          + (f"（最後 20 步 batch 平均 {fmt(res['mse_last20_mean'])}）" if "mse_last20_mean" in res else ""))
        self.compare_json(key, res, mse_key)
        return f"{n} / {self.V} · MSE {mse:.2f}"

    def compare_json(self, key: str, res: dict, mse_key: str) -> None:
        """seed 與 json 相同時，順手報與 w06-data.json 的差。"""
        if not DATA_JSON.exists() or self.seed != self.t["seed"]:
            return
        ref = json.loads(DATA_JSON.read_text(encoding="utf-8"))["dead_codes"]["results"][KEY2NAME[key]]
        d_mse = abs(ref[mse_key] - res[mse_key])
        same_used = ref["codes_used_full_data"] == res["codes_used_full_data"]
        same_assign = ref["assign_full"] == res["assign_full"]
        self.out(f"{C.DIM}對 w06-data.json：codes used {'相同' if same_used else '不同！'}、MSE 差 {d_mse:.1e}、"
                 f"{len(res['assign_full'])} 個點的指派{'全部相同' if same_assign else '有不同！'}。{C.RESET}")
        self.log["sections"].setdefault("vs_w06_data_json", {})[KEY2NAME[key]] = {
            "codes_used_equal": same_used, "mse_abs_diff": d_mse, "assign_equal": same_assign}

    # 1–4 ---------------------------------------------------------------
    def panel(self, key: str) -> None:
        name = KEY2NAME[key]
        res = self.state["results"][name]
        self._header(key)
        self.plots.ensure(self.state["Z"])
        if key == "4" and res["_resets"]:
            self.out(f"重置發生在第 {'、'.join(str(s) for s, _ in res['_resets'][:12])}{'…' if len(res['_resets']) > 12 else ''} 步，共 {sum(len(d) for _, d in res['_resets'])} 次")
        self.plots.animate(key, res, self.V, "")
        trace = self._trace(res)
        label = self._final(key, res, "mse_full_data")
        if self.plots.enabled:
            self.plots.axes[key].texts[-1].set_text(label)
            self.plots._flush()
        if key == "1":
            self.out(f"{C.DIM}（紅叉留在原點附近：它們離資料太遠，永遠不是最近鄰，永遠收不到梯度——兩個成因同時看得見。）{C.RESET}")
        elif key == "2":
            self.out(f"{C.DIM}（還是一樣少。EMA 解的是梯度更新的不穩，不是 collapse——p16 的 ask 答案。）{C.RESET}")
        elif key == "3":
            self.out(f"{C.DIM}（初始化移進資料裡就不死：這裡的 collapse 不是調出來的，是「init 在資料支撐之外」的直接後果。）{C.RESET}")
        elif key == "4":
            self.out(f"{C.DIM}（紅叉從第一步起就全部沒被選到，所以在第 {self.t['reset_after']} 步**同時**到達門檻、一起被搬進資料裡，下一步就被選到、變紫——不是一個一個。之後沒有再死過。）{C.RESET}")
        self.done[key] = {"codes_used_full_data": res["codes_used_full_data"], "codes_used_last_batch": res["codes_used_last_batch"],
                          "mse_full_data": res["mse_full_data"], "mse_last20_mean": res["mse_last20_mean"],
                          "trace": trace, "codebook_final": res["codebook"],
                          "resets": [{"step": s, "codes": d} for s, d in res["_resets"]]}
        self.log["sections"][name] = self.done[key]

    # 5 -------------------------------------------------------------------
    def fsq(self) -> None:
        res = self.state["results"]["fsq_4x4"]
        self._header("5")
        self.plots.ensure(self.state["Z"])
        o = self.out
        cnt = np.bincount(res["_k_all"], minlength=16)
        o()
        o("格點（資料平面）的使用數，列 = x 格、欄 = y 格：")
        for ix in range(4):
            o("   " + "  ".join(f"{cnt[ix * 4 + iy]:>5}" for iy in range(4)))
        label = self._final("5", res, "mse_full_data_in_data_plane")
        self.plots.draw_fsq("5", res, 16, label)
        km = self.state["results"]["kmeans_ema"]["mse_full_data"]
        o(f"{C.DIM}（十六個格點都有人用——沒有 codebook，沒有東西可以死；但 MSE {fmt(res['mse_full_data_in_data_plane'])} 對 k-means 的 {fmt(km)}：格點等距鋪滿，對四團資料是浪費。這是 FSQ 的代價，p18 展開。）{C.RESET}")
        self.done["5"] = {"codes_used_full_data": res["codes_used_full_data"], "mse_full_data_in_data_plane": res["mse_full_data_in_data_plane"],
                          "grid_counts": cnt.tolist()}
        self.log["sections"]["fsq_4x4"] = self.done["5"]

    # i -------------------------------------------------------------------
    def inside_init(self) -> None:
        o = self.out
        lo, hi = CFG["inside"]["low"], CFG["inside"]["high"]
        o.banner(f"i  同一組設定、codebook 從 [{lo:g}, {hi:g}]² 均勻抽（init 移進資料裡）；另一條 rng（seed + 1）")
        self.plots.ensure(self.state["Z"])
        if self.inside is None:
            self.inside = run_inside(self.seed, self.state["Z"], self.every)
        res = {}
        for mode, title in (("grad", "gradient"), ("ema", "EMA")):
            r = self.inside[mode]
            n = r["codes_used_full_data"]
            col = C.GREEN if n >= CFG["selftest"]["inside_min_used"] else C.RED
            o(f"  {title:>8}：{col}{C.BOLD}{n} / {self.V} codes used{C.RESET}，MSE {fmt(r['mse_full_data'])}（第一步 batch MSE {fmt(r['mse_first_batch'])}）")
            res[mode] = {"codes_used_full_data": n, "mse_full_data": r["mse_full_data"], "codes_used_last_batch": r["codes_used_last_batch"]}
        self.plots.animate("i", self.inside["ema"], self.V, f"{res['ema']['codes_used_full_data']} / {self.V} · MSE {res['ema']['mse_full_data']:.2f} (EMA)")
        o(f"{C.DIM}（同樣的更新規則、同樣的資料，只是 E0 落在資料裡：大多數都活。死的是「初始化在資料之外」，不是更新規則——這就是「collapse 不是調出來的」。）{C.RESET}")
        self.done["i"] = res
        self.log["sections"]["inside_init"] = {"low": lo, "high": hi, **res}

    # t -------------------------------------------------------------------
    def table(self) -> None:
        o = self.out
        o.banner("t  投影片 p16 五個面板底下那行字")
        o(f"{'鍵':>3}  {'設定':<28} {'codes used':>11} {'MSE':>7}   註")
        R = self.state["results"]
        rows = []
        for key, name, mode, title, sub in MODES:
            r = R[name]
            mse = r["mse_full_data_in_data_plane"] if mode == "fsq" else r["mse_full_data"]
            n = r["codes_used_full_data"]
            o(f"{key:>3}  {title:<28} {n:>6} / {self.V:<3} {mse:>7.{DIG}f}   {sub}")
            rows.append({"key": key, "name": name, "codes_used": n, "mse": mse})
        if self.inside is not None:
            for mode in ("grad", "ema"):
                r = self.inside[mode]
                o(f"{'i':>3}  {'inside init + ' + mode:<28} {r['codes_used_full_data']:>6} / {self.V:<3} {r['mse_full_data']:>7.{DIG}f}   init in [3, 6]^2 (not on the slide)")
        o()
        o(f"{C.DIM}投影片用的是 mse_full_data（訓完對全部 2000 點）；FSQ 的是映回資料平面算的。seed {self.seed}。{C.RESET}")
        self.log["sections"]["table"] = rows

    # 選單 ---------------------------------------------------------------
    def menu(self) -> None:
        o = self.out
        o(f"\n{C.BOLD}鍵{C.RESET}  1 random+gradient   2 random+EMA   3 k-means init   4 EMA+reset   5 FSQ")
        o(f"    i init 移進資料裡   t 投影片的表   r 換 seed   h 選單   q 離開")

    def dispatch(self, k: str) -> bool:
        if k in ("1", "2", "3", "4"):
            self.panel(k)
        elif k == "5":
            self.fsq()
        elif k == "i":
            self.inside_init()
        elif k == "t":
            self.table()
        elif k == "r":
            self.seed = int(time.time()) % 10_000_000
            self.state = compute_all(self.seed, self.every)
            self.inside = None
            self.done = {}
            if self.plots.fig is not None:
                self.plots.plt.close(self.plots.fig)
                self.plots.fig = None
                self.plots.axes = {}
            self.out(f"換了 seed = {self.seed}，五種設定已重算（按 1–5 看；講稿上的數字是 seed {self.t['seed']} 那組）")
        elif k == "h":
            self.menu()
        elif k == "q":
            return False
        return True

    def run_all(self) -> None:
        """課堂順序全部跑一遍（rehearse 用）。"""
        for k in ("1", "2", "3", "4"):
            self.panel(k)
        self.fsq()
        self.inside_init()
        self.table()


def interactive(d: Demo) -> None:
    d.out.banner("W6 Demo 3：sixteen codes, four clusters, five ways to train" + (f"   [{d.tag}]" if d.tag else ""))
    d.out(f"{C.DIM}四種設定已在啟動時照 make_figs_w06.py 的順序算完（seed {d.seed}）；每個鍵只是把軌跡放出來。{C.RESET}")
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
    t = CFG["toy"]
    C.banner("selftest：同一個 seed 重跑五種設定，對 w06-data.json → dead_codes")
    if not DATA_JSON.exists():
        sys.exit(f"找不到 {DATA_JSON}。")
    j = json.loads(DATA_JSON.read_text(encoding="utf-8"))["dead_codes"]
    for key in ("seed", "V", "n_points", "steps", "batch", "lr", "mu", "reset_after", "std"):
        check(f"config 的 {key} 與 json 相同", j[key] == t[key], f"json {j[key]} vs config {t[key]}")
    check("config 的 centers 與 json 相同", j["centers"] == [list(map(float, c)) for c in t["centers"]])

    t0 = time.time()
    st = compute_all(t["seed"], CFG["display"]["frame_every"])
    print(f"  （五種設定 {time.time() - t0:.1f} 秒）")
    Z, E0 = st["Z"], st["E0"]
    check("2000 個資料點與 json（四位小數）相同", np.allclose(Z.round(4), np.array(j["points"]), atol=0, rtol=0))
    check("初始 codebook E0 與 json（四位小數）相同", np.allclose(E0.round(4), np.array(j["init_codebook"]), atol=0, rtol=0))
    R = st["results"]
    for key, name, mode, title, _ in MODES:
        mine, ref = R[name], j["results"][name]
        mse_key = "mse_full_data_in_data_plane" if mode == "fsq" else "mse_full_data"
        d_mse = abs(mine[mse_key] - ref[mse_key])
        ok = (mine["codes_used_full_data"] == ref["codes_used_full_data"] and d_mse <= tol and mine["assign_full"] == ref["assign_full"])
        if mode != "fsq":
            ok = ok and np.allclose(np.array(mine["codebook"]), np.array(ref["codebook"]), atol=5e-5, rtol=0) \
                and abs(mine["mse_last20_mean"] - ref["mse_last20_mean"]) <= tol
        else:
            ok = ok and np.allclose(np.array(mine["grid_points_data_plane"]), np.array(ref["grid_points_data_plane"]), atol=tol, rtol=0)
        check(f"{key} {title}：{mine['codes_used_full_data']} / {t['V']} used、MSE {mine[mse_key]:.6f} vs json {ref['codes_used_full_data']}、{ref[mse_key]:.6f}",
              ok, f"MSE 差 {d_mse:.1e}，指派{'相同' if mine['assign_full'] == ref['assign_full'] else '不同'}")

    # 該有的性質
    g, e, k, r, f = (R[n] for _, n, *_ in MODES)
    check("1 與 2 用到的 code 一樣少（EMA 不是 collapse 的解）", g["codes_used_full_data"] == e["codes_used_full_data"] and g["codes_used_full_data"] <= 4,
          f"{g['codes_used_full_data']} vs {e['codes_used_full_data']}")
    check("1、2 沒被選到的 code 一步都沒動（紅叉停在 E0）", all(
        np.allclose(np.array(x["codebook"])[~np.isin(np.arange(t["V"]), np.unique(x["assign_full"]))],
                    E0.round(4)[~np.isin(np.arange(t["V"]), np.unique(x["assign_full"]))], atol=5e-5) for x in (g, e)))
    check("3 k-means init 全活", k["codes_used_full_data"] == t["V"])
    check("4 reset 全活，且真的發生過重置", r["codes_used_full_data"] == t["V"] and len(r["_resets"]) > 0, f"{len(r['_resets'])} 次重置事件")
    check("5 FSQ 十六個格點全用，但 MSE 比 k-means 高", f["codes_used_full_data"] == 16 and f["mse_full_data_in_data_plane"] > k["mse_full_data"],
          f"{f['mse_full_data_in_data_plane']:.3f} > {k['mse_full_data']:.3f}")
    check("1、2 的 MSE 比 3、4 高一個數量級", min(g["mse_full_data"], e["mse_full_data"]) > 5 * max(k["mse_full_data"], r["mse_full_data"]))
    ins = run_inside(t["seed"], Z, CFG["display"]["frame_every"])
    mn = CFG["selftest"]["inside_min_used"]
    check(f"i init 移進資料裡：gradient 與 EMA 都至少 {mn} 個活（collapse 不是調出來的）",
          ins["grad"]["codes_used_full_data"] >= mn and ins["ema"]["codes_used_full_data"] >= mn,
          f"gradient {ins['grad']['codes_used_full_data']}、EMA {ins['ema']['codes_used_full_data']}")

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
        png = rehearsal / "panels.png"
        if png.exists():
            C.note(f"圖：{png}（open 開）")
        return

    out = Out()
    plot = CFG["display"]["plot"] and not args.no_plot
    if args.cmd == "rehearse":
        d = Demo(out, Plots(plot, headless=True, save_dir=rehearsal), tag="rehearsal")
        d.run_all()
        save(d, rehearsal)
        print("\n下一步：./present.sh 上課；把 rehearsal.json 的五組 codes used／MSE 與日期對一次 p16 講稿（數字應與投影片相同）。")
        return

    d = Demo(out, Plots(plot, headless=not sys.stdin.isatty(), save_dir=None))
    try:
        interactive(d)
    finally:
        if d.log["sections"]:
            save(d, HERE / "runs" / "live" / C.now().replace(":", ""))


if __name__ == "__main__":
    main()

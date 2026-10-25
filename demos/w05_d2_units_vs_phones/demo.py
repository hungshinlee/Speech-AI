# -*- coding: utf-8 -*-
# 此檔由 bin/publish_demos.py 從 private repo 的 <course>/demos/ 複製而來，請勿在公開 repo 直接編輯。
"""W5 demo 2：unit ≠ 音素——三個老師的純度，與一句話的對齊圖。

  show       課堂：從快取讀表徵與 k-means，開一個視窗；鍵在視窗裡按（下面的鍵）。離開時存 runs/live/<時間>/
  rehearse   課前：不互動，三組 k-means、三個純度、句子的對齊圖、同一個字兩位說話人全部算一遍，存 runs/rehearsal/（進版控 = 現場的退路），
             並寫 slides/assets/w05/w05-metrics.json → demo2
  replay     現場出狀況時：從 runs/rehearsal/ 讀回所有數字與 unit 序列，同一個視窗、同樣的鍵，不碰 torch、不碰快取
  cluster    課前：三組 k-means（MFCC 當場從 wav 算、取 Demo 1 同一組 frame）→ ../_media/w05_d2/（show／rehearse 缺快取會自動跑）
  extract    課前：那句話、念同一個字的其他說話人、自錄的 wav 的整句表徵（第 6、9 層與 MFCC）→ ../_media/w05_d2/seq_<utt>.npz（要模型；用 Demo 1 的 Encoder）
  fetch      課前一週：從對齊資料集的 parquet 只讀 words 欄 → data/words.json（字的區間；要網路）
  fake       沒有模型也能演練流程：合成的特徵與 unit（畫面標明 FAKE，不是給學生看的）
  selftest   不開視窗：純度定義、k-means、DCT／Δ、RLE／LCS 的單元檢查 ＋ fake 流程 ＋ 彩排紀錄對 w05-metrics.json
  export     只把 runs/rehearsal/rehearsal.json 重寫進 w05-metrics.json

畫面上的鍵（順序照投影片 p45 講稿的【操作】）：
  1  三個老師的純度表（MFCC／第 6 層／第 9 層 × phone purity／cluster purity／PNMI；填 p45 的表格）
  2  那句話：字（MFA）、音素段（MFA）、第 6 層的 unit——先放大前兩個字；印「一個音素幾個 unit」「邊界對不對齊」「哪個 unit 是靜音」
  [  ]  放大窗往前／往後一個字      f  整句        u  多畫第 9 層與 MFCC 的 unit 列（再按收回）
  3  同一個字、不同說話人：unit 字串並排（collapsed runs），標出對不上的格；印重疊率
  4  選配：自錄 wav 的 unit 序列並排（demo_config.toml [extra]）
  h  選單    q  離開

數字的定義（HuBERT 原文 §IV-E 的三個度量；聯合分布 p(y, k) 在 frame 上數）：
  phone purity   = Σ_k max_y p(y, k)   每個 cluster 裡最多數的音素佔多少（把 cluster 當音素預測的 frame accuracy）
  cluster purity = Σ_y max_k p(y, k)   每個音素最常落的那個 cluster 佔多少
  PNMI           = I(y; k) / H(y)      音素的不確定性被 cluster 解掉幾成
k-means 在訓練 frame 上擬合，純度報測試 frame（每位說話人後 3 句）；全部 frame 一起算（論文的做法）也印。
三組特徵用的是 Demo 1 抽的同一組 frame（每句 60 個標了音素的 frame，seed 20261011），音素標籤同一份 MFA 對齊。
"""
from __future__ import annotations

import argparse
import importlib.util
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
FEATS = CFG["features"]["order"]
K = CFG["kmeans"]["K"]
DATA = HERE / "data"
MEDIA = HERE.parent / "_media"
MEDIA_D2 = MEDIA / "w05_d2"
METRICS_JSON = HERE.parents[1] / "slides" / "assets" / "w05" / "w05-metrics.json"
DATA_JSON = HERE.parents[1] / "slides" / "assets" / "w05" / "w05-data.json"
SENT_UTT = CFG["sentence"]["utt"]


# ── Demo 1 的程式當模組用（子集、對齊、frame 標籤、log-mel、表徵快取、Encoder）────
def _load_d1():
    p = (HERE / CFG["d1"]["dir"]).resolve() / "demo.py"
    if not p.exists():
        sys.exit(f"找不到 Demo 1 的程式 {p}（demo_config.toml [d1].dir）")
    spec = importlib.util.spec_from_file_location("w05_d1_demo", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


D1 = _load_d1()
FS = D1.FS
FRAME_MS = float(D1.CFG["infer"]["frame_ms"])
MODEL = CFG["d1"]["model"]


def frame_centers_s(T: int) -> np.ndarray:
    return D1.frame_centers_s(T)


# ── MFCC：log-mel → 正交 DCT-II 截斷 → Δ、ΔΔ ─────────────────────
def dct_matrix(n_in: int, n_out: int) -> np.ndarray:
    """正交 DCT-II 的前 n_out 列：D[k, n] = s_k √(2/N) cos(π (n + ½) k / N)，s_0 = 1/√2。D Dᵀ = I。"""
    n = np.arange(n_in)
    k = np.arange(n_out)[:, None]
    D = np.sqrt(2.0 / n_in) * np.cos(np.pi * (n + 0.5) * k / n_in)
    D[0] /= np.sqrt(2.0)
    return D


def deltas(c: np.ndarray, w: int) -> np.ndarray:
    """回歸式差分（HTK／Kaldi 的定義）：d_t = Σ_{n=1..w} n (c_{t+n} − c_{t−n}) / (2 Σ n²)，兩端複製邊界 frame。"""
    cp = np.pad(c, ((w, w), (0, 0)), mode="edge")
    num = sum(n * (cp[w + n: w + n + len(c)] - cp[w - n: w - n + len(c)]) for n in range(1, w + 1))
    return num / (2.0 * sum(n * n for n in range(1, w + 1)))


def mfcc39(x: np.ndarray) -> np.ndarray:
    """(frames, 39)：13 個倒頻譜（含 c0）+ Δ + ΔΔ。log-mel 與 Demo 1 同一支（80 個 Slaney mel、25 ms／20 ms／512 點）。"""
    mel = D1.logmel(x)
    m = CFG["features"]["mfcc"]
    c = mel @ dct_matrix(mel.shape[1], m["n_ceps"]).T
    d = deltas(c, m["delta_window"])
    return np.concatenate([c, d, deltas(d, m["delta_window"])], axis=1)


# ── k-means（純 numpy）─────────────────────────────────────────
def _sqdist_argmin(X: np.ndarray, Cc: np.ndarray, chunk: int) -> tuple[np.ndarray, np.ndarray]:
    """每個 x 最近的 centroid 與平方距離（分塊）。"""
    cn = (Cc * Cc).sum(1)
    lab = np.empty(len(X), dtype=np.int64)
    d2 = np.empty(len(X), dtype=np.float64)
    for a in range(0, len(X), chunk):
        xb = X[a: a + chunk]
        d = (xb * xb).sum(1)[:, None] - 2.0 * (xb @ Cc.T) + cn[None, :]
        j = d.argmin(1)
        lab[a: a + chunk] = j
        d2[a: a + chunk] = np.maximum(d[np.arange(len(xb)), j], 0.0)
    return lab, d2


def kmeans_pp_init(X: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    n = len(X)
    idx = [int(rng.integers(n))]
    dmin = ((X - X[idx[0]]) ** 2).sum(1)
    for _ in range(1, k):
        p = dmin / dmin.sum() if dmin.sum() > 0 else np.full(n, 1.0 / n)
        j = int(rng.choice(n, p=p))
        idx.append(j)
        dmin = np.minimum(dmin, ((X - X[j]) ** 2).sum(1))
    return X[idx].copy()


def kmeans(X: np.ndarray, k: int, seed: int, n_init: int, max_iter: int, chunk: int, log=None) -> tuple[np.ndarray, dict]:
    """Lloyd + k-means++，n_init 次取 inertia 最小的。回 (centroids (k, d) float32, info)。"""
    X = np.ascontiguousarray(X, dtype=np.float32)
    rng = np.random.default_rng(seed)
    best = None
    for r in range(n_init):
        Cc = kmeans_pp_init(X, k, rng)
        lab = None
        for it in range(1, max_iter + 1):
            new_lab, d2 = _sqdist_argmin(X, Cc, chunk)
            inertia = float(d2.sum())
            if lab is not None and np.array_equal(new_lab, lab):
                break
            lab = new_lab
            counts = np.bincount(lab, minlength=k)
            Cn = np.zeros_like(Cc)
            np.add.at(Cn, lab, X)
            nz = counts > 0
            Cn[nz] /= counts[nz][:, None]
            for e in np.flatnonzero(~nz):                      # 空的 cluster：丟到離目前 centroid 最遠的點
                j = int(d2.argmax()); Cn[e] = X[j]; d2[j] = 0.0
            Cc = Cn
        if best is None or inertia < best[1]["inertia"]:
            best = (Cc, {"inertia": inertia, "iters": it, "init": r, "n_empty_final": int((np.bincount(lab, minlength=k) == 0).sum())})
        if log:
            log(f"    init {r}: {it:3d} 次迭代，inertia {inertia:.4g}")
    return best[0], best[1]


def assign(X: np.ndarray, Cc: np.ndarray) -> np.ndarray:
    return _sqdist_argmin(np.ascontiguousarray(X, dtype=np.float32), Cc, CFG["kmeans"]["chunk"])[0]


# ── 三個純度 ───────────────────────────────────────────────────
def purity(y: np.ndarray, k: np.ndarray, n_y: int, n_k: int) -> dict:
    """HuBERT §IV-E：phone purity = Σ_k max_y p(y,k)、cluster purity = Σ_y max_k p(y,k)、PNMI = I(y;k)/H(y)。"""
    J = np.zeros((n_y, n_k))
    np.add.at(J, (y, k), 1.0)
    P = J / J.sum()
    py, pk = P.sum(1), P.sum(0)
    nz = P > 0
    I = float((P[nz] * np.log(P[nz] / (py[:, None] * pk[None, :])[nz])).sum())
    Hy = float(-(py[py > 0] * np.log(py[py > 0])).sum())
    Hk = float(-(pk[pk > 0] * np.log(pk[pk > 0])).sum())
    return {"phone_purity": float(P.max(0).sum()), "cluster_purity": float(P.max(1).sum()), "pnmi": I / Hy if Hy > 0 else 0.0,
            "mi_nat": I, "h_phone_nat": Hy, "h_cluster_nat": Hk, "n_frames": int(J.sum()), "n_clusters_used": int((pk > 0).sum())}


# ── 資料：子集、對齊、字、Demo 1 的 frame 快取 ────────────────────
def load_words() -> dict[str, list] | None:
    p = DATA / "words.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def phone_segments(utt: str) -> list[list]:
    """[[音素（去重音、大寫）, s, e, is_speech], …]；靜音沒有區間，spn 之類 is_speech = False。"""
    out = []
    for ph, s, e in D1.load_alignments()[utt]:
        q = D1.norm_phone(ph)
        out.append([q, float(s), float(e), q not in D1.EXCLUDE])
    return out


def mfcc_frames_path() -> Path:
    return MEDIA_D2 / "mfcc_frames.npz"


def build_mfcc_frames(out=None) -> dict:
    """對子集每句算 MFCC，取 Demo 1 抽表徵時同一組 frame（同 seed、同抽法），並對 Demo 1 的快取逐格核對標籤。"""
    o = out or print
    z = np.load(D1.feats_path(MODEL), allow_pickle=False)
    sub, ali = D1.load_subset(), D1.load_alignments()
    phone_index = {p: i for i, p in enumerate(sub["phones"])}
    rng = np.random.default_rng(D1.CFG["frames"]["seed"])
    per = D1.CFG["frames"]["per_utt"]
    utt_T = z["utt_T"]
    X, ph, uu = [], [], []
    t0 = time.perf_counter()
    for i, u in enumerate(sub["utts"]):
        x = C.load_wav(D1.wav_path(u["id"]), FS)
        T = int(utt_T[i])
        lab = D1.frame_labels(T, ali[u["id"]], phone_index)
        m = mfcc39(x)
        Tm = min(T, m.shape[0])
        lab[Tm:] = -1
        cand = np.flatnonzero(lab >= 0)
        pick = np.sort(rng.choice(cand, size=min(per, len(cand)), replace=False)) if len(cand) else cand
        X.append(m[pick]); ph.append(lab[pick]); uu.append(np.full(len(pick), i))
        if (i + 1) % 100 == 0 or i == len(sub["utts"]) - 1:
            o(f"  MFCC {i + 1:4d}/{len(sub['utts'])}  {u['id']:<18s} T = {T:4d}")
    X, ph, uu = np.concatenate(X).astype(np.float32), np.concatenate(ph), np.concatenate(uu)
    same = bool(np.array_equal(ph, z["phone"]) and np.array_equal(uu, z["utt"]))
    if not same:
        sys.exit("MFCC 取的 frame 與 Demo 1 快取的不同（seed 或子集變了？）。重跑 Demo 1 的 extract 之後再試。")
    meta = {"n_frames": int(len(X)), "dim": int(X.shape[1]), "same_frames_as_d1": same, "seconds": time.perf_counter() - t0,
            "logmel": D1.CFG["logmel"], "mfcc": CFG["features"]["mfcc"], "time": C.now()}
    MEDIA_D2.mkdir(parents=True, exist_ok=True)
    np.savez(mfcc_frames_path(), X=X, phone=ph, utt=uu, meta=json.dumps(meta, ensure_ascii=False))
    o(f"  → {mfcc_frames_path()}（{len(X)} 個 frame × 39，與 Demo 1 的 frame 逐格相同；{meta['seconds']:.0f} s）")
    return meta


class Frames:
    """三組特徵的同一組 frame：X[feat] (Nf, d)、phone (Nf,)、訓／測遮罩。"""

    def __init__(self, out=None):
        z = np.load(D1.feats_path(MODEL), allow_pickle=False)
        self.phone, self.utt = z["phone"], z["utt"]
        split = z["utt_split"][self.utt]
        self.tr, self.te = split == "train", split != "train"
        self.d1_meta = json.loads(str(z["meta"]))
        self.X = {}
        for f in FEATS:
            if f == "mfcc":
                if not mfcc_frames_path().exists():
                    C.note("  沒有 MFCC 的 frame 快取，現在從 wav 算（396 句，約一分鐘）……")
                    build_mfcc_frames(out)
                zm = np.load(mfcc_frames_path(), allow_pickle=False)
                Xm = zm["X"]
                if CFG["features"]["mfcc"]["standardize"]:
                    mu, sd = Xm[self.tr].mean(0), Xm[self.tr].std(0) + 1e-8
                    Xm = (Xm - mu) / sd
                self.X[f] = Xm.astype(np.float32)
            else:
                self.X[f] = z[f"frames_{CFG['features'][f]['layer']}"].astype(np.float32)
        self.n_phones = int(self.phone.max()) + 1


def kmeans_path() -> Path:
    return MEDIA_D2 / "kmeans.npz"


def cluster_all(out=None, frames: Frames | None = None) -> dict:
    """三組 k-means → kmeans.npz（centroids）；回每組的純度與資訊。"""
    o = out or print
    F = frames or Frames(out)
    km = CFG["kmeans"]
    res, cents = {}, {}
    for f in FEATS:
        X = F.X[f]
        o(f"  k-means  {CFG['features'][f]['label']:<22s} K = {K}，{int(F.tr.sum())} 個訓練 frame × {X.shape[1]} 維 ……")
        t0 = time.perf_counter()
        Cc, info = kmeans(X[F.tr], K, km["seed"], km["n_init"], km["max_iter"], km["chunk"], log=o)
        info["seconds"] = time.perf_counter() - t0
        lab_all = assign(X, Cc)
        res[f] = {"label": CFG["features"][f]["label"], "role": CFG["features"][f]["role"], "dim": int(X.shape[1]), "kmeans": info,
                  "test": purity(F.phone[F.te], lab_all[F.te], F.n_phones, K),
                  "train": purity(F.phone[F.tr], lab_all[F.tr], F.n_phones, K),
                  "all": purity(F.phone, lab_all, F.n_phones, K)}
        cents[f] = Cc
        t = res[f]["test"]
        o(f"    test: phone purity {t['phone_purity']:.{DIG}f}  cluster purity {t['cluster_purity']:.{DIG}f}  PNMI {t['pnmi']:.{DIG}f}   ({info['seconds']:.1f} s)")
    MEDIA_D2.mkdir(parents=True, exist_ok=True)
    meta = {"K": K, "seed": km["seed"], "n_init": km["n_init"], "features": FEATS, "n_train": int(F.tr.sum()), "n_test": int(F.te.sum()),
            "time": C.now(), "purities": res}
    np.savez(kmeans_path(), meta=json.dumps(meta, ensure_ascii=False, default=float), **{f"C_{f}": cents[f] for f in FEATS})
    o(f"  → {kmeans_path()}")
    return meta


class Clusters:
    def __init__(self):
        if not kmeans_path().exists():
            C.note("  沒有 k-means 快取，現在算（三組，M5 Max 上估十幾秒）……")
            cluster_all()
        z = np.load(kmeans_path(), allow_pickle=False)
        self.meta = json.loads(str(z["meta"]))
        self.C = {f: z[f"C_{f}"] for f in FEATS}


# ── 整句表徵（extract；要模型）────────────────────────────────────
def seq_path(key: str) -> Path:
    return MEDIA_D2 / f"seq_{key}.npz"


def utts_needed(words: dict | None) -> list[tuple[str, Path]]:
    """要抽整句表徵的：那句話、念同一個字的其他說話人（每人一句，最多 max_others）、自錄的 wav。"""
    need = [(SENT_UTT, D1.wav_path(SENT_UTT))]
    if words:
        for row in word_candidates(words):
            need.append((row["utt"], D1.wav_path(row["utt"])))
    for i, w in enumerate(CFG["extra"]["wavs"]):
        need.append((extra_key(i, w), (HERE / w).resolve()))
    return need


def extra_key(i: int, w: str) -> str:
    labels = CFG["extra"]["labels"]
    return "extra_" + (labels[i] if i < len(labels) and labels[i] else Path(w).stem)


def extract(out=None, device: str | None = None) -> dict:
    o = out or print
    words = load_words()
    if words is None:
        o(f"  {C.AMBER}沒有 data/words.json（./present.sh fetch），只抽那句話；鍵 3 要字的區間{C.RESET}")
    need = utts_needed(words)
    C.note(f"載入 {D1.CFG['models'][MODEL]['repo']}（{device or D1.CFG['infer']['device']}）……")
    enc = D1.Encoder(MODEL, device or D1.CFG["infer"]["device"])
    l6, l9 = CFG["features"]["l6"]["layer"], CFG["features"]["l9"]["layer"]
    MEDIA_D2.mkdir(parents=True, exist_ok=True)
    done = {}
    for key, wav in need:
        if not wav.exists():
            o(f"  {C.AMBER}{key}: 找不到 {wav}{C.RESET}"); continue
        x = C.load_wav(wav, FS)
        t0 = time.perf_counter()
        _, rest = enc.layers(x)
        ms = 1000 * (time.perf_counter() - t0)
        m = mfcc39(x)
        T = min(rest.shape[1], m.shape[0])
        np.savez(seq_path(key), l6=rest[l6 - 1][:T].astype(np.float32), l9=rest[l9 - 1][:T].astype(np.float32), mfcc=m[:T].astype(np.float32),
                 meta=json.dumps({"key": key, "wav": str(wav), "T": int(T), "T_model": int(rest.shape[1]), "T_mfcc": int(m.shape[0]), "ms": ms,
                                  "device": str(enc.device), "repo": D1.CFG["models"][MODEL]["repo"], "commit": D1.read_lock().get(D1.CFG["models"][MODEL]["repo"]),
                                  "time": C.now()}, ensure_ascii=False))
        done[key] = T
        o(f"  {key:<24s} {len(x) / FS:5.2f} s  T = {T:4d}  {ms:6.1f} ms  → {seq_path(key).name}")
    return done


def load_seq(key: str) -> dict:
    p = seq_path(key)
    if not p.exists():
        sys.exit(f"沒有整句表徵 {p}。先 ./present.sh extract（要模型）。")
    z = np.load(p, allow_pickle=False)
    return {"l6": z["l6"], "l9": z["l9"], "mfcc": z["mfcc"], "meta": json.loads(str(z["meta"]))}


# ── unit 序列的小工具 ─────────────────────────────────────────
def rle(ids) -> list[list[int]]:
    """[[id, start_frame, n_frames], …]"""
    out = []
    for t, u in enumerate(ids):
        if out and out[-1][0] == u:
            out[-1][2] += 1
        else:
            out.append([int(u), t, 1])
    return out


def lcs(a: list[int], b: list[int]) -> tuple[int, list[bool], list[bool]]:
    """最長公共子序列的長度，與 a、b 各自哪些位置在 LCS 裡。"""
    n, m = len(a), len(b)
    L = np.zeros((n + 1, m + 1), dtype=np.int32)
    for i in range(n - 1, -1, -1):
        for j in range(m - 1, -1, -1):
            L[i, j] = L[i + 1, j + 1] + 1 if a[i] == b[j] else max(L[i + 1, j], L[i, j + 1])
    ia, ib = [False] * n, [False] * m
    i = j = 0
    while i < n and j < m:
        if a[i] == b[j]:
            ia[i] = ib[j] = True; i += 1; j += 1
        elif L[i + 1, j] >= L[i, j + 1]:
            i += 1
        else:
            j += 1
    return int(L[0, 0]), ia, ib


def frames_in(T: int, s: float, e: float) -> np.ndarray:
    c = frame_centers_s(T)
    return np.flatnonzero((c >= s) & (c < e))


# ── 句子：字、音素段、unit 與統計 ───────────────────────────────
def sentence_stats(units: np.ndarray, segs: list[list], T: int) -> dict:
    """一個特徵的 unit 序列對音素段：每個音素段幾個 run、幾個不同 unit；unit 邊界有幾成落在音素邊界 ±1 frame；靜音 unit；焦點音素各次出現的 unit。"""
    c = frame_centers_s(T)
    speech = [s for s in segs if s[3]]
    runs_per, distinct_per = [], []
    for ph, s, e, _ in speech:
        idx = frames_in(T, s, e)
        if len(idx) == 0:
            continue
        u = units[idx]
        runs_per.append(len(rle(u))); distinct_per.append(len(set(u.tolist())))
    # 邊界：音素段的起點（第一個中心 ≥ s 的 frame）
    pb = sorted({int(np.searchsorted(c, s)) for _, s, _, _ in speech} | {int(np.searchsorted(c, e)) for _, _, e, _ in speech})
    pb = [b for b in pb if 0 < b < T]
    ub = [r[1] for r in rle(units) if r[1] > 0]
    pbs = set(pb)
    exact = sum(b in pbs for b in ub)
    near = sum(any(abs(b - p) <= 1 for p in pb) for b in ub)
    cov1 = {p + d for p in pb for d in (-1, 0, 1)} & set(range(1, T))
    chance_exact, chance = len(pbs) / (T - 1), len(cov1) / (T - 1)   # 一個隨機的 unit 邊界剛好落在音素邊界（±0／±1 frame）的機率
    # 靜音：不在任何區間的 frame
    in_any = np.zeros(T, dtype=bool)
    for _, s, e, _ in segs:
        in_any[frames_in(T, s, e)] = True
    sil = np.flatnonzero(~in_any)
    sil_unit = None
    if len(sil):
        cnt = np.bincount(units[sil], minlength=K)
        u0 = int(cnt.argmax())
        top = [int(k) for k in np.argsort(cnt)[::-1][:3] if cnt[k] > 0]
        sil_unit = {"unit": u0, "share_of_silence_frames": float(cnt[u0] / len(sil)), "share_of_its_frames_in_silence": float(cnt[u0] / max(1, (units == u0).sum())),
                    "n_silence_frames": int(len(sil)),
                    "top": [{"unit": k, "n_silent": int(cnt[k]), "n_total": int((units == k).sum())} for k in top],
                    "top3_share_of_silence_frames": float(cnt[top].sum() / len(sil))}
    # 焦點音素
    fp = CFG["sentence"]["focus_phone"]
    if fp == "auto":
        from collections import Counter
        fp = Counter(s[0] for s in speech).most_common(1)[0][0]
    occ = []
    for ph, s, e, _ in speech:
        if ph == fp:
            idx = frames_in(T, s, e)
            occ.append({"start": s, "end": e, "n_frames": int(len(idx)), "runs": [r[0] for r in rle(units[idx])]})
    return {"n_phone_segments": len(speech), "n_unit_runs": len(rle(units)), "n_unit_runs_speech": int(sum(runs_per)),
            "mean_runs_per_phone": float(np.mean(runs_per)) if runs_per else 0.0, "mean_distinct_per_phone": float(np.mean(distinct_per)) if distinct_per else 0.0,
            "frac_phones_with_one_unit": float(np.mean([d == 1 for d in distinct_per])) if distinct_per else 0.0,
            "unit_boundaries": len(ub), "unit_boundaries_at_phone_boundary": int(exact), "frac_exact": float(exact / max(1, len(ub))), "chance_exact": chance_exact,
            "unit_boundaries_near_phone_boundary": int(near), "frac_near": float(near / max(1, len(ub))), "chance_near": chance,
            "n_phone_boundaries": len(pb), "silence_unit": sil_unit, "focus_phone": fp, "focus_occurrences": occ,
            "n_distinct_units_in_utt": int(len(set(units.tolist())))}


def sentence_bundle(cl: Clusters, words: dict | None) -> dict:
    seq = load_seq(SENT_UTT)
    T = seq["meta"]["T"]
    segs = phone_segments(SENT_UTT)
    units = {f: assign(seq[f], cl.C[f]) for f in FEATS}
    return {"utt": SENT_UTT, "T": T, "frame_ms": FRAME_MS, "dur_s": T * FRAME_MS / 1000, "phones": segs,
            "words": [[str(w).upper(), float(s), float(e)] for w, s, e in words[SENT_UTT]] if words and SENT_UTT in words else None,
            "units": {f: units[f].tolist() for f in FEATS}, "stats": {f: sentence_stats(units[f], segs, T) for f in FEATS},
            "seq_meta": seq["meta"]}


# ── 同一個字、不同說話人 ─────────────────────────────────────────
def word_candidates(words: dict) -> list[dict]:
    """念了 [word].word（auto = 句子裡其他說話人也念過、音素最多的字）的其他說話人，每人取第一句；最多 max_others。"""
    sub = D1.load_subset()
    spk_of = {u["id"]: u["spk"] for u in sub["utts"]}
    sent_words = [str(w).upper() for w, _, _ in words.get(SENT_UTT, [])]
    want = CFG["word"]["word"].upper()
    if want == "AUTO":
        best = None
        for w in dict.fromkeys(sent_words):
            n = len({spk_of[u] for u, ws in words.items() if u in spk_of and spk_of[u] != spk_of.get(SENT_UTT) and any(str(x[0]).upper() == w for x in ws)})
            if n and (best is None or len(w) > len(best[0])):
                best = (w, n)
        want = best[0] if best else sent_words[0]
    rows, seen = [], set()
    for u in sorted(words, key=lambda k: (int(spk_of.get(k, "0")), k)):
        if u not in spk_of or spk_of[u] == spk_of.get(SENT_UTT) or spk_of[u] in seen:
            continue
        hit = [x for x in words[u] if str(x[0]).upper() == want]
        if hit:
            rows.append({"utt": u, "spk": spk_of[u], "word": want, "start": float(hit[0][1]), "end": float(hit[0][2])})
            seen.add(spk_of[u])
        if len(rows) >= CFG["word"]["max_others"]:
            break
    return rows


def word_bundle(cl: Clusters, words: dict | None) -> dict | None:
    if not words or SENT_UTT not in words:
        return None
    rows = word_candidates(words)
    want = rows[0]["word"] if rows else CFG["word"]["word"].upper()
    hit = [x for x in words[SENT_UTT] if str(x[0]).upper() == want]
    if not hit:
        return {"word": want, "error": f"那句話的 words.json 裡沒有 {want}"}
    sub = D1.load_subset()
    ref_spk = next(u["spk"] for u in sub["utts"] if u["id"] == SENT_UTT)
    out_rows = []
    for r in [{"utt": SENT_UTT, "spk": ref_spk, "word": want, "start": float(hit[0][1]), "end": float(hit[0][2])}] + rows:
        if not seq_path(r["utt"]).exists():
            continue
        seq = load_seq(r["utt"])
        idx = frames_in(seq["meta"]["T"], r["start"], r["end"])
        row = dict(r, n_frames=int(len(idx)), units={}, runs={})
        for f in FEATS:
            u = assign(seq[f][idx], cl.C[f])
            row["units"][f] = u.tolist(); row["runs"][f] = [x[0] for x in rle(u)]
        out_rows.append(row)
    ref = out_rows[0]
    for row in out_rows[1:]:
        row["vs_ref"] = {}
        for f in FEATS:
            n, ia, ib = lcs(ref["runs"][f], row["runs"][f])
            A, B = set(ref["runs"][f]), set(row["runs"][f])
            row["vs_ref"][f] = {"lcs": n, "lcs_frac_ref": n / max(1, len(ref["runs"][f])), "lcs_frac_this": n / max(1, len(row["runs"][f])),
                                "jaccard": len(A & B) / max(1, len(A | B)), "in_lcs_ref": ia, "in_lcs_this": ib}
    return {"word": want, "rows": out_rows, "n_candidates": len(rows)}


def extras_bundle(cl: Clusters) -> list[dict]:
    out = []
    for i, w in enumerate(CFG["extra"]["wavs"]):
        key = extra_key(i, w)
        if not seq_path(key).exists():
            continue
        seq = load_seq(key)
        row = {"key": key, "wav": w, "T": seq["meta"]["T"], "units": {}, "runs": {}}
        for f in FEATS:
            u = assign(seq[f], cl.C[f]); row["units"][f] = u.tolist(); row["runs"][f] = [x[0] for x in rle(u)]
        out.append(row)
    return out


# ── 輸出紀錄 ─────────────────────────────────────────────────
class Out:
    def __init__(self):
        self.buf = io.StringIO()

    def __call__(self, s: str = "") -> None:
        print(s, flush=True)
        self.buf.write(re.sub(r"\033\[[0-9;]*m", "", s) + "\n")

    def banner(self, s: str, color: str = C.BLUE) -> None:
        C.banner(s, color)
        self.buf.write(f"\n{'─' * 72}\n{s}\n{'─' * 72}\n")

    def text(self) -> str:
        return self.buf.getvalue()


# ── 資料來源：live（快取）／replay（彩排紀錄）／fake ───────────────
class Live:
    def __init__(self, out):
        self.out = out
        self.cl = Clusters()
        self.words = load_words()
        self._sent = self._word = self._extras = None

    def purities(self) -> dict:
        return self.cl.meta["purities"]

    def kmeans_meta(self) -> dict:
        return {k: v for k, v in self.cl.meta.items() if k != "purities"}

    def sentence(self) -> dict:
        if self._sent is None:
            self._sent = sentence_bundle(self.cl, self.words)
        return self._sent

    def word(self) -> dict | None:
        if self._word is None:
            self._word = word_bundle(self.cl, self.words)
        return self._word

    def extras(self) -> list[dict]:
        if self._extras is None:
            self._extras = extras_bundle(self.cl)
        return self._extras


class Replay:
    def __init__(self, rec: dict):
        self.rec = rec

    def purities(self): return self.rec["purities"]
    def kmeans_meta(self): return self.rec["kmeans"]
    def sentence(self): return self.rec["sentence"]
    def word(self): return self.rec.get("word")
    def extras(self): return self.rec.get("extras", [])


# ── 主體：視窗與鍵 ───────────────────────────────────────────
def unit_color(u: int):
    import colorsys
    h = (u * 0.6180339887) % 1.0
    return colorsys.hls_to_rgb(h, 0.80 if u % 2 else 0.70, 0.75)


PH_COLOR = {"vowel": "#f3d9a4", "cons": "#cfe3f5", "sil": "#e6e6e6"}
VOWELS = {"AA", "AE", "AH", "AO", "AW", "AY", "EH", "ER", "EY", "IH", "IY", "OW", "OY", "UH", "UW"}


class Demo:
    def __init__(self, out: Out, src, tag: str = ""):
        self.out, self.src, self.tag = out, src, tag
        self.log: dict = {"time": C.now(), "versions": versions(), "tag": tag, "sections": {}}
        self.fig = self.plt = None
        self.ax_sent = self.ax_over = self.ax_word = None
        self.win: tuple[float, float] | None = None
        self.extra_rows = False
        self.paper = None
        if DATA_JSON.exists():
            try:
                self.paper = json.loads(DATA_JSON.read_text(encoding="utf-8"))["paper_settings"]["hubert"]["mfcc_teacher_K100"]
            except Exception:
                self.paper = None

    # ── 視窗 ──
    def make_fig(self):
        import matplotlib.pyplot as plt
        self.plt = plt
        self.fig = plt.figure(figsize=(CFG["display"]["fig_w"], CFG["display"]["fig_h"]))
        gs = self.fig.add_gridspec(3, 1, height_ratios=[5.2, 0.9, 3.2], hspace=0.55)
        self.ax_sent, self.ax_over, self.ax_word = (self.fig.add_subplot(gs[i]) for i in range(3))
        for ax in (self.ax_sent, self.ax_over, self.ax_word):
            ax.set_yticks([]); ax.set_xticks([])
            for s in ("top", "right", "left"):
                ax.spines[s].set_visible(False)
        self.ax_sent.set_title("press 2 — one sentence: words, phones (MFA) and k-means units", fontsize=11, loc="left")
        self.ax_word.set_title("press 3 — the same word, different speakers", fontsize=11, loc="left")
        self.fig.suptitle("W5 Demo 2 — units are not phones" + (f"   [{self.tag}]" if self.tag else ""), fontsize=13)
        self.fig.subplots_adjust(left=0.17, right=0.98, top=0.9, bottom=0.07)
        self.fig.canvas.mpl_connect("key_press_event", self.on_key)
        return self.fig

    def redraw(self):
        if self.fig is not None:
            self.fig.canvas.draw_idle(); self.plt.pause(0.05)

    def _px_per_s(self, ax) -> float:
        x0, x1 = ax.get_xlim()
        return ax.get_window_extent().width / max(1e-9, x1 - x0)

    # ── 1：純度表 ──
    def table(self) -> None:
        o = self.out
        P = self.src.purities()
        km = self.src.kmeans_meta()
        o.banner(f"1  three teachers, K = {km['K']}, {km['n_train']} train / {km['n_test']} test frames (phone labels: MFA)")
        o(f"  {'teacher features':<24s} {'phone purity':>13s} {'cluster purity':>15s} {'PNMI':>8s}   {'(all frames: phone / cluster / PNMI)':>36s}")
        for f in FEATS:
            r = P[f]; t, a = r["test"], r["all"]
            o(f"  {r['label']:<24s} {t['phone_purity']:>13.{DIG}f} {t['cluster_purity']:>15.{DIG}f} {t['pnmi']:>8.{DIG}f}   "
              f"{a['phone_purity']:>12.{DIG}f} / {a['cluster_purity']:.{DIG}f} / {a['pnmi']:.{DIG}f}   {r['role']}")
        if self.paper:
            o(f"  {'paper, MFCC K=100, 960 h':<24s} {self.paper['phone_purity']:>13.3f} {self.paper['cluster_purity']:>15.3f} {self.paper['PNMI']:>8.3f}   (arXiv 2106.07447v1 §V-C)")
        o(f"  k-means: {km['n_init']} inits, seed {km['seed']}; " + ", ".join(f"{f} {P[f]['kmeans']['iters']} iters / {P[f]['kmeans']['seconds']:.1f} s" for f in FEATS if 'seconds' in P[f]['kmeans']))
        o(f"  H(phone) = {P[FEATS[0]]['test']['h_phone_nat']:.3f} nat on the test frames; clusters used: " + ", ".join(f"{f} {P[f]['test']['n_clusters_used']}/{K}" for f in FEATS))
        self.log["sections"]["table"] = {f: {"test": P[f]["test"], "all": P[f]["all"]} for f in FEATS}

    # ── 2：句子 ──
    def _default_window(self, S: dict) -> tuple[float, float]:
        if S["words"]:
            n = min(CFG["sentence"]["zoom_words"], len(S["words"]))
            return (S["words"][0][1] - 0.05, S["words"][n - 1][2] + 0.05)
        sp = [s for s in S["phones"] if s[3]]
        a = sp[0][1] if sp else 0.0
        return (a - 0.05, a + CFG["sentence"]["zoom_s"])

    def sentence(self, window: tuple[float, float] | None = None, announce: bool = True) -> None:
        o = self.out
        S = self.src.sentence()
        if self.win is None:
            self.win = self._default_window(S)
        if window is not None:
            self.win = window
        if announce:
            o.banner(f"2  {S['utt']}  T = {S['T']} frames ({FRAME_MS:.0f} ms), {S['dur_s']:.2f} s — phones from the MFA alignment, units from k-means on {', '.join(FEATS)}")
            if S["words"]:
                o("  words: " + " ".join(w for w, _, _ in S["words"]))
            for f in FEATS:
                st = S["stats"][f]
                o(f"  {CFG['features'][f]['label']:<22s} {st['n_phone_segments']} phone segments ↔ {st['n_unit_runs_speech']} unit runs inside them: "
                  f"{st['mean_runs_per_phone']:.2f} runs and {st['mean_distinct_per_phone']:.2f} distinct units per phone; {100 * st['frac_phones_with_one_unit']:.0f} % of phones get a single unit")
                o(f"  {'':<22s} unit boundaries on a phone boundary: {st['unit_boundaries_at_phone_boundary']}/{st['unit_boundaries']} = {100 * st['frac_exact']:.0f} % (chance {100 * st['chance_exact']:.0f} %); "
                  f"within ±1 frame: {st['unit_boundaries_near_phone_boundary']}/{st['unit_boundaries']} = {100 * st['frac_near']:.0f} % (chance {100 * st['chance_near']:.0f} %)")
                if st["silence_unit"]:
                    su = st["silence_unit"]
                    o(f"  {'':<22s} silence ({su['n_silence_frames']} frames outside any interval): " + ", ".join(f"unit {t['unit']} {t['n_silent']}/{t['n_total']} of its frames" for t in su["top"])
                      + f" — the top 3 cover {100 * su['top3_share_of_silence_frames']:.0f} % of the silent frames")
                occ = st["focus_occurrences"]
                o(f"  {'':<22s} phone {st['focus_phone']} × {len(occ)}: " + "  |  ".join(" ".join(str(u) for u in x["runs"]) for x in occ))
            self.log["sections"]["sentence"] = {f: S["stats"][f] for f in FEATS}
        self.draw_sentence(S)

    def draw_sentence(self, S: dict) -> None:
        if self.fig is None:
            return
        ax = self.ax_sent
        ax.cla(); ax.set_yticks([])
        for s in ("top", "right", "left"):
            ax.spines[s].set_visible(False)
        x0, x1 = self.win
        ax.set_xlim(x0, x1)
        rows = ["words", "phones", "l6"] + (["l9", "mfcc"] if self.extra_rows else [])
        if not S["words"]:
            rows.remove("words")
        n = len(rows)
        ax.set_ylim(0, n)
        ax.set_xlabel("time (s)")
        px = self._px_per_s(ax)
        minw = CFG["display"]["unit_text_min_px"]
        c = frame_centers_s(S["T"])
        names = {"words": "words (MFA)", "phones": "phones (MFA)", "l6": CFG["features"]["l6"]["label"] + " units", "l9": CFG["features"]["l9"]["label"] + " units", "mfcc": CFG["features"]["mfcc"]["label"] + " units"}
        for r, kind in enumerate(rows):
            y = n - 1 - r
            ax.text(x0 - 0.01, y + 0.5, names[kind], ha="right", va="center", fontsize=9, clip_on=False)
            # 標籤放在**看得見的那一段**的中點（被窗切到的塊不要把字印到窗外、壓到列標籤）
            def label(s, e, text, fs):
                vs, ve = max(s, x0), min(e, x1)
                if (ve - vs) * px >= minw:
                    ax.text((vs + ve) / 2, y + 0.5, text, ha="center", va="center", fontsize=fs)
            if kind == "words":
                for w, s, e in S["words"]:
                    if e < x0 or s > x1: continue
                    ax.add_patch(self.plt.Rectangle((s, y + 0.15), e - s, 0.7, fc="#dddddd", ec="#555555", lw=0.8))
                    label(s, e, w, 9)
            elif kind == "phones":
                for ph, s, e, sp in S["phones"]:
                    if e < x0 or s > x1: continue
                    fc = PH_COLOR["sil"] if not sp else PH_COLOR["vowel"] if ph in VOWELS else PH_COLOR["cons"]
                    ax.add_patch(self.plt.Rectangle((s, y + 0.1), e - s, 0.8, fc=fc, ec="#444444", lw=0.8))
                    label(s, e, ph, 9)
            else:
                units = S["units"][kind]
                for u, t0, nfr in rle(units):
                    s, e = c[t0] - FRAME_MS / 2000, c[t0 + nfr - 1] + FRAME_MS / 2000
                    if e < x0 or s > x1: continue
                    ax.add_patch(self.plt.Rectangle((s, y + 0.1), e - s, 0.8, fc=unit_color(u), ec="#333333", lw=0.6))
                    label(s, e, str(u), 8)
                # 每個 frame 的細分隔線
                for t in range(S["T"]):
                    s = c[t] - FRAME_MS / 2000
                    if x0 <= s <= x1: ax.plot([s, s], [y + 0.1, y + 0.2], color="#333333", lw=0.4)
        # 音素邊界畫成貫穿 unit 列的虛線，看邊界對不對齊
        for ph, s, e, sp in S["phones"]:
            for b in (s, e):
                if x0 <= b <= x1 and sp:
                    ax.plot([b, b], [0, n - (1 if S["words"] else 0) - 0.0], color="#444444", lw=0.5, ls=":", zorder=0)
        ax.set_title(f"press 2 — {S['utt']}: words, phones (MFA) and k-means units (K = {K}); dotted = phone boundaries; [ ] move, f full, u more rows", fontsize=10, loc="left")
        # overview
        ov = self.ax_over
        ov.cla(); ov.set_yticks([]); ov.set_xticks([])
        for s in ("top", "right", "left"):
            ov.spines[s].set_visible(False)
        ov.set_xlim(0, S["dur_s"]); ov.set_ylim(0, 2)
        for ph, s, e, sp in S["phones"]:
            fc = PH_COLOR["sil"] if not sp else PH_COLOR["vowel"] if ph in VOWELS else PH_COLOR["cons"]
            ov.add_patch(self.plt.Rectangle((s, 1.05), e - s, 0.9, fc=fc, ec="#666666", lw=0.3))
        for u, t0, nfr in rle(S["units"]["l6"]):
            ov.add_patch(self.plt.Rectangle((c[t0] - FRAME_MS / 2000, 0.05), nfr * FRAME_MS / 1000, 0.9, fc=unit_color(u), ec="#666666", lw=0.3))
        ov.add_patch(self.plt.Rectangle((x0, 0), x1 - x0, 2, fc="none", ec="red", lw=1.5))
        ov.text(-0.01 * S["dur_s"], 1.5, "phones", ha="right", va="center", fontsize=8); ov.text(-0.01 * S["dur_s"], 0.5, "l6 units", ha="right", va="center", fontsize=8)
        self.redraw()

    def move(self, direction: int) -> None:
        S = self.src.sentence()
        x0, x1 = self.win
        if S["words"]:
            starts = [w[1] for w in S["words"]]
            cur = int(np.argmin([abs(s - (x0 + 0.05)) for s in starts]))
            i = max(0, min(len(starts) - 1, cur + direction))
            width = x1 - x0
            self.win = (starts[i] - 0.05, starts[i] - 0.05 + width)
        else:
            w = x1 - x0
            self.win = (max(0.0, x0 + direction * w / 2), max(0.0, x0 + direction * w / 2) + w)
        self.sentence(announce=False)

    def full(self) -> None:
        S = self.src.sentence()
        self.sentence((0.0, S["dur_s"]), announce=False)

    def toggle_rows(self) -> None:
        self.extra_rows = not self.extra_rows
        self.out(f"  unit rows: {'l6 + l9 + mfcc' if self.extra_rows else 'l6 only'}")
        self.sentence(announce=False)

    # ── 3：同一個字 ──
    def word(self) -> None:
        o = self.out
        W = self.src.word()
        if W is None:
            o(f"  {C.AMBER}沒有 data/words.json（./present.sh fetch）或那句話不在裡面；鍵 3 沒有東西可畫{C.RESET}"); return
        if "error" in W:
            o(f"  {C.AMBER}{W['error']}{C.RESET}"); return
        rows = W["rows"]
        o.banner(f"3  '{W['word']}' — speaker {rows[0]['spk']} (the sentence) vs. {len(rows) - 1} other speaker(s); collapsed unit runs, | = a run")
        for f in FEATS:
            o(f"  {CFG['features'][f]['label']}:")
            for r in rows:
                vs = r.get("vs_ref", {}).get(f)
                tail = f"   LCS {vs['lcs']}/{len(rows[0]['runs'][f])} of the sentence's runs ({100 * vs['lcs_frac_ref']:.0f} %), Jaccard of unit sets {vs['jaccard']:.2f}" if vs else "   (reference)"
                o(f"    spk {r['spk']:<5s} {r['n_frames']:3d} frames  {' '.join(str(u) for u in r['runs'][f]):<60s}{tail}")
        self.log["sections"]["word"] = {"word": W["word"], "rows": [{k: v for k, v in r.items() if k not in ("units",)} for r in rows]}
        self.draw_word(W)

    def draw_word(self, W: dict) -> None:
        if self.fig is None:
            return
        ax = self.ax_word
        ax.cla(); ax.set_yticks([]); ax.set_xticks([])
        for s in ("top", "right", "left", "bottom"):
            ax.spines[s].set_visible(False)
        rows = W["rows"]
        feats = ["l6"] + (["l9", "mfcc"] if self.extra_rows else [])
        n = len(rows) * len(feats)
        maxlen = max(len(r["runs"][f]) for r in rows for f in feats)
        ax.set_xlim(-0.5, max(8, maxlen) + 0.5); ax.set_ylim(0, n)
        k = 0
        for f in feats:
            for i, r in enumerate(rows):
                y = n - 1 - k; k += 1
                ax.text(-0.6, y + 0.5, f"{CFG['features'][f]['label'] if i == 0 else ''}  spk {r['spk']} ({r['n_frames']} fr)", ha="right", va="center", fontsize=8, clip_on=False)
                mask = r.get("vs_ref", {}).get(f, {}).get("in_lcs_this") if i else None
                for j, u in enumerate(r["runs"][f]):
                    bad = (mask is not None) and (not mask[j])
                    ax.add_patch(self.plt.Rectangle((j - 0.45, y + 0.1), 0.9, 0.8, fc=unit_color(u), ec="red" if bad else "#333333", lw=2.0 if bad else 0.6))
                    ax.text(j, y + 0.5, str(u), ha="center", va="center", fontsize=8)
        ax.set_title(f"press 3 — '{W['word']}' said by {len(rows)} speakers: collapsed unit runs; red = not in the longest common subsequence with the sentence's row", fontsize=10, loc="left")
        self.redraw()

    # ── 4：自錄 ──
    def extras(self) -> None:
        o = self.out
        E = self.src.extras()
        if not E:
            o(f"  {C.AMBER}沒有自錄的 wav（demo_config.toml [extra].wavs，再 ./present.sh extract）{C.RESET}"); return
        o.banner(f"4  {len(E)} own recordings — unit runs (l6)")
        for r in E:
            o(f"  {r['key']:<20s} T = {r['T']:3d}  l6: {' '.join(str(u) for u in r['runs']['l6'])}")
        for a in E:
            for b in E:
                if a is b: continue
                n, _, _ = lcs(a["runs"]["l6"], b["runs"]["l6"])
                o(f"    LCS({a['key']}, {b['key']}) = {n} / {len(a['runs']['l6'])}")
        self.log["sections"]["extras"] = [{k: v for k, v in r.items() if k != "units"} for r in E]
        if self.fig is None:
            return
        ax = self.ax_word
        ax.cla(); ax.set_yticks([]); ax.set_xticks([])
        n = len(E); maxlen = max(len(r["runs"]["l6"]) for r in E)
        ax.set_xlim(-0.5, maxlen + 0.5); ax.set_ylim(0, n)
        for i, r in enumerate(E):
            y = n - 1 - i
            ax.text(-0.6, y + 0.5, r["key"], ha="right", va="center", fontsize=8, clip_on=False)
            for j, u in enumerate(r["runs"]["l6"]):
                ax.add_patch(self.plt.Rectangle((j - 0.45, y + 0.1), 0.9, 0.8, fc=unit_color(u), ec="#333333", lw=0.6))
                ax.text(j, y + 0.5, str(u), ha="center", va="center", fontsize=8)
        ax.set_title("press 4 — own recordings: collapsed l6 unit runs", fontsize=10, loc="left")
        self.redraw()

    # ── 鍵 ──
    def menu(self) -> None:
        o = self.out
        o(f"\n{C.BOLD}p45{C.RESET}  1 三個純度   2 那句話（[ ] 移窗、f 整句、u 多兩列）   3 同一個字不同說話人   4 自錄   h 選單   q 離開")

    def dispatch(self, k: str) -> bool:
        if k == "1": self.table()
        elif k == "2": self.sentence()
        elif k == "[": self.move(-1)
        elif k == "]": self.move(+1)
        elif k == "f": self.full()
        elif k == "u": self.toggle_rows()
        elif k == "3": self.word()
        elif k == "4": self.extras()
        elif k == "h": self.menu()
        elif k == "q": return False
        return True

    def on_key(self, ev) -> None:
        if not self.dispatch(ev.key):
            self.plt.close(self.fig)

    def run_all(self, folder: Path | None = None) -> None:
        """課堂順序全部跑一遍（rehearse 用）；folder 給了就逐張存圖。"""
        self.table()
        self.sentence()
        if folder and self.fig is not None:
            self.fig.savefig(folder / "sentence_zoom.png", dpi=130)
        self.full()
        if folder and self.fig is not None:
            self.fig.savefig(folder / "sentence_full.png", dpi=130)
        self.extra_rows = True
        self.sentence(self._default_window(self.src.sentence()), announce=False)
        self.word()
        if folder and self.fig is not None:
            self.fig.savefig(folder / "sentence_word.png", dpi=130)
        self.extra_rows = False
        self.extras()

    def record(self) -> dict:
        """replay 用的完整紀錄。"""
        return {"time": self.log["time"], "tag": self.tag, "versions": self.log["versions"], "config": CFG,
                "kmeans": self.src.kmeans_meta(), "purities": self.src.purities(), "sentence": self.src.sentence(),
                "word": self.src.word(), "extras": self.src.extras(), "sections": self.log["sections"]}

    def save(self, folder: Path) -> None:
        folder.mkdir(parents=True, exist_ok=True)
        stem = "rehearsal" if folder.name == "rehearsal" else "session"
        rec = self.record() if stem == "rehearsal" else self.log
        (folder / f"{stem}.json").write_text(json.dumps(rec, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
        (folder / f"{stem}.txt").write_text(self.out.text(), encoding="utf-8")
        if self.fig is not None and stem == "session":
            self.fig.savefig(folder / "window.png", dpi=130)
        C.note(f"  存到 {folder}/（{stem}.json、{stem}.txt）")


def versions() -> dict:
    v = C.versions()
    from importlib import metadata
    for pkg in ("torch", "transformers", "soundfile", "pyarrow"):
        try:
            v[pkg] = metadata.version(pkg)
        except metadata.PackageNotFoundError:
            v[pkg] = None
    return v


# ── fake ─────────────────────────────────────────────────────
class Fake:
    """合成：39 個音素、K 個 cluster；三組特徵的 unit 對音素的雜訊依序變小（mfcc 最吵）。句子用真的對齊（有的話），unit 隨機但在音素段內大致一致。"""

    def __init__(self, seed: int = 7):
        rng = np.random.default_rng(seed)
        self.rng = rng
        n_y, N = 39, 6000
        y = rng.integers(0, n_y, N)
        split = rng.random(N) < 0.7
        self._pur = {}
        noise = {"mfcc": 0.7, "l6": 0.35, "l9": 0.3}
        for f in FEATS:
            base = (y * 2 + rng.integers(0, 2, N)) % K              # 每個音素兩個 cluster
            k = np.where(rng.random(N) < noise[f], rng.integers(0, K, N), base)
            self._pur[f] = {"label": CFG["features"][f]["label"] + " [FAKE]", "role": CFG["features"][f]["role"], "dim": 0,
                            "kmeans": {"inertia": 0.0, "iters": 0, "init": 0, "seconds": 0.0},
                            "test": purity(y[~split], k[~split], n_y, K), "train": purity(y[split], k[split], n_y, K), "all": purity(y, k, n_y, K)}
        self._km = {"K": K, "seed": seed, "n_init": 0, "features": FEATS, "n_train": int(split.sum()), "n_test": int((~split).sum()), "time": C.now()}
        # 句子
        if (DATA / "alignments.json").exists() and SENT_UTT in D1.load_alignments():
            segs = phone_segments(SENT_UTT)
        else:
            segs, t = [], 0.3
            for i in range(30):
                d = 0.04 + 0.1 * rng.random()
                segs.append([D1.ARPABET39[rng.integers(0, 39)], t, t + d, True]); t += d
        T = int(round((segs[-1][2] + 0.4) * 1000 / FRAME_MS))
        c = frame_centers_s(T)
        units = {}
        for f in FEATS:
            u = np.full(T, 0)
            for ph, s, e, sp in segs:
                idx = frames_in(T, s, e)
                if not len(idx): continue
                a, b = (sum(map(ord, ph)) % 97) + 1, (sum(map(ord, ph)) * 7 % 97) + 1
                half = len(idx) // 2 + rng.integers(-1, 2)
                u[idx[:max(0, half)]] = a; u[idx[max(0, half):]] = b
            flip = rng.random(T) < noise[f] * 0.3
            u[flip] = rng.integers(1, K, flip.sum())
            units[f] = u
        words = [["WORD" + str(i), segs[i][1], segs[min(len(segs) - 1, i + 2)][2]] for i in range(0, len(segs), 3)]
        self._sent = {"utt": SENT_UTT + " [FAKE]", "T": T, "frame_ms": FRAME_MS, "dur_s": T * FRAME_MS / 1000, "phones": segs, "words": words,
                      "units": {f: units[f].tolist() for f in FEATS}, "stats": {f: sentence_stats(units[f], segs, T) for f in FEATS}, "seq_meta": {"fake": True}}
        rows = []
        for i, spk in enumerate(["1272", "9999", "8888"]):
            idx = frames_in(T, words[0][1], words[0][2])
            u = units["l6"][idx].copy()
            if i:
                sw = rng.random(len(u)) < 0.25
                u[sw] = rng.integers(1, K, sw.sum())
            rows.append({"utt": f"fake-{spk}", "spk": spk, "word": words[0][0], "start": words[0][1], "end": words[0][2], "n_frames": int(len(idx)),
                         "units": {f: u.tolist() for f in FEATS}, "runs": {f: [x[0] for x in rle(u)] for f in FEATS}})
        for r in rows[1:]:
            r["vs_ref"] = {}
            for f in FEATS:
                n, ia, ib = lcs(rows[0]["runs"][f], r["runs"][f]); A, B = set(rows[0]["runs"][f]), set(r["runs"][f])
                r["vs_ref"][f] = {"lcs": n, "lcs_frac_ref": n / max(1, len(rows[0]["runs"][f])), "lcs_frac_this": n / max(1, len(r["runs"][f])),
                                  "jaccard": len(A & B) / max(1, len(A | B)), "in_lcs_ref": ia, "in_lcs_this": ib}
        self._word = {"word": words[0][0], "rows": rows, "n_candidates": 2}

    def purities(self): return self._pur
    def kmeans_meta(self): return self._km
    def sentence(self): return self._sent
    def word(self): return self._word
    def extras(self): return []


# ── fetch：字的區間 ───────────────────────────────────────────
def fetch() -> None:
    import pyarrow.parquet as pq
    C.banner("從對齊資料集的 parquet 讀 words 欄 → data/words.json", C.AMBER)
    sub = D1.load_subset()
    ids = {u["id"] for u in sub["utts"]}
    ds, pqf = D1.CFG["phones"]["hf_dataset"], D1.CFG["phones"]["parquet"]
    try:
        from huggingface_hub import HfFileSystem
        with HfFileSystem().open(f"datasets/{ds}/{pqf}", "rb") as f:
            t = pq.ParquetFile(f).read(columns=["id", "words"])
        print("    （只讀 id、words 兩欄，沒有下載音訊）")
    except Exception as e:
        print(f"    欄位投影失敗（{e}），改下載整個 parquet（約 343 MB，不進版控）……")
        from huggingface_hub import hf_hub_download
        path = hf_hub_download(ds, pqf, repo_type="dataset", cache_dir=str(MEDIA / "hf_datasets"))
        t = pq.read_table(path, columns=["id", "words"])
    words: dict[str, list] = {}
    keys = None
    for row in t.to_pylist():
        if row["id"] not in ids:
            continue
        out = []
        for w in row["words"]:
            if keys is None:
                kw = next((k for k in ("word", "text", "label", "token") if k in w), None)
                ks = next((k for k in ("start", "begin", "xmin") if k in w), None)
                ke = next((k for k in ("end", "stop", "xmax") if k in w), None)
                if not (kw and ks and ke):
                    sys.exit(f"words 欄的欄位認不得：{list(w)}")
                keys = (kw, ks, ke)
                print(f"    words 的欄位：{keys}")
            out.append([str(w[keys[0]]).upper(), round(float(w[keys[1]]), 3), round(float(w[keys[2]]), 3)])
        words[row["id"]] = out
    missing = sorted(ids - set(words))
    if missing:
        print(f"  {C.AMBER}{len(missing)} 句沒有字的區間：{missing[:5]}…{C.RESET}")
    DATA.mkdir(exist_ok=True)
    (DATA / "words.json").write_text(json.dumps(words, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    print(f"  {len(words)} 句 → data/words.json（進版控，CC BY 4.0）")
    if SENT_UTT in words:
        print(f"  {SENT_UTT}: " + " ".join(f"{w}@{s}-{e}" for w, s, e in words[SENT_UTT]))
        rows = word_candidates(words)
        print(f"  '{rows[0]['word'] if rows else CFG['word']['word']}' by other speakers: " + (", ".join(f"{r['spk']} ({r['utt']} {r['start']}–{r['end']})" for r in rows) or "none"))
    print("\n下一步：./present.sh extract（要模型）→ ./present.sh rehearse")


# ── export → w05-metrics.json ────────────────────────────────
def export(rec_path: Path) -> None:
    rec = json.loads(rec_path.read_text(encoding="utf-8"))
    d = json.loads(METRICS_JSON.read_text(encoding="utf-8")) if METRICS_JSON.exists() else {
        "_note": "量測數字（進版控的出處，speech_ai/CLAUDE.md 4.8）。w05-data.json 是 make_figs_w05.py 算出來的、每次重產；量測的放這裡。"}
    S = rec["sentence"]
    demo2 = {"source": "demos/w05_d2_units_vs_phones/runs/rehearsal/rehearsal.json", "time": rec["time"], "tag": rec.get("tag", ""),
             "kmeans": rec["kmeans"],
             "definitions": {"phone_purity": "Σ_k max_y p(y,k)", "cluster_purity": "Σ_y max_k p(y,k)", "pnmi": "I(y;k)/H(y)",
                             "frames": "the 23,726 phone-labelled frames of Demo 1 (60 per utterance, seed 20261011); k-means fitted on the train frames, purities on the test frames; 'all' = fitted on train, scored on every frame",
                             "mfcc": "13 cepstra (incl. c0) of the 80-band Slaney log-mel (25/20 ms, 512-pt) by orthonormal DCT-II, + Δ + ΔΔ (±2 frames), not standardised",
                             "phone_labels": "MFA alignments (gilkeyio/librispeech-alignments, CC BY 4.0), stress stripped, 39 ARPAbet classes"},
             "purities": {f: {"label": rec["purities"][f]["label"], "test": {k: round(v, 4) if isinstance(v, float) else v for k, v in rec["purities"][f]["test"].items()},
                              "all": {k: round(v, 4) if isinstance(v, float) else v for k, v in rec["purities"][f]["all"].items()}} for f in FEATS},
             "sentence": {"utt": S["utt"], "T": S["T"], "n_words": len(S["words"]) if S["words"] else None,
                          "stats": {f: {k: (round(v, 4) if isinstance(v, float) else v) for k, v in S["stats"][f].items() if k != "focus_occurrences"} for f in FEATS},
                          "focus_phone": S["stats"]["l6"]["focus_phone"], "focus_units_l6": [x["runs"] for x in S["stats"]["l6"]["focus_occurrences"]]}}
    W = rec.get("word")
    if W and "rows" in W:
        demo2["word"] = {"word": W["word"], "reference_speaker": W["rows"][0]["spk"], "n_frames": [r["n_frames"] for r in W["rows"]],
                         "runs_l6": {r["spk"]: r["runs"]["l6"] for r in W["rows"]},
                         "others": [{"spk": r["spk"], "utt": r["utt"], **{f: {"lcs": r["vs_ref"][f]["lcs"], "lcs_frac_ref": round(r["vs_ref"][f]["lcs_frac_ref"], 4), "jaccard": round(r["vs_ref"][f]["jaccard"], 4)} for f in FEATS}}
                                    for r in W["rows"][1:]]}
    d["demo2"] = demo2
    METRICS_JSON.parent.mkdir(parents=True, exist_ok=True)
    METRICS_JSON.write_text(json.dumps(d, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"  → {METRICS_JSON}（demo2）")


# ── selftest ─────────────────────────────────────────────────
def selftest() -> None:
    import matplotlib
    matplotlib.use("Agg")
    C.banner("selftest：純度定義、k-means、DCT／Δ、RLE／LCS、fake 流程、彩排對 w05-metrics.json", C.AMBER)
    fails = []

    def ok(cond: bool, msg: str):
        print(f"  {'OK ' if cond else 'FAIL'} {msg}")
        if not cond:
            fails.append(msg)

    rng = np.random.default_rng(1)
    # 1. 純度：完美分群 → 三個都是 1；全部同一個 cluster → cluster purity 1、PNMI 0、phone purity = max_y p(y)
    y = rng.integers(0, 5, 1000)
    p = purity(y, y, 5, 5)
    ok(abs(p["phone_purity"] - 1) < 1e-12 and abs(p["cluster_purity"] - 1) < 1e-12 and abs(p["pnmi"] - 1) < 1e-12, f"purity(完美分群) = 1/1/1：{p['phone_purity']:.3f}/{p['cluster_purity']:.3f}/{p['pnmi']:.3f}")
    p1 = purity(y, np.zeros_like(y), 5, 5)
    ok(abs(p1["cluster_purity"] - 1) < 1e-12 and abs(p1["pnmi"]) < 1e-12 and abs(p1["phone_purity"] - np.bincount(y).max() / 1000) < 1e-12,
       f"purity(全部一個 cluster)：phone {p1['phone_purity']:.3f} = max p(y)、cluster 1、PNMI 0")
    k2 = y * 2 + rng.integers(0, 2, 1000)                   # 每個音素拆兩個 cluster：phone purity 1、cluster purity ≈ 0.5、PNMI 1
    p2 = purity(y, k2, 5, 10)
    ok(abs(p2["phone_purity"] - 1) < 1e-12 and 0.4 < p2["cluster_purity"] < 0.6 and abs(p2["pnmi"] - 1) < 1e-12, f"purity(每音素兩 cluster)：phone 1、cluster {p2['cluster_purity']:.3f}、PNMI 1")
    # 2. k-means：五個可分的群
    mu = rng.standard_normal((5, 16)) * 6
    lab = rng.integers(0, 5, 3000)
    X = mu[lab] + rng.standard_normal((3000, 16))
    Cc, info = kmeans(X, 5, 3, 2, 50, 1024)
    a = assign(X, Cc)
    pk = purity(lab, a, 5, 5)
    ok(pk["phone_purity"] > 0.98 and pk["cluster_purity"] > 0.98, f"k-means 五個可分的群：purity {pk['phone_purity']:.3f}/{pk['cluster_purity']:.3f}（{info['iters']} 次迭代）")
    Cc2, info2 = kmeans(X, 5, 3, 2, 50, 1024)
    ok(np.array_equal(Cc, Cc2), "k-means 同 seed 逐位可重現")
    # 3. DCT 正交、Δ 對線性斜坡是常數
    D = dct_matrix(80, 13)
    ok(np.allclose(D @ D.T, np.eye(13), atol=1e-10), "DCT-II 的 13 列正交歸一")
    ramp = np.outer(np.arange(50), np.ones(3)) * 2.0
    dd = deltas(ramp, 2)
    ok(np.allclose(dd[2:-2], 2.0), f"Δ 對斜率 2 的斜坡：中段 = {dd[25, 0]:.3f}")
    # 4. RLE／LCS
    ok(rle([3, 3, 5, 5, 5, 3]) == [[3, 0, 2], [5, 2, 3], [3, 5, 1]], "rle")
    n, ia, ib = lcs([1, 2, 3, 4], [2, 9, 3, 4, 7])
    ok(n == 3 and ia == [False, True, True, True] and ib == [True, False, True, True, False], f"lcs = {n}，標記正確")
    # 5. frames_in 與 Demo 1 的 frame 中心一致
    ok(frames_in(10, 0.0, 0.05).tolist() == [0, 1] and frames_in(10, 0.05, 0.1).tolist() == [2, 3, 4], "frames_in：中心 12.5 + 20t ms")
    # 6. fake 流程
    out = Out()
    d = Demo(out, Fake(), tag="FAKE")
    d.make_fig()
    t0 = time.perf_counter()
    d.run_all()
    print(f"  fake 全流程 {time.perf_counter() - t0:.1f} s")
    P = d.src.purities()
    ok(P["mfcc"]["test"]["pnmi"] < P["l6"]["test"]["pnmi"] <= P["l9"]["test"]["pnmi"] + 0.05, f"fake 純度的次序 mfcc < l6 ≲ l9：{P['mfcc']['test']['pnmi']:.3f} < {P['l6']['test']['pnmi']:.3f} ≲ {P['l9']['test']['pnmi']:.3f}")
    st = d.src.sentence()["stats"]["l6"]
    ok(st["mean_distinct_per_phone"] > 1.2, f"fake 句子：每個音素 {st['mean_distinct_per_phone']:.2f} 個 unit > 1.2")
    ok("word" in d.log["sections"] and d.log["sections"]["word"]["rows"][1]["vs_ref"]["l6"]["lcs"] >= 1, "fake 同一個字：LCS 算得出來")
    rec = d.record()
    json.dumps(rec, default=float)
    d2 = Demo(Out(), Replay(json.loads(json.dumps(rec, default=float))), tag="REPLAY of FAKE")
    d2.make_fig(); d2.run_all()
    ok(d2.src.purities()["l6"]["test"]["pnmi"] == P["l6"]["test"]["pnmi"], "replay 讀回同樣的數字")
    # 7. 彩排紀錄對 w05-metrics.json（有才比）
    rec_p = HERE / "runs" / "rehearsal" / "rehearsal.json"
    if rec_p.exists() and METRICS_JSON.exists():
        r = json.loads(rec_p.read_text(encoding="utf-8"))
        m = json.loads(METRICS_JSON.read_text(encoding="utf-8")).get("demo2", {})
        for f in FEATS:
            if f in m.get("purities", {}):
                ok(abs(round(r["purities"][f]["test"]["pnmi"], 4) - m["purities"][f]["test"]["pnmi"]) < 1e-9, f"[{f}] rehearsal.json 的 PNMI == w05-metrics.json")
    else:
        print("  （沒有彩排紀錄或 w05-metrics.json，跳過比對）")
    # 8. Demo 1 的快取在的話：MFCC frame 快取與它逐格同 frame
    if mfcc_frames_path().exists() and D1.feats_path(MODEL).exists():
        zm, z = np.load(mfcc_frames_path(), allow_pickle=False), np.load(D1.feats_path(MODEL), allow_pickle=False)
        ok(np.array_equal(zm["phone"], z["phone"]) and np.array_equal(zm["utt"], z["utt"]), "MFCC frame 快取與 Demo 1 的 frame 逐格相同")
    print(f"\n  {'全部通過' if not fails else f'{len(fails)} 項失敗'}")
    if fails:
        sys.exit(1)


# ── main ─────────────────────────────────────────────────────
def build(cmd: str, out: Out) -> Demo:
    if cmd == "fake":
        return Demo(out, Fake(), tag="FAKE — synthetic units, not a model")
    if cmd == "replay":
        p = HERE / "runs" / "rehearsal" / "rehearsal.json"
        if not p.exists():
            sys.exit(f"沒有彩排紀錄 {p}。先跑 ./present.sh rehearse。")
        return Demo(out, Replay(json.loads(p.read_text(encoding="utf-8"))), tag="REPLAY — numbers from the rehearsal, not live")
    if not D1.feats_path(MODEL).exists():
        sys.exit(f"沒有 Demo 1 的表徵快取 {D1.feats_path(MODEL)}。先在 ../w05_d1_layer_probe 跑 ./present.sh fetch 與 extract。")
    return Demo(out, Live(out))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["show", "rehearse", "replay", "cluster", "extract", "fetch", "fake", "selftest", "export"])
    ap.add_argument("--device", help="extract：蓋過 Demo 1 demo_config.toml 的 device（cpu／mps／auto）")
    ap.add_argument("--no-plot", action="store_true", help="show：只在終端機，不開圖")
    args = ap.parse_args()

    if args.cmd == "fetch":
        fetch(); return
    if args.cmd == "selftest":
        selftest(); return
    if args.cmd == "export":
        export(HERE / "runs" / "rehearsal" / "rehearsal.json"); return
    if args.cmd == "cluster":
        cluster_all(); print("\n下一步：./present.sh extract（要模型）→ ./present.sh rehearse"); return
    if args.cmd == "extract":
        extract(device=args.device); print("\n下一步：./present.sh rehearse"); return

    if args.cmd == "rehearse" or args.no_plot:
        import matplotlib
        matplotlib.use("Agg")

    out = Out()
    d = build(args.cmd, out)
    d.make_fig()

    if args.cmd == "rehearse":
        d.tag = "rehearsal"
        folder = HERE / "runs" / "rehearsal"
        folder.mkdir(parents=True, exist_ok=True)
        d.run_all(folder)
        d.save(folder)
        export(folder / "rehearsal.json")
        print("\n下一步：./present.sh 自己按一遍 1 2 ] ] f u 3 4；把 1 的九格填進 p45 的表、2 與 3 的數字填進講稿。")
        return

    out.banner("W5 Demo 2：units are not phones" + (f"   [{d.tag}]" if d.tag else ""))
    d.menu()
    if args.no_plot:
        C.note("  --no-plot：鍵在終端機按。")
        while d.dispatch(C.getkey()):
            pass
    else:
        C.note("  鍵在視窗裡按（視窗要有焦點）。終端機這裡是紀錄。")
        d.plt.show()
    if d.log["sections"] and args.cmd not in ("replay", "fake"):
        d.save(HERE / "runs" / "live" / C.now().replace(":", ""))


if __name__ == "__main__":
    main()

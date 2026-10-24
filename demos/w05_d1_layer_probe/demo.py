# -*- coding: utf-8 -*-
# 此檔由 bin/publish_demos.py 從 private repo 的 <course>/demos/ 複製而來，請勿在公開 repo 直接編輯。
"""W5 demo 1：layer-wise probing——HuBERT-base 與 wav2vec 2.0 base 全部 13 層的三條曲線。

  show       課堂：從快取讀表徵，開一個三格的空視窗；鍵在視窗裡按（下面的鍵）。離開時存 runs/live/<時間>/
  rehearse   課前：不互動，兩個模型的三條曲線全部算一遍，存 runs/rehearsal/（進版控 = 現場的退路），並寫 slides/assets/w05/w05-metrics.json
  replay     現場出狀況時：從 runs/rehearsal/ 讀回曲線，同一個視窗、同樣的鍵，不碰 torch、不碰快取
  extract    課前：兩個模型對子集抽表徵（每層平均池化 + 每句 60 個 frame）→ ../_media/w05_d1/feats_<model>.npz（慢的那一步；show 與 rehearse 缺快取時會自動跑）
  check      課前體檢：版本、兩個模型在不在快取、HuBERT 的 output_hidden_states 在 MPS 上跑不跑、T 對 stride、mps 對 cpu → runs/check.json
  fetch      課前一週：下載兩個模型（鎖 commit → model.lock）、dev-clean 子集（與 W3 Demo 3 共用 tar）、音素對齊（要網路）
  fake       沒有模型也能演練流程：合成的表徵（畫面標明 FAKE，不是給學生看的）
  selftest   不開視窗：探針與 CCA 的單元檢查（可分的群、旋轉過的同一組資料、獨立的資料）＋ fake 流程 ＋ 彩排紀錄對 w05-metrics.json

畫面上的鍵（順序照投影片 p37 講稿的【操作】）：
  1  播一句（W3 那句 1272-128104-0000）、印它 13 層的 shape        2  說話人探針曲線（13 個點逐個出現）
  3  音素探針曲線                                                   4  CCA 曲線（對第 0 層）；l 切換成對 log-mel
  w  換模型（hubert ↔ w2v2，同樣三條）                              a  兩個模型並排（都算完才畫）
  t  印摘要表（兩個模型的峰值層、CCA 最低／頂層；p37 的表格）      h  選單    q  離開

數字的定義：第 0 層 = CNN 前端直出（model.feature_extractor，512 維，feature projection 之前）、第 1–12 層 = 每個 Transformer block 的輸出（768 維）。
說話人探針 = frame 級、40 類 logistic regression（句級平均池化的另印，每層都接近 1.0），報測試 frame 的 accuracy；音素探針 = frame 級、39 個 ARPAbet 音素（去重音），報測試 frame 的 accuracy；
CCA = 該層與第 0 層（或 log-mel）在訓練 frame 上做 PCA（保留 99% 變異）→ ridge CCA，測試 frame 上每對 canonical 方向的相關，以 PWCCA 加權平均（Morcos et al. 2018；Pasad et al. 2021 用的度量）。
訓練／測試依句子分（每位說話人最後 3 句測試），探針與 CCA 都只報測試集。
"""
from __future__ import annotations

import argparse
import io
import json
import re
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
FS = CFG["data"]["fs"]
DIG = CFG["display"]["digits"]
NL = CFG["infer"]["n_layers"]
MODELS = CFG["models"]["order"]
DATA = HERE / "data"
MEDIA = HERE.parent / "_media"
MEDIA_D1 = MEDIA / "w05_d1"
MODEL_LOCK = HERE / "model.lock"
METRICS_JSON = HERE.parents[1] / "slides" / "assets" / "w05" / "w05-metrics.json"     # parents[1] = speech_ai/（第一版寫成 [2]，寫到 Course-Hub/slides/ 去了）
W3_UTT = "1272-128104-0000"
# 兩個 repo 都只有 pytorch_model.bin（沒有 safetensors；2026-10-11 對過 HF 的檔案清單），所以要收 *.bin。tf_model.h5 不收。
MODEL_PATTERNS = ["*.json", "*.txt", "*.safetensors", "*.bin"]
# 排除的標籤與 norm_phone 一樣比大寫（對齊檔裡是小寫 spn；第一版沒統一，SPN 混進了 39 個音素裡變成第 40 類）。
EXCLUDE = {p.strip().upper() for p in CFG["phones"]["exclude"]}

ARPABET39 = ["AA", "AE", "AH", "AO", "AW", "AY", "B", "CH", "D", "DH", "EH", "ER", "EY", "F", "G", "HH", "IH", "IY",
             "JH", "K", "L", "M", "N", "NG", "OW", "OY", "P", "R", "S", "SH", "T", "TH", "UH", "UW", "V", "W", "Y", "Z", "ZH"]


# ── 探針與 CCA（純 numpy）──────────────────────────────────────
def standardize(Xtr: np.ndarray, Xte: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mu = Xtr.mean(0, keepdims=True)
    sd = Xtr.std(0, keepdims=True) + 1e-6
    return (Xtr - mu) / sd, (Xte - mu) / sd


def logreg(Xtr: np.ndarray, ytr: np.ndarray, Xte: np.ndarray, yte: np.ndarray, n_classes: int,
           steps: int, lr: float, l2: float, seed: int) -> dict:
    """多類 logistic regression：softmax 交叉熵 + L2，full-batch Adam。回傳訓練／測試 accuracy。"""
    Xtr, Xte = standardize(Xtr.astype(np.float32), Xte.astype(np.float32))
    n, d = Xtr.shape
    rng = np.random.default_rng(seed)
    W = (0.01 * rng.standard_normal((d, n_classes))).astype(np.float32)
    b = np.zeros(n_classes, dtype=np.float32)
    mW, vW, mb, vb = np.zeros_like(W), np.zeros_like(W), np.zeros_like(b), np.zeros_like(b)
    b1, b2, eps = 0.9, 0.999, 1e-8
    Y = np.zeros((n, n_classes), dtype=np.float32)
    Y[np.arange(n), ytr] = 1.0
    for s in range(1, steps + 1):
        S = Xtr @ W + b
        S -= S.max(1, keepdims=True)
        P = np.exp(S)
        P /= P.sum(1, keepdims=True)
        G = (P - Y) / n
        gW = Xtr.T @ G + l2 * W
        gb = G.sum(0)
        for p, g, m, v in ((W, gW, mW, vW), (b, gb, mb, vb)):
            m *= b1; m += (1 - b1) * g
            v *= b2; v += (1 - b2) * g * g
            p -= lr * (m / (1 - b1 ** s)) / (np.sqrt(v / (1 - b2 ** s)) + eps)
    acc_tr = float(np.mean((Xtr @ W + b).argmax(1) == ytr))
    acc_te = float(np.mean((Xte @ W + b).argmax(1) == yte))
    return {"acc_train": acc_tr, "acc_test": acc_te}


def _pca_whiten(Xtr: np.ndarray, keep: float, reg: float):
    """中心化 → PCA 保留 keep 的變異 → 回傳 (平均, 投影矩陣到白化的 PCA 座標)。"""
    mu = Xtr.mean(0, keepdims=True)
    Xc = Xtr - mu
    cov = Xc.T @ Xc / len(Xc)
    w, V = np.linalg.eigh(cov)
    idx = np.argsort(w)[::-1]
    w, V = w[idx], V[:, idx]
    k = int(np.searchsorted(np.cumsum(w) / w.sum(), keep) + 1)
    k = max(1, min(k, len(w)))
    return mu, V[:, :k], w[:k]


def cca_mean(Xtr: np.ndarray, Ytr: np.ndarray, Xte: np.ndarray, Yte: np.ndarray, reg: float, keep: float = 0.99) -> dict:
    """PWCCA：兩邊各做 PCA（保留 keep 的變異）→ ridge CCA 在訓練 frame 上擬合 → 測試 frame 上每對 canonical 方向的 Pearson 相關，投影加權平均（"mean"）；"plain_mean" 是不加權的。"""
    Xtr, Xte, Ytr, Yte = (a.astype(np.float64) for a in (Xtr, Xte, Ytr, Yte))
    mx, Px, _ = _pca_whiten(Xtr, keep, reg)
    my, Py, _ = _pca_whiten(Ytr, keep, reg)
    A, B = (Xtr - mx) @ Px, (Ytr - my) @ Py
    n = len(A)
    Sxx = A.T @ A / n
    Syy = B.T @ B / n
    Sxy = A.T @ B / n
    Sxx += reg * np.trace(Sxx) / len(Sxx) * np.eye(len(Sxx))
    Syy += reg * np.trace(Syy) / len(Syy) * np.eye(len(Syy))

    def inv_sqrt(S):
        w, V = np.linalg.eigh(S)
        return V @ np.diag(1.0 / np.sqrt(np.maximum(w, 1e-12))) @ V.T

    Wx, Wy = inv_sqrt(Sxx), inv_sqrt(Syy)
    U, s, Vt = np.linalg.svd(Wx @ Sxy @ Wy)
    k = min(len(s), A.shape[1], B.shape[1])
    Dx, Dy = Wx @ U[:, :k], Wy @ Vt[:k].T
    Xc = Xte - mx
    Ate, Bte = (Xc @ Px) @ Dx, ((Yte - my) @ Py) @ Dy
    Ate -= Ate.mean(0); Bte -= Bte.mean(0)
    num = (Ate * Bte).sum(0)
    den = np.sqrt((Ate ** 2).sum(0) * (Bte ** 2).sum(0)) + 1e-12
    rho = num / den
    # PWCCA（Morcos et al. 2018）：第 i 對 canonical 方向的權重 = 它與 X 每一維活化的 |協方差| 總和——解釋 X 越多的方向權重越大，
    # 純雜訊的方向（相關接近 0）幾乎不佔權重。單純平均會被幾百對相關為 0 的方向稀釋，PWCCA 是 Pasad et al. 2021 用的度量。
    alpha = np.abs(Ate.T @ (Xc - Xc.mean(0))).sum(1)
    alpha /= alpha.sum() + 1e-12
    return {"mean": float((alpha * rho).sum()), "plain_mean": float(rho.mean()), "n_components": int(k), "k_x": int(Px.shape[1]), "k_y": int(Py.shape[1]),
            "train_mean": float(s[:k].mean())}


# ── 資料：子集、對齊、log-mel ───────────────────────────────────
def norm_phone(p: str) -> str:
    return re.sub(r"\d+$", "", p.strip()).upper() if CFG["phones"]["strip_stress"] else p.strip().upper()


def load_subset() -> dict:
    p = DATA / "subset.json"
    if not p.exists():
        sys.exit(f"找不到 {p}。先 ./present.sh fetch。")
    return json.loads(p.read_text(encoding="utf-8"))


def load_alignments() -> dict[str, list]:
    p = DATA / "alignments.json"
    if not p.exists():
        sys.exit(f"找不到 {p}。先 ./present.sh fetch。")
    return json.loads(p.read_text(encoding="utf-8"))


def frame_centers_s(T: int) -> np.ndarray:
    return (np.arange(T) * CFG["infer"]["frame_ms"] + CFG["infer"]["receptive_ms"] / 2) / 1000.0


def frame_labels(T: int, phones: list, phone_index: dict[str, int]) -> np.ndarray:
    """每個 frame 的音素 id（-1 = 靜音／排除的標籤）。frame 中心落在哪個區間就是哪個。"""
    lab = np.full(T, -1, dtype=np.int64)
    c = frame_centers_s(T)
    excl = EXCLUDE
    for ph, s, e in phones:
        q = norm_phone(ph)
        if q in excl or q not in phone_index:
            continue
        lab[(c >= s) & (c < e)] = phone_index[q]
    return lab


def logmel(x: np.ndarray) -> np.ndarray:
    """(frames, n_mels) 的 log-mel；25 ms Hann、20 ms hop、512 點、Slaney mel。frame m 的中心是 m·hop。"""
    L = int(round(CFG["logmel"]["win_ms"] * FS / 1000))
    H = int(round(CFG["logmel"]["hop_ms"] * FS / 1000))
    N = CFG["logmel"]["n_fft"]
    X = dsp.stft(x, L, H, N)
    M = dsp.mel_filterbank(FS, N, CFG["logmel"]["n_mels"], CFG["logmel"]["scale"])
    return np.log(M @ (np.abs(X) ** 2) + 1e-10).T


def wav_path(utt: str) -> Path:
    return MEDIA_D1 / "wav" / f"{utt}.wav"


# ── 模型 ─────────────────────────────────────────────────────
def model_dir(key: str, offline: bool = True) -> str:
    from huggingface_hub import snapshot_download
    repo = CFG["models"][key]["repo"]
    rev = read_lock().get(repo)
    return snapshot_download(repo, revision=rev, local_files_only=offline, allow_patterns=MODEL_PATTERNS)


def read_lock() -> dict[str, str]:
    if not MODEL_LOCK.exists():
        return {}
    out = {}
    for line in MODEL_LOCK.read_text(encoding="utf-8").splitlines():
        if line.strip() and not line.startswith("#"):
            repo, sha = line.split()
            out[repo] = sha
    return out


class Encoder:
    """transformers 的 HubertModel／Wav2Vec2Model（沒有 CTC head、沒有 quantizer）。回傳 numpy：第 0 層 = CNN 前端直出 (T, 512)、第 1–12 層 (12, T, 768)。"""

    def __init__(self, key: str, device: str):
        import torch
        from transformers import HubertModel, Wav2Vec2FeatureExtractor, Wav2Vec2Model
        self.torch = torch
        self.key = key
        path = model_dir(key)
        self.fe = Wav2Vec2FeatureExtractor.from_pretrained(path)
        cls = HubertModel if "hubert" in CFG["models"][key]["repo"] else Wav2Vec2Model
        self.model = cls.from_pretrained(path).eval()
        if device == "auto":
            device = "mps" if torch.backends.mps.is_available() else "cpu"
        self.device = torch.device(device)
        self.model.to(self.device)
        self.n_params = sum(p.numel() for p in self.model.parameters())
        self.do_normalize = bool(getattr(self.fe, "do_normalize", False))

    def layers(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        torch = self.torch
        inp = self.fe(x.astype(np.float32), sampling_rate=FS, return_tensors="pt")
        iv = inp.input_values.to(self.device)
        with torch.no_grad():
            out = self.model(iv, output_hidden_states=True)
            # 第 0 層：直接叫 CNN 前端（兩個模型都叫 feature_extractor，輸出 (B, 512, T)），不靠回傳物件——
            # HubertModel 回的是 BaseModelOutput，沒有 extract_features（Wav2Vec2Model 才有）；這樣兩個模型的第 0 層
            # 定義也一致：都是 CNN 直出、feature projection 之前。
            cnn = self.model.feature_extractor(iv).transpose(1, 2)
        if self.device.type == "mps":
            torch.mps.synchronize()
        l0 = cnn[0].float().cpu().numpy()
        hs = out.hidden_states
        assert len(hs) == NL, f"hidden_states 有 {len(hs)} 個，預期 {NL}"
        rest = np.stack([h[0].float().cpu().numpy() for h in hs[1:]])
        return l0, rest


# ── 抽表徵 ───────────────────────────────────────────────────
def feats_path(key: str) -> Path:
    return MEDIA_D1 / f"feats_{key}.npz"


def extract(key: str, out=None, device: str | None = None) -> dict:
    o = out or print
    sub = load_subset()
    ali = load_alignments()
    utts = sub["utts"]
    phone_index = {p: i for i, p in enumerate(sub["phones"])}
    spk_index = {s: i for i, s in enumerate(sub["speakers"])}
    rng = np.random.default_rng(CFG["frames"]["seed"])
    per = CFG["frames"]["per_utt"]
    C.note(f"載入 {CFG['models'][key]['repo']}（{device or CFG['infer']['device']}）……")
    enc = Encoder(key, device or CFG["infer"]["device"])
    C.note(f"  {enc.n_params / 1e6:.1f} M 參數，device = {enc.device}，do_normalize = {enc.do_normalize}")
    pooled = [[] for _ in range(NL)]
    frames = [[] for _ in range(NL)]
    f_phone, f_utt, f_mel, utt_T, utt_spk, utt_split, ms = [], [], [], [], [], [], []
    t_all = time.perf_counter()
    for i, u in enumerate(utts):
        x = C.load_wav(wav_path(u["id"]), FS)
        t0 = time.perf_counter()
        l0, rest = enc.layers(x)
        ms.append(1000 * (time.perf_counter() - t0))
        T = l0.shape[0]
        lab = frame_labels(T, ali[u["id"]], phone_index)
        mel = logmel(x)
        Tm = min(T, mel.shape[0])
        lab[Tm:] = -1
        cand = np.flatnonzero(lab >= 0)
        pick = np.sort(rng.choice(cand, size=min(per, len(cand)), replace=False)) if len(cand) else cand
        pooled[0].append(l0.mean(0)); frames[0].append(l0[pick])
        for l in range(1, NL):
            pooled[l].append(rest[l - 1].mean(0)); frames[l].append(rest[l - 1][pick])
        f_phone.append(lab[pick]); f_utt.append(np.full(len(pick), i)); f_mel.append(mel[pick])
        utt_T.append(T); utt_spk.append(spk_index[u["spk"]]); utt_split.append(u["split"])
        if (i + 1) % 50 == 0 or i == len(utts) - 1:
            o(f"  {i + 1:4d}/{len(utts)}  {u['id']:<18s} T = {T:4d}  {ms[-1]:6.1f} ms  frames picked {len(pick)}")
    total = time.perf_counter() - t_all
    MEDIA_D1.mkdir(parents=True, exist_ok=True)
    arrays = {f"pooled_{l}": np.stack(pooled[l]).astype(np.float32) for l in range(NL)}
    arrays.update({f"frames_{l}": np.concatenate(frames[l]).astype(np.float16) for l in range(NL)})
    meta = {"model": key, "repo": CFG["models"][key]["repo"], "commit": read_lock().get(CFG["models"][key]["repo"]),
            "device": str(enc.device), "n_params": enc.n_params, "do_normalize": enc.do_normalize,
            "n_utts": len(utts), "n_frames": int(sum(len(p) for p in f_phone)), "ms_per_utt_median": float(np.median(ms)),
            "seconds_total": total, "time": C.now(), "versions": versions()}
    np.savez(feats_path(key), phone=np.concatenate(f_phone), utt=np.concatenate(f_utt), logmel=np.concatenate(f_mel).astype(np.float32),
             utt_T=np.array(utt_T), utt_spk=np.array(utt_spk), utt_split=np.array(utt_split),
             meta=json.dumps(meta, ensure_ascii=False), **arrays)
    o(f"  → {feats_path(key)}（{feats_path(key).stat().st_size / 1e6:.0f} MB；{total:.0f} s，每句中位數 {np.median(ms):.1f} ms）")
    return meta


def versions() -> dict:
    v = C.versions()
    from importlib import metadata
    for pkg in ("torch", "transformers", "huggingface_hub", "soundfile", "pyarrow"):
        try:
            v[pkg] = metadata.version(pkg)
        except metadata.PackageNotFoundError:
            v[pkg] = None
    return v


class Feats:
    """一個模型的快取。frames_l (Nf, d_l) float16、pooled_l (n_utt, d_l)、phone (Nf,)、utt (Nf,)、logmel (Nf, 80)。"""

    def __init__(self, key: str, z=None, tag: str = ""):
        self.key, self.tag = key, tag
        self.z = z if z is not None else np.load(feats_path(key), allow_pickle=False)
        self.meta = json.loads(str(self.z["meta"]))
        self.phone, self.utt, self.mel = self.z["phone"], self.z["utt"], self.z["logmel"]
        self.utt_T, self.utt_spk, self.utt_split = self.z["utt_T"], self.z["utt_spk"], self.z["utt_split"]
        self.tr_u = self.utt_split == "train"
        self.te_u = ~self.tr_u
        self.tr_f, self.te_f = self.tr_u[self.utt], self.te_u[self.utt]

    def frames(self, l: int) -> np.ndarray:
        return self.z[f"frames_{l}"]

    def pooled(self, l: int) -> np.ndarray:
        return self.z[f"pooled_{l}"]

    @property
    def label(self) -> str:
        return CFG["models"][self.key]["label"] + (f" [{self.tag}]" if self.tag else "")


# ── 曲線 ─────────────────────────────────────────────────────
def speaker_curve(F: Feats, l: int, n_spk: int) -> dict:
    """說話人探針：frame 級（與音素探針同一組 frame，每個 frame 自己認說話人）是畫出來的那條；句級平均池化的也算、另印。
    句級的在 40 位說話人、276 句訓練下每一層都接近 1.0（M5 Max 第一次彩排：0.96–1.00，看不出形狀），所以曲線用 frame 級。"""
    P = CFG["probe"]
    X = F.frames(l)
    y = F.utt_spk[F.utt]
    r = logreg(X[F.tr_f], y[F.tr_f], X[F.te_f], y[F.te_f], n_spk, P["steps"], P["lr"], P["l2"], CFG["frames"]["seed"])
    ru = logreg(F.pooled(l)[F.tr_u], F.utt_spk[F.tr_u], F.pooled(l)[F.te_u], F.utt_spk[F.te_u], n_spk,
                P["steps"], P["lr"], P["l2"], CFG["frames"]["seed"])
    r["acc_test_utt"], r["acc_train_utt"] = ru["acc_test"], ru["acc_train"]
    return r


def phone_curve(F: Feats, l: int, n_ph: int) -> dict:
    P = CFG["probe"]
    X = F.frames(l)
    return logreg(X[F.tr_f], F.phone[F.tr_f], X[F.te_f], F.phone[F.te_f], n_ph, P["steps"], P["lr"], P["l2"], CFG["frames"]["seed"])


def cca_curve(F: Feats, l: int, target: str) -> dict:
    X = F.frames(l)
    Y = F.frames(0) if target == "layer0" else F.mel
    return cca_mean(X[F.tr_f], Y[F.tr_f], X[F.te_f], Y[F.te_f], CFG["probe"]["cca_reg"])


def summarize(c: dict) -> dict:
    """p37 表格的四格：說話人峰值層、音素峰值層、CCA（對第 0 層）最低／頂層。"""
    spk = [r["acc_test"] for r in c["speaker"]]
    ph = [r["acc_test"] for r in c["phone"]]
    cc = [r["mean"] for r in c["cca_layer0"]]
    low = int(np.argmax(spk[:7]))                                           # 低層（0–6）的峰：說話人探針的「低層最好」看這個；整條的 argmax 可能在頂層（折回）
    out = {"speaker_peak_layer": int(np.argmax(spk)), "speaker_peak_acc": float(max(spk)),
           "speaker_low_peak_layer": low, "speaker_low_peak_acc": float(spk[low]),
           "speaker_min_layer": int(np.argmin(spk)), "speaker_min_acc": float(min(spk)), "speaker_top_acc": float(spk[-1]),
           "phone_peak_layer": int(np.argmax(ph)), "phone_peak_acc": float(max(ph)),
           "cca0_min_layer": int(np.argmin(cc)), "cca0_min": float(min(cc)), "cca0_top": float(cc[-1]),
           "cca0_turnback": float(cc[-1] - min(cc)),
           "speaker_layer0_minus_top": float(spk[0] - spk[-1]), "phone_top_minus_peak": float(ph[-1] - max(ph))}
    if "cca_logmel" in c and c["cca_logmel"]:
        cm = [r["mean"] for r in c["cca_logmel"]]
        out.update({"ccamel_min_layer": int(np.argmin(cm)), "ccamel_min": float(min(cm)), "ccamel_top": float(cm[-1])})
    return out


# ── 終端機輸出（同時存成純文字，rehearse 用）──────────────────
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


# ── 主體 ─────────────────────────────────────────────────────
COLORS = {"hubert": "#5b3fa0", "w2v2": "#c8742a"}
STYLE = {"hubert": "-", "w2v2": "--"}


class Demo:
    def __init__(self, out: Out, feats: dict[str, Feats], subset: dict, tag: str = "", recorded: dict | None = None):
        self.out, self.feats, self.subset, self.tag = out, feats, subset, tag
        self.n_spk, self.n_ph = len(subset["speakers"]), len(subset["phones"])
        self.cur = MODELS[0]
        self.cca_target = "layer0"
        self.curves: dict[str, dict] = {k: {} for k in MODELS}     # curves[model][kind] = list of 13 dicts
        if recorded:
            for k in MODELS:
                if k in recorded:
                    self.curves[k] = recorded[k]
        self.log: dict = {"time": C.now(), "versions": versions(), "tag": tag, "subset": {kk: subset[kk] for kk in ("n_speakers", "n_utts", "n_train", "n_test", "phones")},
                          "meta": {k: f.meta for k, f in feats.items()}, "curves": self.curves, "sections": {}}
        self.fig = self.axes = self.plt = None
        self.lines: dict[tuple, object] = {}

    # ── 視窗 ──
    def make_fig(self):
        import matplotlib.pyplot as plt
        self.plt = plt
        self.fig, self.axes = plt.subplots(1, 3, figsize=(CFG["display"]["fig_w"], CFG["display"]["fig_h"]))
        titles = ["speaker probe (frame-level test accuracy, 40 speakers)", "phone probe (test accuracy, 39 phones)", "PWCCA with layer 0 (held-out)"]
        for ax, t in zip(self.axes, titles):
            ax.set_title(t, fontsize=11)
            ax.set_xlabel("layer  (0 = CNN output, 1–12 = transformer blocks)")
            ax.set_xticks(range(NL))
            ax.set_xlim(-0.5, NL - 0.5)
            ax.grid(alpha=0.3)
        self.axes[0].set_ylim(0, 1.02); self.axes[1].set_ylim(0, 1.02); self.axes[2].set_ylim(0, 1.02)
        self.fig.suptitle("W5 Demo 1 — three curves across thirteen layers" + (f"   [{self.tag}]" if self.tag else ""), fontsize=13)
        self.fig.tight_layout()
        self.fig.canvas.mpl_connect("key_press_event", self.on_key)
        return self.fig

    def redraw(self):
        if self.fig is not None:
            self.fig.canvas.draw_idle()
            self.plt.pause(0.05)

    def plot_curve(self, key: str, kind: str, vals: list[float], ax_i: int, label_suffix: str = ""):
        if self.fig is None:
            return
        ax = self.axes[ax_i]
        tag = (key, kind)
        if tag in self.lines:
            ln = self.lines[tag]
            ln.set_data(range(len(vals)), vals)
        else:
            (ln,) = ax.plot(range(len(vals)), vals, STYLE[key], marker="o", color=COLORS[key], lw=2,
                            label=CFG["models"][key]["label"] + label_suffix)
            self.lines[tag] = ln
            ax.legend(loc="lower left", fontsize=9)
        self.redraw()

    # ── 鍵 ──
    def play(self) -> None:
        o = self.out
        utt = W3_UTT if W3_UTT in {u["id"] for u in self.subset["utts"]} else self.subset["utts"][0]["id"]
        i = next(j for j, u in enumerate(self.subset["utts"]) if u["id"] == utt)
        F = self.feats[self.cur]
        T = int(F.utt_T[i])
        o.banner(f"1  {utt}（{self.subset['utts'][i]['dur_s']:.2f} s）— {F.label}")
        p = wav_path(utt)
        if self.tag.startswith("FAKE"):
            o("  （FAKE：沒有音訊可播）")
        elif p.exists():
            threading.Thread(target=C.play, args=(p,), daemon=True).start()     # 背景播，不要卡住視窗的事件迴圈
        else:
            o(f"  找不到 {p}（replay 不需要音訊；要播就先 fetch）")
        o(f"  T = {T} frames（20 ms 一格；(L − 400)/320 + 1 = {(int(self.subset['utts'][i]['dur_s'] * FS) - 400) // 320 + 1}）")
        o(f"  layer  0  (CNN output, before projection)      {T} × {F.frames(0).shape[1]}")
        for l in range(1, NL):
            o(f"  layer {l:2d}  (transformer block {l:2d} output)        {T} × {F.frames(l).shape[1]}")
        o(f"  抽表徵：每句中位數 {F.meta['ms_per_utt_median']:.1f} ms（{F.meta['device']}）；子集 {F.meta['n_utts']} 句、{F.meta['n_frames']} 個標了音素的 frame")
        self.log["sections"]["play"] = {"utt": utt, "T": T}

    def _ensure(self, key: str, kind: str, compute, ax_i: int, fmt, title: str) -> list:
        """算一條曲線（沒算過才算），一層一列印、圖上逐點加。"""
        o = self.out
        o.banner(f"{title} — {self.feats[key].label}")
        rows = self.curves[key].get(kind)
        if rows and len(rows) == NL:
            o("  （已算過，重畫）")
            for l, r in enumerate(rows):
                o(f"  layer {l:2d}  {fmt(r)}")
            self.plot_curve(key, kind, [self._val(kind, r) for r in rows], ax_i)
            return rows
        rows = []
        t0 = time.perf_counter()
        for l in range(NL):
            t1 = time.perf_counter()
            r = compute(l)
            r["seconds"] = time.perf_counter() - t1
            rows.append(r)
            o(f"  layer {l:2d}  {fmt(r)}   {r['seconds']:4.1f} s")
            self.plot_curve(key, kind, [self._val(kind, x) for x in rows], ax_i)
        self.curves[key][kind] = rows
        o(f"  共 {time.perf_counter() - t0:.0f} s")
        return rows

    @staticmethod
    def _val(kind: str, r: dict) -> float:
        return r["mean"] if kind.startswith("cca") else r["acc_test"]

    def speaker(self, key: str | None = None) -> None:
        key = key or self.cur
        F = self.feats[key]
        rows = self._ensure(key, "speaker", lambda l: speaker_curve(F, l, self.n_spk), 0,
                            lambda r: f"frame test acc {r['acc_test']:.{DIG}f}   (train {r['acc_train']:.{DIG}f};  utterance-level {r.get('acc_test_utt', float('nan')):.{DIG}f})", "2  speaker probe")
        acc = [r["acc_test"] for r in rows]
        self.out(f"  {C.BOLD}低層峰值 layer {int(np.argmax(acc[:7]))}（{max(acc[:7]):.{DIG}f}）；最低 layer {int(np.argmin(acc))}（{min(acc):.{DIG}f}）；layer 12 {acc[-1]:.{DIG}f}{C.RESET}")

    def phone(self, key: str | None = None) -> None:
        key = key or self.cur
        F = self.feats[key]
        rows = self._ensure(key, "phone", lambda l: phone_curve(F, l, self.n_ph), 1,
                            lambda r: f"test acc {r['acc_test']:.{DIG}f}   (train {r['acc_train']:.{DIG}f})", "3  phone probe")
        acc = [r["acc_test"] for r in rows]
        self.out(f"  {C.BOLD}峰值在 layer {int(np.argmax(acc))}（{max(acc):.{DIG}f}）；layer 12 {acc[-1]:.{DIG}f}，比峰值低 {max(acc) - acc[-1]:.{DIG}f}{C.RESET}")

    def cca(self, key: str | None = None, target: str | None = None) -> None:
        key = key or self.cur
        target = target or self.cca_target
        F = self.feats[key]
        kind = "cca_layer0" if target == "layer0" else "cca_logmel"
        name = "layer 0 (CNN output)" if target == "layer0" else "log-mel (80 bands)"
        if self.fig is not None:
            self.axes[2].set_title(f"PWCCA with {name} (held-out)", fontsize=11)
        rows = self._ensure(key, kind, lambda l: cca_curve(F, l, target), 2,
                            lambda r: f"PWCCA {r['mean']:.{DIG}f}   (plain mean {r['plain_mean']:.{DIG}f}; PCA dims {r['k_x']} × {r['k_y']})", f"4  PWCCA with {name}")
        v = [r["mean"] for r in rows]
        self.out(f"  {C.BOLD}最低在 layer {int(np.argmin(v))}（{min(v):.{DIG}f}）；layer 12 {v[-1]:.{DIG}f}，回升 {v[-1] - min(v):+.{DIG}f}{C.RESET}")

    def toggle_target(self) -> None:
        self.cca_target = "logmel" if self.cca_target == "layer0" else "layer0"
        self.out(f"  CCA 的對象 → {'log-mel' if self.cca_target == 'logmel' else 'layer 0'}（再按 4）")

    def switch(self) -> None:
        i = MODELS.index(self.cur)
        self.cur = MODELS[(i + 1) % len(MODELS)]
        self.out(f"  模型 → {self.feats[self.cur].label}（按 2 3 4 畫它的三條）")

    def both(self) -> None:
        for k in MODELS:
            self.speaker(k); self.phone(k); self.cca(k, "layer0")
        self.table()

    def table(self) -> None:
        o = self.out
        o.banner("t  摘要（p37 的表格）")
        o(f"  {'':<18s} {'speaker: low peak · min · top':>38s} {'phone peak':>14s} {'CCA0 min / top':>18s}")
        res = {}
        for k in MODELS:
            c = self.curves[k]
            if not all(c.get(x) for x in ("speaker", "phone", "cca_layer0")):
                o(f"  {CFG['models'][k]['label']:<18s}  （還沒算完三條）")
                continue
            s = summarize(c)
            res[k] = s
            c1 = f"l{s['speaker_low_peak_layer']} ({s['speaker_low_peak_acc']:.2f}) · min l{s['speaker_min_layer']} ({s['speaker_min_acc']:.2f}) · top {s['speaker_top_acc']:.2f}"
            c2 = f"layer {s['phone_peak_layer']} ({s['phone_peak_acc']:.2f})"
            c3 = f"{s['cca0_min']:.2f} (l{s['cca0_min_layer']}) / {s['cca0_top']:.2f}"
            o(f"  {CFG['models'][k]['label']:<18s} {c1:>38s} {c2:>14s} {c3:>18s}")
        self.log["sections"]["table"] = res

    def menu(self) -> None:
        o = self.out
        o(f"\n{C.BOLD}p37{C.RESET}  1 播一句＋13 層 shape   2 說話人探針   3 音素探針   4 CCA（l 切換 layer 0 ↔ log-mel）")
        o(f"     w 換模型（現在 {CFG['models'][self.cur]['label']}）   a 兩個模型並排   t 摘要表   h 選單   q 離開")

    def dispatch(self, k: str) -> bool:
        if k == "1":
            self.play()
        elif k == "2":
            self.speaker()
        elif k == "3":
            self.phone()
        elif k == "4":
            self.cca()
        elif k == "l":
            self.toggle_target()
        elif k == "w":
            self.switch()
        elif k == "a":
            self.both()
        elif k == "t":
            self.table()
        elif k == "h":
            self.menu()
        elif k == "q":
            return False
        return True

    def on_key(self, ev) -> None:
        if not self.dispatch(ev.key):
            self.plt.close(self.fig)

    def run_all(self) -> None:
        """課堂順序全部跑一遍（rehearse 用；不播音訊）：兩個模型的三條（CCA 對第 0 層與對 log-mel 都算）。"""
        self.play()
        for k in MODELS:
            self.speaker(k); self.phone(k); self.cca(k, "layer0"); self.cca(k, "logmel")
        self.table()

    def save(self, folder: Path) -> None:
        folder.mkdir(parents=True, exist_ok=True)
        stem = "rehearsal" if folder.name == "rehearsal" else "session"
        (folder / f"{stem}.json").write_text(json.dumps(self.log, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
        (folder / f"{stem}.txt").write_text(self.out.text(), encoding="utf-8")
        if self.fig is not None:
            self.fig.savefig(folder / "curves.png", dpi=130)
        C.note(f"  存到 {folder}/（{stem}.json、{stem}.txt、curves.png）")


# ── fake ─────────────────────────────────────────────────────
def fake_feats(key: str, seed: int = 7) -> tuple[Feats, dict]:
    """合成的表徵：說話人成分隨層遞減、音素成分在中層最強、與第 0 層共用的「輸入」成分 U 形。只用來演練流程。"""
    rng = np.random.default_rng(seed + MODELS.index(key))
    n_spk, per_spk, per_utt = 40, 10, 20                      # fake 每句 20 個 frame（真的是 60），selftest 才跑得快
    phones = ARPABET39
    n_ph, d0, d = len(phones), 512, 768
    spk_emb = rng.standard_normal((n_spk, 16))
    ph_emb = rng.standard_normal((n_ph, 16))
    utts, utt_spk, utt_split, utt_T = [], [], [], []
    for s in range(n_spk):
        for j in range(per_spk):
            utts.append({"id": f"fake-{s:02d}-{j:02d}", "spk": f"fake{s:02d}", "dur_s": 5.0, "split": "train" if j < per_spk - CFG["data"]["test_per_speaker"] else "test"})
            utt_spk.append(s); utt_split.append(utts[-1]["split"]); utt_T.append(250)
    n_utt = len(utts)
    Nf = n_utt * per_utt
    phone = rng.integers(0, n_ph, Nf)
    utt = np.repeat(np.arange(n_utt), per_utt)
    inp = rng.standard_normal((Nf, 32))                    # 「輸入」的局部成分（第 0 層 = 它）
    lat_s = spk_emb[np.array(utt_spk)[utt]]
    lat_p = ph_emb[phone]
    l = np.arange(NL)
    w_s = 1.2 * np.exp(-l / 4.0)                                     # 說話人：遞減
    w_p = 0.04 + 0.16 * np.exp(-((l - 6.5) / 3.0) ** 2)             # 音素：中層最強（弱到探針不會每層都滿分）
    w_i = 0.3 + 1.5 * ((l - 7) / 7.0) ** 2                           # 輸入：U 形（頂層折回）
    if key == "hubert":
        w_p = 0.04 + 0.16 * np.exp(-((l - 8.5) / 3.0) ** 2); w_i = 0.3 + 1.0 * ((l - 7) / 7.0) ** 2
    arrays = {}
    pooled = []
    for li in range(NL):
        dl = d0 if li == 0 else d
        P = rng.standard_normal((64, dl)) / np.sqrt(64)
        ws, wp = (0.2, 0.04) if li == 0 else (w_s[li], w_p[li])        # 第 0 層：以「輸入」成分為主，說話人／音素只有一點
        lat = np.concatenate([ws * lat_s, wp * lat_p, w_i[li] * inp], axis=1)
        X = lat @ P + 0.3 * rng.standard_normal((Nf, dl))
        arrays[f"frames_{li}"] = X.astype(np.float16)
        pooled_l = np.stack([X[utt == i].mean(0) for i in range(n_utt)])
        arrays[f"pooled_{li}"] = pooled_l.astype(np.float32)
    mel = (inp @ rng.standard_normal((32, 80)) + 0.5 * rng.standard_normal((Nf, 80))).astype(np.float32)
    meta = {"model": key, "repo": "FAKE", "commit": None, "device": "fake", "n_params": 0, "do_normalize": None, "n_utts": n_utt,
            "n_frames": Nf, "ms_per_utt_median": 0.0, "seconds_total": 0.0, "time": C.now(), "versions": versions()}
    z = dict(phone=phone, utt=utt, logmel=mel, utt_T=np.array(utt_T), utt_spk=np.array(utt_spk), utt_split=np.array(utt_split),
             meta=np.array(json.dumps(meta)), **arrays)
    subset = {"speakers": [f"fake{s:02d}" for s in range(n_spk)], "phones": phones, "utts": utts, "n_speakers": n_spk, "n_utts": n_utt,
              "n_train": int(sum(u["split"] == "train" for u in utts)), "n_test": int(sum(u["split"] == "test" for u in utts))}
    return Feats(key, z, tag="FAKE"), subset


# ── 載入：真的、fake、replay ──────────────────────────────────
def build(cmd: str, out: Out, auto_extract: bool = True) -> Demo:
    if cmd == "fake":
        feats, subset = {}, None
        for k in MODELS:
            feats[k], subset = fake_feats(k)
        return Demo(out, feats, subset, tag="FAKE — synthetic features, not a model")
    if cmd == "replay":
        folder = HERE / "runs" / "rehearsal"
        if not (folder / "rehearsal.json").exists():
            sys.exit(f"沒有彩排紀錄 {folder}/rehearsal.json。先跑 ./present.sh rehearse。")
        rec = json.loads((folder / "rehearsal.json").read_text(encoding="utf-8"))
        if (DATA / "subset.json").exists():
            subset = load_subset()
        else:
            subset = dict(rec["subset"])
            subset["speakers"] = [None] * subset["n_speakers"]
            subset["utts"] = [{"id": rec["sections"].get("play", {}).get("utt", W3_UTT), "dur_s": 0.0}]
        feats = {k: ReplayFeats(k, rec["meta"][k], rec["sections"].get("play", {})) for k in MODELS}
        return Demo(out, feats, subset, tag="REPLAY — curves computed in rehearsal, not live", recorded=rec["curves"])
    subset = load_subset()
    feats = {}
    for k in MODELS:
        if not feats_path(k).exists():
            if not auto_extract:
                sys.exit(f"沒有快取 {feats_path(k)}。先跑 ./present.sh extract。")
            C.note(f"沒有快取 {feats_path(k)}，現在抽（{CFG['models'][k]['repo']}，{subset['n_utts']} 句）……")
            extract(k, out)
        feats[k] = Feats(k)
    return Demo(out, feats, subset)


class ReplayFeats:
    """replay 用：沒有陣列，只有 meta 與 shape。"""

    def __init__(self, key: str, meta: dict, play: dict):
        self.key, self.tag, self.meta = key, "replay", meta
        self.utt_T = np.array([int(play.get("T", 0))])

    def frames(self, l: int):
        from types import SimpleNamespace
        return SimpleNamespace(shape=(0, 512 if l == 0 else 768))

    @property
    def label(self) -> str:
        return CFG["models"][self.key]["label"] + " [replay]"


# ── check ────────────────────────────────────────────────────
def check() -> None:
    C.banner("課前體檢", C.AMBER)
    res = {"time": C.now(), "versions": versions()}
    for k, v in res["versions"].items():
        print(f"  {k:<16s} {v}")
    try:
        import torch
    except ImportError:
        sys.exit("  沒有 torch：先跑 ./setup.sh")
    mps = bool(torch.backends.mps.is_available())
    print(f"  {'mps':<16s} built={torch.backends.mps.is_built()} available={mps}")
    res["mps_available"] = mps
    lock = read_lock()
    sub = load_subset() if (DATA / "subset.json").exists() else None
    utt = W3_UTT if sub and W3_UTT in {u["id"] for u in sub["utts"]} else (sub["utts"][0]["id"] if sub else None)
    x = C.load_wav(wav_path(utt), FS) if utt and wav_path(utt).exists() else None
    if x is None:
        print("  沒有子集音訊，先跑 ./present.sh fetch；下面只檢查模型載入")
    res["models"] = {}
    for k in MODELS:
        repo = CFG["models"][k]["repo"]
        r = {"repo": repo, "commit": lock.get(repo)}
        print(f"\n  {CFG['models'][k]['label']}  {repo} @ {str(lock.get(repo))[:12]}")
        try:
            path = model_dir(k)
            r["path"] = path
        except Exception as e:
            print(f"    不在快取：{e}\n    先跑 ./present.sh fetch")
            r["error"] = str(e)
            res["models"][k] = r
            continue
        outs = {}
        for dev in ["cpu"] + (["mps"] if mps else []):
            try:
                enc = Encoder(k, dev)
                r["n_params"] = enc.n_params
                r["do_normalize"] = enc.do_normalize
                if x is not None:
                    enc.layers(x[:FS])                                   # 暖機
                    t0 = time.perf_counter()
                    l0, rest = enc.layers(x)
                    ms = 1000 * (time.perf_counter() - t0)
                    outs[dev] = (l0, rest)
                    r[f"infer_{dev}"] = {"ms": round(ms, 1), "T": int(l0.shape[0]), "d0": int(l0.shape[1]), "d": int(rest.shape[2]), "n_hidden": int(rest.shape[0]) + 1}
                    print(f"    {dev}: {ms:7.1f} ms   layer 0 {l0.shape}   layers 1–12 {rest.shape}   do_normalize={enc.do_normalize}   {enc.n_params / 1e6:.1f} M")
                else:
                    print(f"    {dev}: 載入 OK，{enc.n_params / 1e6:.1f} M 參數")
            except Exception as e:
                r[f"infer_{dev}"] = {"error": str(e)}
                print(f"    {dev}: {C.RED}失敗{C.RESET} {e}")
        if x is not None and "cpu" in outs:
            T = outs["cpu"][0].shape[0]
            r["T_predicted_from_stride"] = (len(x) - 400) // 320 + 1
            print(f"    T = {T}，(L − 400)/320 + 1 = {r['T_predicted_from_stride']}")
            if "mps" in outs:
                d0 = float(np.max(np.abs(outs["mps"][0] - outs["cpu"][0])))
                d1 = float(np.max(np.abs(outs["mps"][1] - outs["cpu"][1])))
                r["mps_vs_cpu_maxabs"] = {"layer0": d0, "layers": d1}
                print(f"    mps 對 cpu 最大差：layer 0 {d0:.2e}、layers 1–12 {d1:.2e}")
        res["models"][k] = r
    (HERE / "runs").mkdir(exist_ok=True)
    (HERE / "runs" / "check.json").write_text(json.dumps(res, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    print(f"\n  → runs/check.json\n下一步：./present.sh extract（抽表徵，約幾分鐘）→ ./present.sh rehearse")


# ── fetch ────────────────────────────────────────────────────
def fetch() -> None:
    import tarfile
    import urllib.request
    import soundfile as sf
    from huggingface_hub import HfApi, snapshot_download
    C.banner("下載兩個模型、dev-clean 子集、音素對齊", C.AMBER)
    lock = read_lock()
    for k in MODELS:
        repo = CFG["models"][k]["repo"]
        sha = lock.get(repo) or HfApi().model_info(repo).sha
        path = snapshot_download(repo, revision=sha, allow_patterns=MODEL_PATTERNS)
        lock[repo] = sha
        print(f"  {repo} @ {sha[:12]} → {path}")
    MODEL_LOCK.write_text("".join(f"{r} {s}\n" for r, s in lock.items()), encoding="utf-8")
    print("  commit 寫進 model.lock（進版控）")

    D = CFG["data"]
    tgz = MEDIA / Path(D["librispeech_url"]).name
    MEDIA.mkdir(exist_ok=True)
    if not tgz.exists():
        print(f"  下載 {D['librispeech_url']}（約 337 MB）→ {tgz} ……")
        urllib.request.urlretrieve(D["librispeech_url"], tgz)
    else:
        print(f"  dev-clean.tar.gz 已在 {tgz}（與 W3 Demo 3 共用）")

    # 第一遍：只列檔名，每位說話人取排序後的前 utts_per_speaker + 5 句當候選
    print("  第一遍：列 tar 裡的檔名……")
    names = []
    with tarfile.open(tgz, "r|gz") as tf:
        for m in tf:
            if m.name.endswith(".flac"):
                names.append(m.name)
    by_spk: dict[str, list[str]] = {}
    for n in names:
        utt = Path(n).stem
        by_spk.setdefault(utt.split("-")[0], []).append(utt)
    speakers = sorted(by_spk, key=int)
    want = {}
    for s in speakers:
        for u in sorted(by_spk[s])[: D["utts_per_speaker"] + 5]:
            want[u] = s
    print(f"  dev-clean：{len(speakers)} 位說話人、{len(names)} 句；候選 {len(want)} 句")

    # 第二遍：解候選的 flac，量長度，留每人前 utts_per_speaker 句落在 [min_s, max_s] 的
    print("  第二遍：解候選的 flac……")
    (MEDIA_D1 / "wav").mkdir(parents=True, exist_ok=True)
    durs: dict[str, float] = {}
    with tarfile.open(tgz, "r|gz") as tf:
        for m in tf:
            if not m.name.endswith(".flac"):
                continue
            utt = Path(m.name).stem
            if utt not in want:
                continue
            x, fs_in = sf.read(io.BytesIO(tf.extractfile(m).read()), dtype="float64")
            x = dsp.resample(dsp.to_mono_float(x), fs_in, FS)
            durs[utt] = len(x) / FS
            if D["min_s"] <= durs[utt] <= D["max_s"]:
                C.save_wav(wav_path(utt), x, FS)
    utts = []
    for s in speakers:
        ok = [u for u in sorted(by_spk[s]) if u in durs and D["min_s"] <= durs[u] <= D["max_s"]][: D["utts_per_speaker"]]
        if len(ok) < D["utts_per_speaker"]:
            print(f"  {C.AMBER}說話人 {s} 只有 {len(ok)} 句落在長度範圍{C.RESET}")
        n_test = D["test_per_speaker"]
        for j, u in enumerate(ok):
            utts.append({"id": u, "spk": s, "dur_s": round(durs[u], 3), "split": "train" if j < len(ok) - n_test else "test"})
    print(f"  子集 {len(utts)} 句（{sum(u['split'] == 'train' for u in utts)} 訓、{sum(u['split'] == 'test' for u in utts)} 測）→ {MEDIA_D1 / 'wav'}（不進版控）")

    # 音素對齊：只讀 parquet 的兩欄
    print(f"  音素對齊 {CFG['phones']['hf_dataset']} / {CFG['phones']['parquet']} ……")
    table = read_alignment_table()
    ids = set(u["id"] for u in utts)
    ali: dict[str, list] = {}
    for row in table.to_pylist():
        if row["id"] in ids:
            ali[row["id"]] = [[p["phoneme"], round(float(p["start"]), 3), round(float(p["end"]), 3)] for p in row["phonemes"]]
    missing = [u["id"] for u in utts if u["id"] not in ali]
    if missing:
        print(f"  {C.AMBER}{len(missing)} 句沒有對齊，從子集拿掉：{missing[:5]}…{C.RESET}")
        utts = [u for u in utts if u["id"] in ali]
    phones = sorted({norm_phone(p[0]) for a in ali.values() for p in a} - EXCLUDE)
    unknown = sorted(set(phones) - set(ARPABET39))
    if unknown:
        print(f"  {C.AMBER}ARPAbet 39 以外的標籤：{unknown}（照樣當類別）{C.RESET}")
    DATA.mkdir(exist_ok=True)
    (DATA / "alignments.json").write_text(json.dumps(ali, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    subset = {"source": {"audio": D["librispeech_url"], "alignments": CFG["phones"]["hf_dataset"], "license": "CC BY 4.0", "time": C.now()},
              "speakers": speakers, "phones": phones, "utts": utts, "n_speakers": len(speakers), "n_utts": len(utts),
              "n_train": int(sum(u["split"] == "train" for u in utts)), "n_test": int(sum(u["split"] == "test" for u in utts))}
    (DATA / "subset.json").write_text(json.dumps(subset, ensure_ascii=False, indent=1), encoding="utf-8")
    ex = ali.get(W3_UTT) or next(iter(ali.values()))
    print(f"  {len(phones)} 個音素類別；對齊例（{W3_UTT if W3_UTT in ali else '第一句'}）：{ex[:6]} …")
    print(f"  → data/subset.json、data/alignments.json（進版控，CC BY 4.0）\n\n下一步：./present.sh check → ./present.sh extract")


def read_alignment_table():
    """先試 HfFileSystem + pyarrow 的欄位投影（只抓兩欄，range request）；不行就整個 parquet 下載到 ../_media/。"""
    import pyarrow.parquet as pq
    ds, pqf = CFG["phones"]["hf_dataset"], CFG["phones"]["parquet"]
    try:
        from huggingface_hub import HfFileSystem
        with HfFileSystem().open(f"datasets/{ds}/{pqf}", "rb") as f:
            t = pq.ParquetFile(f).read(columns=["id", "phonemes"])
        print("    （只讀 id、phonemes 兩欄，沒有下載音訊）")
        return t
    except Exception as e:
        print(f"    欄位投影失敗（{e}），改下載整個 parquet（約 343 MB，含音訊，不進版控）……")
        from huggingface_hub import hf_hub_download
        path = hf_hub_download(ds, pqf, repo_type="dataset", cache_dir=str(MEDIA / "hf_datasets"))
        return pq.read_table(path, columns=["id", "phonemes"])


# ── export → w05-metrics.json ────────────────────────────────
def export(rec_path: Path) -> None:
    rec = json.loads(rec_path.read_text(encoding="utf-8"))
    d = json.loads(METRICS_JSON.read_text(encoding="utf-8")) if METRICS_JSON.exists() else {
        "_note": "量測數字（進版控的出處，speech_ai/CLAUDE.md 4.8）。w05-data.json 是 make_figs_w05.py 算出來的、每次重產；量測的放這裡。"}
    demo1 = {"source": "demos/w05_d1_layer_probe/runs/rehearsal/rehearsal.json", "time": rec["time"], "tag": rec.get("tag", ""),
             "subset": {k: rec["subset"][k] for k in ("n_speakers", "n_utts", "n_train", "n_test")}, "n_phones": len(rec["subset"]["phones"]),
             "layers": "0 = CNN front-end output before the feature projection (model.feature_extractor, 512-d); 1–12 = transformer block outputs (768-d)",
             "definitions": {"speaker": "frame-level 40-way logistic regression, test accuracy (utterance mean-pool accuracy also recorded as speaker_acc_utt)",
                             "phone": "frame-level → 39-way logistic regression (ARPAbet, stress stripped), test accuracy",
                             "cca": "PWCCA (Morcos et al. 2018): PCA 99% each side, ridge CCA fit on train frames, projection-weighted mean of held-out canonical correlations"},
             "models": {}}
    for k in MODELS:
        c = rec["curves"].get(k, {})
        if not all(c.get(x) for x in ("speaker", "phone", "cca_layer0")):
            continue
        demo1["models"][k] = {"repo": rec["meta"][k]["repo"], "commit": rec["meta"][k].get("commit"), "device": rec["meta"][k].get("device"),
                              "ms_per_utt_median": rec["meta"][k].get("ms_per_utt_median"),
                              "speaker_acc": [round(r["acc_test"], 4) for r in c["speaker"]],
                              "speaker_acc_utt": [round(r.get("acc_test_utt", float("nan")), 4) for r in c["speaker"]],
                              "phone_acc": [round(r["acc_test"], 4) for r in c["phone"]],
                              "cca_layer0": [round(r["mean"], 4) for r in c["cca_layer0"]],
                              "cca_logmel": [round(r["mean"], 4) for r in c["cca_logmel"]] if c.get("cca_logmel") else None,
                              "summary": summarize(c)}
    d["demo1"] = demo1
    METRICS_JSON.parent.mkdir(parents=True, exist_ok=True)
    METRICS_JSON.write_text(json.dumps(d, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"  → {METRICS_JSON}（demo1）")


# ── selftest ─────────────────────────────────────────────────
def selftest() -> None:
    import matplotlib
    matplotlib.use("Agg")
    C.banner("selftest：探針與 CCA 的單元檢查、fake 流程、彩排對 w05-metrics.json", C.AMBER)
    fails = []

    def ok(cond: bool, msg: str):
        print(f"  {'OK ' if cond else 'FAIL'} {msg}")
        if not cond:
            fails.append(msg)

    rng = np.random.default_rng(1)
    P = CFG["probe"]
    # 1. 可分的群：logistic regression 要認得出來
    mu = rng.standard_normal((5, 32)) * 3
    y = rng.integers(0, 5, 2000)
    X = mu[y] + rng.standard_normal((2000, 32))
    r = logreg(X[:1500], y[:1500], X[1500:], y[1500:], 5, P["steps"], P["lr"], P["l2"], 0)
    ok(r["acc_test"] > 0.95, f"logreg 五個可分的群：test acc {r['acc_test']:.3f} > 0.95")
    # 2. 打亂標籤：接近亂猜
    yp = rng.permutation(y)
    r2 = logreg(X[:1500], yp[:1500], X[1500:], yp[1500:], 5, P["steps"], P["lr"], P["l2"], 0)
    ok(r2["acc_test"] < 0.35, f"logreg 打亂的標籤：test acc {r2['acc_test']:.3f} < 0.35")
    # 3. CCA：同一組資料旋轉過 → 接近 1；獨立的資料 → 小
    A = rng.standard_normal((3000, 40))
    R = np.linalg.qr(rng.standard_normal((40, 40)))[0]
    c1 = cca_mean(A[:2000], (A @ R)[:2000], A[2000:], (A @ R)[2000:], P["cca_reg"])
    ok(c1["mean"] > 0.97, f"PWCCA(X, XR)：held-out {c1['mean']:.3f} > 0.97")
    Bi = rng.standard_normal((3000, 40))
    c2 = cca_mean(A[:2000], Bi[:2000], A[2000:], Bi[2000:], P["cca_reg"])
    ok(abs(c2["mean"]) < 0.15, f"PWCCA(X, 獨立)：held-out {c2['mean']:.3f}，|·| < 0.15")
    Bh = A[:, :20] @ rng.standard_normal((20, 40)) + 2.0 * rng.standard_normal((3000, 40))
    c3 = cca_mean(A[:2000], Bh[:2000], A[2000:], Bh[2000:], P["cca_reg"])
    ok(c2["mean"] < c3["mean"] < c1["mean"], f"CCA 的次序：獨立 {c2['mean']:.3f} < 一半相關 {c3['mean']:.3f} < 旋轉 {c1['mean']:.3f}")
    # 4. frame 標籤：區間 → frame
    lab = frame_labels(10, [["AH1", 0.0, 0.05], ["spn", 0.05, 0.10], ["S", 0.10, 0.2]], {"AH": 0, "S": 1})
    ok(lab.tolist() == [0, 0, -1, -1, -1, 1, 1, 1, 1, 1], f"frame_labels：{lab.tolist()}（中心 12.5 + 20t ms；spn 與區間外 = −1）")
    # 5. fake 流程：合成的結構要被三條曲線抓到
    out = Out()
    d = build("fake", out)
    d.make_fig()
    t0 = time.perf_counter()
    d.run_all()
    print(f"  fake 全流程 {time.perf_counter() - t0:.0f} s")
    for k in MODELS:
        s = summarize(d.curves[k])
        ok(s["speaker_layer0_minus_top"] > 0.1, f"[{k}] fake 說話人：layer 0 比 layer 12 高 {s['speaker_layer0_minus_top']:.3f}")
        ok(3 <= s["phone_peak_layer"] <= 10, f"[{k}] fake 音素：峰值在 layer {s['phone_peak_layer']}")
        ok(d.curves[k]["cca_layer0"][0]["mean"] > 0.95, f"[{k}] fake CCA(layer 0, layer 0) = {d.curves[k]['cca_layer0'][0]['mean']:.3f} > 0.95")
        ok(s["cca0_turnback"] > 0.02 and 3 <= s["cca0_min_layer"] <= 10, f"[{k}] fake CCA 最低在 layer {s['cca0_min_layer']}、頂層回升 {s['cca0_turnback']:+.3f}")
    # 6. 彩排紀錄對 w05-metrics.json（有才比）
    rec_p = HERE / "runs" / "rehearsal" / "rehearsal.json"
    if rec_p.exists() and METRICS_JSON.exists():
        rec = json.loads(rec_p.read_text(encoding="utf-8"))
        m = json.loads(METRICS_JSON.read_text(encoding="utf-8")).get("demo1", {})
        for k in MODELS:
            if k in rec["curves"] and k in m.get("models", {}):
                a = [round(r["acc_test"], 4) for r in rec["curves"][k]["speaker"]]
                ok(a == m["models"][k]["speaker_acc"], f"[{k}] rehearsal.json 的說話人曲線 == w05-metrics.json")
    else:
        print("  （沒有彩排紀錄或 w05-metrics.json，跳過比對）")
    print(f"\n  {'全部通過' if not fails else f'{len(fails)} 項失敗'}")
    if fails:
        sys.exit(1)


# ── main ─────────────────────────────────────────────────────
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["show", "rehearse", "replay", "extract", "check", "fetch", "fake", "selftest", "export"])
    ap.add_argument("--model", choices=MODELS, help="extract：只抽這一個模型")
    ap.add_argument("--device", help="extract／check：蓋過 demo_config.toml 的 device（cpu／mps／auto）")
    ap.add_argument("--no-plot", action="store_true", help="show：只在終端機，不開圖")
    args = ap.parse_args()

    if args.cmd == "check":
        check(); return
    if args.cmd == "fetch":
        fetch(); return
    if args.cmd == "selftest":
        selftest(); return
    if args.cmd == "export":
        export(HERE / "runs" / "rehearsal" / "rehearsal.json"); return
    if args.cmd == "extract":
        for k in ([args.model] if args.model else MODELS):
            extract(k, device=args.device)
        print("\n下一步：./present.sh rehearse")
        return

    if args.cmd == "rehearse" or args.no_plot:
        import matplotlib
        matplotlib.use("Agg")

    out = Out()
    d = build(args.cmd, out)
    d.make_fig()

    if args.cmd == "rehearse":
        d.tag = "rehearsal"
        d.run_all()
        d.save(HERE / "runs" / "rehearsal")
        export(HERE / "runs" / "rehearsal" / "rehearsal.json")
        print("\n下一步：./present.sh 自己按一遍 1 2 3 4 l 4 w 2 3 4 a t；把 t 的四格填進 p37 的表、曲線的數字填進講稿。")
        return

    out.banner("W5 Demo 1：three curves across thirteen layers" + (f"   [{d.tag}]" if d.tag else ""))
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

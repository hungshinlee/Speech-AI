# -*- coding: utf-8 -*-
# 此檔由 bin/publish_demos.py 從 private repo 的 <course>/demos/ 複製而來，請勿在公開 repo 直接編輯。
"""W6 demo 2：語音的字母長什麼樣——token 矩陣、codebook 使用率、殘差能量曲線、算帳。

  show       課堂：載三個 codec（Demo 1 的 Codec），讀子集快取（沒有就先 extract），開一個視窗；鍵在視窗裡按（下面的鍵）。離開時存 runs/live/<時間>/
  extract    課前：dev-clean 子集（W5 Demo 1 的 40 人 × 10 句）每句經三個 codec encode 一次，存 token 矩陣與各層殘差能量到 ../_media/w06_d2/（幾分鐘，一次）
  rehearse   課前：extract（若還沒有）→ 全部統計 → runs/rehearsal/（進版控 = 現場的退路）、slides/assets/w06/w06-metrics.json → demo2
  replay     現場出狀況時：從 runs/rehearsal/rehearsal.json 讀回全部數字，同一個視窗、同樣的鍵，不碰 torch
  check      課前體檢：對 W3 那句，三個 codec 的「各層 codeword 累加」對 quantizer 自己的 decode、從殘差重推 code 是否逐格相同、
             各路的維度與層數；tokenizer 載不載得起 → runs/check.json
  fake       沒有模型也能演練流程（合成的 Markov token 與遞減曲線，畫面標明 FAKE）
  selftest   不開視窗：熵／run／bigram 的函數、純 numpy RVQ 的殘差恆等、算帳對 w06-data.json、fake 流程；有彩排紀錄時對 w06-metrics.json
  export     只重寫 w06-metrics.json → demo2（從 runs/rehearsal/rehearsal.json）

畫面上的鍵（順序照投影片 p37／p46 講稿的【操作】）：
  1  W3 那句經三個 codec 的 token 矩陣（終端機：前幾層 × 一個時間窗；[ ] 移窗；視窗：整句的矩陣，同一個 code 同一個顏色）
  2  子集上每一層：用到幾個 code／V、unigram 熵 H1／log2 V、perplexity、重複率、平均 run 長（三個 codec 並排）
  3  殘差能量曲線 E‖r_i‖²／E‖z‖²（Mimi 分 semantic 路與 acoustic 路）——視窗右下
  4  bigram 條件熵 H2 = H(c_t | c_{t-1}) 對 H1（每層都印，第一層是投影片要的那個），附同句內打亂順序的基線
  5  算帳：三組 codec 設定的 token rate、30 分鐘 context、flatten／RQ-Transformer 每秒步數、1/f_r；同一段 30 分鐘用 Gemma 4 的 tokenizer 算文字 token
  t  投影片 p37 的表（三列四欄）與 p46 文字那一列    h 選單    q 離開

定義（README〈數字怎麼定義〉有完整版）：
  使用率 = 子集全部 frame 上該層出現過的 code 數 ÷ V；H1 = 該層 code 的 unigram 熵（bit），除以 log2 V 得 [0, 1]；
  殘差能量比 = Σ_t‖r_{t,i}‖² ÷ Σ_t‖z_t‖²（整個子集的和的比），r_{t,i} = z_t − Σ_{j≤i} q_{t,j}，q_{t,j} 從 transformers 的 quantizer 取第 j 層的 codeword
  （EnCodec：encoder 輸出的 128 維；Mimi：各路 input_proj 之後的 256 維，semantic 路與 acoustic 路各自算；DAC：1024 維，第 j 層 codeword 經它的 out_proj）；
  H2 = Σ_a p(a) H(c_t | c_{t-1} = a) 的 plug-in 估計，只數同一句內相鄰的 pair——樣本少時偏低，所以同時印同句內打亂順序的基線。
"""
from __future__ import annotations

import argparse
import importlib.util
import io
import json
import math
import re
import shutil
import sys
import time
import tomllib
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import common as C  # noqa: E402
import dsp          # noqa: E402

CFG = tomllib.loads((HERE / "demo_config.toml").read_text(encoding="utf-8"))
METRICS_JSON = HERE.parents[1] / "slides" / "assets" / "w06" / "w06-metrics.json"     # parents[1] = speech_ai/
DATA_JSON = HERE.parents[1] / "slides" / "assets" / "w06" / "w06-data.json"
CACHE = (HERE / CFG["subset"]["cache_dir"]).resolve()


# ── Demo 1 的程式當模組用（Codec、FS、三個 codec 的 config）────
def _load_d1():
    p = (HERE / CFG["d1"]["dir"]).resolve() / "demo.py"
    if not p.exists():
        sys.exit(f"找不到 Demo 1 的程式 {p}（demo_config.toml [d1].dir）")
    spec = importlib.util.spec_from_file_location("w06_d1_demo", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


D1 = _load_d1()
FS = D1.FS
CODECS = D1.CODECS
LABEL = {k: D1.CFG["codecs"][k]["label"] for k in CODECS}


def versions() -> dict:
    return D1.versions()


# ── 統計 ─────────────────────────────────────────────────────
def unigram(seq: np.ndarray, V: int) -> dict:
    """一層的 code 序列（全部句子接起來）：用到幾個 code、H1（bit）、H1/log2 V、perplexity。"""
    cnt = np.bincount(seq.astype(np.int64), minlength=V)
    n = int(cnt.sum())
    p = cnt[cnt > 0] / n
    H1 = max(0.0, float(-(p * np.log2(p)).sum())) if n else float("nan")
    used = int((cnt > 0).sum())
    return {"codes_used": used, "V": int(V), "usage_ratio": used / V, "H1_bits": H1, "H1_norm": H1 / math.log2(V),
            "perplexity": float(2 ** H1), "n_frames": n}


def repeat_stats(seqs: list[np.ndarray]) -> dict:
    """同一句內相鄰 frame 同一個 code 的比例，與平均 run 長度。"""
    same = sum(int((s[1:] == s[:-1]).sum()) for s in seqs)
    n_pairs = sum(max(0, len(s) - 1) for s in seqs)
    n_runs = sum(1 + int((s[1:] != s[:-1]).sum()) for s in seqs if len(s))
    n_frames = sum(len(s) for s in seqs)
    return {"repeat_rate": same / n_pairs if n_pairs else float("nan"), "mean_run": n_frames / n_runs if n_runs else float("nan")}


def bigram_cond_entropy(seqs: list[np.ndarray], V: int) -> float:
    """H(c_t | c_{t-1})，bit，plug-in；只數同一句內相鄰的 pair。"""
    pairs = np.concatenate([s[:-1].astype(np.int64) * V + s[1:].astype(np.int64) for s in seqs if len(s) > 1])
    if len(pairs) == 0:
        return float("nan")
    u, c = np.unique(pairs, return_counts=True)
    prev = u // V
    c_prev = np.bincount(prev, weights=c, minlength=V)
    N = c.sum()
    return float(np.sum(c / N * np.log2(c_prev[prev] / c)))


def shuffled_bigram(seqs: list[np.ndarray], V: int, repeats: int, seed: int) -> float:
    """同一句內把順序打亂（unigram 不變），同樣的 H2——偏低多少是樣本量造成的，這個基線看得到。"""
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(repeats):
        vals.append(bigram_cond_entropy([rng.permutation(s) for s in seqs], V))
    return float(np.mean(vals))


def stage_label(codec: str, i: int, n_semantic: int) -> str:
    """表格裡的層名：Mimi 第一層是 S（semantic VQ），之後 A1…；其餘 1…"""
    if codec == "mimi" and n_semantic:
        return "S" if i < n_semantic else f"A{i - n_semantic + 1}"
    return str(i + 1)


def residual_energies(z: np.ndarray, contribs: np.ndarray) -> tuple[float, np.ndarray]:
    """z: (D, T)；contribs: (K, D, T) 第 j 層的 codeword。回 Σ‖z‖² 與 Σ‖z − Σ_{j≤i} q_j‖²（i = 1…K）。"""
    e_z = float(np.sum(z.astype(np.float64) ** 2))
    acc = np.zeros_like(z, dtype=np.float64)
    e_r = np.zeros(contribs.shape[0])
    for i in range(contribs.shape[0]):
        acc += contribs[i]
        e_r[i] = float(np.sum((z - acc) ** 2))
    return e_z, e_r


def compute_stats(codes_all: np.ndarray, offsets: np.ndarray, V: int, codec: str, n_semantic: int, paths: dict) -> dict:
    """子集的全部統計。codes_all: (N_q, total_T)；offsets: 每句的起點；paths: {name: {stage_offset, energy_z: (n,), energy_r: (n, K)}}。"""
    N_q = codes_all.shape[0]
    seqs_by_stage = [[codes_all[i, offsets[u]:offsets[u + 1]] for u in range(len(offsets) - 1)] for i in range(N_q)]
    st = CFG["stats"]
    stages = []
    for i in range(N_q):
        row = {"stage": i + 1, "label": stage_label(codec, i, n_semantic), **unigram(codes_all[i], V), **repeat_stats(seqs_by_stage[i])}
        H2 = bigram_cond_entropy(seqs_by_stage[i], V)
        H2s = shuffled_bigram(seqs_by_stage[i], V, st["shuffle_repeats"], st["shuffle_seed"] + i)
        row.update({"H2_bits": H2, "H2_over_H1": H2 / row["H1_bits"] if row["H1_bits"] else float("nan"),
                    "H2_shuffled_bits": H2s, "H2_over_shuffled": H2 / H2s if H2s else float("nan")})
        stages.append(row)
    residual = {}
    for name, p in paths.items():
        e_z = float(np.sum(p["energy_z"])); e_r = np.sum(np.asarray(p["energy_r"]), axis=0)
        ratios = (e_r / e_z).tolist() if e_z > 0 else [float("nan")] * len(e_r)
        K = len(ratios)
        residual[name] = {"stage_offset": int(p["stage_offset"]), "n_stages": K, "ratios": ratios,
                          "after": {str(n): ratios[n - 1] for n in (1, 2, 4, K) if 1 <= n <= K}}
    return {"N_q": int(N_q), "V": int(V), "n_semantic": int(n_semantic), "n_utts": int(len(offsets) - 1), "n_frames": int(codes_all.shape[1]),
            "stages": stages, "residual": residual}


# ── 從 transformers 的 quantizer 取 codeword ──────────────────
class Latents:
    """包住 Demo 1 的 Codec：encode 一次拿 (N_q, T) 的 code 矩陣，同時抓 quantizer 的輸入 z_t，再從各層 codebook 取 codeword 算殘差。
    三個類別的 quantizer 介面不同，各寫一段；check 用 quantizer 自己的 decode 與「從殘差重推 code」對過。"""

    def __init__(self, key: str, device: str):
        self.codec = D1.Codec(key, device)
        self.key, self.label = key, self.codec.label
        self.torch = self.codec.torch
        self._z = None
        q = self.codec.model.quantizer
        if key == "dac":
            q.register_forward_pre_hook(self._hook_dac)
        else:
            orig = q.encode

            def enc(embeddings, *a, **kw):
                self._z = embeddings.detach()
                return orig(embeddings, *a, **kw)
            q.encode = enc
        self.n_semantic = self.codec.n_semantic

    def _hook_dac(self, module, args, kwargs=None):
        self._z = args[0].detach()

    @property
    def device(self):
        return self.codec.device

    def info(self) -> dict:
        return self.codec.info()

    # paths: {name: (rvq-like object or None, stage_offset, n_stages)}
    def _paths(self):
        q = self.codec.model.quantizer
        if self.key == "mimi":
            ns = self.n_semantic
            return {"semantic": (q.semantic_residual_vector_quantizer, 0, ns),
                    "acoustic": (q.acoustic_residual_vector_quantizer, ns, len(q.acoustic_residual_vector_quantizer.layers))}
        if self.key == "encodec":
            return {"rvq": (q, 0, len(q.layers))}
        return {"rvq": (q, 0, len(q.quantizers))}

    def encode(self, x: np.ndarray) -> tuple[np.ndarray, dict]:
        """回 codes (N_q, T) 與 paths {name: {"z": (D, T) tensor, "contrib": (K, D, T) tensor, "stage_offset": int}}（tensor 在 device 上，float32）。"""
        torch = self.torch
        self._z = None
        codes = self.codec.encode_all(x)
        assert self._z is not None, "沒抓到 quantizer 的輸入"
        z = self._z[0]                                                   # (D, T)
        ct = torch.as_tensor(codes, dtype=torch.long, device=self.device)
        out = {}
        with torch.no_grad():
            for name, (rvq, off, K) in self._paths().items():
                K = min(K, codes.shape[0] - off)
                if K <= 0:
                    continue
                sub = ct[off:off + K]                                        # (K, T)
                if self.key == "dac":
                    zp = z
                    contrib = torch.stack([rvq.quantizers[i].out_proj(rvq.quantizers[i].codebook(sub[i][None]).transpose(1, 2))[0] for i in range(K)])
                else:
                    zp = rvq.input_proj(z[None])[0] if getattr(rvq, "input_proj", None) is not None else z
                    contrib = torch.stack([rvq.layers[i].decode(sub[i][None])[0] for i in range(K)])
                out[name] = {"z": zp.float(), "contrib": contrib.float(), "stage_offset": off}
        return codes, out

    @staticmethod
    def energies(paths: dict) -> dict:
        """paths → {name: {"stage_offset", "energy_z": float, "energy_r": (K,)}}（numpy）。"""
        r = {}
        for name, p in paths.items():
            e_z, e_r = residual_energies(p["z"].cpu().numpy(), p["contrib"].cpu().numpy())
            r[name] = {"stage_offset": p["stage_offset"], "energy_z": e_z, "energy_r": e_r}
        return r

    # ── check 用 ──
    def official_quantized(self, codes: np.ndarray):
        """quantizer 自己從 code 還原的 latent（送進 decoder 的那個），(D, T)。"""
        torch = self.torch
        q = self.codec.model.quantizer
        ct = torch.as_tensor(codes, dtype=torch.long, device=self.device)
        with torch.no_grad():
            if self.key == "encodec":
                y = q.decode(ct[:, None, :])[0]                               # codes (K, B, T)
            elif self.key == "mimi":
                y = q.decode(ct[None])[0]                                      # (B, K, T)
            else:
                y = q.from_codes(ct[None])[0][0]
        return y.float()

    def mine_quantized(self, paths: dict):
        """我的「各層 codeword 累加」映回 latent 空間（Mimi 經各路的 output_proj），(D, T)。"""
        torch = self.torch
        total = None
        with torch.no_grad():
            for name, (rvq, off, K) in self._paths().items():
                if name not in paths:
                    continue
                s = paths[name]["contrib"].sum(0)
                if self.key == "mimi" and getattr(rvq, "output_proj", None) is not None:
                    s = rvq.output_proj(s[None])[0]
                total = s if total is None else total + s
        return total

    def rederive_codes(self, paths: dict) -> dict:
        """從殘差重推 code：第 i 層的 code 應該是 r_{i-1} 在第 i 層 codebook 的最近鄰（DAC：in_proj 之後 ℓ2 正規化的 cosine）。回每路逐層的一致率。"""
        torch = self.torch
        res = {}
        with torch.no_grad():
            for name, (rvq, off, K) in self._paths().items():
                if name not in paths:
                    continue
                z, contrib = paths[name]["z"], paths[name]["contrib"]
                codes_path = torch.as_tensor(self._last_codes[off:off + contrib.shape[0]], device=self.device)
                resid = z.clone()
                agree = []
                for i in range(contrib.shape[0]):
                    if self.key == "dac":
                        qz = rvq.quantizers[i]
                        e = qz.in_proj(resid[None])[0].T                              # (T, 8)
                        e = torch.nn.functional.normalize(e, dim=1)
                        cb = torch.nn.functional.normalize(qz.codebook.weight, dim=1)
                        idx = (e @ cb.T).argmax(1)
                    else:
                        cb = rvq.layers[i].codebook.embed                              # (V, D)
                        idx = torch.cdist(resid.T.float()[None], cb.float()[None])[0].argmin(1)
                    agree.append(float((idx == codes_path[i]).float().mean()))
                    resid = resid - contrib[i]
                res[name] = agree
        return res

    def encode_check(self, x: np.ndarray):
        codes, paths = self.encode(x)
        self._last_codes = codes
        return codes, paths


class FakeLatents:
    """沒有 torch 時的替身：Markov 鏈的 token（第一層黏、後面幾層接近均勻）與一條遞減的殘差曲線。畫面標 FAKE。"""
    DEFAULTS = {"mimi": (12.5, 2048, 8, 1), "encodec": (75.0, 1024, 8, 0), "dac": (75.0, 1024, 8, 0)}

    def __init__(self, key: str, device: str = "cpu"):
        self.key, self.label = key, LABEL[key] + " [FAKE]"
        self.frame_rate, self.V, self.N_q, self.n_semantic = self.DEFAULTS[key]
        self.rng = np.random.default_rng(CFG["fake"]["seed"] + len(key))
        self.device = type("D", (), {"type": "fake"})()

    def info(self) -> dict:
        return {"repo": "fake", "commit": None, "device": "fake", "n_params": 0, "sampling_rate": FS, "frame_rate_cfg": self.frame_rate,
                "codebook_size": self.V, "n_q_cfg": self.N_q, "n_semantic_quantizers": self.n_semantic}

    def encode(self, x: np.ndarray):
        T = max(2, int(math.ceil(len(x) / FS * self.frame_rate)))
        stick = CFG["fake"]["stickiness"]
        codes = np.zeros((self.N_q, T), dtype=np.int64)
        for i in range(self.N_q):
            # 第 i 層只用 V·(1 − i/N_q·0.6) 個 code，再 zipf 一點
            n_use = max(8, int(self.V * (1 - 0.6 * i / self.N_q)))
            w = 1 / np.arange(1, n_use + 1) ** (0.8 * (1 - i / self.N_q))
            w /= w.sum()
            s = stick[min(i, len(stick) - 1)]
            c = self.rng.choice(n_use, p=w)
            for t in range(T):
                if t and self.rng.random() < s:
                    codes[i, t] = codes[i, t - 1]
                else:
                    codes[i, t] = self.rng.choice(n_use, p=w)
        paths = {}
        if self.key == "mimi":
            paths["semantic"] = {"stage_offset": 0, "energy_z": 1.0, "energy_r": np.array([0.55])}
            paths["acoustic"] = {"stage_offset": 1, "energy_z": 1.0, "energy_r": 0.4 ** np.arange(1, self.N_q) ** 0.5}
        else:
            paths["rvq"] = {"stage_offset": 0, "energy_z": 1.0, "energy_r": 0.45 ** np.arange(1, self.N_q + 1) ** 0.5}
        return codes, paths

    @staticmethod
    def energies(paths):
        return paths


# ── 素材 ─────────────────────────────────────────────────────
def load_subset() -> list[dict]:
    p = (HERE / CFG["subset"]["list"]).resolve()
    if not p.exists():
        sys.exit(f"找不到 W5 Demo 1 的子集名單 {p}：先在 ../w05_d1_layer_probe 跑 ./present.sh fetch")
    return json.loads(p.read_text(encoding="utf-8"))["utts"]


def subset_wav(utt: str) -> Path:
    return (HERE / CFG["subset"]["wav_dir"]).resolve() / f"{utt}.wav"


def load_wav24(p: Path) -> np.ndarray:
    return C.load_wav(p, FS)


def sentence_wav() -> np.ndarray:
    p = (HERE / CFG["sentence"]["path"]).resolve()
    if not p.exists():
        sys.exit(f"找不到 W3 那句 {p}：先在 ../w06_d1_layer_listen 跑 ./present.sh fetch")
    return load_wav24(p)


def cache_path(key: str) -> Path:
    return CACHE / f"codes_{key}.npz"


# ── extract：子集 × 三個 codec ────────────────────────────────
def extract(codecs: dict, out, only_missing: bool = True) -> None:
    utts = load_subset()
    missing = [u for u in utts if not subset_wav(u["id"]).exists()]
    if missing:
        sys.exit(f"子集的 wav 快取缺 {len(missing)} 句（例如 {subset_wav(missing[0]['id'])}）：先在 ../w05_d1_layer_probe 跑 ./present.sh fetch")
    CACHE.mkdir(parents=True, exist_ok=True)
    for key, cd in codecs.items():
        p = cache_path(key)
        if only_missing and p.exists():
            out(f"  {LABEL[key]:<16s} 快取已有：{p}")
            continue
        out(f"  {LABEL[key]:<16s} encode {len(utts)} 句 …")
        codes_list, offsets, paths_acc, ms = [], [0], {}, []
        t_all = time.perf_counter()
        for n, u in enumerate(utts):
            x = load_wav24(subset_wav(u["id"]))
            t0 = time.perf_counter()
            codes, paths = cd.encode(x)
            en = cd.energies(paths)
            ms.append(1000 * (time.perf_counter() - t0))
            codes_list.append(codes.astype(np.int32)); offsets.append(offsets[-1] + codes.shape[1])
            for name, e in en.items():
                a = paths_acc.setdefault(name, {"stage_offset": e["stage_offset"], "energy_z": [], "energy_r": []})
                a["energy_z"].append(float(e["energy_z"])); a["energy_r"].append(np.asarray(e["energy_r"], dtype=np.float64))
            if (n + 1) % 40 == 0 or n + 1 == len(utts):
                out(f"      {n + 1:4d}/{len(utts)}   {np.median(ms):6.0f} ms/句（中位）   累計 {time.perf_counter() - t_all:5.0f} s")
        info = cd.info()
        info.update({"N_q": int(codes_list[0].shape[0]), "time": C.now(), "versions": versions(), "n_utts": len(utts), "fs": FS,
                     "seconds": float(sum(u["dur_s"] for u in utts)), "encode_ms_median": float(np.median(ms))})
        np.savez_compressed(p, codes=np.concatenate(codes_list, axis=1), offsets=np.asarray(offsets, dtype=np.int64),
                            utts=np.asarray([u["id"] for u in utts]), info=json.dumps(info, ensure_ascii=False),
                            paths=json.dumps({k: {"stage_offset": v["stage_offset"], "energy_z": v["energy_z"], "energy_r": np.stack(v["energy_r"]).tolist()} for k, v in paths_acc.items()}))
        out(f"      → {p}（{p.stat().st_size / 1e6:.1f} MB）")


def load_cache(key: str) -> dict | None:
    p = cache_path(key)
    if not p.exists():
        return None
    z = np.load(p, allow_pickle=False)
    paths = json.loads(str(z["paths"]))
    for v in paths.values():
        v["energy_z"] = np.asarray(v["energy_z"]); v["energy_r"] = np.asarray(v["energy_r"])
    return {"codes": z["codes"], "offsets": z["offsets"], "utts": list(z["utts"]), "info": json.loads(str(z["info"])), "paths": paths}


# ── 算帳 ─────────────────────────────────────────────────────
def budget_rows() -> dict:
    d = json.loads(DATA_JSON.read_text(encoding="utf-8"))["paper_settings"]["budget"]
    return {k: v for k, v in d.items() if isinstance(v, dict)}


def budget_arith(f_r: float, N_q: int, minutes: float) -> dict:
    tok = f_r * N_q
    return {"f_r": f_r, "N_q": N_q, "tokens_per_s": tok, "context": tok * minutes * 60, "steps_flatten": tok, "steps_rq": f_r, "floor_ms": 1000 / f_r}


def load_tokenizer():
    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer
    repo = CFG["budget"]["tokenizer_repo"]
    path = snapshot_download(repo, local_files_only=True, allow_patterns=["tokenizer*", "special_tokens_map.json", "*.model", "config.json", "generation_config.json", "chat_template*"])
    try:
        return AutoTokenizer.from_pretrained(path), path
    except Exception as e:
        # transformers 不認得這個 model_type 時，直接從 tokenizer.json 載（算 token 數只需要這個）
        tj = Path(path) / "tokenizer.json"
        if not tj.exists():
            raise
        from transformers import PreTrainedTokenizerFast
        C.note(f"  AutoTokenizer 失敗（{e!r}），改從 tokenizer.json 載")
        return PreTrainedTokenizerFast(tokenizer_file=str(tj)), path


def text_sources() -> dict:
    """兩句實測語音的文字與秒數：華語（W1 Demo 2 的 result.json）、英語（W3 那句）。"""
    b = CFG["budget"]
    src = {}
    pm = (HERE / b["mandarin_result"]).resolve()
    if pm.exists():
        r = json.loads(pm.read_text(encoding="utf-8"))
        seg = r["endpoint_report"]["speech_segments_s"]
        text = r["runs"][0]["transcript"]
        src["mandarin"] = {"text": text, "n_chars": len(text), "seconds_span": seg[-1][1] - seg[0][0],
                           "seconds_speech": float(sum(e - s for s, e in seg)), "source": str(Path(b["mandarin_result"]))}
    pe = (HERE / b["english_text"]).resolve(); pw = (HERE / b["english_wav"]).resolve()
    if pe.exists() and pw.exists():
        text = pe.read_text(encoding="utf-8").strip()
        if b["english_normalize_case"]:
            text = text.lower().capitalize() + "."
        x = load_wav24(pw)
        src["english"] = {"text": text, "n_chars": len(text), "n_words": len(text.split()), "seconds_span": len(x) / FS,
                          "seconds_speech": len(x) / FS, "source": str(Path(b["english_text"]))}
    return src


def budget_text(out, tok=None) -> dict:
    """文字那一列：兩句的 token 數 → tokens/s → 30 分鐘。tok=None 時試著載 tokenizer；載不起來就只回估計值。"""
    b = CFG["budget"]
    res = {"minutes": b["minutes"], "tokenizer": b["tokenizer_repo"], "rows": {}}
    try:
        if tok is None:
            tok, path = load_tokenizer()
            res["tokenizer_path"] = str(path)
        res["tokenizer_class"] = type(tok).__name__
    except Exception as e:
        res["error"] = repr(e)
        out(f"  {C.AMBER}tokenizer 載不起來：{e!r}{C.RESET}（W1 Demo 2 跑過 mlx-lm 的話 HF cache 裡應該有；文字列先用大綱的估計 5–10 k）")
        return res
    for key, s in text_sources().items():
        ids = tok(s["text"], add_special_tokens=False)["input_ids"]
        n = len(ids)
        row = dict(s)
        row.update({"n_tokens": n, "tokens_per_char": n / s["n_chars"], "tokens_per_s": n / s["seconds_span"],
                    "tokens_30min": n / s["seconds_span"] * b["minutes"] * 60,
                    "tokens_30min_speech_only": n / s["seconds_speech"] * b["minutes"] * 60,
                    "pieces": [tok.decode([i]) for i in ids]})
        res["rows"][key] = row
    return res


# ── 紀錄 ─────────────────────────────────────────────────────
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
class Demo:
    """一個 session。資料全部是 numpy／dict（live 從快取算、replay 從 json 讀、fake 合成），所以三種模式同一個視窗、同樣的鍵。"""

    def __init__(self, out: Out, tag: str = ""):
        self.out, self.tag = out, tag
        self.info: dict[str, dict] = {}
        self.stats: dict[str, dict] = {}
        self.sentence: dict[str, np.ndarray] = {}
        self.budget: dict = {}
        self.win0 = 0
        self.fig = self.axes = self.plt = None
        self.log = {"time": C.now(), "versions": versions(), "tag": tag, "fs": FS, "subset": {}, "sentence": {"utt": CFG["sentence"]["utt"], "T": {}}}
        self.folder = HERE / "runs" / "live" / Path(C.now().replace(":", "") + (".fake" if "FAKE" in tag else ""))

    # ── 準備 ──
    def from_codecs(self, codecs: dict, do_extract: bool = True) -> None:
        o = self.out
        if do_extract:
            o.banner("子集快取", C.AMBER)
            extract(codecs, o, only_missing=True)
        o.banner("統計", C.AMBER)
        for key, cd in codecs.items():
            cache = load_cache(key)
            if cache is None:
                o(f"  {LABEL[key]} 沒有快取，跳過"); continue
            info = cache["info"]
            t0 = time.perf_counter()
            self.stats[key] = compute_stats(cache["codes"], cache["offsets"], info["codebook_size"], key, info.get("n_semantic_quantizers", 0), cache["paths"])
            self.info[key] = info
            self.log["subset"] = {"n_utts": info["n_utts"], "seconds": info["seconds"], "source": "W5 Demo 1: LibriSpeech dev-clean, 40 speakers × 10 sentences (CC BY 4.0)"}
            o(f"  {LABEL[key]:<16s} {info['n_utts']} 句、{cache['codes'].shape[1]} frame、N_q {cache['codes'].shape[0]}、V {info['codebook_size']}   統計 {time.perf_counter() - t0:4.1f} s")
        o.banner("W3 那句", C.AMBER)
        x = sentence_wav()
        for key, cd in codecs.items():
            if key not in self.stats:
                continue
            t0 = time.perf_counter()
            codes, _ = cd.encode(x)
            self.sentence[key] = codes
            self.log["sentence"]["T"][key] = int(codes.shape[1])
            o(f"  {LABEL[key]:<16s} {codes.shape[0]} × {codes.shape[1]}   {1000 * (time.perf_counter() - t0):5.0f} ms")
        self.log["sentence"]["seconds"] = len(x) / FS

    def from_replay(self, rec: dict) -> None:
        self.info = rec["codecs"]; self.stats = rec["stats"]
        self.sentence = {k: np.asarray(v, dtype=np.int64) for k, v in rec["sentence"]["codes"].items()}
        self.budget = rec.get("budget", {})
        self.log["subset"] = rec["subset"]; self.log["sentence"] = {k: v for k, v in rec["sentence"].items() if k != "codes"}

    # ── 視窗 ──
    def make_fig(self):
        import matplotlib.pyplot as plt
        self.plt = plt
        d = CFG["display"]
        self.fig = plt.figure(figsize=(d["fig_w"], d["fig_h"]))
        gs = self.fig.add_gridspec(2, 2, height_ratios=[1.0, 1], hspace=0.3, wspace=0.22, top=0.92, bottom=0.08, left=0.07, right=0.98)
        self.ax_mat = [self.fig.add_subplot(gs[0, :])]
        self.ax_use = self.fig.add_subplot(gs[1, 0])
        self.ax_res = self.fig.add_subplot(gs[1, 1])
        self.fig.suptitle("W6 Demo 2 — what the letters look like" + (f"   [{self.tag}]" if self.tag else ""), fontsize=13)
        self.fig.canvas.mpl_connect("key_press_event", self.on_key)
        for ax in (self.ax_mat[0], self.ax_use, self.ax_res):
            ax.set_axis_off()
        self.ax_mat[0].text(0.5, 0.5, "press 1 — token matrix of the W3 sentence", ha="center", va="center", color="#888", transform=self.ax_mat[0].transAxes)
        self.ax_use.text(0.5, 0.5, "press 2 — usage and entropy per stage", ha="center", va="center", color="#888", transform=self.ax_use.transAxes)
        self.ax_res.text(0.5, 0.5, "press 3 — residual energy per stage", ha="center", va="center", color="#888", transform=self.ax_res.transAxes)
        self.redraw()
        return self.fig

    def redraw(self):
        if self.fig is not None:
            self.fig.canvas.draw_idle()
            self.plt.pause(0.05)

    COLORS = {"mimi": "#5b3fa0", "encodec": "#2a7fc8", "dac": "#c8742a"}

    def draw_matrix(self):
        if self.fig is None:
            return
        ax = self.ax_mat[0]
        ax.clear(); ax.set_axis_off()
        keys = [k for k in CODECS if k in self.sentence]
        n = len(keys)
        K = CFG["display"]["matrix_stages"]
        sub = ax.get_subplotspec().subgridspec(n, 1, hspace=0.55)
        # 把舊的子軸清掉
        for a in self.ax_mat[1:]:
            a.remove()
        self.ax_mat = [ax]
        for r, key in enumerate(keys):
            a = self.fig.add_subplot(sub[r])
            self.ax_mat.append(a)
            m = self.sentence[key][:K]
            info = self.info[key]
            # 同一個 code 同一個顏色（code 對 20 取餘數進 tab20 —— 顏色只是要讓 run 看得見）
            a.imshow(m % 20, aspect="auto", cmap="tab20", interpolation="nearest", vmin=0, vmax=19)
            ns = info.get("n_semantic_quantizers", 0)
            a.set_yticks(range(m.shape[0])); a.set_yticklabels([stage_label(key, i, ns) for i in range(m.shape[0])], fontsize=7)
            fr = info.get("frame_rate_cfg") or 1
            step = max(1, int(round(fr)))          # 每秒一個刻度；只有最下面那條帶子標秒數
            if r == n - 1:
                a.set_xticks(range(0, m.shape[1], step)); a.set_xticklabels([f"{t / fr:.0f} s" for t in range(0, m.shape[1], step)], fontsize=7)
            else:
                a.set_xticks([])
            rep = [s["repeat_rate"] for s in self.stats[key]["stages"][:K]]
            a.set_title(f"{LABEL[key]}   {self.sentence[key].shape[0]} × {m.shape[1]}   (first {m.shape[0]} stages; repeat rate on the subset, top→bottom: " +
                        " ".join(f"{v:.2f}" for v in rep) + ")", fontsize=8, loc="left")
            a.tick_params(length=2)
        self.redraw()

    def draw_usage(self):
        if self.fig is None:
            return
        ax = self.ax_use
        ax.clear(); ax.set_axis_on()
        K = CFG["sentence"]["max_stage_plot"]
        for key in CODECS:
            if key not in self.stats:
                continue
            st = self.stats[key]["stages"][:K]
            xs = [s["stage"] for s in st]
            ax.plot(xs, [s["usage_ratio"] for s in st], "o-", color=self.COLORS[key], label=f"{LABEL[key]}: codes used / V")
            ax.plot(xs, [s["H1_norm"] for s in st], "s--", color=self.COLORS[key], alpha=0.7, label=f"{LABEL[key]}: $H_1/\\log_2 V$")
        ax.set_xlabel("stage"); ax.set_ylim(0, 1.05); ax.set_xticks(range(1, K + 1)); ax.grid(alpha=0.3)
        ax.set_title("codebook usage and unigram entropy, per stage (dev-clean 40 × 10)", fontsize=10, loc="left")
        ax.legend(fontsize=7, ncol=1, loc="lower right")
        self.redraw()

    def draw_residual(self):
        if self.fig is None:
            return
        ax = self.ax_res
        ax.clear(); ax.set_axis_on()
        K = CFG["sentence"]["max_stage_plot"]
        for key in CODECS:
            if key not in self.stats:
                continue
            for name, r in self.stats[key]["residual"].items():
                ratios = [1.0] + r["ratios"][:K]
                lab = LABEL[key] + (f" — {name} path" if len(self.stats[key]["residual"]) > 1 else "")
                style = "o-" if name != "semantic" else "D"
                ax.plot(range(len(ratios)), ratios, style, color=self.COLORS[key], label=lab, ms=5, alpha=0.9 if name != "semantic" else 1)
        ax.set_xlabel("stage $i$"); ax.set_ylabel(r"$\sum_t\|\mathbf{r}_{t,i}\|^2 / \sum_t\|\mathbf{z}_t\|^2$"); ax.set_ylim(0, 1.05)
        ax.set_xticks(range(0, K + 1)); ax.grid(alpha=0.3)
        ax.set_title("residual energy after stage $i$ — measured, not schematic", fontsize=10, loc="left")
        ax.legend(fontsize=7)
        self.redraw()

    # ── 鍵 ──
    def key_matrix(self, move: int = 0) -> None:
        o = self.out
        if move == 0:
            o.banner("1  token matrix — the W3 sentence through three codecs", C.BLUE)
        W = CFG["sentence"]["window"]
        for key in CODECS:
            if key not in self.sentence:
                continue
            m = self.sentence[key]
            info = self.info[key]
            fr = info.get("frame_rate_cfg") or 1
            if move == 0 and self.win0 == 0:
                self.win0 = 0
            start = int(round(CFG["sentence"]["start_s"] * fr)) + int(round(self.win0 * fr))
            start = max(0, min(start, m.shape[1] - 1))
            end = min(m.shape[1], start + W)
            ns = info.get("n_semantic_quantizers", 0)
            o(f"\n  {C.BOLD}{LABEL[key]}{C.RESET}   N_q × T = {m.shape[0]} × {m.shape[1]}   f_r {fr:g} Hz   V {info.get('codebook_size')}   frames {start}–{end - 1} ({start / fr:.2f}–{end / fr:.2f} s)")
            K = CFG["display"]["matrix_stages"]
            for i in range(min(K, m.shape[0])):
                cells = []
                for t in range(start, end):
                    same = t > start and m[i, t] == m[i, t - 1]
                    cells.append((C.DIM + "    ·" + C.RESET) if same else f"{m[i, t]:5d}")
                o(f"    {stage_label(key, i, ns):>3s} │" + "".join(cells))
            o(f"    {C.GREY}· = same code as the previous frame{C.RESET}")
        o(f"  {C.GREY}[ ] 移窗（目前從 {CFG['sentence']['start_s'] + self.win0:.1f} s 起）{C.RESET}")
        if move == 0:
            self.draw_matrix()

    def key_usage(self) -> None:
        o = self.out
        o.banner("2  codes used / V and H1 / log2 V, per stage (dev-clean 40 × 10)", C.BLUE)
        K = CFG["sentence"]["max_stage_plot"]
        keys = [k for k in CODECS if k in self.stats]
        o(f"  {'':>5s}" + "".join(f"{LABEL[k]:^34s}" for k in keys))
        o(f"  {'stage':>5s}" + "".join(f"{'used/V':>9s}{'H1/lg V':>9s}{'pplx':>8s}{'rep':>7s} " for _ in keys))
        for i in range(max(self.stats[k]["N_q"] for k in keys)):
            if i >= K and i + 1 != max(self.stats[k]["N_q"] for k in keys):
                if i == K:
                    o(f"  {'…':>5s}")
                continue
            row = f"  {i + 1:>5d}"
            for k in keys:
                st = self.stats[k]["stages"]
                if i < len(st):
                    s = st[i]
                    row += f"{s['codes_used']:>4d}/{s['V']:<4d}{s['H1_norm']:>9.3f}{s['perplexity']:>8.0f}{s['repeat_rate']:>7.2f} "
                else:
                    row += " " * 34
            o(row)
        o(f"  {C.GREY}used/V: distinct codes in the subset ÷ codebook size;  H1/lg V: unigram entropy over log2 V;  pplx = 2^H1;  rep: P(c_t = c_(t-1)) within a sentence{C.RESET}")
        o(f"  {C.GREY}Mimi: S = semantic VQ, A1… = acoustic RVQ (its own path, not the residual of S){C.RESET}")
        self.draw_usage()

    def key_residual(self) -> None:
        o = self.out
        o.banner("3  residual energy after stage i — Σ‖r_i‖² / Σ‖z‖², measured on the subset", C.BLUE)
        for key in CODECS:
            if key not in self.stats:
                continue
            for name, r in self.stats[key]["residual"].items():
                lab = LABEL[key] + (f"  {name} path" if len(self.stats[key]["residual"]) > 1 else "")
                o(f"  {lab:<28s} " + "  ".join(f"{i + 1}:{v:.3f}" for i, v in enumerate(r["ratios"][:CFG['sentence']['max_stage_plot']])) +
                  (f"  …  {r['n_stages']}:{r['ratios'][-1]:.3f}" if r["n_stages"] > CFG["sentence"]["max_stage_plot"] else ""))
        o(f"  {C.GREY}z: EnCodec = encoder output (128-d); Mimi = after each path's input_proj (256-d), semantic and acoustic separately; DAC = encoder output (1024-d), codeword via out_proj{C.RESET}")
        self.draw_residual()

    def key_bigram(self) -> None:
        o = self.out
        o.banner("4  H2 = H(c_t | c_(t-1)) against H1 — how much structure in time (stage 1 is the one on the slide)", C.BLUE)
        K = CFG["sentence"]["max_stage_plot"]
        for key in CODECS:
            if key not in self.stats:
                continue
            o(f"\n  {C.BOLD}{LABEL[key]}{C.RESET}   {self.stats[key]['n_frames']} frames")
            o(f"    {'stage':>5s}{'H1':>7s}{'H2':>7s}{'H2/H1':>7s}{'H2 shuf':>9s}{'H2/shuf':>9s}")
            for s in self.stats[key]["stages"][:K]:
                hl = C.BOLD if s["stage"] == 1 else ""
                o(f"    {hl}{s['label']:>5s}{s['H1_bits']:>7.2f}{s['H2_bits']:>7.2f}{s['H2_over_H1']:>7.2f}{s['H2_shuffled_bits']:>9.2f}{s['H2_over_shuffled']:>9.2f}{C.RESET}")
        o(f"  {C.GREY}plug-in estimates; 'shuf' = the same H2 after shuffling each sentence's order (unigram unchanged) — the gap H1 → shuf is small-sample bias, shuf → H2 is real structure{C.RESET}")

    def key_budget(self) -> None:
        o = self.out
        o.banner("5  the bill — token rate, context, steps, floor; and the same 30 minutes as text", C.BLUE)
        mins = CFG["budget"]["minutes"]
        rows = budget_rows()
        o(f"  {'':<26s}{'f_r':>7s}{'N_q':>5s}{'tok/s':>8s}{f'ctx {mins} min':>14s}{'flatten/s':>11s}{'RQ-T/s':>8s}{'1/f_r':>9s}")
        comp = {}
        for name, r in rows.items():
            a = budget_arith(r["f_r"], r["N_q"], mins)
            comp[name] = a
            o(f"  {name:<26s}{a['f_r']:>7g}{a['N_q']:>5d}{a['tokens_per_s']:>8.0f}{a['context']:>14,.0f}{a['steps_flatten']:>11.0f}{a['steps_rq']:>8g}{a['floor_ms']:>8.1f} ms")
        o(f"  {C.GREY}arithmetic from the papers' settings (w06-data.json → paper_settings → budget): tok/s = f_r·N_q; ctx = tok/s·{mins}·60; flatten = f_r·N_q steps/s; RQ-Transformer = f_r steps/s; floor = 1/f_r{C.RESET}")
        if not self.budget or "rows" not in self.budget:
            self.budget = budget_text(o)
        bt = self.budget
        if bt.get("rows"):
            o(f"\n  text, {bt['tokenizer']}:")
            for key, r in bt["rows"].items():
                o(f"    {key:<10s} {r['n_tokens']:3d} tokens for {r['n_chars']} chars in {r['seconds_span']:.2f} s  →  {r['tokens_per_s']:.2f} tok/s  →  {mins} min ≈ {r['tokens_30min']:,.0f} tokens"
                  f"   (speech-only {r['seconds_speech']:.2f} s → {r['tokens_30min_speech_only']:,.0f})")
                o(f"    {C.GREY}{r['text']}{C.RESET}")
            m = bt["rows"].get("mandarin") or next(iter(bt["rows"].values()))
            o(f"  {C.GREY}speaking rate is extrapolated from one sentence each (the W1 Demo 2 sentence; the W3 sentence) — an order-of-magnitude figure, not a corpus statistic{C.RESET}")
            o(f"  {C.BOLD}Mimi {comp['mimi']['context']:,.0f} tokens  vs  text ≈ {m['tokens_30min']:,.0f}  →  ×{comp['mimi']['context'] / m['tokens_30min']:.0f};  EnCodec 6 kbps ×{comp['encodec_6kbps']['context'] / m['tokens_30min']:.0f}{C.RESET}")
        self.log["budget"] = {"codecs": comp, "text": {k: v for k, v in bt.items()}}
        if "rows" in self.log["budget"]["text"]:
            self.log["budget"]["text"]["rows"] = {k: {kk: vv for kk, vv in v.items() if kk != "pieces"} for k, v in bt["rows"].items()}

    def key_table(self) -> None:
        o = self.out
        o.banner("t  slide p37 table (fill by hand) · p46 text row", C.BLUE)
        K = CFG["sentence"]["max_stage_plot"]
        o(f"  {'':<28s}{'codes used / V  (stage 1 … N_q)':<44s}{'H1/log2 V  (stage 1 … N_q)':<40s}{'residual after 1 / 2 / 4 / N_q':<36s}{'H2/H1 st.1'}")
        for key in CODECS:
            if key not in self.stats:
                continue
            st = self.stats[key]["stages"]
            pick = [s for s in st if s["stage"] in (1, 2, 4, 8) or s["stage"] == len(st)]
            use = " ".join(f"{s['usage_ratio']:.2f}" for s in pick)
            ent = " ".join(f"{s['H1_norm']:.2f}" for s in pick)
            res = " | ".join(f"{name[:3]} " + "/".join(f"{v:.2f}" for v in r["after"].values()) for name, r in self.stats[key]["residual"].items())
            o(f"  {LABEL[key]:<28s}{use:<44s}{ent:<40s}{res:<36s}{st[0]['H2_over_H1']:.2f}  (shuf {st[0]['H2_over_shuffled']:.2f})")
        o(f"  {C.GREY}stages picked: 1, 2, 4, 8 and N_q{C.RESET}")
        if self.budget.get("rows"):
            for key, r in self.budget["rows"].items():
                o(f"  text ({key}): {r['tokens_30min']:,.0f} tokens in {CFG['budget']['minutes']} min ({r['tokens_per_s']:.2f} tok/s, {r['tokens_per_char']:.2f} tok/char)")

    def menu(self) -> None:
        o = self.out
        o(f"\n{C.BOLD}p37／p46{C.RESET}  1 token 矩陣（[ ] 移窗）   2 使用率與熵   3 殘差曲線   4 bigram H2/H1   5 算帳   t 投影片的表   h 選單   q 離開")

    def dispatch(self, k: str) -> bool:
        if k == "1":
            self.key_matrix()
        elif k == "[":
            self.win0 -= 1; self.key_matrix(move=-1)
        elif k == "]":
            self.win0 += 1; self.key_matrix(move=1)
        elif k == "2":
            self.key_usage()
        elif k == "3":
            self.key_residual()
        elif k == "4":
            self.key_bigram()
        elif k == "5":
            self.key_budget()
        elif k == "t":
            self.key_table()
        elif k == "h":
            self.menu()
        elif k == "q":
            return False
        return True

    def on_key(self, ev) -> None:
        if not self.dispatch(ev.key):
            self.plt.close(self.fig)

    # ── 彩排 ──
    def run_all(self) -> None:
        self.key_matrix(); self.key_usage(); self.key_residual(); self.key_bigram(); self.key_budget(); self.key_table()

    def record(self) -> dict:
        rec = dict(self.log)
        rec.update({"codecs": self.info, "stats": self.stats, "budget": self.log.get("budget", {}),
                    "sentence": {**self.log["sentence"], "codes": {k: v.tolist() for k, v in self.sentence.items()}}})
        return rec

    def save(self, folder: Path | None = None) -> Path:
        folder = folder or self.folder
        folder.mkdir(parents=True, exist_ok=True)
        stem = "rehearsal" if folder.name == "rehearsal" else "session"
        (folder / f"{stem}.json").write_text(json.dumps(self.record(), ensure_ascii=False, indent=1, default=float), encoding="utf-8")
        (folder / f"{stem}.txt").write_text(self.out.text(), encoding="utf-8")
        if self.fig is not None:
            self.fig.savefig(folder / "figure.png", dpi=110)
        C.note(f"  存到 {folder}/（{stem}.json、{stem}.txt、figure.png）")
        return folder / f"{stem}.json"


# ── build ────────────────────────────────────────────────────
def build(cmd: str, out: Out, device: str | None = None) -> Demo:
    if cmd == "fake":
        codecs = {k: FakeLatents(k) for k in CODECS}
        d = Demo(out, tag="FAKE")
        # fake 不碰 ../_media：子集用合成的短句，當場算
        rng = np.random.default_rng(CFG["fake"]["seed"])
        utts = load_subset()[:60] if (HERE / CFG["subset"]["list"]).resolve().exists() else [{"id": f"fake-{i}", "dur_s": 3.0} for i in range(60)]
        for key, cd in codecs.items():
            codes_list, offsets, paths_acc = [], [0], {}
            for u in utts:
                x = np.zeros(int(u["dur_s"] * FS))
                codes, paths = cd.encode(x)
                codes_list.append(codes); offsets.append(offsets[-1] + codes.shape[1])
                for name, e in paths.items():
                    a = paths_acc.setdefault(name, {"stage_offset": e["stage_offset"], "energy_z": [], "energy_r": []})
                    a["energy_z"].append(e["energy_z"]); a["energy_r"].append(e["energy_r"] * (1 + 0.1 * rng.standard_normal()))
            for v in paths_acc.values():
                v["energy_z"] = np.asarray(v["energy_z"]); v["energy_r"] = np.stack(v["energy_r"])
            info = cd.info(); info.update({"n_utts": len(utts), "seconds": float(sum(u["dur_s"] for u in utts))})
            d.info[key] = info
            d.stats[key] = compute_stats(np.concatenate(codes_list, axis=1), np.asarray(offsets), cd.V, key, cd.n_semantic, paths_acc)
            d.sentence[key], _ = cd.encode(np.zeros(int(5.9 * FS)))
            d.log["sentence"]["T"][key] = int(d.sentence[key].shape[1])
        d.log["subset"] = {"n_utts": len(utts), "seconds": float(sum(u["dur_s"] for u in utts)), "source": "FAKE"}
        return d
    try:
        import torch  # noqa: F401
    except ImportError:
        sys.exit("沒有 torch：先跑 ../w03_d3_blank_heatmap/setup.sh；或 ./present.sh fake 演練流程")
    dev = device or CFG["infer"]["device"]
    codecs = {}
    for k in CODECS:
        t0 = time.perf_counter()
        try:
            codecs[k] = Latents(k, dev)
            c = codecs[k].codec
            out(f"  載入 {c.label:<16s} {c.n_params / 1e6:6.1f} M 參數   {c.device.type}   {1000 * (time.perf_counter() - t0):5.0f} ms   f_r {c.frame_rate_cfg:g} Hz · V {c.codebook_size} · N_q {c.n_q_cfg}")
        except Exception as e:
            out(f"  {C.RED}{LABEL[k]} 載入失敗：{e!r}{C.RESET}（跳過；先在 ../w06_d1_layer_listen 跑 ./present.sh check）")
    if not codecs:
        sys.exit("三個 codec 都載不起來")
    d = Demo(out, tag="rehearsal" if cmd == "rehearse" else "")
    d.from_codecs(codecs, do_extract=True)
    return d


def build_replay(out: Out) -> Demo:
    rec_path = HERE / "runs" / "rehearsal" / "rehearsal.json"
    if not rec_path.exists():
        sys.exit("沒有 runs/rehearsal/rehearsal.json：先 ./present.sh rehearse")
    rec = json.loads(rec_path.read_text(encoding="utf-8"))
    d = Demo(out, tag="replay " + rec["time"])
    d.from_replay(rec)
    out(f"  replay：{rec['time']} 的彩排，{len(d.stats)} 個 codec、子集 {rec['subset'].get('n_utts')} 句")
    return d


# ── check ────────────────────────────────────────────────────
def check(device: str | None = None) -> None:
    C.banner("課前體檢", C.AMBER)
    res = {"time": C.now(), "versions": versions(), "codecs": {}}
    for k, v in res["versions"].items():
        print(f"  {k:<16s} {v}")
    try:
        import torch
    except ImportError:
        sys.exit("沒有 torch：先跑 ../w03_d3_blank_heatmap/setup.sh")
    dev = device or CFG["infer"]["device"]
    x = sentence_wav()
    for k in CODECS:
        print(f"\n  {C.BOLD}{LABEL[k]}{C.RESET}")
        r = {}
        try:
            L = Latents(k, dev)
            r["info"] = L.info()
            t0 = time.perf_counter()
            codes, paths = L.encode_check(x)
            r["encode_ms"] = 1000 * (time.perf_counter() - t0)
            r["N_q"], r["T"] = int(codes.shape[0]), int(codes.shape[1])
            r["paths"] = {n: {"stage_offset": p["stage_offset"], "n_stages": int(p["contrib"].shape[0]), "dim": int(p["z"].shape[0])} for n, p in paths.items()}
            print(f"    codes {codes.shape}   {r['encode_ms']:.0f} ms   paths: " + ", ".join(f"{n}: {v['n_stages']} stages × {v['dim']}-d" for n, v in r["paths"].items()))
            # (a) 各層 codeword 累加 == quantizer 自己的 decode
            off = L.official_quantized(codes); mine = L.mine_quantized(paths)
            diff = float((off - mine).abs().max()); scale = float(off.abs().max())
            r["sum_vs_decode_maxabs"], r["sum_vs_decode_rel"] = diff, diff / max(scale, 1e-12)
            print(f"    Σ codeword vs quantizer.decode: max |diff| {diff:.2e} (rel {diff / max(scale, 1e-12):.1e})   {'OK' if diff / max(scale, 1e-12) < 1e-4 else C.RED + '不同' + C.RESET}")
            # (b) 從殘差重推 code
            agree = L.rederive_codes(paths)
            r["rederive_agree"] = agree
            for n, a in agree.items():
                print(f"    re-derive codes from residuals, {n}: " + " ".join(f"{v:.3f}" for v in a[:12]) + (" …" if len(a) > 12 else "") + f"   min {min(a):.3f}")
            # (c) 殘差能量
            en = L.energies(paths)
            r["residual_ratios"] = {n: (e["energy_r"] / e["energy_z"]).tolist() for n, e in en.items()}
            for n, v in r["residual_ratios"].items():
                print(f"    residual ratios, {n}: " + " ".join(f"{u:.3f}" for u in v[:8]) + (" …" if len(v) > 8 else ""))
                mono = all(a >= b - 1e-9 for a, b in zip(v, v[1:]))
                r.setdefault("monotone", {})[n] = mono
                if not mono:
                    print(f"    {C.AMBER}殘差不是單調遞減（{n}）——看哪一層{C.RESET}")
        except Exception as e:
            r["error"] = repr(e)
            print(f"    {C.RED}{e!r}{C.RESET}")
        res["codecs"][k] = r
    print(f"\n  {C.BOLD}tokenizer{C.RESET}")
    out = Out()
    bt = budget_text(out)
    res["tokenizer"] = {k: v for k, v in bt.items() if k != "rows"}
    if bt.get("rows"):
        res["tokenizer"]["rows"] = {k: {kk: vv for kk, vv in v.items() if kk != "pieces"} for k, v in bt["rows"].items()}
        for k, v in bt["rows"].items():
            print(f"    {k}: {v['n_tokens']} tokens / {v['n_chars']} chars   {' | '.join(v['pieces'])}")
    (HERE / "runs").mkdir(exist_ok=True)
    (HERE / "runs" / "check.json").write_text(json.dumps(res, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    print(f"\n  → runs/check.json")


# ── export ───────────────────────────────────────────────────
def export(rec_path: Path) -> None:
    rec = json.loads(rec_path.read_text(encoding="utf-8"))
    d = json.loads(METRICS_JSON.read_text(encoding="utf-8")) if METRICS_JSON.exists() else {
        "_note": "量測數字（進版控的出處，speech_ai/CLAUDE.md 4.8）。w06-data.json 是 make_figs_w06.py 算出來的、每次重產；量測的放這裡。"}
    demo2 = {"source": "demos/w06_d2_token_matrix/runs/rehearsal/rehearsal.json", "time": rec["time"], "tag": rec.get("tag", ""),
             "subset": rec["subset"], "sentence": {k: v for k, v in rec["sentence"].items() if k != "codes"},
             "definitions": {"usage_ratio": "distinct codes seen on the whole subset ÷ V, per stage",
                             "H1_norm": "unigram entropy of the stage's codes (bits) ÷ log2 V; perplexity = 2^H1",
                             "repeat_rate": "P(c_t = c_{t-1}) over within-sentence adjacent frames; mean_run = frames ÷ runs",
                             "H2_bits": "plug-in H(c_t | c_{t-1}) over within-sentence adjacent pairs; H2_shuffled = the same after shuffling each sentence (small-sample bias)",
                             "residual.ratios[i-1]": "Σ_t‖z_t − Σ_{j≤i} q_{t,j}‖² ÷ Σ_t‖z_t‖² over the whole subset; z and q in the quantizer's own space (EnCodec 128-d encoder output; Mimi after each path's input_proj, semantic and acoustic separately; DAC 1024-d with out_proj)",
                             "budget.codecs": "arithmetic from paper settings (w06-data.json)", "budget.text": "tokens of one measured sentence ÷ its seconds × 1800 s (Gemma 4 tokenizer); extrapolated from one sentence"},
             "codecs": {}}
    for k, st in rec["stats"].items():
        demo2["codecs"][k] = {"info": rec["codecs"][k], "N_q": st["N_q"], "V": st["V"], "n_frames": st["n_frames"],
                              "stages": [{kk: (round(vv, 4) if isinstance(vv, float) else vv) for kk, vv in s.items()} for s in st["stages"]],
                              "residual": {n: {"stage_offset": r["stage_offset"], "n_stages": r["n_stages"], "ratios": [round(v, 5) for v in r["ratios"]],
                                               "after": {a: round(b, 5) for a, b in r["after"].items()}} for n, r in st["residual"].items()}}
    demo2["budget"] = rec.get("budget", {})
    d["demo2"] = demo2
    METRICS_JSON.parent.mkdir(parents=True, exist_ok=True)
    METRICS_JSON.write_text(json.dumps(d, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"  → {METRICS_JSON}（demo2）")


# ── selftest ─────────────────────────────────────────────────
def selftest() -> None:
    C.banner("selftest", C.AMBER)
    fails = []

    def ok(cond: bool, msg: str):
        print(f"  {'OK ' if cond else C.RED + 'FAIL' + C.RESET}  {msg}")
        if not cond:
            fails.append(msg)

    rng = np.random.default_rng(7)
    # 1. unigram：均勻 → H1 = log2 V；單一 code → 0
    V = 64
    u = unigram(np.tile(np.arange(V), 50), V)
    ok(abs(u["H1_norm"] - 1) < 1e-12 and u["codes_used"] == V, "unigram 均勻：H1/log2 V = 1、全部用到")
    u = unigram(np.zeros(100, dtype=int), V)
    ok(u["H1_bits"] == 0 and u["codes_used"] == 1, "unigram 單一 code：H1 = 0")
    # 2. repeat／run
    r = repeat_stats([np.array([1, 1, 1, 2, 2, 3]), np.array([5, 5])])
    ok(abs(r["repeat_rate"] - 4 / 6) < 1e-12 and abs(r["mean_run"] - 8 / 4) < 1e-12, f"repeat_stats：rate {r['repeat_rate']:.3f}、mean run {r['mean_run']:.2f}")
    # 3. bigram：週期序列 H2 = 0；iid 均勻大樣本 H2 ≈ H1；打亂的基線 ≈ H2 本身
    per = [np.tile(np.array([0, 1, 2, 3]), 200)]
    ok(abs(bigram_cond_entropy(per, V)) < 1e-12, "bigram 週期序列：H2 = 0")
    iid = [rng.integers(0, 8, 20000)]
    H1 = unigram(iid[0], 8)["H1_bits"]; H2 = bigram_cond_entropy(iid, 8)
    ok(abs(H2 - H1) < 0.01, f"bigram iid 均勻：H2 {H2:.3f} ≈ H1 {H1:.3f}")
    sticky = [np.array([x for x in np.repeat(rng.integers(0, 8, 500), 4)])]
    H2s = bigram_cond_entropy(sticky, 8); H2sh = shuffled_bigram(sticky, 8, 3, 1)
    ok(H2s < 0.5 * H2sh, f"bigram 黏的序列：H2 {H2s:.3f} 遠低於打亂基線 {H2sh:.3f}")
    # 4. 純 numpy RVQ：greedy encode → residual_energies 單調遞減、最後一層殘差 = z − Σq、從殘差重推的 code 一致
    D, T, K, Vq = 16, 300, 6, 32
    z = rng.standard_normal((D, T))
    cbs = [rng.standard_normal((Vq, D)) * (0.7 ** i) for i in range(K)]
    resid = z.copy(); codes = []; contribs = []
    for i in range(K):
        idx = np.argmin(((resid.T[:, None, :] - cbs[i][None]) ** 2).sum(-1), axis=1)
        q = cbs[i][idx].T
        codes.append(idx); contribs.append(q); resid = resid - q
    e_z, e_r = residual_energies(z, np.stack(contribs))
    ok(all(a >= b for a, b in zip(e_r, e_r[1:])), f"numpy RVQ：殘差能量單調遞減 {[round(v / e_z, 3) for v in e_r]}")
    ok(abs(e_r[-1] - np.sum(resid ** 2)) < 1e-9, "numpy RVQ：最後一層殘差能量 = ‖z − Σq‖²")
    # 5. compute_stats 走一遍（兩句）
    codes_all = np.stack(codes); offsets = np.array([0, 120, T])
    st = compute_stats(codes_all, offsets, Vq, "encodec", 0, {"rvq": {"stage_offset": 0, "energy_z": np.array([e_z]), "energy_r": e_r[None]}})
    ok(len(st["stages"]) == K and abs(st["residual"]["rvq"]["ratios"][-1] - e_r[-1] / e_z) < 1e-12 and st["residual"]["rvq"]["after"]["1"] == st["residual"]["rvq"]["ratios"][0],
       "compute_stats：K 層、殘差比對得上、after 表")
    ok(stage_label("mimi", 0, 1) == "S" and stage_label("mimi", 1, 1) == "A1" and stage_label("encodec", 3, 0) == "4", "stage_label：S／A1／4")
    # 6. 算帳對 w06-data.json
    rows = budget_rows()
    bad = []
    for name, r in rows.items():
        a = budget_arith(r["f_r"], r["N_q"], 30)
        if abs(a["tokens_per_s"] - r["tokens_per_s"]) > 1e-9 or abs(a["context"] - r["context_30min"]) > 1e-6 or abs(a["steps_rq"] - r["steps_rq"]) > 1e-9 or abs(a["floor_ms"] - r["floor_ms"]) > 0.05:
            bad.append(name)
    ok(not bad, f"算帳對 w06-data.json → budget：{len(rows)} 列" + (f"，不一致 {bad}" if bad else ""))
    src = text_sources()
    ok("mandarin" in src and src["mandarin"]["n_chars"] == len(src["mandarin"]["text"]) and src["mandarin"]["seconds_span"] > src["mandarin"]["seconds_speech"] > 0,
       f"文字來源：華語 {src.get('mandarin', {}).get('n_chars')} 字、{src.get('mandarin', {}).get('seconds_span', 0):.2f} s" if "mandarin" in src else "文字來源：找不到 W1 Demo 2 的 result.json")
    ok("english" in src and src["english"]["text"][0].isupper() and src["english"]["text"].endswith("."), f"文字來源：英語 {src.get('english', {}).get('n_words')} 字、{src.get('english', {}).get('seconds_span', 0):.2f} s")
    # 7. fake 流程（不開視窗）
    import matplotlib
    matplotlib.use("Agg")
    out = Out()
    d = build("fake", out)
    d.folder = HERE / "runs" / "live" / "selftest.fake"
    ok(set(d.stats) == set(CODECS) and all(len(d.stats[k]["stages"]) == d.stats[k]["N_q"] for k in CODECS), "fake：三個 codec 的統計")
    s1 = d.stats["encodec"]["stages"]
    ok(s1[0]["repeat_rate"] > s1[-1]["repeat_rate"], f"fake：第一層重複率 {s1[0]['repeat_rate']:.2f} > 最後一層 {s1[-1]['repeat_rate']:.2f}")
    ok(set(d.stats["mimi"]["residual"]) == {"semantic", "acoustic"}, "fake：Mimi 兩路")
    d.make_fig()
    d.key_matrix(); d.dispatch("]"); d.dispatch("["); d.key_usage(); d.key_residual(); d.key_bigram(); d.key_table()
    # 算帳：tokenizer 可能不在（容器）——不算失敗
    d.key_budget()
    p = d.save()
    rec = json.loads(p.read_text(encoding="utf-8"))
    ok("stats" in rec and "codes" in rec["sentence"] and (d.folder / "figure.png").exists(), "fake：存檔（json、圖）")
    d2 = Demo(Out(), tag="replay-test"); d2.from_replay(rec)
    ok(d2.stats["mimi"]["stages"][0]["H1_bits"] == d.stats["mimi"]["stages"][0]["H1_bits"] and d2.sentence["dac"].shape == d.sentence["dac"].shape, "fake：replay 讀回同樣的數字")
    shutil.rmtree(d.folder, ignore_errors=True)
    # 8. 有彩排紀錄：對 w06-metrics.json
    rec_path = HERE / "runs" / "rehearsal" / "rehearsal.json"
    if rec_path.exists() and METRICS_JSON.exists():
        rec = json.loads(rec_path.read_text(encoding="utf-8"))
        mj = json.loads(METRICS_JSON.read_text(encoding="utf-8")).get("demo2", {})
        n_bad = 0
        for k, st in rec["stats"].items():
            e = mj.get("codecs", {}).get(k)
            if not e:
                n_bad += 1; continue
            for s, es in zip(st["stages"], e["stages"]):
                if abs(s["H1_norm"] - es["H1_norm"]) > 1e-3 or s["codes_used"] != es["codes_used"]:
                    n_bad += 1
            for n, r in st["residual"].items():
                if abs(r["ratios"][-1] - e["residual"][n]["ratios"][-1]) > 1e-4:
                    n_bad += 1
        ok(n_bad == 0, f"彩排紀錄對 w06-metrics.json → demo2：{n_bad} 處不一致")
    else:
        print("  --   沒有彩排紀錄或 w06-metrics.json，跳過比對")
    print(f"\n  {'全部通過' if not fails else str(len(fails)) + ' 項失敗'}")
    sys.exit(1 if fails else 0)


# ── main ─────────────────────────────────────────────────────
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["show", "extract", "rehearse", "replay", "check", "fake", "selftest", "export"])
    ap.add_argument("--device", help="show／extract／rehearse／check：蓋過 demo_config.toml 的 device（cpu／mps／auto）")
    ap.add_argument("--force", action="store_true", help="extract：快取已有也重算")
    ap.add_argument("--no-plot", action="store_true", help="show／replay／fake：只在終端機，不開圖")
    args = ap.parse_args()

    if args.cmd == "check":
        check(args.device); return
    if args.cmd == "selftest":
        selftest(); return
    if args.cmd == "export":
        export(HERE / "runs" / "rehearsal" / "rehearsal.json"); return

    if args.cmd in ("rehearse", "extract") or args.no_plot:
        import matplotlib
        matplotlib.use("Agg")

    out = Out()
    out.banner("W6 Demo 2：what the letters look like — token matrix, usage, residual, the bill")
    if args.cmd == "extract":
        try:
            import torch  # noqa: F401
        except ImportError:
            sys.exit("沒有 torch：先跑 ../w03_d3_blank_heatmap/setup.sh")
        codecs = {k: Latents(k, args.device or CFG["infer"]["device"]) for k in CODECS}
        extract(codecs, out, only_missing=not args.force)
        return
    d = build_replay(out) if args.cmd == "replay" else build(args.cmd, out, args.device)

    if args.cmd == "rehearse":
        d.make_fig()
        d.run_all()
        p = d.save(HERE / "runs" / "rehearsal")
        export(p)
        print("\n下一步：./present.sh 自己按一遍 1 ] ] 2 3 4 5 t；把 t 印的數字填進 p37 的表格與 p46 文字那一列（講稿【提示】記出處與日期）；"
              "python3 slides/scripts/make_figs_w06.py 重畫 p22 的殘差曲線（讀 w06-metrics.json → demo2，標 measured）。")
        return

    d.make_fig()
    d.menu()
    if args.no_plot:
        C.note("  --no-plot：鍵在終端機按。")
        while d.dispatch(C.getkey()):
            pass
    else:
        C.note("  鍵在視窗裡按（視窗要有焦點）。終端機這裡是紀錄。")
        d.plt.show()
    if args.cmd == "show":
        d.save()


if __name__ == "__main__":
    main()

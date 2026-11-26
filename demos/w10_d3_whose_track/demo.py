# -*- coding: utf-8 -*-
# 此檔由 bin/publish_demos.py 從 private repo 的 <course>/demos/ 複製而來，請勿在公開 repo 直接編輯。
"""W10 demo 3：兩個人同時說——哪一軌是使用者；同一份輸出、三個 DER。

  show       課堂：啟動時全部算好（分離、兩個 encoder、Whisper、diarization），單鍵操作（下面的鍵）→ runs/live/
  rehearse   課前：同樣算一遍 → runs/rehearsal/（進版控 = 現場的退路），寫 w10-metrics.json → demo3
             --device cpu  全部放 CPU     --no-diar  不跑 pyannote（還沒拿到 token 時）
  replay     現場模型出狀況時：只讀彩排紀錄，同樣的鍵，不碰 torch；--from runs/live 讀別的資料夾
  check      課前體檢：版本、模型在不在、分離模型在 cpu／mps 上跑不跑與要多久、兩軌的極性、
             pyannote 在 mps 與 cpu 上的時間戳一不一致、der.py 對 pyannote.metrics → runs/check.json
  fetch      課前一週：從 dev-clean.tar.gz 解出要用的句子；下載 SepFormer、ECAPA、pyannote 的 pipeline（要網路；pyannote 要 token）
  fake       沒有模型也能演練流程（替身；畫面標明 FAKE，數字沒有意義）
  selftest   不需要模型：接對話、切塊、SI-SDR、DER（對投影片 p86 的玩具）、fake 整條流程
  export     runs/rehearsal/ 的摘要 → slides/assets/w10/w10-metrics.json（鍵 demo3）

畫面上的鍵（順序照投影片 p78、p87 講稿的【操作】）：
  1  完全重疊的那一段：播混合的、再播分出來的兩軌        p  播「對話」那一段（混合的）
  2  「對話」切成一塊一塊：每一塊的兩軌各自對註冊語音的 SIM（encoder A）      3  同一張表，encoder B
  o  揭曉：每一塊裡使用者其實在哪一軌（對乾淨的答案算的）
  n  播「每一塊都取第一軌」接起來的樣子
  4  照 SIM 挑軌、接起來、送 Whisper：幾種接法的 WER 與逐字稿        l  播照 SIM 挑軌接起來的
  5  使用者沒說話的那一塊：SIM 仍然挑了一軌——播它        a  門檻：兩個 encoder 各自擋不擋得住
  c  換塊的長度        x  整段不切、一次處理        g  換另一位說話人
  6  diarization：參考、系統輸出、三種錯的著色        7  同一份輸出、三個 DER        k  換成土法聚類的輸出
  t  投影片上那兩張表     ?  選單     q  離開

定義（課堂上要念出來的）：
  SIM        兩段聲音各自過 speaker encoder 得到的向量的 cosine（W9 p18）。一邊永遠是註冊語音。
  「其實在哪一軌」  因為混合是自己接的，使用者那一軌乾淨的答案還在：對它算 SI-SDR，高的那一軌。
  參考標註    每位說話人乾淨的那一軌、每 10 ms 一格的能量門檻（mix.activity）；不是人標的。
  DER        (FA + miss + conf) / 參考標註裡的語音總長；der.py。collar = 每個參考邊界前後各 0.25 秒不計。
"""
from __future__ import annotations

import argparse
import importlib.util
import io
import json
import os
import re
import subprocess
import sys
import tempfile
import time
import tomllib
import warnings
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
import common as C   # noqa: E402
import der as D      # noqa: E402
import dsp           # noqa: E402
import mix as M      # noqa: E402

CFG = tomllib.loads((HERE / "demo_config.toml").read_text(encoding="utf-8"))
FS = int(CFG["audio"]["fs"])
MIXC, SEP, ENC, DIAR, NAIVE = CFG["mix"], CFG["sep"], CFG["enc"], CFG["diar"], CFG["naive"]
PAIRS = {p["id"]: p for p in CFG["pair"]}
MAIN = MIXC["main"]
MODEL_LOCK = HERE / "model.lock"
SENTENCES = HERE / "data" / "sentences.json"
REHEARSAL = HERE / "runs" / "rehearsal"
LIVE = HERE / "runs" / "live"
METRICS_JSON = HERE.parents[1] / "slides" / "assets" / "w10" / "w10-metrics.json"     # parents[1] = speech_ai/
DATA_JSON = HERE.parents[1] / "slides" / "assets" / "w10" / "w10-data.json"
WORKER = HERE / "diar_worker.py"
ENCS = ("a", "b")
CHUNKINGS = [float(SEP["chunk_s"])] + [float(c) for c in SEP["alt_chunks"]]
COLLAR = float(CFG["der"]["collar"])
SCORINGS = [(n, (COLLAR if c else 0.0), s) for n, c, s in D.SCORINGS]
VERSIONS = [("mix", "the mixture, untouched"), ("track1", "track 1 of every chunk"), ("pick_a", f"picked by SIM, {ENC['a']['label']}"),
            ("pick_b", f"picked by SIM, {ENC['b']['label']}"), ("gate_a", f"… and silenced below {ENC['a']['threshold']:g}"),
            ("gate_b", f"… and silenced below {ENC['b']['threshold']:g}"), ("truth", "the track the user is really in"),
            ("clean", "the user's own clean track")]
SPK_COLOR = {"user": "#2a7fc8", "other": "#c8742a"}
ERR_COLOR = {"miss": "#c8742a", "fa": "#2a7fc8", "conf": "#c0392b"}
SEP_PATTERNS = ["hyperparams.yaml", "encoder.ckpt", "decoder.ckpt", "masknet.ckpt"]
ECAPA_PATTERNS = ["hyperparams.yaml", "embedding_model.ckpt", "mean_var_norm_emb.ckpt", "classifier.ckpt", "label_encoder.txt"]
WAVLM_PATTERNS = ["*.json", "*.txt", "*.safetensors"]


# ── 小工具 ──────────────────────────────────────────────────────────────
def load_sibling(rel: str, name: str):
    p = (HERE / rel / "demo.py").resolve()
    if not p.exists():
        sys.exit(f"找不到 {os.path.relpath(p, HERE)}（demo_config.toml 指過去的那個 demo）")
    spec = importlib.util.spec_from_file_location(name, p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


W2 = load_sibling(CFG["whisper"]["from"], "w02_d3_demo")     # Whisper、WER 的函數、解 tar 的寫法


def _p(rel: str) -> Path:
    return (HERE / rel).resolve()


def ck(chunk_s: float | str) -> str:
    return chunk_s if isinstance(chunk_s, str) else f"{chunk_s:g}"


def f1(v, nd: int = 1) -> str:
    return "—" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:.{nd}f}"


def f3(v) -> str:
    return "—" if v is None else f"{v:+.3f}"


def pct(v) -> str:
    return "—" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{100 * v:.1f}"


def clean_num(v):
    """json 用：nan → None、numpy 的數 → Python 的數。"""
    if isinstance(v, (np.floating, float)):
        return None if np.isnan(v) else float(v)
    if isinstance(v, (np.integer,)):
        return int(v)
    return v


def read_lock() -> dict[str, str]:
    lock = {}
    paths = [MODEL_LOCK, _p(ENC["b"]["lock_from"])]
    for p in paths:
        if not p.exists():
            continue
        for line in p.read_text(encoding="utf-8").splitlines():
            parts = line.split()
            if len(parts) == 2 and not line.startswith("#"):
                lock.setdefault(parts[0], parts[1])
    return lock


def model_dir(repo: str, patterns: list[str] | None = None, offline: bool = True) -> str:
    from huggingface_hub import snapshot_download
    return snapshot_download(repo, revision=read_lock().get(repo), local_files_only=offline, allow_patterns=patterns)


def resolve_device(d: str) -> str:
    import torch
    if d == "auto":
        return "mps" if torch.backends.mps.is_available() else "cpu"
    return d


def cos(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def write_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    tmp.replace(path)


class Out:
    def __init__(self, echo: bool = True):
        self.buf, self.echo = io.StringIO(), echo

    def __call__(self, s: str = "") -> None:
        if self.echo:
            print(s, flush=True)
        self.buf.write(re.sub(r"\033\[[0-9;]*m", "", s) + "\n")

    def banner(self, s: str, color: str = C.BLUE) -> None:
        if self.echo:
            C.banner(s, color)
        self.buf.write(f"\n{'─' * 72}\n{s}\n{'─' * 72}\n")

    def text(self) -> str:
        return self.buf.getvalue()


# ── 模型 ────────────────────────────────────────────────────────────────
class Separator:
    """SpeechBrain 的 SepformerSeparation。一段 16 kHz 的混合 → 兩軌 16 kHz。

    模型輸出的音量是任意的（訓練目標是 SI-SNR，對音量與正負號都不敏感），所以兩軌各乘一個增益：
    讓 g1·軌1 + g2·軌2 最貼近混合訊號的那一組（最小平方）。g 可以是負的——這份權重的輸出有一軌極性是反的；
    SI-SDR 與 speaker encoder 都不看極性，但要把兩軌加回去比、或接起來聽，極性得先擺正。
    """

    def __init__(self, device: str = "auto"):
        import torch
        from speechbrain.inference.separation import SepformerSeparation
        self.torch = torch
        self.device = resolve_device(device)
        self.fs = int(SEP["fs"])
        path = model_dir(SEP["repo"], SEP_PATTERNS)
        self._tmp = tempfile.mkdtemp(prefix="w10_d3_sep_")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.m = SepformerSeparation.from_hparams(source=path, savedir=self._tmp, run_opts={"device": self.device})
        self.n_params = sum(p.numel() for p in self.m.mods.parameters())
        self.last_gains = (1.0, 1.0)

    def raw(self, y: np.ndarray) -> np.ndarray:
        """(n,) 16 kHz → (n, 2) 16 kHz，沒有調音量。"""
        torch = self.torch
        z = dsp.resample(y, FS, self.fs) if self.fs != FS else y
        with torch.no_grad(), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            w = self.m.separate_batch(torch.tensor(z[None], dtype=torch.float32, device=self.device))[0].float().cpu().numpy().astype(np.float64)
        cols = []
        for j in range(w.shape[1]):
            e = dsp.resample(w[:, j], self.fs, FS) if self.fs != FS else w[:, j]
            cols.append(np.pad(e, (0, max(0, len(y) - len(e))))[: len(y)])
        return np.stack(cols, axis=1)

    def separate(self, y: np.ndarray, srcs=None) -> list[np.ndarray]:
        e = self.raw(y)
        g = np.linalg.lstsq(e, y, rcond=None)[0]
        self.last_gains = (float(g[0]), float(g[1]))
        return [e[:, 0] * g[0], e[:, 1] * g[1]]


class Ecapa:
    def __init__(self, device: str = "cpu"):
        import torch
        from speechbrain.inference.speaker import EncoderClassifier
        self.torch = torch
        self.device = resolve_device(device)
        src = model_dir(ENC["a"]["repo"], ECAPA_PATTERNS)
        self._tmp = tempfile.mkdtemp(prefix="w10_d3_ecapa_")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # hyperparams.yaml 裡的 pretrained_path 是 Hub 上的 repo 名；蓋成本機快照，離線時才不會另外向 Hub 要
            self.m = EncoderClassifier.from_hparams(source=src, savedir=self._tmp, overrides={"pretrained_path": src}, run_opts={"device": self.device})
        self.m.eval()
        self.n_params = sum(p.numel() for p in self.m.mods.embedding_model.parameters())

    def embed(self, x: np.ndarray) -> np.ndarray:
        torch = self.torch
        with torch.no_grad(), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            e = self.m.encode_batch(torch.tensor(x, dtype=torch.float32)[None].to(self.device))
        return e.detach().float().cpu().numpy().reshape(-1).astype(np.float64)


class WavLMSV:
    """W9 Demo 1 的 SIM 裁判，同一份權重、同一個寫法。"""

    def __init__(self, device: str = "auto"):
        import torch
        from transformers import Wav2Vec2FeatureExtractor, WavLMForXVector
        W2.quiet_transformers()
        self.torch = torch
        self.device = resolve_device(device)
        path = model_dir(ENC["b"]["repo"], WAVLM_PATTERNS)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.fe = Wav2Vec2FeatureExtractor.from_pretrained(path)
            self.model = WavLMForXVector.from_pretrained(path).eval().to(self.device)
        self.n_params = sum(p.numel() for p in self.model.parameters())

    def embed(self, x: np.ndarray) -> np.ndarray:
        torch = self.torch
        inp = self.fe([x.astype(np.float32)], sampling_rate=FS, return_tensors="pt", padding=True)
        with torch.no_grad(), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            e = self.model(**{k: v.to(self.device) for k, v in inp.items()}).embeddings
        return e.detach().float().cpu().numpy().reshape(-1).astype(np.float64)


class Real:
    """真的模型，用到才載。"""
    fake = False

    def __init__(self, device: str | None = None):
        self.device = device
        self._m: dict = {}

    def _get(self, name: str, make):
        if name not in self._m:
            t0 = time.perf_counter()
            self._m[name] = make()
            C.note(f"  載入 {name}：{time.perf_counter() - t0:.1f} s")
        return self._m[name]

    @property
    def sep(self) -> Separator:
        return self._get("sep", lambda: Separator(self.device or SEP["device"]))

    def separate(self, y: np.ndarray, srcs=None) -> list[np.ndarray]:
        return self.sep.separate(y)

    def embed(self, k: str, x: np.ndarray) -> np.ndarray:
        if k == "a":
            return self._get("enc_a", lambda: Ecapa(ENC["a"]["device"])).embed(x)
        return self._get("enc_b", lambda: WavLMSV(self.device or ENC["b"]["device"])).embed(x)

    def asr(self, x: np.ndarray, hint=None) -> str:
        wh = self._get("whisper", W2.Whisper)
        return wh.transcribe(wh.features_hf(x))["text"]

    def normalize(self, text: str) -> str:
        return self._get("whisper", W2.Whisper).normalize(text)

    def info(self) -> dict:
        lock = read_lock()
        d = {"sep": {"repo": SEP["repo"], "commit": lock.get(SEP["repo"]), "fs": int(SEP["fs"])},
             "enc_a": {"repo": ENC["a"]["repo"], "commit": lock.get(ENC["a"]["repo"]), "threshold": ENC["a"]["threshold"]},
             "enc_b": {"repo": ENC["b"]["repo"], "commit": lock.get(ENC["b"]["repo"]), "threshold": ENC["b"]["threshold"]},
             "whisper": {"repo": W2.CFG["model"]["repo"], "commit": W2.read_lock(), "device": W2.CFG["model"]["device"]},
             "diar": {"repo": DIAR["repo"], "commit": lock.get(DIAR["repo"])}}
        for k in ("sep", "enc_a", "enc_b"):
            if k in self._m:
                d[k]["device"], d[k]["n_params"] = self._m[k].device, self._m[k].n_params
        return d


class Fake:
    """沒有模型時的替身。分離：乾淨的答案各混一點對方、每一塊隨機對調；encoder：粗頻譜包絡；辨識器：照聲音裡使用者的成分留字。
    數字沒有意義，只用來演練流程與 selftest。"""
    fake = True

    def __init__(self):
        self.rng = np.random.default_rng(int(CFG["fake"]["seed"]))
        self.proj = {k: np.random.default_rng(i + 7).standard_normal((24, 16)) for i, k in enumerate(ENCS)}

    def separate(self, y: np.ndarray, srcs=None) -> list[np.ndarray]:
        u, o = srcs if srcs is not None else (y, np.zeros_like(y))
        e = [u + 0.1 * o, o + 0.1 * u]
        return e[::-1] if self.rng.random() < 0.5 else e

    def embed(self, k: str, x: np.ndarray) -> np.ndarray:
        n = 512
        if len(x) < n:
            x = np.pad(x, (0, n - len(x)))
        fr = np.lib.stride_tricks.sliding_window_view(x, n)[:: n // 2] * np.hanning(n)
        p = (np.abs(np.fft.rfft(fr, axis=1)) ** 2).mean(0)[:192].reshape(24, 8).mean(1)
        v = np.log(p + 1e-10)
        return (v - v.mean()) @ self.proj[k] + (0.3 if k == "b" else 0.0)

    def asr(self, x: np.ndarray, hint=None) -> str:
        if hint is None:
            return ""
        clean, text = hint
        words = text.split()
        n = min(len(x), len(clean))
        den = np.linalg.norm(x[:n]) * np.linalg.norm(clean[:n]) + 1e-12
        keep = float(np.clip(abs(np.dot(x[:n], clean[:n])) / den, 0, 1))
        return " ".join(w for i, w in enumerate(words) if (i * 0.37) % 1.0 < keep)

    def normalize(self, text: str) -> str:
        return W2.norm_simple(text)

    def info(self) -> dict:
        return {"fake": True}


# ── 素材 ────────────────────────────────────────────────────────────────
def wanted_utts() -> list[str]:
    u = [CFG["user"]["enrol"], CFG["user"]["overlap_utt"]]
    for p in CFG["pair"]:
        u.append(p["overlap_utt"])
        u += [t["utt"] for t in p["turns"]]
    return list(dict.fromkeys(u))


def wav_path(utt: str) -> Path:
    return _p(CFG["audio"]["wav_dir"]) / f"{utt}.wav"


def material() -> tuple[dict[str, np.ndarray], dict[str, str]]:
    if not SENTENCES.exists():
        sys.exit(f"沒有 {SENTENCES.relative_to(HERE)}，先跑 ./present.sh fetch（從 dev-clean.tar.gz 解出要用的句子）")
    texts = json.loads(SENTENCES.read_text(encoding="utf-8"))["texts"]
    audio = {}
    for u in wanted_utts():
        if not wav_path(u).exists() or u not in texts:
            sys.exit(f"沒有 {u} 的 wav 或參考文字，先跑 ./present.sh fetch")
        audio[u] = C.load_wav(wav_path(u), FS)
    return audio, texts


def fake_material() -> tuple[dict[str, np.ndarray], dict[str, str]]:
    """沒有 wav 的機器上（selftest）：合成的母音串當「句子」，兩位「說話人」的基頻與共振峰不同。"""
    rng = np.random.default_rng(int(CFG["fake"]["seed"]))
    vowels = {"user": ([730, 1090, 2440], [270, 2290, 3010], [300, 870, 2240]), "other": ([850, 1220, 2810], [310, 2790, 3310], [370, 950, 2670])}
    audio, texts = {}, {}
    for i, u in enumerate(wanted_utts()):
        who = "user" if u.startswith(CFG["user"]["spk"] + "-") else "other"
        f0 = 110 if who == "user" else 165
        n_syl = 8 + (i % 4)
        parts = [np.zeros(int(0.2 * FS))]
        for j in range(n_syl):
            parts.append(dsp.synth_vowel(FS, 0.3, f0 * (1 + 0.05 * rng.standard_normal()), vowels[who][j % 3]))
            parts.append(np.zeros(int((0.25 if j == n_syl // 2 else 0.04) * FS)))
        x = np.concatenate(parts + [np.zeros(int(0.2 * FS))])
        audio[u] = 0.1 * x / (np.max(np.abs(x)) + 1e-9)
        texts[u] = " ".join(f"W{i}X{j}" for j in range(n_syl))
    return audio, texts


# ── 土法聚類的 diarization（對照用）─────────────────────────────────────
def naive_diarize(mixed: np.ndarray, embed, win_s: float, hop_s: float, cut: float) -> list[tuple[float, float, str]]:
    """能量找出有聲的格子 → 滑動窗 → 每個窗一個向量 → average linkage、cosine 距離 > cut 不併 → 每格取蓋到它的窗的多數決。
    一格只會有一位說話人，所以兩個人同時說的地方它一定漏掉一位。"""
    from scipy.cluster.hierarchy import fcluster, linkage
    speech = M.activity(mixed, FS, float(MIXC["floor_db"]), float(MIXC["min_pause_s"]))
    n_fr = len(speech)
    win, hop = int(round(win_s / M.STEP)), int(round(hop_s / M.STEP))
    step = int(round(M.STEP * FS))
    starts = [a for a in range(0, max(n_fr - win // 2, 1), hop) if speech[a: a + win].mean() >= 0.5]
    if not starts:
        return []
    embs = np.stack([embed(mixed[a * step: min((a + win) * step, len(mixed))]) for a in starts])
    embs = embs / (np.linalg.norm(embs, axis=1, keepdims=True) + 1e-12)
    labels = fcluster(linkage(embs, method="average", metric="cosine"), t=cut, criterion="distance") if len(starts) > 1 else np.array([1])
    ids = sorted(set(labels), key=lambda c: -int((labels == c).sum()))
    votes = np.zeros((len(ids), n_fr))
    for a, lab in zip(starts, labels):
        votes[ids.index(lab), a: a + win] += 1
    best = votes.argmax(0)
    covered = votes.sum(0) > 0
    segs = []
    for j in range(len(ids)):
        for a, b in M.segments(speech & covered & (best == j)):
            segs.append((a, b, f"cluster {j + 1}"))
    return sorted(segs)


def run_worker(args: list[str], timeout: float | None = None) -> dict:
    """pyannote 住在另一個環境：開子程序跑 diar_worker.py，結果從 json 讀回來。"""
    py = _p(DIAR["env"]) / "bin" / "python"
    if not py.exists():
        return {"error": f"沒有 {DIAR['env']}：先跑 ./setup.sh"}
    with tempfile.TemporaryDirectory(prefix="w10_d3_") as td:
        out = Path(td) / "out.json"
        env = dict(os.environ, PYANNOTE_METRICS_ENABLED="0", HF_HUB_OFFLINE=os.environ.get("HF_HUB_OFFLINE", "1"))
        try:
            r = subprocess.run([str(py), str(WORKER), *args, "--out", str(out)], capture_output=True, text=True, env=env,
                               timeout=timeout or float(DIAR["timeout_s"]))
        except subprocess.TimeoutExpired:
            return {"error": "pyannote 的子程序逾時"}
        if out.exists():
            return json.loads(out.read_text(encoding="utf-8"))
        tail = (r.stderr or r.stdout or "").strip().splitlines()[-3:]
        return {"error": "pyannote 的子程序沒有交回結果：" + " / ".join(tail)[:400]}


def pyannote_diarize(mixed: np.ndarray, device: str | None = None) -> dict:
    lock = read_lock()
    try:
        path = model_dir(DIAR["repo"])
        if not (Path(path) / "config.yaml").exists():
            raise FileNotFoundError("config.yaml")
    except Exception as e:
        return {"error": f"快取裡沒有 {DIAR['repo']}（{type(e).__name__}）：先同意使用條款、登入，再跑 ./present.sh fetch"}
    with tempfile.TemporaryDirectory(prefix="w10_d3_") as td:
        wav = Path(td) / "mix.wav"
        C.save_wav(wav, mixed, FS)
        dev = device or DIAR["device"]
        r = run_worker(["run", "--wav", str(wav), "--model", path, "--device", dev])
    if "error" not in r:
        r["commit"] = lock.get(DIAR["repo"])
    return r


# ── 計算 ────────────────────────────────────────────────────────────────
class Lab:
    def __init__(self, backend, audio: dict[str, np.ndarray], texts: dict[str, str], out: Out):
        self.b, self.audio, self.texts, self.out = backend, audio, texts, out
        self.enrol = M.trim(audio[CFG["user"]["enrol"]], FS, float(MIXC["floor_db"]))
        self.enrol = self.enrol * (float(MIXC["level"]) / M.rms(self.enrol))
        self.e_enrol = {k: backend.embed(k, self.enrol) for k in ENCS}
        self.wavs: dict[str, np.ndarray] = {"enrol": self.enrol}

    def sims(self, x: np.ndarray) -> dict:
        return {k: cos(self.b.embed(k, x), self.e_enrol[k]) for k in ENCS}

    def track_row(self, est: np.ndarray, y: np.ndarray, s_user: np.ndarray, s_other: np.ndarray) -> dict:
        return {"si_user": clean_num(M.si_sdr(est, s_user)), "si_other": clean_num(M.si_sdr(est, s_other)),
                "level_db": float(20 * np.log10(M.rms(est) / M.rms(y))), "sim": self.sims(est)}

    def truth(self, tracks: list[dict], user_s: float) -> int | None:
        si = [t["si_user"] if t["si_user"] is not None else -np.inf for t in tracks]
        j = int(np.argmax(si))
        return j if user_s >= float(CFG["truth"]["min_user_s"]) and si[j] >= float(CFG["truth"]["min_si_sdr"]) else None

    def overlap(self, pair: dict) -> dict:
        fl, lv = float(MIXC["floor_db"]), float(MIXC["level"])
        u = M.trim(self.audio[CFG["user"]["overlap_utt"]], FS, fl)
        o = M.trim(self.audio[pair["overlap_utt"]], FS, fl)
        n = min(len(u), len(o))
        u, o = u[:n] * (lv / M.rms(u[:n])), o[:n] * (lv / M.rms(o[:n]))
        y = u + o
        t0 = time.perf_counter()
        est = self.b.separate(y, (u, o))
        secs = time.perf_counter() - t0
        tracks = [self.track_row(e, y, u, o) for e in est]
        pid = pair["id"]
        self.wavs.update({f"{pid}_ov_mix": y, f"{pid}_ov_t1": est[0], f"{pid}_ov_t2": est[1]})
        ref = self.b.normalize(self.texts[CFG["user"]["overlap_utt"]])
        tr = self.truth(tracks, n / FS)
        asr = {"mix": self.score(y, ref, (u, ref))}
        if tr is not None:
            asr["user_track"] = self.score(est[tr], ref, (u, ref))
        return {"seconds": n / FS, "utts": [CFG["user"]["overlap_utt"], pair["overlap_utt"]], "ref_text": ref, "tracks": tracks, "truth": tr,
                "pick": {k: int(np.argmax([t["sim"][k] for t in tracks])) for k in ENCS}, "asr": asr, "compute_s": secs}

    def score(self, x: np.ndarray, ref: str, hint=None, split: int | None = None) -> dict:
        """Whisper 轉一次、對參考文字算 WER。split 給定時在那個樣本切成兩段各轉一次（一個視窗只有 30 秒）。"""
        if split:
            h1 = None if hint is None else (hint[0][:split], " ".join(hint[1].split()[: hint[2]]))
            h2 = None if hint is None else (hint[0][split:], " ".join(hint[1].split()[hint[2]:]))
            hyp = (self.b.asr(x[:split], h1) + " " + self.b.asr(x[split:], h2)).strip()
        else:
            hyp = self.b.asr(x, None if hint is None else hint[:2])
        hyp = self.b.normalize(hyp)
        w = W2.corpus_wer([ref], [hyp])
        return {"text": hyp, **{k: w[k] for k in ("S", "D", "I", "n_ref", "wer")}}

    def chunking(self, conv: dict, chunk_s: float | None, pid: str, keep_audio: bool) -> tuple[dict, list[list[np.ndarray]]]:
        y, n = conv["mix"], conv["n"]
        spans = M.chunks(n, FS, chunk_s) if chunk_s else [(0, n)]
        rows, ests = [], []
        t_all = time.perf_counter()
        for k, (a, b) in enumerate(spans):
            s_u, s_o = conv["tracks"]["user"][a:b], conv["tracks"]["other"][a:b]
            est = self.b.separate(y[a:b], (s_u, s_o))
            user_s, other_s = M.active_s(conv["ref"]["user"], a, b, FS), M.active_s(conv["ref"]["other"], a, b, FS)
            tracks = [self.track_row(e, y[a:b], s_u, s_o) for e in est]
            rows.append({"k": k + 1, "a_s": a / FS, "b_s": b / FS, "user_s": user_s, "other_s": other_s, "tracks": tracks,
                         "truth": self.truth(tracks, user_s), "pick": {e: int(np.argmax([t["sim"][e] for t in tracks])) for e in ENCS},
                         "max_sim": {e: float(max(t["sim"][e] for t in tracks)) for e in ENCS}})
            ests.append(est)
            if keep_audio:
                for j in range(2):
                    self.wavs[f"{pid}_k{k + 1:02d}_t{j + 1}"] = est[j]
            self.out(f"    {C.GREY}{'whole' if not chunk_s else f'{chunk_s:g} s'}  塊 {k + 1}/{len(spans)}{C.RESET}")
        present = [r["truth"] for r in rows if r["truth"] is not None]
        return ({"chunk_s": chunk_s, "rows": rows, "compute_s": time.perf_counter() - t_all,
                 "n_present": len(present), "n_swaps": int(sum(a != b for a, b in zip(present[:-1], present[1:]))),
                 "n_right": {e: int(sum(r["pick"][e] == r["truth"] for r in rows if r["truth"] is not None)) for e in ENCS}}, ests)

    def stitch(self, rows: list[dict], ests: list[list[np.ndarray]], version: str, conv: dict) -> np.ndarray:
        if version == "mix":
            return conv["mix"]
        if version == "clean":
            return conv["tracks"]["user"]
        parts = []
        for r, est in zip(rows, ests):
            if version == "track1":
                j = 0
            elif version == "truth":
                j = r["truth"]
            else:
                e = version[-1]
                j = r["pick"][e]
                if version.startswith("gate") and r["max_sim"][e] < float(ENC[e]["threshold"]):
                    j = None
            parts.append(np.zeros_like(est[0]) if j is None else est[j])
        return np.concatenate(parts)

    def pair(self, pair: dict, diar: bool = True, device: str | None = None) -> dict:
        pid, o = pair["id"], self.out
        o(f"  {C.BOLD}{pair['label']}{C.RESET}")
        conv = M.build(pair["turns"], self.audio, FS, float(MIXC["level"]), float(MIXC["head_s"]), float(MIXC["tail_s"]),
                       float(MIXC["floor_db"]), float(MIXC["min_pause_s"]))
        self.wavs[f"{pid}_mix"] = conv["mix"]
        turns = [{**t, "text": self.b.normalize(self.texts[t["utt"]])} for t in conv["turns"]]
        user_turns = [t for t in turns if t["spk"] == "user"]
        ref_text = " ".join(t["text"] for t in user_turns)
        # 一個視窗 30 秒：在使用者最長的一次沉默正中間切
        gaps = [(b["start"] - a["end"], (a["end"] + b["start"]) / 2, i + 1) for i, (a, b) in enumerate(zip(user_turns[:-1], user_turns[1:]))]
        _, split_s, n_before = max(gaps)
        n_words = len(" ".join(t["text"] for t in user_turns[:n_before]).split())
        split = int(round(split_s * FS))
        rec = {"label": pair["label"], "other": pair["other"], "seconds": conv["seconds"], "stats": M.overlap_stats(conv["ref"]), "turns": turns,
               "ref_segments": {s: [list(x) for x in M.segments(m)] for s, m in conv["ref"].items()}, "split_s": split_s, "user_ref_text": ref_text}
        o("    完全重疊的那一段")
        rec["overlap"] = self.overlap(pair)
        rec["chunkings"] = {}
        main_ests = None
        for cs in CHUNKINGS:
            r, ests = self.chunking(conv, cs, pid, keep_audio=(cs == CHUNKINGS[0]))
            rec["chunkings"][ck(cs)] = r
            if cs == CHUNKINGS[0]:
                main_ests = ests
        if SEP["whole"]:
            rec["chunkings"]["whole"], _ = self.chunking(conv, None, pid, keep_audio=False)
        o("    Whisper")
        rows = rec["chunkings"][ck(CHUNKINGS[0])]["rows"]
        hint = (conv["tracks"]["user"], ref_text, n_words)
        rec["asr"] = {v: self.score(self.stitch(rows, main_ests, v, conv), ref_text, hint, split) for v, _ in VERSIONS}
        o("    diarization")
        rec["diar"] = {}
        naive = naive_diarize(conv["mix"], lambda x: self.b.embed("a", x), float(NAIVE["win_s"]), float(NAIVE["hop_s"]), float(NAIVE["cut"]))
        rec["diar"]["naive"] = self.diar_entry([list(s) for s in naive], conv)
        if self.b.fake:
            segs = [[max(a + 0.08 * ((i % 3) - 1), 0.0), b + 0.12, f"SPEAKER_{0 if s == 'user' else 1:02d}"]
                    for s, ss in rec["ref_segments"].items() for i, (a, b) in enumerate(ss)]
            rec["diar"]["pyannote"] = self.diar_entry(sorted(segs), conv)
        elif diar:
            r = pyannote_diarize(conv["mix"], device)
            if "error" in r:
                rec["diar"]["pyannote"] = r
                o(f"    {C.AMBER}pyannote：{r['error']}{C.RESET}")
            else:
                rec["diar"]["pyannote"] = {**self.diar_entry(r["segments"], conv), **{k: r.get(k) for k in ("device", "compute_s", "commit", "versions")}}
        else:
            rec["diar"]["pyannote"] = {"error": "這次沒有跑（--no-diar）"}
        return rec

    def diar_entry(self, segs: list[list], conv: dict) -> dict:
        hyp = D.to_grid([tuple(s) for s in segs], conv["seconds"]) if segs else {}
        hyp = {k: v[: len(conv["ref"]["user"])] for k, v in hyp.items()}
        res = []
        for name, c, s in SCORINGS:
            d = D.der(conv["ref"], hyp, c, s)
            res.append({"name": name, "collar": c, "skip_overlap": s, **{k: clean_num(d[k]) for k in ("der", "fa", "miss", "conf", "total", "scored_s")}, "map": d["map"]})
        return {"segments": [[round(float(a), 3), round(float(b), 3), str(l)] for a, b, l in segs], "n_speakers": len(hyp), "der": res}


def compute(backend, audio, texts, out: Out, tag: str, diar: bool = True, device: str | None = None, pairs: list[str] | None = None) -> tuple[dict, dict]:
    lab = Lab(backend, audio, texts, out)
    rec = {"time": C.now(), "tag": tag, "fake": backend.fake, "main": MAIN, "chunk_s": CHUNKINGS[0], "collar": COLLAR,
           "enrol": {"utt": CFG["user"]["enrol"], "seconds": len(lab.enrol) / FS}, "thresholds": {e: float(ENC[e]["threshold"]) for e in ENCS},
           "enc_labels": {e: ENC[e]["label"] for e in ENCS}, "pairs": {}}
    for pid in (pairs or list(PAIRS)):
        rec["pairs"][pid] = lab.pair(PAIRS[pid], diar, device)
    rec["models"] = backend.info()
    return rec, lab.wavs


def write_run(folder: Path, rec: dict, wavs: dict, out: Out | None = None) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    for old in folder.glob("*.wav"):
        old.unlink()
    peak = max(float(np.max(np.abs(x))) for x in wavs.values()) + 1e-9
    for name, x in wavs.items():
        C.save_wav(folder / f"{name}.wav", x, FS, peak)
    write_json(folder / "rehearsal.json", rec)
    if out is not None:
        (folder / "rehearsal.txt").write_text(out.text(), encoding="utf-8")


# ── 畫面 ────────────────────────────────────────────────────────────────
class Demo:
    def __init__(self, out: Out, rec: dict, folder: Path):
        self.out, self.rec, self.folder = out, rec, folder
        self.pid = rec["main"] if rec["main"] in rec["pairs"] else next(iter(rec["pairs"]))
        self.cs = ck(float(rec["chunk_s"]))
        self.enc = "a"
        self.reveal = False
        self.system = "pyannote"
        self.scored = False
        self.view = "blank"
        self.fig = self.plt = None
        self.el = rec["enc_labels"]
        self.th = rec["thresholds"]

    # ── 取資料 ──
    @property
    def p(self) -> dict:
        return self.rec["pairs"][self.pid]

    @property
    def chunking(self) -> dict:
        return self.p["chunkings"][self.cs]

    @property
    def main_rows(self) -> list[dict]:
        return self.p["chunkings"][ck(float(self.rec["chunk_s"]))]["rows"]

    def diar(self) -> tuple[str, dict]:
        d = self.p["diar"]
        if self.system == "pyannote" and "error" in d.get("pyannote", {"error": "沒有紀錄"}):
            return "naive", d["naive"]
        return self.system, d[self.system]

    def sys_label(self, name: str) -> str:
        return DIAR["label"] if name == "pyannote" else "home-made clustering (ECAPA windows)"

    def head(self, s: str) -> None:
        tag = f"  {C.RED}[FAKE：替身，數字沒有意義]{C.RESET}" if self.rec.get("fake") else ""
        self.out.banner(s + tag)

    def wav(self, name: str) -> Path:
        return self.folder / f"{name}.wav"

    def play(self, name: str, say: str) -> None:
        p = self.wav(name)
        self.out(f"  ♪ {say}")
        if p.exists():
            C.play(p)
        else:
            self.out(f"    {C.AMBER}（沒有 {p.name}）{C.RESET}")

    def play_stitched(self, version: str, say: str) -> None:
        """照紀錄裡的挑法，把每一塊存下來的那一軌接起來播。"""
        if version == "mix":
            return self.play(f"{self.pid}_mix", say)
        from scipy.io import wavfile
        parts = []
        for r in self.main_rows:
            j = 0 if version == "track1" else (r["truth"] if version == "truth" else r["pick"][version[-1]])
            if version.startswith("gate") and r["max_sim"][version[-1]] < self.th[version[-1]]:
                j = None
            p = self.wav(f"{self.pid}_k{r['k']:02d}_t{(j or 0) + 1}")
            if not p.exists():
                self.out(f"    {C.AMBER}（沒有 {p.name}）{C.RESET}")
                return
            x = wavfile.read(str(p))[1]
            parts.append(np.zeros_like(x) if j is None else x)
        tmp = LIVE / "_play.wav"
        tmp.parent.mkdir(parents=True, exist_ok=True)
        wavfile.write(str(tmp), FS, np.concatenate(parts))
        self.out(f"  ♪ {say}")
        C.play(tmp)

    # ── 前半 ──
    def overlap(self) -> None:
        ov, o = self.p["overlap"], self.out
        self.head(f"完全重疊的 {ov['seconds']:.1f} 秒：{self.p['label']}")
        self.play(f"{self.pid}_ov_mix", "混合的")
        o(f"    Whisper 聽混合的：{C.GREY}{ov['asr']['mix']['text'] or '（空的）'}{C.RESET}   WER {pct(ov['asr']['mix']['wer'])}%")
        o(f"\n  {SEP['label']} 分出兩軌：")
        o(f"    {'':8s}{'SI-SDR 對使用者':>16s}{'對另一位':>12s}{'SIM ' + self.el['a']:>14s}{'SIM ' + self.el['b']:>16s}")
        for j, t in enumerate(ov["tracks"]):
            o(f"    track {j + 1} {f1(t['si_user']):>14s} dB{f1(t['si_other']):>9s} dB{f3(t['sim']['a']):>14s}{f3(t['sim']['b']):>16s}")
        for j in range(2):
            self.play(f"{self.pid}_ov_t{j + 1}", f"track {j + 1}")
        if "user_track" in ov["asr"]:
            a = ov["asr"]["user_track"]
            o(f"    Whisper 聽使用者所在的那一軌（track {ov['truth'] + 1}）：{a['text']}   WER {pct(a['wer'])}%")

    def chunk_table(self, enc: str | None = None) -> None:
        if enc:
            self.enc = enc
        e, o, c = self.enc, self.out, self.chunking
        what = "整段不切、一次處理" if self.cs == "whole" else f"每 {self.cs} 秒一塊"
        self.head(f"「對話」{self.p['seconds']:.1f} 秒，{what}：兩軌各自對註冊語音的 SIM（{self.el[e]}）")
        hdr = f"    {'塊':>3s} {'時間':>13s}   {'track 1':>8s} {'track 2':>8s}   {'挑':>3s}"
        if self.reveal:
            hdr += f"   {'其實在':>5s}  {'使用者說了':>7s}  {'SI-SDR 對使用者（兩軌）':>18s}"
        o(hdr)
        for r in c["rows"]:
            s = [t["sim"][e] for t in r["tracks"]]
            cells = [f"{C.BOLD}{f3(v):>8s}{C.RESET}" if j == r["pick"][e] else f"{f3(v):>8s}" for j, v in enumerate(s)]
            line = f"    {r['k']:>3d} {r['a_s']:5.1f}–{r['b_s']:5.1f} s   {cells[0]} {cells[1]}   → {r['pick'][e] + 1}"
            if self.reveal:
                tr = "—" if r["truth"] is None else str(r["truth"] + 1)
                ok = "" if r["truth"] is None else (f" {C.GREEN}✓{C.RESET}" if r["truth"] == r["pick"][e] else f" {C.RED}✗{C.RESET}")
                line += f"      {tr}{ok:<3s}   {r['user_s']:4.1f} s     {f1(r['tracks'][0]['si_user']):>6s} {f1(r['tracks'][1]['si_user']):>6s} dB"
            o(line)
        if self.reveal:
            o(f"\n    使用者在場的 {c['n_present']} 塊裡，軌的編號換了 {C.BOLD}{c['n_swaps']}{C.RESET} 次；"
              f"照 SIM 挑對的：{self.el['a']} {c['n_right']['a']}/{c['n_present']}、{self.el['b']} {c['n_right']['b']}/{c['n_present']}")
            o(f"    {C.GREY}「其實在」＝對使用者乾淨的那一軌 SI-SDR 較高、而且 ≥ {CFG['truth']['min_si_sdr']:g} dB 的那一軌；— ＝使用者不在任何一軌{C.RESET}")
        if enc == "b" or (self.enc == "b" and enc is None):
            diff = [r["k"] for r in c["rows"] if r["pick"]["a"] != r["pick"]["b"]]
            o(f"    兩個 encoder 挑得不一樣的塊：{diff if diff else '沒有'}；同一軌的 SIM："
              f"{self.el['a']} {min(t['sim']['a'] for r in c['rows'] for t in r['tracks']):+.2f}…{max(t['sim']['a'] for r in c['rows'] for t in r['tracks']):+.2f}、"
              f"{self.el['b']} {min(t['sim']['b'] for r in c['rows'] for t in r['tracks']):+.2f}…{max(t['sim']['b'] for r in c['rows'] for t in r['tracks']):+.2f}")

    def asr_table(self) -> None:
        o, a = self.out, self.p["asr"]
        self.head(f"接起來送 Whisper（每 {self.rec['chunk_s']:g} 秒一塊）：只算使用者說的那 {a['clean']['n_ref']} 個詞")
        o(f"    {'接法':<44s}{'WER':>7s}{'S':>5s}{'D':>5s}{'I':>5s}")
        for v, label in VERSIONS:
            r = a[v]
            o(f"    {label:<44s}{pct(r['wer']):>6s}%{r['S']:>5d}{r['D']:>5d}{r['I']:>5d}")
        ref = self.p["user_ref_text"]
        for v in ("track1", "pick_a"):
            o(f"\n  {dict(VERSIONS)[v]}：")
            o("    " + W2.colored(ref, a[v]["text"]))
        o(f"\n    {C.GREY}紅色＝替換或插入（+）；灰色 ⟨⟩＝使用者說了、沒轉出來的詞。插入的多半是另一位說話人的話。{C.RESET}")

    def absent_rows(self) -> list[dict]:
        return [r for r in self.main_rows if r["truth"] is None]

    def absent(self) -> None:
        o = self.out
        rows = self.absent_rows()
        self.head("使用者不在的那幾塊：分離照樣吐兩軌，SIM 照樣有一個比較高")
        if not rows:
            o("    這一組沒有這樣的塊。")
            return
        o(f"    {'塊':>3s} {'使用者說了':>8s}   {self.el['a']:>10s} t1／t2 {'挑':>4s}   {self.el['b']:>10s} t1／t2 {'挑':>4s}   {'兩軌的音量（對混合）':>14s}")
        for r in rows:
            t = r["tracks"]
            o(f"    {r['k']:>3d} {r['user_s']:7.1f} s   {f3(t[0]['sim']['a']):>9s} {f3(t[1]['sim']['a']):>7s}  → {r['pick']['a'] + 1}   "
              f"{f3(t[0]['sim']['b']):>9s} {f3(t[1]['sim']['b']):>7s}  → {r['pick']['b'] + 1}   {t[0]['level_db']:6.1f} {t[1]['level_db']:6.1f} dB")
        r = min(rows, key=lambda r: (r["user_s"], r["k"]))
        j = r["pick"]["a"]
        self.play(f"{self.pid}_k{r['k']:02d}_t{j + 1}", f"第 {r['k']} 塊，{self.el['a']} 挑的 track {j + 1}（使用者這一塊說了 {r['user_s']:.1f} 秒）")

    def gate(self) -> None:
        o = self.out
        self.head("加一個門檻：SIM 最高的那一軌也不到門檻，就當作使用者不在")
        for e in ENCS:
            pres = [r["max_sim"][e] for r in self.main_rows if r["truth"] is not None]
            abse = [r["max_sim"][e] for r in self.main_rows if r["truth"] is None]
            th = self.th[e]
            o(f"  {C.BOLD}{self.el[e]}{C.RESET}（門檻 {th:g}，出處見設定檔）")
            o(f"    使用者在場的 {len(pres)} 塊，挑到的那一軌的 SIM：" + "  ".join(f"{C.RED if v < th else ''}{v:+.2f}{C.RESET}" for v in pres))
            o(f"    使用者不在的 {len(abse)} 塊：                  " + "  ".join(f"{C.RED if v >= th else ''}{v:+.2f}{C.RESET}" for v in abse))
            sep = (min(pres) > max(abse)) if pres and abse else None
            o(f"    在場的被擋掉 {sum(v < th for v in pres)} 塊、不在的放過 {sum(v >= th for v in abse)} 塊；"
              + ("" if sep is None else (f"有一個門檻分得開（{max(abse):+.2f} 與 {min(pres):+.2f} 之間）" if sep else "沒有任何一個門檻分得開這兩組")))
        a = self.p["asr"]
        o(f"\n    接起來送 Whisper：不設門檻 {pct(a['pick_a']['wer'])}%／{pct(a['pick_b']['wer'])}%，設門檻 {pct(a['gate_a']['wer'])}%／{pct(a['gate_b']['wer'])}%"
          f"（{self.el['a']}／{self.el['b']}；插入 {a['pick_a']['I']}／{a['pick_b']['I']} → {a['gate_a']['I']}／{a['gate_b']['I']}，刪除 {a['pick_a']['D']}／{a['pick_b']['D']} → {a['gate_a']['D']}／{a['gate_b']['D']}）")

    # ── 後半 ──
    def grids(self) -> tuple[dict, dict]:
        ref = D.to_grid([(a, b, s) for s, ss in self.p["ref_segments"].items() for a, b in ss], self.p["seconds"])
        for s in self.p["ref_segments"]:
            ref.setdefault(s, np.zeros(int(np.ceil(self.p["seconds"] / D.STEP)), dtype=bool))
        name, d = self.diar()
        hyp = D.to_grid([tuple(s) for s in d["segments"]], self.p["seconds"]) if d["segments"] else {}
        return ref, hyp

    def diar_summary(self) -> None:
        o = self.out
        name, d = self.diar()
        st = self.p["stats"]
        self.head(f"diarization：{self.sys_label(name)} 聽同一段「對話」")
        if name != self.system:
            o(f"    {C.AMBER}沒有 pyannote 的輸出（{self.p['diar']['pyannote'].get('error', '')}）——下面是土法聚類的。{C.RESET}")
        o(f"    參考標註：使用者 {st['speech_s']['user']:.1f} s、另一位 {st['speech_s']['other']:.1f} s；同時說 {st['overlap_s']:.1f} s"
          f"（有人說話的時間的 {100 * st['overlap_ratio']:.0f}%）；沒人說話 {st['silence_s']:.1f} s")
        o(f"    系統輸出：{len(d['segments'])} 段、{d['n_speakers']} 位說話人" + (f"；{d['device']}，{d['compute_s']:.1f} s" if d.get("compute_s") else ""))
        strict = d["der"][-1]
        o(f"    系統的編號 → 參考的說話人：{strict['map'] or '（一個都對不上）'}")
        o(f"    最嚴的計分下，三種錯各幾秒：false alarm {strict['fa']:.2f}、missed {strict['miss']:.2f}、wrong speaker {strict['conf']:.2f}")

    def der_table(self) -> None:
        o = self.out
        name, d = self.diar()
        self.scored = True
        self.head(f"同一份輸出（{self.sys_label(name)}）、三種計分")
        o(f"    {'計分':<34s}{'DER':>8s}   {'FA':>6s} {'miss':>6s} {'conf':>6s}   {'÷ 語音總長':>10s}   {'計分的時間':>8s}")
        for r in d["der"]:
            o(f"    {r['name']:<34s}{C.BOLD}{pct(r['der']):>7s}%{C.RESET}   {r['fa']:6.2f} {r['miss']:6.2f} {r['conf']:6.2f}   {r['total']:9.2f} s   {r['scored_s']:7.2f} s")
        ds = [r["der"] for r in d["der"] if r["der"] is not None]
        if len(ds) == 3 and min(ds) > 0:
            o(f"    最嚴的是最寬的 {max(ds) / min(ds):.1f} 倍")
        o(f"    {C.GREY}collar：參考標註的每個邊界前後各 {self.rec['collar']:g} 秒不計；overlap skipped：參考標註裡兩位同時說的格子不計。10 ms 一格。{C.RESET}")

    def whole(self) -> None:
        if "whole" not in self.p["chunkings"]:
            self.out("  這份紀錄沒有整段一次處理的結果。")
            return
        keep = self.cs
        self.cs = "whole"
        self.chunk_table()
        r = self.chunking["rows"][0]
        self.out(f"    整段一次處理：SI-SDR 對使用者 {f1(r['tracks'][0]['si_user'])}／{f1(r['tracks'][1]['si_user'])} dB、對另一位 {f1(r['tracks'][0]['si_other'])}／{f1(r['tracks'][1]['si_other'])} dB"
                 f"（{self.chunking['compute_s']:.1f} s）")
        self.redraw()
        self.cs = keep

    def slide_tables(self) -> None:
        o = self.out
        self.head("投影片 p78 的表")
        o("    | chunk | SIM to the enrolment: track 1 · track 2 (" + self.el["a"] + ") | the same, " + self.el["b"] + " | picked | the user is really in |")
        for r in self.main_rows:
            t = r["tracks"]
            tr = "neither" if r["truth"] is None else f"track {r['truth'] + 1}"
            pk = f"{r['pick']['a'] + 1}" if r["pick"]["a"] == r["pick"]["b"] else f"{r['pick']['a'] + 1} / {r['pick']['b'] + 1}"
            o(f"    | {r['k']} ({r['a_s']:.0f}–{r['b_s']:.0f} s; user {r['user_s']:.1f} s) | {t[0]['sim']['a']:.2f} · {t[1]['sim']['a']:.2f} | {t[0]['sim']['b']:.2f} · {t[1]['sim']['b']:.2f} | {pk} | {tr} |")
        self.head("投影片 p87 的表")
        for name in ("pyannote", "naive"):
            d = self.p["diar"].get(name, {})
            if "error" in d or not d:
                o(f"    {self.sys_label(name)}：{C.AMBER}{d.get('error', '沒有紀錄')}{C.RESET}")
                continue
            o(f"    {self.sys_label(name)}（{d['n_speakers']} speakers）")
            o("    | DER | " + " | ".join(f"{pct(r['der'])}%" for r in d["der"]) + " |")
            o("    | FA · missed · wrong speaker (s) | " + " | ".join(f"{r['fa']:.2f} · {r['miss']:.2f} · {r['conf']:.2f}" for r in d["der"]) + " |")

    # ── 圖 ──
    def make_fig(self):
        import matplotlib.pyplot as plt
        self.plt = plt
        for k in [k for k in plt.rcParams if k.startswith("keymap.")]:
            plt.rcParams[k] = []
        self.fig = plt.figure(figsize=(float(CFG["display"]["fig_w"]), float(CFG["display"]["fig_h"])))
        self.draw()
        return self.fig

    def redraw(self) -> None:
        if self.fig is not None:
            self.draw()
            self.fig.canvas.draw_idle()

    def show_view(self, view: str) -> None:
        self.view = view
        self.redraw()

    def _ref_rows(self, ax, y0: float) -> None:
        for i, s in enumerate(("user", "other")):
            for a, b in self.p["ref_segments"][s]:
                ax.broken_barh([(a, b - a)], (y0 - i, 0.7), color=SPK_COLOR[s])
            ax.text(-0.4, y0 - i + 0.35, f"reference: {s}", ha="right", va="center", fontsize=11)

    def draw(self) -> None:
        fig = self.fig
        fig.clf()
        tag = "  [FAKE]" if self.rec.get("fake") else ""
        if self.view == "chunks":
            self._draw_chunks(fig.add_subplot(111), tag)
        elif self.view == "der":
            self._draw_der(tag)
        else:
            ax = fig.add_subplot(111)
            ax.axis("off")
            ax.text(0.5, 0.5, "W10 · Demo 3\ntwo people at once: which track is the user?" + tag, ha="center", va="center", fontsize=20)
        fig.tight_layout()

    def _draw_chunks(self, ax, tag: str) -> None:
        import matplotlib.cm as cm
        e, c = self.enc, self.chunking
        self._ref_rows(ax, 4.0)
        sims = [t["sim"][e] for r in c["rows"] for t in r["tracks"]]
        lo, hi = min(sims), max(sims)
        for r in c["rows"]:
            ax.axvline(r["a_s"], color="#9ca3af", lw=0.8, ls=":")
            for j, t in enumerate(r["tracks"]):
                v = (t["sim"][e] - lo) / (hi - lo + 1e-9)
                picked = r["pick"][e] == j
                ax.broken_barh([(r["a_s"] + 0.06, r["b_s"] - r["a_s"] - 0.12)], (1.0 - j * 1.1, 0.9), facecolor=cm.Greens(0.15 + 0.7 * v),
                               edgecolor="#111827" if picked else "none", linewidth=2.5 if picked else 0)
                ax.text((r["a_s"] + r["b_s"]) / 2, 1.45 - j * 1.1, f"{t['sim'][e]:.2f}", ha="center", va="center", fontsize=11, fontweight="bold" if picked else "normal")
                if self.reveal and r["truth"] == j:
                    ax.text((r["a_s"] + r["b_s"]) / 2, 1.12 - j * 1.1, "user", ha="center", va="center", fontsize=9, color=SPK_COLOR["user"])
            if self.reveal and r["truth"] is None:
                ax.text((r["a_s"] + r["b_s"]) / 2, 0.92, "user\nabsent", ha="center", va="center", fontsize=8, color="#c0392b")
        for j in range(2):
            ax.text(-0.4, 1.45 - j * 1.1, f"track {j + 1}", ha="right", va="center", fontsize=11)
        ax.set_xlim(-0.2, self.p["seconds"])
        ax.set_ylim(-0.4, 5.0)
        ax.set_yticks([])
        ax.set_xlabel("time (s)")
        for s in ("top", "right", "left"):
            ax.spines[s].set_visible(False)
        what = "one pass, no chunking" if self.cs == "whole" else f"{self.cs} s chunks"
        ax.set_title(f"SIM of each separated track to the enrolment — {self.el[e]}, {what}; box = the track picked{tag}\n{self.p['label']}", fontsize=12, loc="left")

    def _draw_der(self, tag: str) -> None:
        fig = self.fig
        gs = fig.add_gridspec(1, 4)
        ax = fig.add_subplot(gs[0, :3] if self.scored else gs[0, :])
        name, d = self.diar()
        ref, hyp = self.grids()
        self._ref_rows(ax, 6.0)
        res = D.der(ref, hyp, 0.0, False)
        inv = {h: r for h, r in res["map"].items()}
        for i, (h, m) in enumerate(sorted(hyp.items())):
            col = SPK_COLOR.get(inv.get(h), "#6b7280")
            for a, b in M.segments(m):
                ax.broken_barh([(a, b - a)], (3.4 - i * 0.8, 0.6), facecolor="white", edgecolor=col, linewidth=1.8, hatch="//")
            ax.text(-0.4, 3.7 - i * 0.8, f"system: {h}", ha="right", va="center", fontsize=10)
        y_err = 3.4 - max(len(hyp), 1) * 0.8 - 0.4
        for k in ("miss", "fa", "conf"):
            for a, b in M.segments(res["per_frame"][k] > 0):
                ax.broken_barh([(a, b - a)], (y_err, 0.7), color=ERR_COLOR[k])
        ax.text(-0.4, y_err + 0.35, "errors", ha="right", va="center", fontsize=11)
        if self.scored:
            keep = D.scored_mask(ref, COLLAR, False)
            for a, b in M.segments(~keep):
                ax.axvspan(a, b, color="#9ca3af", alpha=0.25, lw=0)
            ov = np.sum(np.stack(list(ref.values())), axis=0) > 1
            for a, b in M.segments(ov):
                ax.axvspan(a, b, ymin=0.86, ymax=1.0, color="#5b3fa0", alpha=0.25, lw=0)
        from matplotlib.patches import Patch
        handles = [Patch(color=ERR_COLOR["miss"], label="missed"), Patch(color=ERR_COLOR["fa"], label="false alarm"), Patch(color=ERR_COLOR["conf"], label="wrong speaker")]
        if self.scored:
            handles += [Patch(color="#9ca3af", alpha=0.4, label=f"collar ±{COLLAR:g} s"), Patch(color="#5b3fa0", alpha=0.4, label="overlap")]
        ax.legend(handles=handles, loc="lower right", ncol=len(handles), fontsize=9, frameon=False)
        ax.set_xlim(-0.2, self.p["seconds"])
        ax.set_ylim(y_err - 1.0, 7.1)
        ax.set_yticks([])
        ax.set_xlabel("time (s)")
        for s in ("top", "right", "left"):
            ax.spines[s].set_visible(False)
        ax.set_title(f"who spoke when — {self.sys_label(name)}{tag}\n{self.p['label']}", fontsize=12, loc="left")
        if self.scored:
            bx = fig.add_subplot(gs[0, 3])
            vals = [100 * (r["der"] or 0) for r in d["der"]]
            bx.bar(range(3), vals, color=["#9ca3af", "#6b7280", "#111827"])
            for i, v in enumerate(vals):
                bx.text(i, v, f"{v:.1f}%", ha="center", va="bottom", fontsize=12)
            bx.set_xticks(range(3))
            bx.set_xticklabels(["collar,\nno overlap", "collar,\noverlap", "no collar,\noverlap"], fontsize=9)
            bx.set_ylabel("DER (%)")
            bx.set_title("one output, three scorings", fontsize=11)
            for s in ("top", "right"):
                bx.spines[s].set_visible(False)

    # ── 鍵 ──
    def menu(self) -> None:
        o = self.out
        o(f"\n  {C.BOLD}前半{C.RESET}  1 完全重疊：混合 → 兩軌   p 播「對話」   2 逐塊的 SIM（{self.el['a']}）   3 換 {self.el['b']}   o 揭曉   n 每塊都取第一軌")
        o(f"        4 挑軌、接起來、Whisper   l 播挑出來的   5 使用者不在的那一塊   a 門檻   c 換塊長   x 整段一次   g 換另一位")
        o(f"  {C.BOLD}後半{C.RESET}  6 diarization 的時間軸   7 三個 DER   k 換成土法聚類   t 投影片的表   ? 選單   q 離開")
        o(f"  {C.GREY}現在：{self.p['label']}；{'整段' if self.cs == 'whole' else self.cs + ' 秒一塊'}{C.RESET}")

    def dispatch(self, k: str | None) -> bool:
        if k is None:
            return True
        if k in ("q", "\x03", "\x04"):
            return False
        if k == "1":
            self.overlap()
        elif k == "p":
            self.head(f"「對話」{self.p['seconds']:.1f} 秒（混合的）")
            self.play(f"{self.pid}_mix", "兩個人、輪流說、交界有一點重疊")
        elif k in ("2", "3"):
            self.chunk_table("a" if k == "2" else "b")
            self.show_view("chunks")
        elif k == "o":
            self.reveal = not self.reveal
            self.chunk_table()
            self.show_view("chunks")
        elif k == "n":
            self.head("每一塊都取第一軌，接起來")
            self.play_stitched("track1", "track 1 · track 1 · track 1 …")
        elif k == "4":
            self.asr_table()
        elif k == "l":
            self.head(f"照 SIM（{self.el['a']}）挑軌，接起來")
            self.play_stitched("pick_a", "每一塊取 SIM 較高的那一軌")
        elif k == "5":
            self.absent()
        elif k == "a":
            self.gate()
        elif k == "c":
            opts = [ck(c) for c in CHUNKINGS if ck(c) in self.p["chunkings"]]
            self.cs = opts[(opts.index(self.cs) + 1) % len(opts)] if self.cs in opts else opts[0]
            self.chunk_table()
            self.show_view("chunks")
        elif k == "x":
            self.whole()
        elif k == "g":
            ids = list(self.rec["pairs"])
            self.pid = ids[(ids.index(self.pid) + 1) % len(ids)]
            self.out(f"  另一位說話人 → {self.p['label']}")
            self.redraw()
        elif k == "6":
            self.diar_summary()
            self.show_view("der")
        elif k == "7":
            self.der_table()
            self.show_view("der")
        elif k == "k":
            self.system = "naive" if self.system == "pyannote" else "pyannote"
            self.out(f"  系統 → {self.sys_label(self.diar()[0])}")
            self.diar_summary()
            if self.scored:
                self.der_table()
            self.show_view("der")
        elif k == "t":
            self.slide_tables()
        elif k in ("?", "h"):
            self.menu()
        return True

    def on_key(self, ev) -> None:
        if ev.key is None:
            return
        if not self.dispatch(ev.key):
            self.plt.close(self.fig)

    def save_figs(self) -> None:
        import matplotlib
        matplotlib.use("Agg")
        self.make_fig()
        keep = (self.pid, self.enc, self.reveal, self.system, self.scored, self.view)
        for pid in self.rec["pairs"]:
            self.pid, self.reveal, self.scored = pid, True, True
            for e in ENCS:
                self.enc, self.view = e, "chunks"
                self.draw()
                self.fig.savefig(self.folder / f"{pid}_chunks_{e}.png", dpi=110)
            for system in ("pyannote", "naive"):
                if "error" in self.p["diar"].get(system, {"error": 1}):
                    continue
                self.system, self.view = system, "der"
                self.draw()
                self.fig.savefig(self.folder / f"{pid}_der_{system}.png", dpi=110)
        self.pid, self.enc, self.reveal, self.system, self.scored, self.view = keep
        self.plt.close(self.fig)
        self.fig = None

    def print_all(self) -> None:
        keep = (self.pid, self.reveal, self.system, self.cs, self.enc)
        play, C.play = C.play, (lambda p: None)
        try:
            for pid in self.rec["pairs"]:
                self.pid, self.reveal = pid, True
                self.overlap()
                for cs in [ck(c) for c in CHUNKINGS]:
                    self.cs = cs
                    self.chunk_table("a")
                    self.chunk_table("b")
                self.cs = ck(float(self.rec["chunk_s"]))
                self.whole()
                self.asr_table()
                self.absent()
                self.gate()
                for system in ("pyannote", "naive"):
                    self.system = system
                    self.diar_summary()
                    self.der_table()
                self.slide_tables()
        finally:
            C.play = play
        self.pid, self.reveal, self.system, self.cs, self.enc = keep
        self.scored = False


def run(d: Demo, plot: bool) -> None:
    d.menu()
    if plot:
        import matplotlib.pyplot as plt
        d.make_fig()
        d.fig.canvas.mpl_connect("key_press_event", d.on_key)
        plt.show()
    else:
        while d.dispatch(C.getkey()):
            pass


# ── 組裝 ────────────────────────────────────────────────────────────────
def versions() -> dict:
    from importlib import metadata
    v = C.versions()
    for pkg in ("torch", "torchaudio", "transformers", "speechbrain", "huggingface_hub"):
        try:
            v[pkg] = metadata.version(pkg)
        except metadata.PackageNotFoundError:
            v[pkg] = None
    return v


def build_replay(out: Out, folder: Path) -> Demo:
    p = folder / "rehearsal.json"
    if not p.exists():
        sys.exit(f"沒有 {os.path.relpath(p, HERE)}：先跑 ./present.sh rehearse")
    return Demo(out, json.loads(p.read_text(encoding="utf-8")), folder)


def fill_from_rehearsal(rec: dict, out: Out) -> None:
    """show 時 pyannote 沒跑起來：有彩排的那一份就拿來用（畫面會說明）。"""
    p = REHEARSAL / "rehearsal.json"
    if not p.exists():
        return
    old = json.loads(p.read_text(encoding="utf-8"))
    for pid, pr in rec["pairs"].items():
        got = old.get("pairs", {}).get(pid, {}).get("diar", {}).get("pyannote", {})
        if "error" in pr["diar"]["pyannote"] and got and "error" not in got and old["pairs"][pid]["ref_segments"] == pr["ref_segments"]:
            pr["diar"]["pyannote"] = {**got, "from_rehearsal": old.get("time")}
            out(f"  {C.AMBER}{pid}：pyannote 現場沒有跑起來，用彩排（{old.get('time')}）的輸出。{C.RESET}")


# ── fetch ───────────────────────────────────────────────────────────────
def fetch() -> None:
    import soundfile as sf
    from huggingface_hub import HfApi, snapshot_download
    C.banner("解出要用的句子；下載分離模型、ECAPA 與 pyannote 的 pipeline", C.AMBER)
    tar = _p(CFG["audio"]["dev_clean_tar"])
    want = wanted_utts()
    if not tar.exists():
        sys.exit(f"沒有 {tar}（W3 Demo 3 的 fetch 會下載 dev-clean.tar.gz）")
    import tarfile
    names = {f"{u}.flac": u for u in want}
    chapters = {"-".join(u.split("-")[:2]) + ".trans.txt" for u in want}
    texts, done = {}, 0
    with tarfile.open(tar, "r|gz") as tf:
        for m in tf:
            base = Path(m.name).name
            if base in chapters:
                for line in tf.extractfile(m).read().decode("utf-8").splitlines():
                    parts = line.strip().split(" ", 1)
                    if len(parts) == 2 and parts[0] in want:
                        texts[parts[0]] = parts[1]
            elif base in names:
                x, fs_in = sf.read(io.BytesIO(tf.extractfile(m).read()), dtype="float64")
                C.save_wav(wav_path(names[base]), dsp.resample(dsp.to_mono_float(x), fs_in, FS), FS)
                done += 1
    print(f"  {tar.name}：解出 {done}/{len(want)} 個 wav → {os.path.relpath(_p(CFG['audio']['wav_dir']), HERE)}；參考文字 {len(texts)} 句")
    write_json(SENTENCES, {"source": {"audio": "https://www.openslr.org/resources/12/dev-clean.tar.gz", "license": "CC BY 4.0", "time": C.now()},
                           "texts": {u: texts[u] for u in want if u in texts}})
    lock, lines = read_lock(), []
    for repo, pats in ((SEP["repo"], SEP_PATTERNS), (ENC["a"]["repo"], ECAPA_PATTERNS)):
        sha = lock.get(repo) or HfApi().model_info(repo).sha
        p = snapshot_download(repo, revision=sha, allow_patterns=pats)
        mb = sum(f.stat().st_size for f in Path(p).rglob("*") if f.is_file()) / 1e6
        print(f"  {repo} @ {sha[:12]}  {mb:.0f} MB")
        lines.append(f"{repo} {sha}")
    repo = DIAR["repo"]
    try:
        sha = lock.get(repo) or HfApi().model_info(repo).sha
        p = snapshot_download(repo, revision=sha)
        mb = sum(f.stat().st_size for f in Path(p).rglob("*") if f.is_file()) / 1e6
        print(f"  {repo} @ {sha[:12]}  {mb:.0f} MB")
        lines.append(f"{repo} {sha}")
    except Exception as e:
        print(f"  {C.AMBER}{repo}：{type(e).__name__}: {str(e).strip()[:300]}{C.RESET}")
        print(f"  {C.AMBER}這個模型要先在 https://huggingface.co/{repo} 同意使用條款，"
              f"再登入（../.venv-tts/bin/hf auth login，或設 HF_TOKEN），然後重跑 ./present.sh fetch。{C.RESET}")
        print(f"  {C.AMBER}沒有它，前半照跑；後半只有土法聚類的輸出。{C.RESET}")
    MODEL_LOCK.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"  → {MODEL_LOCK.name}")
    repo = ENC["b"]["repo"]
    try:
        snapshot_download(repo, revision=read_lock().get(repo), allow_patterns=WAVLM_PATTERNS)
        print(f"  W9 Demo 1 · {repo} @ {(read_lock().get(repo) or '最新')[:12]}：在")
    except Exception as e:
        print(f"  {C.RED}{repo}：{type(e).__name__}: {str(e)[:160]}{C.RESET}")
    try:
        W2.model_dir(offline=False)
        print(f"  W2 Demo 3 · {W2.CFG['model']['repo']} @ {(W2.read_lock() or '最新')[:12]}：在")
    except Exception as e:
        print(f"  {C.RED}{W2.CFG['model']['repo']}：{type(e).__name__}: {str(e)[:160]}{C.RESET}")
    py = _p(DIAR["env"]) / "bin" / "python"
    print(f"  {DIAR['env']}（pyannote 的環境）：{'在' if py.exists() else C.AMBER + '不在 ← ./setup.sh' + C.RESET}")
    print("下一步：./present.sh check")


# ── check ───────────────────────────────────────────────────────────────
def check() -> None:
    import torch
    C.banner("課前體檢", C.AMBER)
    rec = {"time": C.now(), "versions": versions(), "mps_available": bool(torch.backends.mps.is_available()),
           "mps_fallback_env": os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK")}
    for k, v in rec["versions"].items():
        print(f"  {k:<16s} {v}")
    print(f"  MPS：{rec['mps_available']}   PYTORCH_ENABLE_MPS_FALLBACK={rec['mps_fallback_env']}（0 = 嚴格：不支援的運算直接報錯）")
    audio, texts = material()
    conv = M.build(PAIRS[MAIN]["turns"], audio, FS, float(MIXC["level"]), float(MIXC["head_s"]), float(MIXC["tail_s"]), float(MIXC["floor_db"]), float(MIXC["min_pause_s"]))
    a, b = M.chunks(conv["n"], FS, CHUNKINGS[0])[0]
    y = conv["mix"][a:b]
    devs = ["cpu"] + (["mps"] if rec["mps_available"] else [])
    print(f"\n  {SEP['label']}：第一塊（{(b - a) / FS:g} 秒）")
    rec["sep"], raw = {}, {}
    for dev in devs:
        try:
            s = Separator(dev)
            s.raw(y)
            t0 = time.perf_counter()
            raw[dev] = s.raw(y)
            dt = time.perf_counter() - t0
            g = np.linalg.lstsq(raw[dev], y, rcond=None)[0]
            resid = float(10 * np.log10(np.sum((y - raw[dev] @ g) ** 2) / np.sum(y ** 2)))
            rec["sep"][dev] = {"ok": True, "seconds": dt, "gains": [float(g[0]), float(g[1])], "residual_db": resid, "n_params": s.n_params}
            print(f"    {dev:<4s} {dt:6.2f} s   兩軌的增益 {g[0]:+.4f}／{g[1]:+.4f}（負的＝極性反的）   兩軌加回去對混合訊號的殘差 {resid:.1f} dB")
        except Exception as e:
            rec["sep"][dev] = {"ok": False, "error": f"{type(e).__name__}: {str(e)[:300]}"}
            print(f"    {dev:<4s} {C.RED}不行：{type(e).__name__}: {str(e)[:200]}{C.RESET}")
    if "cpu" in raw and "mps" in raw:
        rel = float(np.max(np.abs(raw["mps"] - raw["cpu"])) / (np.max(np.abs(raw["cpu"])) + 1e-12))
        rec["sep"]["mps_vs_cpu_rel"] = rel
        print(f"    mps 對 cpu：最大相對差 {rel:.2e}")
    enr = M.trim(audio[CFG["user"]["enrol"]], FS)
    other = M.trim(audio[PAIRS[MAIN]["overlap_utt"]], FS)
    same = M.trim(audio[CFG["user"]["overlap_utt"]], FS)
    rec["enc"] = {}
    print("\n  兩個 speaker encoder：同一位的另一句／另一位的一句，對註冊語音的 SIM")
    for k, make in (("a", lambda d: Ecapa(d)), ("b", lambda d: WavLMSV(d))):
        rec["enc"][k] = {"label": ENC[k]["label"]}
        embs = {}
        for dev in devs:
            try:
                m = make(dev)
                t0 = time.perf_counter()
                embs[dev] = [m.embed(x) for x in (enr, same, other)]
                dt = (time.perf_counter() - t0) / 3
                rec["enc"][k][dev] = {"ok": True, "seconds": dt, "sim_same": cos(embs[dev][0], embs[dev][1]), "sim_other": cos(embs[dev][0], embs[dev][2]), "dim": int(len(embs[dev][0]))}
                print(f"    {ENC[k]['label']:<9s} {dev:<4s} {dt:5.2f} s   同一位 {rec['enc'][k][dev]['sim_same']:+.3f}   另一位 {rec['enc'][k][dev]['sim_other']:+.3f}   （門檻 {ENC[k]['threshold']:g}）")
            except Exception as e:
                rec["enc"][k][dev] = {"ok": False, "error": f"{type(e).__name__}: {str(e)[:300]}"}
                print(f"    {ENC[k]['label']:<9s} {dev:<4s} {C.RED}不行：{type(e).__name__}: {str(e)[:200]}{C.RESET}")
        if "cpu" in embs and "mps" in embs:
            rec["enc"][k]["mps_vs_cpu_cos"] = cos(embs["cpu"][0], embs["mps"][0])
            print(f"    {'':<14s} mps 對 cpu 的向量 cosine {rec['enc'][k]['mps_vs_cpu_cos']:.6f}")
    print(f"\n  {DIAR['label']}：整段「對話」（{conv['seconds']:.1f} 秒）")
    rec["diar"], outs = {}, {}
    for dev in devs:
        r = pyannote_diarize(conv["mix"], dev)
        if "error" in r:
            rec["diar"][dev] = {"ok": False, "error": r["error"]}
            print(f"    {dev:<4s} {C.AMBER}{r['error']}{C.RESET}")
            if dev == "cpu":
                break
            continue
        outs[dev] = r
        hyp = D.to_grid([tuple(s) for s in r["segments"]], conv["seconds"])
        d = D.der(conv["ref"], {k: v[: len(conv["ref"]["user"])] for k, v in hyp.items()}, 0.0, False)
        rec["diar"][dev] = {"ok": True, "seconds": r.get("compute_s"), "n_segments": len(r["segments"]), "n_speakers": len(hyp), "der_strict": d["der"], "versions": r.get("versions")}
        print(f"    {dev:<4s} {r.get('compute_s', 0):6.1f} s   {len(r['segments'])} 段、{len(hyp)} 位   DER（不給 collar、重疊照計）{100 * d['der']:.1f}%")
    if "cpu" in outs and "mps" in outs:
        gc = D.to_grid([tuple(s) for s in outs["cpu"]["segments"]], conv["seconds"])
        gm = D.to_grid([tuple(s) for s in outs["mps"]["segments"]], conv["seconds"])
        d = D.der(gc, gm, 0.0, False)
        ca, mb = sorted(outs["cpu"]["segments"]), sorted(outs["mps"]["segments"])
        same_n = len(ca) == len(mb)
        dmax = max((max(abs(x[0] - z[0]), abs(x[1] - z[1])) for x, z in zip(ca, mb)), default=0.0) if same_n else None
        rec["diar"]["mps_vs_cpu"] = {"same_n_segments": same_n, "max_boundary_diff_s": dmax, "der_mps_against_cpu": d["der"]}
        print(f"    mps 對 cpu：段數{'相同' if same_n else '不同'}" + (f"、邊界最大差 {1000 * dmax:.0f} ms" if dmax is not None else "")
              + f"；把 cpu 的輸出當參考，mps 的 DER {100 * d['der']:.2f}%（0 ＝ 逐格相同）")
    # der.py 對 pyannote.metrics（同一份參考、同一份系統輸出）
    hyp_segs = outs["cpu"]["segments"] if "cpu" in outs else [list(s) for s in naive_diarize(conv["mix"], Ecapa("cpu").embed, float(NAIVE["win_s"]), float(NAIVE["hop_s"]), float(NAIVE["cut"]))]
    ref_segs = [[a, b, s] for s, m in conv["ref"].items() for a, b in M.segments(m)]
    with tempfile.TemporaryDirectory(prefix="w10_d3_") as td:
        fr, fh = Path(td) / "ref.json", Path(td) / "hyp.json"
        fr.write_text(json.dumps(ref_segs))
        fh.write_text(json.dumps(hyp_segs))
        r = run_worker(["score", "--ref", str(fr), "--hyp", str(fh), "--seconds", str(conv["seconds"]), "--collar", str(COLLAR)])
    if "error" in r:
        rec["der_vs_pyannote_metrics"] = {"ok": False, "error": r["error"]}
        print(f"\n  der.py 對 pyannote.metrics：{C.AMBER}{r['error']}{C.RESET}")
    else:
        hyp = D.to_grid([tuple(s) for s in hyp_segs], conv["seconds"])
        hyp = {k: v[: len(conv["ref"]["user"])] for k, v in hyp.items()}
        rows = []
        print("\n  der.py 對 pyannote.metrics（同一份參考、同一份系統輸出）")
        for (name, c, s), theirs in zip(SCORINGS, r["scores"]):
            mine = D.der(conv["ref"], hyp, c, s)["der"]
            rows.append({"name": name, "der_py": mine, "pyannote_metrics": theirs["der"]})
            print(f"    {name:<34s} der.py {100 * mine:6.2f}%   pyannote.metrics {100 * theirs['der']:6.2f}%")
        rec["der_vs_pyannote_metrics"] = {"ok": True, "rows": rows, "versions": r.get("versions")}
    write_json(HERE / "runs" / "check.json", rec)
    print(f"\n  → runs/check.json\n下一步：./present.sh rehearse")


# ── export ──────────────────────────────────────────────────────────────
def export(rec_path: Path) -> None:
    rec = json.loads(rec_path.read_text(encoding="utf-8"))
    if rec.get("fake"):
        sys.exit("這份紀錄是 fake 的，不匯出")
    out = {"source": f"demos/{HERE.name}/runs/rehearsal/rehearsal.json", "time": rec["time"], "models": rec["models"],
           "chunk_s": rec["chunk_s"], "collar": rec["collar"], "thresholds": rec["thresholds"], "enc_labels": rec["enc_labels"], "main": rec["main"], "pairs": {}}
    for pid, p in rec["pairs"].items():
        out["pairs"][pid] = {
            "label": p["label"], "seconds": p["seconds"], "stats": p["stats"],
            "overlap": {"seconds": p["overlap"]["seconds"], "si_sdr_user": [t["si_user"] for t in p["overlap"]["tracks"]],
                        "si_sdr_other": [t["si_other"] for t in p["overlap"]["tracks"]], "sim": [t["sim"] for t in p["overlap"]["tracks"]],
                        "wer_mix": p["overlap"]["asr"]["mix"]["wer"], "wer_user_track": p["overlap"]["asr"].get("user_track", {}).get("wer")},
            "chunkings": {cs: {"n_present": c["n_present"], "n_swaps": c["n_swaps"], "n_right": c["n_right"],
                               "rows": [{"k": r["k"], "a_s": r["a_s"], "b_s": r["b_s"], "user_s": round(r["user_s"], 2), "truth": r["truth"], "pick": r["pick"],
                                         "sim": [t["sim"] for t in r["tracks"]], "si_sdr_user": [t["si_user"] for t in r["tracks"]]} for r in c["rows"]]}
                          for cs, c in p["chunkings"].items()},
            "asr": {v: {k: a[k] for k in ("wer", "S", "D", "I", "n_ref")} for v, a in p["asr"].items()},
            "diar": {name: ({"error": d["error"]} if "error" in d else {"n_speakers": d["n_speakers"], "n_segments": len(d["segments"]), "der": d["der"]}) for name, d in p["diar"].items()}}
    allm = json.loads(METRICS_JSON.read_text(encoding="utf-8")) if METRICS_JSON.exists() else {}
    allm["demo3"] = out
    write_json(METRICS_JSON, allm)
    print(f"  → {os.path.relpath(METRICS_JSON, HERE)}（鍵 demo3）")


# ── selftest ────────────────────────────────────────────────────────────
def selftest() -> None:
    C.banner("selftest（不需要模型）", C.AMBER)
    n_ok = [0, 0]

    def ok(cond: bool, msg: str) -> None:
        n_ok[0 if cond else 1] += 1
        print(f"  {C.GREEN + '✓' if cond else C.RED + '✗'}{C.RESET} {msg}")

    rng = np.random.default_rng(0)
    # 1 接對話
    def tone(sec, f):
        t = np.arange(int(sec * FS)) / FS
        return np.concatenate([np.zeros(1600), np.sin(2 * np.pi * f * t), np.zeros(1600)])
    audio = {"u1": tone(2.0, 200), "o1": tone(3.0, 310), "u2": tone(1.5, 200)}
    turns = [{"spk": "user", "utt": "u1", "lead": 0.0}, {"spk": "other", "utt": "o1", "lead": -0.5}, {"spk": "user", "utt": "u2", "lead": 0.3}]
    c = M.build(turns, audio, FS, 0.05, 0.5, 0.5)
    st = M.overlap_stats(c["ref"])
    ok(np.allclose(c["mix"], c["tracks"]["user"] + c["tracks"]["other"]), "build：混合 = 兩位各自的軌相加")
    ok(abs(st["overlap_s"] - 0.5) <= 0.03, f"build：lead = −0.5 → 重疊 {st['overlap_s']:.2f} s")
    ok(abs(c["turns"][2]["start"] - c["turns"][1]["end"] - 0.3) < 1e-3, "build：lead = +0.3 → 隔 0.3 s 才接話")
    ok(abs(st["speech_s"]["user"] - 3.5) <= 0.05 and abs(st["speech_s"]["other"] - 3.0) <= 0.05, f"activity：使用者 {st['speech_s']['user']} s、另一位 {st['speech_s']['other']} s（裁掉頭尾的靜音）")
    a0 = int(round(c["turns"][0]["start"] * FS))
    ok(abs(M.rms(c["tracks"]["user"][a0: a0 + 2 * FS]) - 0.05) < 2e-3 and abs(c["seconds"] - (c["turns"][-1]["end"] + 0.5)) < 1e-3, "build：每一句調到同一個 RMS；尾巴留 0.5 s")
    # 2 格子
    m = rng.random(700) > 0.5
    m = M._fill(M._fill(m, False, 5), True, 5)
    g = D.to_grid([(a, b, "x") for a, b in M.segments(m)], len(m) * M.STEP)["x"]
    ok(np.array_equal(g, m), "segments ↔ to_grid：來回一樣")
    sp = M.chunks(int(39.4 * FS), FS, 4.0)
    ok(sp[0] == (0, 4 * FS) and sp[-1][1] == int(39.4 * FS) and all(a2 == b1 for (_, b1), (a2, _) in zip(sp[:-1], sp[1:])) and len(sp) == 10, f"chunks：39.4 s／4 s → {len(sp)} 塊、首尾相接")
    ok(len(M.chunks(int(33.0 * FS), FS, 4.0)) == 8, "chunks：最後一塊短於半塊就併進前一塊")
    # 3 SI-SDR
    s, v = rng.standard_normal(16000), rng.standard_normal(16000)
    e = s + 0.1 * v
    ok(abs(M.si_sdr(e, s) - 20.0) < 0.3, f"si_sdr：s + 0.1 v → {M.si_sdr(e, s):.2f} dB")
    ok(abs(M.si_sdr(-3.7 * e, s) - M.si_sdr(e, s)) < 1e-9, "si_sdr：不看音量、不看極性")
    A = np.stack([-41.0 * (s + 0.01 * v), 37.0 * (v + 0.01 * s)], 1)
    gsol = np.linalg.lstsq(A, s + v, rcond=None)[0]
    ok(gsol[0] < 0 < gsol[1] and np.sum((s + v - A @ gsol) ** 2) / np.sum((s + v) ** 2) < 1e-3, "兩軌的增益：最小平方把反的極性擺正")
    # 4 DER：投影片 p86 的玩具
    ref_t = {"A": ((0.5, 6.0), (9.0, 13.0)), "B": ((5.2, 9.4), (13.5, 18.0))}
    hyp_t = {"A": ((0.6, 6.6), (9.2, 13.3)), "B": ((6.6, 9.2), (13.8, 18.4))}
    R = D.to_grid([(a, b, k) for k, ss in ref_t.items() for a, b in ss], 20.0)
    H = D.to_grid([(a, b, "sys_" + k) for k, ss in hyp_t.items() for a, b in ss], 20.0)
    got = [D.der(R, H, cc, so) for _, cc, so in D.SCORINGS]
    want = [(13.8, 0.15, 0.05, 0.35, 4.0), (14.4, 0.15, 0.35, 0.35, 5.9), (18.2, 0.7, 1.6, 0.6, 15.9)]
    for gk, wk, (name, _, _) in zip(got, want, D.SCORINGS):
        mine = (round(gk["total"], 2), round(gk["fa"], 2), round(gk["miss"], 2), round(gk["conf"], 2), round(100 * gk["der"], 1))
        ok(mine == wk, f"DER 玩具（p86）· {name}：{mine}")
    if DATA_JSON.exists():
        sc = json.loads(DATA_JSON.read_text(encoding="utf-8")).get("der", {}).get("scoring", {})
        ok([round(100 * gk["der"], 1) for gk in got] == [v["DER_percent"] for v in sc.values()], "DER 玩具：與 w10-data.json → der 相同")
    ok(got[2]["map"] == {"sys_A": "A", "sys_B": "B"}, "DER：系統的編號自己配到參考的說話人")
    ok(D.der(R, {"q": R["B"], "z": R["A"]})["der"] == 0.0, "DER：輸出與參考相同、只是編號不同 → 0")
    ok(abs(D.der(R, {})["der"] - 1.0) < 1e-12 and D.der(R, {})["fa"] == 0, "DER：什麼都沒輸出 → 100%，全部是 missed")
    one = {"only": R["A"] | R["B"]}
    d1 = D.der(R, one)
    ok(d1["fa"] == 0 and abs(d1["miss"] - float((R["A"] & R["B"]).sum() * D.STEP)) < 1e-9 and d1["conf"] > 0, "DER：全部標成同一個人 → 重疊處各漏一位、其餘是 wrong speaker")
    # 5 fake 整條流程
    audio, texts = fake_material()
    out = Out(echo=False)
    rec, wavs = compute(Fake(), audio, texts, out, "selftest")
    ok(set(rec["pairs"]) == set(PAIRS) and all(set(p["chunkings"]) >= {ck(cs) for cs in CHUNKINGS} for p in rec["pairs"].values()), "fake：每一組、每一種塊長都算了")
    p = rec["pairs"][MAIN]
    ok(all(len(r["tracks"]) == 2 for r in p["chunkings"][ck(CHUNKINGS[0])]["rows"]) and p["chunkings"][ck(CHUNKINGS[0])]["rows"][-1]["b_s"] == p["seconds"], "fake：每一塊兩軌、塊蓋滿整段")
    ok(p["asr"]["clean"]["wer"] <= p["asr"]["mix"]["wer"] + 1e-9 and p["asr"]["clean"]["n_ref"] > 0, "fake：使用者自己的那一軌 WER 不比混合的高")
    ok("error" not in p["diar"]["pyannote"] and len(p["diar"]["pyannote"]["der"]) == 3 and "segments" in p["diar"]["naive"], "fake：兩個 diarization 的輸出都有三個 DER")
    blob = json.dumps(rec, ensure_ascii=False)
    ok("/Users/" not in blob and "/home/" not in blob and str(HERE) not in blob, "紀錄裡沒有本機的絕對路徑")
    tmp = HERE / "runs" / "_selftest"
    write_run(tmp, rec, wavs, out)
    d = build_replay(Out(echo=False), tmp)
    play, C.play = C.play, (lambda p: None)
    try:
        d.print_all()
        for k in "1p23on4l5acxg67kt?":
            d.dispatch(k)
    finally:
        C.play = play
    ok(len(d.out.text()) > 3000, "replay：每一個鍵都走過一遍")
    try:
        d.save_figs()
        ok((tmp / f"{MAIN}_chunks_a.png").exists() and (tmp / f"{MAIN}_der_naive.png").exists(), "圖：存得出來")
    except ImportError:
        print(f"  {C.GREY}（沒有 matplotlib，跳過圖）{C.RESET}")
    print(f"\n  {n_ok[0]} 項通過、{n_ok[1]} 項失敗")
    if n_ok[1]:
        sys.exit(1)


# ── main ────────────────────────────────────────────────────────────────
def main() -> None:
    ap = argparse.ArgumentParser(description="W10 demo 3")
    ap.add_argument("cmd", nargs="?", default="show", choices=["show", "rehearse", "replay", "check", "fetch", "fake", "selftest", "export"])
    ap.add_argument("--device", default=None, help="分離模型與 WavLM-SV、pyannote 放哪裡（cpu／mps）；預設照設定檔")
    ap.add_argument("--no-diar", action="store_true", help="不跑 pyannote")
    ap.add_argument("--no-plot", action="store_true", help="只在終端機，不開圖")
    ap.add_argument("--from", dest="src", default=None, help="replay 讀哪一個資料夾（預設 runs/rehearsal）")
    ap.add_argument("--pairs", default=None, help="只算哪幾組（逗號分隔的 id）")
    a = ap.parse_args()
    if a.cmd == "selftest":
        return selftest()
    if a.cmd == "fetch":
        return fetch()
    if a.cmd == "check":
        return check()
    if a.cmd == "export":
        return export(REHEARSAL / "rehearsal.json")
    out = Out()
    if a.cmd == "replay":
        d = build_replay(out, _p(a.src) if a.src else REHEARSAL)
        out.banner(f"REPLAY：{d.rec['time']} 的紀錄（{d.rec['tag']}）", C.AMBER)
        return run(d, not a.no_plot)
    pairs = a.pairs.split(",") if a.pairs else None
    if a.cmd == "fake":
        audio, texts = fake_material()
        out.banner("FAKE：替身，數字沒有意義", C.RED)
        rec, wavs = compute(Fake(), audio, texts, out, "fake", pairs=pairs)
        folder = HERE / "runs" / "fake"
        write_run(folder, rec, wavs)
        return run(Demo(out, rec, folder), not a.no_plot)
    audio, texts = material()
    out.banner("全部算一遍：分離、兩個 encoder、Whisper、diarization", C.AMBER)
    t0 = time.perf_counter()
    rec, wavs = compute(Real(a.device), audio, texts, out, a.cmd, diar=not a.no_diar, device=a.device, pairs=pairs)
    rec["versions"] = versions()
    rec["compute_s"] = time.perf_counter() - t0
    out(f"  算完：{rec['compute_s']:.0f} s")
    if a.cmd == "rehearse":
        d = Demo(out, rec, REHEARSAL)
        write_run(REHEARSAL, rec, wavs)
        d.print_all()
        (REHEARSAL / "rehearsal.txt").write_text(out.text(), encoding="utf-8")
        d.save_figs()
        export(REHEARSAL / "rehearsal.json")
        print(f"\n  → runs/rehearsal/（rehearsal.json、rehearsal.txt、圖、要聽的 .wav）\n下一步：./present.sh，照 p78、p87 的【操作】按一遍")
        return
    fill_from_rehearsal(rec, out)
    write_run(LIVE, rec, wavs)
    run(Demo(out, rec, LIVE), not a.no_plot)


if __name__ == "__main__":
    main()

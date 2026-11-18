# -*- coding: utf-8 -*-
# 此檔由 bin/publish_demos.py 從 private repo 的 <course>/demos/ 複製而來，請勿在公開 repo 直接編輯。
"""W8 demo 2：NFE、CFG 與第一包——品質–速度曲線的膝點（前半，p62）、兩種範式的第一包（後半，p81）。

  show       課堂：載 F5（Demo 1 的載法）與 MLX worker（Qwen3-TTS、Kokoro），開一個視窗；鍵在視窗裡按。**按鍵時才生成**——等待本身是這個 demo 的內容。
  rehearse   課前：不互動，全部算一遍，.wav、mel（.npy）、圖、時間存 runs/rehearsal/（進版控 = 現場的退路），寫 slides/assets/w08/w08-metrics.json → demo2
  replay     現場出狀況時：從 runs/rehearsal/ 讀回，同一個視窗、同樣的鍵，不碰 torch 也不開 worker
  check      課前體檢：版本、mlx-audio 裡有沒有 qwen3_tts；自寫的 sampler 對 f5-tts 的 sample()（同 seed、同 NFE 逐格相同）；一步兩次前向在 cpu／mps 各多久 → runs/check.json
  fetch      課前一週：下載 Qwen3-TTS 與 Kokoro（鎖 commit → model.lock；F5 用 Demo 1 的）（要網路）
  fake       沒有模型也能演練流程（F5 用正弦波冒充、worker --fake；畫面標明 FAKE）
  selftest   不開視窗、不需要模型：三句的字數與 F5 的長度、時刻表（等距／EPSS／sway）、CFG 的 w ↔ α、fake 流程；有彩排紀錄時對 w08-metrics.json
  export     只重寫 w08-metrics.json → demo2（從 runs/rehearsal/rehearsal.json）

畫面上的鍵（順序照投影片 p62、p81 講稿的【操作】）：
  1  NFE 掃描：同一句（訂票句）、同一個 prompt、同一個 seed，NFE = 1, 2, 4, 8, 16, 32 各生成一次——每次印秒數、播；六張 mel 排成兩列
  e  同一掃描換另一句（英文，W3 那句；NFE 32 這一格與 Demo 1 的 seed 1 逐格相同）
  2  CFG 掃描：NFE 16 固定，w = 0, 1, 2, 3, 4（投影片 p55 的尺度；F5 的 cfg_strength α = w − 1，預設 w = 3）——w = 0、1 每步一次前向，其餘兩次
  3  sway 開（s = −1）對關（s = 0），同樣 NFE 16 等距
  p  F5 自己的 16 步時刻表（EPSS ＋ sway）當對照
  4  第一包 · F5 整句：10／30／90 個漢字，NFE 32——第一包＝整句生成完
  5  第一包 · Qwen3-TTS 串流（mlx-audio，0.32 s = 4 個 token 一包）：第一個 chunk 送出的時間、之後每包的間隔對它的長度（RTF）
  6  第一包 · Kokoro（W1 Demo 2 的設定，整句、iSTFT）
  t  數字表     r  再播     h  選單     q  離開

F5 的載入、prompt 前處理、長度公式、前端都沿用 Demo 1（importlib 載 ../w08_d1_mean_vs_sample/demo.py）。
**取樣這一段自己寫**（照 f5_tts/model/cfm.py 的 sample() 逐行，三處不同）：(1) 時刻表一律等距再套 sway（f5-tts 對 NFE 16 預設用 EPSS 那張表，掃描要一次只變一件事）；
(2) CFG 寫成投影片的 w：ṽ = v(∅) + w (v(c) − v(∅))，w = 0 是純無條件場——f5-tts 的 cfg_strength < 1e-5 會退成純條件，表達不出 w = 0；
(3) 每一步計時。check 對 f5-tts 的 sample()：同 seed、同 NFE、w = 3、sway −1 時逐格相同。
"""
from __future__ import annotations

import argparse
import importlib.util
import io
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time
import tomllib
import warnings
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import common as C  # noqa: E402
import dsp          # noqa: E402

CFG = tomllib.loads((HERE / "demo_config.toml").read_text(encoding="utf-8"))


def _load_d1():
    p = (HERE / CFG["d1"]["dir"] / "demo.py").resolve()
    if not p.exists():
        sys.exit(f"找不到 Demo 1 的程式 {p}（demo_config.toml [d1].dir）")
    spec = importlib.util.spec_from_file_location("w08_d1_demo", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


D1 = _load_d1()
D1_DIR = (HERE / CFG["d1"]["dir"]).resolve()
PROMPT_WAV = D1_DIR / D1.CFG["audio"]["prompt_path"]
PROMPT_TEXT = D1.TXT["prompt_text"]
MODEL_LOCK = HERE / "model.lock"
METRICS_JSON = D1.METRICS_JSON
REHEARSAL = HERE / "runs" / "rehearsal"
WORKER_PY = HERE / "mlx_worker.py"
WORKER_PY_ENV = HERE.parent / ".venv" / "bin" / "python"           # mlx-audio 住在共用環境
F5C, QC, KC = CFG["f5"], CFG["qwen3"], CFG["kokoro"]
TEXTS = CFG["text"]
FP_TEXTS = CFG["first_packet"]["texts"]
HOP, FS24 = 256, 24000
REPOS = {"qwen3": QC["repo"], "kokoro": KC["repo"]}


def cjk(s: str) -> int:
    return len(re.findall(r"[一-鿿]", s))


def short(s: str, n: int = 14) -> str:
    return s if len(s) <= n else s[: n - 1] + "…"


def read_lock() -> dict[str, str]:
    out = {}
    if MODEL_LOCK.exists():
        for line in MODEL_LOCK.read_text(encoding="utf-8").splitlines():
            if line.strip() and not line.startswith("#"):
                repo, sha = line.split()[:2]
                out[repo] = sha
    return out


# ── 時刻表與 CFG 的兩個小函數（純 numpy，selftest 對）────────
EPSS = {5: [0, 2, 4, 8, 16, 32], 6: [0, 2, 4, 6, 8, 16, 32], 7: [0, 2, 4, 6, 8, 16, 24, 32], 10: [0, 2, 4, 6, 8, 12, 16, 20, 24, 28, 32],
        12: [0, 2, 4, 6, 8, 10, 12, 14, 16, 20, 24, 28, 32], 16: [0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 16, 20, 24, 28, 32]}   # f5_tts.model.utils.get_epss_timesteps


def time_grid(nfe: int, sway: float | None, epss: bool) -> np.ndarray:
    """τ 的格點（nfe + 1 個）：等距或 F5 的 EPSS 表，再套 sway：τ ← τ + s (cos(πτ/2) − 1 + τ)（F5 Eq. 7）。"""
    t = np.array(EPSS[nfe], dtype=np.float64) / 32 if (epss and nfe in EPSS) else np.linspace(0.0, 1.0, nfe + 1)
    if sway is not None and sway != 0:
        t = t + sway * (np.cos(np.pi / 2 * t) - 1 + t)
    return t


def alpha_of(w: float) -> float:
    """投影片的 w（加在無條件場上）↔ F5 的 cfg_strength α（Ho & Salimans 型）：w = α + 1。"""
    return w - 1


def passes_per_step(w: float) -> int:
    return 1 if w in (0, 1) else 2


# ── F5 的取樣（自寫，照 cfm.py 的 sample）─────────────────────
class F5Sampler:
    """包住 Demo 1 的 F5（模型、vocoder、prompt 前處理），取樣自己做：時刻表、CFG 的 w、每步計時。"""
    label = "F5-TTS"

    def __init__(self, m):
        self.m = m
        self.torch = m.torch
        self.device, self.vdev = m.device, m.vdev
        p16 = C.load_wav(PROMPT_WAV, 16000)
        self.prompt24 = m.prompt(p16)
        self.prompt_seconds = len(self.prompt24) / FS24
        from f5_tts.model.utils import lens_to_mask, list_str_to_idx
        from torchdiffeq import odeint
        self._lens_to_mask, self._to_idx, self._odeint = lens_to_mask, list_str_to_idx, odeint
        self._convert = m.f["utils_infer"].convert_char_to_pinyin
        self.warm = False

    def info(self) -> dict:
        i = self.m.info()
        i.update(prompt_utt=D1.TXT["prompt_utt"], prompt_text=PROMPT_TEXT, prompt_seconds=round(self.prompt_seconds, 3),
                 sampler="demo.py F5Sampler (cfm.py sample() re-written: uniform or EPSS grid, sway, CFG as the slides' w, per-step timing)")
        return i

    def length(self, text: str) -> dict:
        return D1.f5_length(len(self.prompt24), PROMPT_TEXT, text, D1.CFG["f5"]["speed"])

    def sample(self, text: str, seed: int, nfe: int, w: float, sway: float | None, epss: bool = False, target_rms: float = 0.1) -> dict:
        torch, model = self.torch, self.m.model
        dev = self.device
        L = self.length(text)
        audio = torch.tensor(self.prompt24, dtype=torch.float32)[None]
        rms = torch.sqrt(torch.mean(torch.square(audio)))
        if rms < target_rms:
            audio = audio * target_rms / rms
        audio = audio.to(dev)
        ref_text = D1.f5_ref_text(PROMPT_TEXT)                               # infer_batch_process 用的 prompt 文字（句尾 ". " 再補一個空格）
        text_ids = self._to_idx(self._convert([ref_text + text]), model.vocab_char_map).to(dev)
        duration = L["ref_frames"] + L["gen_frames"]
        step_s: list[float] = []
        t0 = time.perf_counter()
        with torch.inference_mode(), D1._quiet():
            cond = model.mel_spec(audio).permute(0, 2, 1).to(next(model.parameters()).dtype)
            cond_len = cond.shape[1]
            lens = torch.full((1,), cond_len, device=dev, dtype=torch.long)
            dur = torch.full((1,), duration, device=dev, dtype=torch.long)
            dur = torch.maximum(torch.maximum((text_ids != -1).sum(dim=-1), lens) + 1, dur)
            total = int(dur.amax())
            cond = torch.nn.functional.pad(cond, (0, 0, 0, total - cond_len), value=0.0)
            cond_mask = self._lens_to_mask(lens)
            cond_mask = torch.nn.functional.pad(cond_mask, (0, total - cond_mask.shape[-1]), value=False).unsqueeze(-1)
            step_cond = torch.where(cond_mask, cond, torch.zeros_like(cond))
            tr = model.transformer

            def field(t, x):
                s0 = time.perf_counter()
                if w == 1:
                    v = tr(x=x, cond=step_cond, text=text_ids, time=t, mask=None, drop_audio_cond=False, drop_text=False, cache=True)
                elif w == 0:
                    v = tr(x=x, cond=step_cond, text=text_ids, time=t, mask=None, drop_audio_cond=True, drop_text=True, cache=True)
                else:
                    both = tr(x=x, cond=step_cond, text=text_ids, time=t, mask=None, cfg_infer=True, cache=True)
                    v_c, v_u = torch.chunk(both, 2, dim=0)
                    v = v_u + w * (v_c - v_u)                                  # = v_c + (w − 1)(v_c − v_u) = cfm.py 的 pred + α (pred − null_pred)
                D1.sync(dev)
                step_s.append(time.perf_counter() - s0)
                return v

            torch.manual_seed(seed)
            y0 = torch.randn(total, model.num_channels, device=model.device, dtype=step_cond.dtype)[None]
            # 時刻表照 cfm.py 的寫法在 torch 裡算（float32 的 linspace 再套 sway），check 才對得到逐位相同；time_grid() 是同一條式子的 numpy 版（selftest 用）
            if epss and nfe in EPSS:
                grid = (1 / 32) * torch.tensor(EPSS[nfe], device=dev, dtype=step_cond.dtype)
            else:
                grid = torch.linspace(0, 1, nfe + 1, device=dev, dtype=step_cond.dtype)
            if sway is not None and sway != 0:
                grid = grid + sway * (torch.cos(torch.pi / 2 * grid) - 1 + grid)
            traj = self._odeint(field, y0, grid, **model.odeint_kwargs)
            tr.clear_cache()
            out = torch.where(cond_mask, cond, traj[-1])
            gen = out[:, L["ref_frames"]:, :].permute(0, 2, 1).float()        # infer_batch_process 從 ⌊samples/hop⌋ 切，不是 mel_spec 的 frame 數（center STFT 多一格）
            D1.sync(dev)
            t_ac = time.perf_counter() - t0
            self.m.vocoder.seconds = 0.0
            wav = self.m.vocoder.decode(gen)
            if rms < target_rms:
                wav = wav * rms / target_rms
            wav = wav.squeeze().float().cpu().numpy().astype(np.float64)
            mel = gen[0].cpu().numpy()
        total_s = time.perf_counter() - t0
        return {"mel": mel, "wav": wav, "fs": FS24, "T": int(mel.shape[1]), "seconds": mel.shape[1] * HOP / FS24, "nfe": nfe, "w": w, "alpha": alpha_of(w),
                "sway": sway, "epss": bool(epss and nfe in EPSS), "seed": seed, "passes_per_step": passes_per_step(w), "n_field_calls": len(step_s),
                "acoustic_s": t_ac, "vocoder_s": self.m.vocoder.seconds, "total_s": total_s, "step_s": [round(s, 4) for s in step_s],
                "grid": [round(float(g), 4) for g in time_grid(nfe, sway, epss)], "length": L, "text": text, "cjk": cjk(text), "bytes": len(text.encode("utf-8"))}

    def warmup(self) -> None:
        if not self.warm:
            self.sample("Hi there.", 0, 1, F5C["w_default"], F5C["sway"])
            self.warm = True


class FakeF5:
    """沒有模型時的替身：正弦波加噪音，NFE 越多噪音越少、w 越大音高越誇張；時間照 [fake].f5_seconds_per_pass 算。畫面標 FAKE。"""
    label = "F5-TTS (FAKE)"
    device = vdev = "fake"
    prompt_seconds = 4.91

    def info(self) -> dict:
        return {"fake": True, "prompt_text": PROMPT_TEXT}

    def length(self, text: str) -> dict:
        return D1.f5_length(int(self.prompt_seconds * FS24), PROMPT_TEXT, text, 1.0)

    def sample(self, text: str, seed: int, nfe: int, w: float, sway: float | None, epss: bool = False, **k) -> dict:
        L = self.length(text)
        T = L["gen_frames"]
        n = T * HOP
        rng = np.random.default_rng(seed)
        tt = np.arange(n) / FS24
        f0 = 150 * (1 + 0.15 * (w - 1)) * (1 + 0.1 * np.sin(2 * np.pi * 1.3 * tt))
        wav = 0.08 * np.sin(2 * np.pi * np.cumsum(f0) / FS24) + 0.05 / math.sqrt(nfe) * rng.standard_normal(n)
        fe = D1.front_ends(real=False)["f5"]
        mel = fe.mel(wav, FS24)[:, :T] if T > 0 else np.zeros((100, 0))
        per = CFG["fake"]["f5_seconds_per_pass"] * passes_per_step(w) * (T + int(self.prompt_seconds * FS24 / HOP)) / 700
        step = [per] * nfe
        time.sleep(min(0.3, sum(step)))
        return {"mel": mel, "wav": wav, "fs": FS24, "T": int(mel.shape[1]), "seconds": mel.shape[1] * HOP / FS24, "nfe": nfe, "w": w, "alpha": alpha_of(w),
                "sway": sway, "epss": bool(epss and nfe in EPSS), "seed": seed, "passes_per_step": passes_per_step(w), "n_field_calls": nfe,
                "acoustic_s": sum(step), "vocoder_s": 0.01, "total_s": sum(step) + 0.01, "step_s": [round(s, 4) for s in step],
                "grid": [round(float(g), 4) for g in time_grid(nfe, sway, epss)], "length": L, "text": text, "cjk": cjk(text), "bytes": len(text.encode("utf-8"))}

    def warmup(self) -> None:
        pass


# ── MLX worker（Qwen3-TTS、Kokoro）───────────────────────────
class Worker:
    """共用 ../.venv 裡的 mlx_worker.py，一行 JSON 進、一行 JSON 出。"""

    def __init__(self, out_dir: Path, fake: bool = False, out=print):
        self.fake, self.out_dir, self.out = fake, out_dir, out
        py = WORKER_PY_ENV if WORKER_PY_ENV.exists() else Path(sys.executable)
        if fake and not WORKER_PY_ENV.exists():
            py = Path(sys.executable)
        env = dict(os.environ, HF_HUB_OFFLINE=os.environ.get("HF_HUB_OFFLINE", "1"), HF_HUB_DISABLE_PROGRESS_BARS="1", TOKENIZERS_PARALLELISM="false")
        args = [str(py), str(WORKER_PY), "--out", str(out_dir)] + (["--fake"] if fake else [])
        self.p = subprocess.Popen(args, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=sys.stderr, text=True, encoding="utf-8", bufsize=1, env=env)
        self.ready = self._read()
        self.versions = self.ready.get("versions", {})
        self.loaded: dict = {}
        if not self.ready.get("ok"):
            raise RuntimeError(f"mlx worker 起不來：{self.ready.get('error')}")

    def _read(self) -> dict:
        line = self.p.stdout.readline()
        if not line:
            raise RuntimeError("mlx worker 沒有回應（看 stderr）")
        return json.loads(line)

    def call(self, **req) -> dict:
        self.p.stdin.write(json.dumps(req, ensure_ascii=False) + "\n")
        self.p.stdin.flush()
        return self._read()

    def load(self, qwen3: bool = True, kokoro: bool = True) -> dict:
        lock = read_lock()
        r = self.call(cmd="load", qwen3=REPOS["qwen3"] if qwen3 else None, kokoro=REPOS["kokoro"] if kokoro else None,
                      revision={k: lock.get(v) for k, v in REPOS.items()})
        if not r.get("ok"):
            raise RuntimeError(f"載入失敗：{r.get('error')}")
        self.loaded = r.get("loaded", {})
        return r

    def qwen3(self, name: str, text: str) -> dict:
        return self.call(cmd="qwen3", name=name, text=text, ref_audio=str(PROMPT_WAV), ref_text=PROMPT_TEXT, interval=QC["streaming_interval"],
                         seed=QC["seed"], temperature=QC["temperature"], lang_code=QC["lang_code"])

    def kokoro(self, name: str, text: str) -> dict:
        return self.call(cmd="kokoro", name=name, text=text, voice=KC["voice"], lang_code=KC["lang_code"])

    def close(self) -> None:
        try:
            self.call(cmd="quit")
        except Exception:
            pass
        try:
            self.p.stdin.close(); self.p.wait(timeout=10)
        except Exception:
            self.p.kill()


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
SWEEP_KEY = TEXTS["sweep"]
OTHER_KEY = "en" if SWEEP_KEY == "zh" else "zh"


def nfe_name(key: str, nfe: int) -> str:
    return f"nfe_{key}_{nfe:02d}"


def cfg_name(w: float) -> str:
    return f"cfg_w{w:g}"


def sway_name(s: float, epss: bool = False) -> str:
    return "epss16" if epss else ("sway_on" if s else "sway_off")


def fp_name(system: str, i: int) -> str:
    return f"fp_{system}_{i}"


class Demo:
    """items[name] = 一次生成（dict：mel、wav、時間……）；sections 記每個鍵做了什麼、照順序。"""

    def __init__(self, out: Out, f5, worker: Worker | None, folder: Path, tag: str = ""):
        self.out, self.f5, self.worker, self.folder, self.tag = out, f5, worker, folder, tag
        self.items: dict[str, dict] = {}
        self.fe = D1.front_ends(real=False)["f5"]
        self.log: dict = {"time": C.now(), "versions": versions(), "tag": tag, "texts": {"zh": TEXTS["zh"], "en": TEXTS["en"], "sweep": SWEEP_KEY, "first_packet": FP_TEXTS},
                          "env": {"PYTORCH_ENABLE_MPS_FALLBACK_at_start": os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK")},
                          "models": {"f5": f5.info() if f5 else None, "mlx": {"versions": worker.versions, "loaded": worker.loaded} if worker else None},
                          "config": {"f5": F5C, "qwen3": QC, "kokoro": KC}, "items": {}, "played": [], "first_packet": {}}
        self.fig = self.plt = None
        self.mode = "nfe"
        self.mode_key = SWEEP_KEY
        self.last = None
        self.ref_rms = D1.rms(C.load_wav((HERE / CFG["audio"]["match_rms_to"]).resolve(), 16000)) if (HERE / CFG["audio"]["match_rms_to"]).resolve().exists() else 0.05
        self.folder.mkdir(parents=True, exist_ok=True)

    # ── 生成 ──
    def _store(self, name: str, r: dict, play: bool = True, what: str = "") -> dict:
        r["name"] = name
        self.items[name] = r
        y = r["wav"]
        g = self.ref_rms / D1.rms(y) if len(y) else 1.0
        z = y * g
        pk = float(np.max(np.abs(z))) if len(z) else 0.0
        if pk > 0.99:
            z = z * 0.99 / pk
        p = self.folder / f"{name}.wav"
        C.save_wav(p, z, r["fs"])
        r["wav_path"] = p
        rec = {k: v for k, v in r.items() if k not in ("mel", "wav", "wav_path")}
        for k in ("seconds", "acoustic_s", "vocoder_s", "total_s"):
            if k in rec:
                rec[k] = round(rec[k], 4)
        rec["wav"] = os.path.relpath(p, HERE)
        rec["play_gain"] = round(float(g), 4)
        self.log["items"][name] = rec
        if play:
            self._play(p, what or name)
        return r

    def _play(self, path: Path, what: str) -> None:
        self.out(f"  ▶ {what}")
        self.last = (path, what)
        self.log["played"].append(what)
        C.play(path)

    def gen_f5(self, name: str, text: str, nfe: int, w: float, sway: float | None, epss: bool = False, play: bool = True, what: str = "") -> dict:
        if name in self.items:
            r = self.items[name]
            if play:
                self._play(r["wav_path"], what or name)
            return r
        if self.f5 is None:
            self.out(f"  {C.AMBER}沒有 F5（replay 裡沒有 {name}）{C.RESET}"); return {}
        self.f5.warmup()
        r = self.f5.sample(text, F5C["seed"], nfe, w, sway, epss)
        return self._store(name, r, play, what)

    # ── 鍵 1／e：NFE 掃描 ──
    def nfe_sweep(self, key: str) -> None:
        text = TEXTS[key]
        self.mode, self.mode_key = "nfe", key
        self.out(f"\n{C.BOLD}NFE 掃描{C.RESET}  「{text}」  prompt {D1.TXT['prompt_utt']}  seed {F5C['seed']}  w = {F5C['w_default']:g}（α = {alpha_of(F5C['w_default']):g}）  sway {F5C['sway']:g}  等距時刻表")
        if self.f5 is not None:
            L = self.f5.length(text)
            self.out(f"  長度 T = ⌊{L['ref_frames']} · {L['gen_bytes']} / {L['ref_bytes']}⌋ = {L['gen_frames']} frames = {L['gen_frames'] * HOP / FS24:.2f} s（六次都一樣——長度不是抽的）")
        for nfe in F5C["nfe_sweep"]:
            n = nfe_name(key, nfe)
            r = self.gen_f5(n, text, nfe, F5C["w_default"], F5C["sway"], False, play=False)
            if not r:
                continue
            self.draw()
            self.out(f"  NFE {nfe:2d}：{r['total_s']:5.2f} s（速度場 {r['acoustic_s']:5.2f} s = {r['n_field_calls']} 步 × {r['passes_per_step']} 次前向，每步 {np.mean(r['step_s']) * 1000:4.0f} ms；Vocos {r['vocoder_s']:.2f} s）  T = {r['T']}")
            self._play(r["wav_path"], f"NFE {nfe}")
        self.draw()

    # ── 鍵 2：CFG 掃描 ──
    def cfg_sweep(self) -> None:
        text = TEXTS[SWEEP_KEY]
        self.mode = "cfg"
        nfe = F5C["nfe_fixed"]
        self.out(f"\n{C.BOLD}CFG 掃描{C.RESET}  「{text}」  NFE {nfe}  sway {F5C['sway']:g}  w 是 p55 的尺度（F5 的 α = w − 1；F5 預設 w = 3）")
        for w in F5C["w_sweep"]:
            n = cfg_name(w)
            r = self.gen_f5(n, text, nfe, w, F5C["sway"], False, play=False)
            if not r:
                continue
            self.draw()
            tag = {0: "純無條件場：不看文字、不看 prompt", 1: "純條件", 2: "", 3: "F5 的預設", 4: ""}.get(w, "")
            self.out(f"  w = {w:g}（α = {alpha_of(w):+g}）：{r['total_s']:5.2f} s，每步 {r['passes_per_step']} 次前向{('  ' + tag) if tag else ''}")
            self._play(r["wav_path"], f"w = {w:g}")
        self.draw()

    # ── 鍵 3／p：sway 開關、EPSS ──
    def sway_pair(self) -> None:
        text = TEXTS[SWEEP_KEY]
        self.mode = "sway"
        nfe = F5C["nfe_fixed"]
        self.out(f"\n{C.BOLD}sway{C.RESET}  「{text}」  NFE {nfe}  w = {F5C['w_default']:g}  等距時刻表 × {{s = {F5C['sway']:g}, s = 0}}")
        for s in (F5C["sway"], 0.0):
            n = sway_name(s)
            r = self.gen_f5(n, text, nfe, F5C["w_default"], s, False, play=False)
            if not r:
                continue
            self.draw()
            g = r["grid"]
            self.out(f"  s = {s:g}：{r['total_s']:5.2f} s   前 8 步到 τ = {g[8]:.2f}、後 8 步從那裡到 1（τ 格點：{', '.join(f'{v:.2f}' for v in g[:5])} …）")
            self._play(r["wav_path"], f"sway s = {s:g}")
        self.draw()

    def epss16(self) -> None:
        text = TEXTS[SWEEP_KEY]
        self.mode = "sway"
        nfe = F5C["nfe_fixed"]
        n = sway_name(F5C["sway"], epss=True)
        r = self.gen_f5(n, text, nfe, F5C["w_default"], F5C["sway"], True, play=False)
        if r:
            self.draw()
            self.out(f"  F5 自己的 16 步（EPSS 表 {EPSS[16]} /32，再套 sway）：{r['total_s']:5.2f} s——這是 f5-tts 預設 NFE 16 時真正用的時刻表")
            self._play(r["wav_path"], "EPSS 16 + sway")

    # ── 鍵 4／5／6：第一包 ──
    def fp_f5(self) -> None:
        self.mode = "fp"
        nfe = F5C["first_packet_nfe"]
        self.out(f"\n{C.BOLD}第一包 · F5 整句{C.RESET}  NFE {nfe}  w = {F5C['w_default']:g}  sway {F5C['sway']:g}——第一包 = 整句生成完（＝最後一包）")
        for i, text in enumerate(FP_TEXTS):
            n = fp_name("f5", i)
            r = self.gen_f5(n, text, nfe, F5C["w_default"], F5C["sway"], False, play=False)
            if not r:
                continue
            fp = r["total_s"]
            self.log["first_packet"][n] = {"system": "f5", "i": i, "cjk": r["cjk"], "bytes": r["bytes"], "first_packet_s": round(fp, 3), "total_s": round(fp, 3),
                                           "audio_s": round(r["seconds"], 3), "T": r["T"], "rtf": round(fp / r["seconds"], 3) if r["seconds"] else None,
                                           "acoustic_s": round(r["acoustic_s"], 3), "vocoder_s": round(r["vocoder_s"], 3), "nfe": nfe}
            self.draw()
            self.out(f"  {r['cjk']:3d} 字（{r['bytes']} bytes）：第一包 {fp:6.2f} s   音訊 {r['seconds']:5.2f} s（T = {r['T']} + prompt {r['length']['ref_frames']} frames）   RTF {fp / r['seconds']:.2f}   每步 {np.mean(r['step_s']) * 1000:4.0f} ms")
            self._play(r["wav_path"], f"F5 {r['cjk']} 字")
        self.draw()

    def fp_mlx(self, system: str) -> None:
        self.mode = "fp"
        label = "Qwen3-TTS 串流（mlx-audio）" if system == "qwen3" else "Kokoro（W1 Demo 2 的設定）"
        self.out(f"\n{C.BOLD}第一包 · {label}{C.RESET}" + (f"  chunk = {QC['streaming_interval']} s = {int(QC['streaming_interval'] * 12.5)} 個 token（12.5 Hz）  同一個 prompt（ICL cloning）  temperature {QC['temperature']}  seed {QC['seed']}" if system == "qwen3" else f"  voice {KC['voice']}  lang {KC['lang_code']}  整句一次、iSTFT"))
        for i, text in enumerate(FP_TEXTS):
            n = fp_name(system, i)
            if n in self.items:
                r = self.items[n]
            else:
                if self.worker is None:
                    self.out(f"  {C.AMBER}沒有 MLX worker（replay 裡沒有 {n}）{C.RESET}"); continue
                if not self.worker.loaded.get(system):
                    try:
                        self.worker.load(qwen3=(system == "qwen3"), kokoro=(system == "kokoro"))
                        ld = self.worker.loaded.get(system, {})
                        self.log["models"]["mlx"]["loaded"] = dict(self.worker.loaded)              # 建構時還沒載，這裡補進紀錄
                        self.out(f"  載入 {system}：{ld.get('load_s')} s" + (f"，{ld.get('n_params_M')} M 參數" if ld.get("n_params_M") else ""))
                    except Exception as e:
                        self.out(f"  {C.RED}{system} 載不起來：{e}{C.RESET}（退路：只放 F5 與另一個）"); return
                    if system == "qwen3" and not self.worker.fake:                   # 暖機：第一次呼叫含編譯，不算
                        self.out("  暖機一次（第一次呼叫含 Metal 編譯，不計）……")
                        self.worker.qwen3("_warm", FP_TEXTS[0])
                    if system == "kokoro" and not self.worker.fake:
                        self.worker.kokoro("_warm", FP_TEXTS[0])
                q = getattr(self.worker, system)(n, text)
                if not q.get("ok"):
                    self.out(f"  {C.RED}{system} 失敗：{q.get('error')}{C.RESET}"); continue
                y = C.load_wav(Path(q["wav"]), q["sr"])
                r = {k: v for k, v in q.items() if k not in ("ok", "cmd", "wav")}
                r.update(wav=y, fs=q["sr"], seconds=len(y) / q["sr"], T=None, system=system, text=text, cjk=cjk(text), bytes=len(text.encode("utf-8")))
                self._store(n, r, play=False)
            fp = r["t_first"]
            rec = {"system": system, "i": i, "cjk": r["cjk"], "bytes": r["bytes"], "first_packet_s": round(fp, 3) if fp is not None else None,
                   "total_s": round(r["total"], 3), "audio_s": round(r["seconds"], 3), "rtf": round(r["total"] / r["seconds"], 3) if r["seconds"] else None}
            if system == "qwen3":
                ch = r.get("chunks", [])
                gaps = [round(ch[k]["t"] - ch[k - 1]["t"], 3) for k in range(1, len(ch))]
                durs = [ch[k]["seconds"] for k in range(1, len(ch))]
                rec.update(n_chunks=len(ch), chunk_gaps_s=gaps, chunk_audio_s=[round(d, 3) for d in durs],
                           later_chunks_rtf=round(sum(gaps) / sum(durs), 3) if durs and sum(durs) > 0 else None,
                           later_chunks_rtf_max=round(max(g / d for g, d in zip(gaps, durs) if d > 0), 3) if durs else None,
                           n_tokens=r.get("n_tokens"), ref_encode_s=r.get("ref_encode_s"), ref_tokens=r.get("ref_tokens"))
            else:
                rec.update(n_results=r.get("n_results"))
            self.log["first_packet"][n] = rec
            self.draw()
            if system == "qwen3":
                self.out(f"  {r['cjk']:3d} 字：第一包 {fp:6.3f} s（{len(ch)} 包；prompt 編碼 {r.get('ref_encode_s', float('nan')):.2f} s 含在內）   之後每包 {np.mean(gaps) if gaps else float('nan'):.3f} s 對 {np.mean(durs) if durs else float('nan'):.2f} s 音訊 → RTF {rec['later_chunks_rtf']}（最差 {rec['later_chunks_rtf_max']}）   整句 {r['total']:.2f} s / 音訊 {r['seconds']:.2f} s")
            else:
                self.out(f"  {r['cjk']:3d} 字：第一包 {fp:6.3f} s = 整句（{r.get('n_results')} 段）   音訊 {r['seconds']:5.2f} s   RTF {rec['rtf']}")
            self._play(r["wav_path"], f"{system} {r['cjk']} 字")
        self.draw()

    # ── 視窗 ──
    def make_fig(self):
        import matplotlib.pyplot as plt
        self.plt = plt
        d = CFG["display"]
        self.fig = plt.figure(figsize=(d["fig_w"], d["fig_h"]))
        self.fig.canvas.mpl_connect("key_press_event", self.on_key)
        self.draw()
        return self.fig

    def redraw(self):
        if self.fig is not None:
            self.fig.canvas.draw_idle()
            self.plt.pause(0.05)

    def _mel(self, ax, r: dict, title: str):
        d = CFG["display"]
        fe = self.fe
        Sdb = r["mel"] * fe.to_db
        top = fe.bin_of(d["max_hz"]) + 0.5
        S = Sdb[: int(math.ceil(top))]
        if S.size == 0:
            self._hint(ax, title); return
        vmax = float(S.max())
        ax.imshow(S, origin="lower", aspect="auto", cmap="magma", vmin=vmax - d["db_range"], vmax=vmax, extent=[0, S.shape[1] / fe.fps, -0.5, S.shape[0] - 0.5], interpolation="nearest")
        ticks = [250, 500, 1000, 2000, 4000, 7000]
        ax.set_yticks([fe.bin_of(f) for f in ticks]); ax.set_yticklabels([f"{f / 1000:g}k" for f in ticks], fontsize=8)
        ax.set_ylim(-0.5, top)
        ax.set_title(title, fontsize=9.5, loc="left")
        ax.tick_params(labelsize=8); ax.set_xlabel("time (s)", fontsize=8)

    def _hint(self, ax, text: str):
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_color("#cccccc")
        ax.text(0.5, 0.5, text, ha="center", va="center", fontsize=10, color="#777777", transform=ax.transAxes, wrap=True)

    def draw(self):
        if self.fig is None:
            return
        self.fig.clf()
        sup = {"nfe": f"W8 Demo 2 — the NFE sweep ({'Mandarin' if self.mode_key == 'zh' else 'English'} sentence, one prompt, one seed, w = {F5C['w_default']:g})",
               "cfg": f"W8 Demo 2 — the CFG sweep at NFE {F5C['nfe_fixed']} (w on the slides' scale; F5's default is w = 3)",
               "sway": f"W8 Demo 2 — where the {F5C['nfe_fixed']} steps are spent: sway on / off, and F5's EPSS table",
               "fp": "W8 Demo 2 — the first packet on this machine: F5 (whole utterance), Qwen3-TTS (streaming), Kokoro"}[self.mode]
        self.fig.suptitle(sup + (f"   [{self.tag}]" if self.tag else ""), fontsize=12)
        if self.mode == "nfe":
            gs = self.fig.add_gridspec(2, 3)
            for k, nfe in enumerate(F5C["nfe_sweep"]):
                ax = self.fig.add_subplot(gs[k // 3, k % 3])
                n = nfe_name(self.mode_key, nfe)
                if n in self.items:
                    r = self.items[n]
                    self._mel(ax, r, f"NFE = {nfe} · {r['total_s']:.2f} s ({r['n_field_calls']} × {r['passes_per_step']} forward passes)")
                else:
                    self._hint(ax, f"NFE = {nfe}")
        elif self.mode == "cfg":
            gs = self.fig.add_gridspec(2, 3)
            for k, w in enumerate(F5C["w_sweep"]):
                ax = self.fig.add_subplot(gs[k // 3, k % 3])
                n = cfg_name(w)
                if n in self.items:
                    r = self.items[n]
                    note = {0: "unconditional", 1: "plain conditional", 3: "F5 default"}.get(w, "")
                    self._mel(ax, r, f"w = {w:g} (α = {alpha_of(w):+g}) · {r['total_s']:.2f} s · {r['passes_per_step']} pass/step" + (f" · {note}" if note else ""))
                else:
                    self._hint(ax, f"w = {w:g}")
            ax = self.fig.add_subplot(gs[1, 2])
            ax.axis("off")
            ax.text(0, 0.95, r"$\tilde{\mathbf{v}} = \mathbf{v}_\theta(\mathbf{X}_\tau,\tau) + w\,(\mathbf{v}_\theta(\mathbf{X}_\tau,\tau,\mathbf{c}) - \mathbf{v}_\theta(\mathbf{X}_\tau,\tau))$" "\n\n"
                    "w = 0: unconditional field only (one pass)\nw = 1: conditional field only (one pass)\nw > 1: two passes per step; F5's cfg_strength α = w − 1\n"
                    f"same sentence, prompt, seed, NFE {F5C['nfe_fixed']}, sway {F5C['sway']:g}", va="top", fontsize=10, transform=ax.transAxes)
        elif self.mode == "sway":
            gs = self.fig.add_gridspec(2, 3, height_ratios=[1.3, 1])
            cols = [(sway_name(F5C["sway"]), f"sway s = {F5C['sway']:g} (F5's choice)"), (sway_name(0.0), "s = 0: uniform τ"), (sway_name(F5C["sway"], True), "F5's EPSS table + sway")]
            for k, (n, lab) in enumerate(cols):
                ax = self.fig.add_subplot(gs[0, k])
                if n in self.items:
                    r = self.items[n]
                    self._mel(ax, r, f"{lab} · NFE {r['nfe']} · {r['total_s']:.2f} s")
                else:
                    self._hint(ax, lab + ("" if k < 2 else "  (key p)"))
            ax = self.fig.add_subplot(gs[1, :])
            for k, (n, lab) in enumerate(cols):
                g = self.items[n]["grid"] if n in self.items else time_grid(F5C["nfe_fixed"], [F5C["sway"], 0.0, F5C["sway"]][k], k == 2)
                ax.plot(g, np.full(len(g), 2 - k), "|", ms=18, mew=2, color=["#5b3fa0", "#c8742a", "#2a7fc8"][k], label=lab)
            ax.set_yticks([2, 1, 0]); ax.set_yticklabels([c[1].split(" (")[0] for c in cols], fontsize=9)
            ax.set_xlim(-0.02, 1.02); ax.set_xlabel("τ (0 = noise, 1 = data) — where the 16 steps land", fontsize=9); ax.grid(alpha=0.3, axis="x")
            ax.set_title(r"sway: $\tau \leftarrow \tau + s\,(\cos\frac{\pi}{2}\tau - 1 + \tau)$, F5 Eq. 7 — more steps at small τ, where the large structure is decided", fontsize=9.5, loc="left")
        else:
            gs = self.fig.add_gridspec(1, 2, width_ratios=[1, 1.3])
            ax = self.fig.add_subplot(gs[0, 0])
            syst = [("f5", "F5-TTS, whole utterance (NFE 32)", "#5b3fa0", "o-"), ("qwen3", f"Qwen3-TTS streaming, {int(QC['streaming_interval'] * 12.5)}-token chunks", "#c8742a", "s-"), ("kokoro", "Kokoro (W1 Demo 2 setting)", "#2a7fc8", "^-")]
            any_pt = False
            for sysk, lab, col, mk in syst:
                xs, ys = [], []
                for i in range(len(FP_TEXTS)):
                    rec = self.log["first_packet"].get(fp_name(sysk, i))
                    if rec and rec.get("first_packet_s") is not None:
                        xs.append(rec["cjk"]); ys.append(rec["first_packet_s"])
                if xs:
                    any_pt = True
                    ax.plot(xs, ys, mk, color=col, label=lab, ms=7)
                    for x, y in zip(xs, ys):
                        ax.annotate(f"{y:.2f}", (x, y), textcoords="offset points", xytext=(4, 4), fontsize=8, color=col)
            ax.set_xlabel("characters in the text (CJK)"); ax.set_ylabel("first packet (s, wall clock on this machine)")
            ax.set_xticks([cjk(t) for t in FP_TEXTS]); ax.grid(alpha=0.3)
            if any_pt:
                ax.legend(fontsize=8.5, loc="upper left")
                ax.set_ylim(bottom=0)
            else:
                self._hint(ax, "4 F5 · 5 Qwen3-TTS · 6 Kokoro")
            ax.set_title("first packet vs. length", fontsize=10, loc="left")
            ax = self.fig.add_subplot(gs[0, 1])
            drawn = False
            for i in reversed(range(len(FP_TEXTS))):
                n = fp_name("qwen3", i)
                if n in self.items and self.items[n].get("chunks"):
                    r = self.items[n]
                    ch = r["chunks"]
                    t_audio = 0.0
                    for k, c in enumerate(ch):
                        ax.plot([c["t"], c["t"]], [0, c["seconds"]], color="#c8742a", lw=1.2)
                        ax.plot(c["t"], c["seconds"], "s", color="#c8742a", ms=4)
                        t_audio += c["seconds"]
                    # 播放的死線：第一包送出後開始播，第 k 包必須在 t_first + Σ(前面各包的秒數) 之前到
                    deadline = np.cumsum([0.0] + [c["seconds"] for c in ch[:-1]]) + ch[0]["t"]
                    arrive = np.array([c["t"] for c in ch])
                    ax.step(deadline, [c["seconds"] for c in ch], where="post", color="#999999", lw=0.8, ls="--")
                    late = arrive > deadline + 1e-9
                    if late.any():
                        ax.plot(arrive[late], [ch[k]["seconds"] for k in np.flatnonzero(late)], "x", color="red", ms=8, mew=2, label="arrives after its play-out deadline")
                    ax.set_title(f"Qwen3-TTS chunks for the {r['cjk']}-character text: when each chunk arrived (bars = its audio length); "
                                 f"first {ch[0]['t']:.2f} s, then {len(ch) - 1} chunks, later-chunk RTF {self.log['first_packet'][n].get('later_chunks_rtf')}", fontsize=9, loc="left", wrap=True)
                    ax.set_xlabel("wall-clock time since the call (s)"); ax.set_ylabel("audio in the chunk (s)")
                    ax.set_ylim(0, max(0.5, max(c["seconds"] for c in ch) * 1.6)); ax.grid(alpha=0.3)
                    ax.plot([], [], "--", color="#999999", label="play-out deadline if playback starts at the first chunk")
                    ax.legend(fontsize=8, loc="upper right")
                    drawn = True
                    break
            if not drawn:
                self._hint(ax, "5 — Qwen3-TTS: when each chunk arrives")
        self.fig.tight_layout()
        self.redraw()

    # ── 表、選單、鍵 ──
    def table(self) -> None:
        o = self.out
        o(f"\n{C.BOLD}W8 Demo 2 — numbers{C.RESET}   prompt {D1.TXT['prompt_utt']}（{self.f5.prompt_seconds if self.f5 else self.log['models'].get('f5', {}).get('prompt_seconds', 0):.2f} s）  F5 seed {F5C['seed']}")
        for key in (SWEEP_KEY, OTHER_KEY):
            rows = [(nfe, self.items[nfe_name(key, nfe)]) for nfe in F5C["nfe_sweep"] if nfe_name(key, nfe) in self.items]
            if rows:
                o(f"  NFE sweep · 「{TEXTS[key]}」  w = {F5C['w_default']:g}  sway {F5C['sway']:g}  T = {rows[0][1]['T']} frames = {rows[0][1]['seconds']:.2f} s")
                o(f"    {'NFE':>5s}{'total s':>9s}{'field s':>9s}{'ms/step':>9s}{'Vocos s':>9s}   what we heard")
                for nfe, r in rows:
                    o(f"    {nfe:>5d}{r['total_s']:>9.2f}{r['acoustic_s']:>9.2f}{np.mean(r['step_s']) * 1000:>9.0f}{r['vocoder_s']:>9.2f}   (fill in after listening)")
        rows = [(w, self.items[cfg_name(w)]) for w in F5C["w_sweep"] if cfg_name(w) in self.items]
        if rows:
            o(f"  CFG sweep · NFE {F5C['nfe_fixed']}")
            o(f"    {'w':>5s}{'α':>6s}{'passes':>8s}{'total s':>9s}   note")
            for w, r in rows:
                o(f"    {w:>5g}{alpha_of(w):>+6g}{r['passes_per_step']:>8d}{r['total_s']:>9.2f}   {({0: 'unconditional', 1: 'plain conditional', 3: 'F5 default'}).get(w, '')}")
        rows = [(n, self.items[n]) for n in (sway_name(F5C["sway"]), sway_name(0.0), sway_name(F5C["sway"], True)) if n in self.items]
        if rows:
            o(f"  sway / schedule · NFE {F5C['nfe_fixed']}  w = {F5C['w_default']:g}")
            for n, r in rows:
                o(f"    {n:<9s} total {r['total_s']:5.2f} s   τ grid: {' '.join(f'{v:.2f}' for v in r['grid'])}")
        if self.log["first_packet"]:
            o("  first packet (this machine; wall clock from the call)")
            o(f"    {'system':<9s}{'chars':>6s}{'bytes':>6s}{'first s':>9s}{'total s':>9s}{'audio s':>9s}{'RTF':>7s}   later chunks")
            for sysk in ("f5", "qwen3", "kokoro"):
                for i in range(len(FP_TEXTS)):
                    rec = self.log["first_packet"].get(fp_name(sysk, i))
                    if not rec:
                        continue
                    later = f"{rec['n_chunks']} chunks, gap/audio RTF {rec['later_chunks_rtf']} (max {rec['later_chunks_rtf_max']})" if sysk == "qwen3" else "— (first = last packet)"
                    o(f"    {sysk:<9s}{rec['cjk']:>6d}{rec['bytes']:>6d}{rec['first_packet_s']:>9.3f}{rec['total_s']:>9.2f}{rec['audio_s']:>9.2f}{(rec['rtf'] or float('nan')):>7.2f}   {later}")
            o("  F5: first packet = the whole utterance (acoustic NFE 32 × 2 passes + Vocos). Qwen3-TTS: ICL cloning, the prompt's speech-tokenizer encode is inside the first packet;"
              f" chunk = {int(QC['streaming_interval'] * 12.5)} tokens = {QC['streaming_interval']} s of audio. Kokoro: whole utterance, iSTFT (W1 Demo 2's row).")
        o("  Times are this machine's (F5 on " + (self.f5.device if self.f5 else str(self.log['models'].get('f5', {}).get('device'))) + ", MLX models on Metal); do not put them in one table with the papers' RTFs.")

    def menu(self) -> None:
        self.out(f"\n{C.BOLD}p62{C.RESET}  1 NFE 掃描   e 另一句   2 CFG 掃描   3 sway 開／關   p EPSS 16     {C.BOLD}p81{C.RESET}  4 F5 整句   5 Qwen3-TTS 串流   6 Kokoro     t 數字表   r 再播   h 選單   q 離開")

    def dispatch(self, k: str) -> bool:
        if k == "1":
            self.nfe_sweep(SWEEP_KEY)
        elif k == "e":
            self.nfe_sweep(OTHER_KEY)
        elif k == "2":
            self.cfg_sweep()
        elif k == "3":
            self.sway_pair()
        elif k == "p":
            self.epss16()
        elif k == "4":
            self.fp_f5()
        elif k == "5":
            self.fp_mlx("qwen3")
        elif k == "6":
            self.fp_mlx("kokoro")
        elif k == "t":
            self.table()
        elif k == "r":
            if self.last:
                self._play(*self.last)
        elif k == "h":
            self.menu()
        elif k == "q":
            return False
        return True

    def on_key(self, ev) -> None:
        if not self.dispatch(ev.key):
            self.plt.close(self.fig)

    # ── 彩排 ──
    def rehearse_all(self) -> None:
        C.play = lambda *a, **k: None
        self.nfe_sweep(SWEEP_KEY)
        self.nfe_sweep(OTHER_KEY)
        self.cfg_sweep()
        self.sway_pair()
        self.epss16()
        self.fp_f5()
        if self.worker is not None:
            self.fp_mlx("qwen3")
            self.fp_mlx("kokoro")
        # NFE 32、seed 1 的英文對 Demo 1 的 f5_seed1（同機同裝置應逐格相同）
        d1_mel = D1_DIR / "runs" / "rehearsal" / "f5_seed1_mel.npy"
        n = nfe_name("en", 32)
        if d1_mel.exists() and n in self.items:
            A, B = np.load(d1_mel), self.items[n]["mel"].astype(np.float32)
            same = A.shape == B.shape
            diff = float(np.max(np.abs(A - B))) if same else None
            self.log["vs_demo1_f5_seed1"] = {"same_shape": same, "max_abs_diff": diff, "d1_device": json.loads((D1_DIR / "runs" / "rehearsal" / "rehearsal.json").read_text())["models"]["f5"]["device"]
                                             if (D1_DIR / "runs" / "rehearsal" / "rehearsal.json").exists() else None}
            self.out(f"\n  英文 NFE 32 對 Demo 1 的 f5_seed1_mel.npy：{'同形狀，最大差 %.2e' % diff if same else '形狀不同 ' + str((A.shape, B.shape))}（同機同裝置應為 0；Demo 1 在 {self.log['vs_demo1_f5_seed1']['d1_device']}）")

    def rehearse_figs(self) -> None:
        import matplotlib.pyplot as plt
        self.plt = plt
        d = CFG["display"]
        for mode, key, nm in (("nfe", SWEEP_KEY, f"nfe_{SWEEP_KEY}.png"), ("nfe", OTHER_KEY, f"nfe_{OTHER_KEY}.png"), ("cfg", None, "cfg.png"), ("sway", None, "sway.png"), ("fp", None, "first_packet.png")):
            self.fig = plt.figure(figsize=(d["fig_w"], d["fig_h"]))
            self.mode = mode
            if key:
                self.mode_key = key
            self.draw(); self.fig.savefig(self.folder / nm, dpi=100); plt.close(self.fig)
        self.fig = None

    def save(self) -> None:
        stem = "rehearsal" if self.folder.name == "rehearsal" else "session"
        self.folder.mkdir(parents=True, exist_ok=True)
        for name, r in self.items.items():
            if r.get("mel") is not None:
                np.save(self.folder / f"{name}_mel.npy", r["mel"].astype(np.float32))
        (self.folder / f"{stem}.json").write_text(json.dumps(self.log, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
        (self.folder / f"{stem}.txt").write_text(self.out.text(), encoding="utf-8")
        C.note(f"  存到 {self.folder}/（{stem}.json、{stem}.txt、.wav、*_mel.npy）")


def versions() -> dict:
    v = C.versions()
    from importlib import metadata
    for pkg in ("torch", "torchaudio", "transformers", "huggingface_hub", "f5-tts", "vocos", "torchdiffeq", "soundfile"):
        try:
            v[pkg] = metadata.version(pkg)
        except metadata.PackageNotFoundError:
            v[pkg] = None
    return v


# ── replay ───────────────────────────────────────────────────
def build_replay(out: Out) -> Demo:
    rec_path = REHEARSAL / "rehearsal.json"
    if not rec_path.exists():
        sys.exit("沒有 runs/rehearsal/rehearsal.json：先 ./present.sh rehearse")
    rec = json.loads(rec_path.read_text(encoding="utf-8"))
    d = Demo(out, None, None, REHEARSAL, tag="replay " + rec["time"])
    d.log = rec
    d.log["played"] = []
    for name, r in rec["items"].items():
        item = dict(r)
        p = (HERE / r["wav"]).resolve()
        item["wav_path"] = p
        item["wav"] = C.load_wav(p, r["fs"]) if p.exists() else np.zeros(0)
        mp = REHEARSAL / f"{name}_mel.npy"
        item["mel"] = np.load(mp) if mp.exists() else None
        d.items[name] = item
    d.f5 = None
    out(f"  replay：{rec['time']} 的彩排，{len(d.items)} 個版本（按鍵只播、不生成）")
    return d


def build(cmd: str, out: Out, device: str | None = None, no_mlx: bool = False) -> Demo:
    folder = REHEARSAL if cmd == "rehearse" else HERE / "runs" / "live" / (C.now().replace(":", "") + (".fake" if cmd == "fake" else ""))
    folder.mkdir(parents=True, exist_ok=True)
    if cmd == "fake":
        worker = None if no_mlx else Worker(folder, fake=True, out=out)
        return Demo(out, FakeF5(), worker, folder, tag="FAKE")
    try:
        import torch  # noqa: F401
    except ImportError:
        sys.exit("沒有 torch：先跑 ../w08_d1_mean_vs_sample/setup.sh（建 ../.venv-tts）；或 ./present.sh fake 演練流程")
    D1.quiet_hf()
    c = CFG["infer"]
    t0 = time.perf_counter()
    m = D1.F5(device or c["f5_device"], device or c["vocoder_device"])
    f5 = F5Sampler(m)
    out(f"  載入 F5-TTS {m.n_params / 1e6:6.1f} M 參數   {m.device} / vocoder {m.vdev}   {time.perf_counter() - t0:5.1f} s   prompt {f5.prompt_seconds:.2f} s")
    worker = None
    if not no_mlx:
        try:
            worker = Worker(folder, fake=False, out=out)
            out(f"  MLX worker 起來了（mlx-audio {worker.versions.get('mlx-audio')}、mlx {worker.versions.get('mlx')}）；模型在第一次按 5／6 時載")
        except Exception as e:
            out(f"  {C.AMBER}MLX worker 起不來：{e}（鍵 5、6 不能用；只放 F5）{C.RESET}")
    return Demo(out, f5, worker, folder, tag="rehearsal" if cmd == "rehearse" else "")


# ── check ────────────────────────────────────────────────────
def check(device: str | None = None) -> None:
    C.banner("課前體檢", C.AMBER)
    res = {"time": C.now(), "versions": versions(), "env": {"PYTORCH_ENABLE_MPS_FALLBACK": os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK")}}
    for k, v in res["versions"].items():
        print(f"  {k:<16s} {v}")
    # 1. MLX 那一半：共用 .venv 的 mlx-audio 有沒有 qwen3_tts、kokoro
    try:
        py = WORKER_PY_ENV if WORKER_PY_ENV.exists() else Path(sys.executable)
        p = subprocess.run([str(py), str(WORKER_PY), "--selfcheck"], capture_output=True, text=True, timeout=120)
        res["mlx"] = json.loads(p.stdout.strip().splitlines()[-1]) if p.stdout.strip() else {"ok": False, "stderr": p.stderr[-500:]}
    except Exception as e:
        res["mlx"] = {"ok": False, "error": repr(e)}
    mv = res["mlx"].get("versions", {})
    print(f"  mlx-audio {mv.get('mlx-audio')}、mlx {mv.get('mlx')}、misaki {mv.get('misaki')}；qwen3_tts 與 kokoro import：{mv.get('qwen3_tts_in_mlx_audio')}")
    lock = read_lock()
    for k, repo in REPOS.items():
        print(f"  {repo:<48s} @ {str(lock.get(repo))[:12]}")
    # 2. F5：自寫的 sampler 對 f5-tts 的 sample()
    try:
        import torch
    except ImportError:
        sys.exit("  沒有 torch：先跑 ../w08_d1_mean_vs_sample/setup.sh")
    D1.quiet_hf()
    mps = bool(torch.backends.mps.is_available())
    res["mps_available"] = mps
    devs = [device] if device else (["cpu"] + (["mps"] if mps else []))
    res["f5"] = {}
    for dev in devs:
        r = {}
        try:
            t0 = time.perf_counter(); m = D1.F5(dev, dev); r["load_s"] = round(time.perf_counter() - t0, 2)
            s = F5Sampler(m)
            text = TEXTS["zh"]
            # (a) w = 3（α = 2）、NFE 2（等距，f5-tts 對 2 也等距）、sway −1：對 infer_batch_process
            ours = s.sample(text, F5C["seed"], 2, 3, F5C["sway"], False)
            lib = m.synth(s.prompt24, PROMPT_TEXT, text, F5C["seed"], nfe=2, cfg_strength=2.0, sway=F5C["sway"])
            r["w3_nfe2_vs_library"] = {"same_T": ours["T"] == lib["T"], "mel_max_abs": float(np.max(np.abs(ours["mel"] - lib["mel"]))) if ours["T"] == lib["T"] else None}
            # (b) w = 1（純條件、一次前向）對 f5-tts cfg_strength = 0 的捷徑
            ours1 = s.sample(text, F5C["seed"], 2, 1, F5C["sway"], False)
            lib1 = m.synth(s.prompt24, PROMPT_TEXT, text, F5C["seed"], nfe=2, cfg_strength=0.0, sway=F5C["sway"])
            r["w1_nfe2_vs_library_cfg0"] = {"same_T": ours1["T"] == lib1["T"], "mel_max_abs": float(np.max(np.abs(ours1["mel"] - lib1["mel"]))) if ours1["T"] == lib1["T"] else None}
            # (c) w = 0 跟 w = 1 要不一樣（f5-tts 表達不出 w = 0）
            ours0 = s.sample(text, F5C["seed"], 2, 0, F5C["sway"], False)
            r["w0_differs_from_w1"] = {"mel_max_abs": float(np.max(np.abs(ours0["mel"] - ours1["mel"])))}
            # (d) EPSS 16 對 f5-tts 預設的 NFE 16（它用 EPSS 表）
            oe = s.sample(text, F5C["seed"], 16, 3, F5C["sway"], True)
            le = m.synth(s.prompt24, PROMPT_TEXT, text, F5C["seed"], nfe=16, cfg_strength=2.0, sway=F5C["sway"])
            r["epss16_vs_library_nfe16"] = {"mel_max_abs": float(np.max(np.abs(oe["mel"] - le["mel"]))) if oe["T"] == le["T"] else None}
            # (e) 一步的成本：兩次前向（w = 3）對一次（w = 1），用 NFE 16 的平均
            r["ms_per_step"] = {"two_passes_w3": round(float(np.mean(oe["step_s"])) * 1000, 1), "one_pass_w1": round(float(np.mean(ours1["step_s"])) * 1000, 1),
                                "T_total_frames": oe["length"]["ref_frames"] + oe["T"]}
            r["vocoder_s"] = round(oe["vocoder_s"], 4)
            print(f"  F5 {dev}: 載入 {r['load_s']} s · 自寫 sampler 對 f5-tts sample()：w=3/NFE2 最大差 {r['w3_nfe2_vs_library']['mel_max_abs']:.2e}，w=1 對 cfg_strength=0 {r['w1_nfe2_vs_library_cfg0']['mel_max_abs']:.2e}，"
                  f"EPSS16 對預設 NFE16 {r['epss16_vs_library_nfe16']['mel_max_abs']:.2e}；w=0 對 w=1 差 {r['w0_differs_from_w1']['mel_max_abs']:.2f}（要大）"
                  f" · 每步 {r['ms_per_step']['two_passes_w3']} ms（兩次前向）對 {r['ms_per_step']['one_pass_w1']} ms（一次），T = {r['ms_per_step']['T_total_frames']} frames")
        except Exception as e:
            import traceback; traceback.print_exc()
            r["error"] = repr(e)
            print(f"  F5 {dev}: {C.RED}失敗{C.RESET} {e!r}")
        res["f5"][dev] = r
    (HERE / "runs").mkdir(exist_ok=True)
    (HERE / "runs" / "check.json").write_text(json.dumps(res, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    print(f"\n  → runs/check.json。三個「最大差」要是 0 或 1e-5 量級；不是的話 sampler 跟 f5-tts 的 sample() 走岔了，先修再彩排。下一步：./present.sh rehearse")


# ── fetch ────────────────────────────────────────────────────
def fetch() -> None:
    C.banner("下載 Qwen3-TTS 與 Kokoro（F5 用 Demo 1 的）", C.AMBER)
    if not PROMPT_WAV.exists():
        print(f"  {C.AMBER}沒有 {PROMPT_WAV}：先跑 ../w08_d1_mean_vs_sample/present.sh fetch{C.RESET}")
    py = WORKER_PY_ENV if WORKER_PY_ENV.exists() else Path(sys.executable)
    env = dict(os.environ, HF_HUB_OFFLINE="0")
    p = subprocess.Popen([str(py), str(WORKER_PY), "--selfcheck"], stdout=subprocess.PIPE, text=True, env=env)
    print(f"  mlx worker：{p.stdout.read().strip()[:300]}")
    p.wait()
    w = subprocess.Popen([str(py), str(WORKER_PY), "--fake"], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, env=env, bufsize=1)
    w.stdout.readline()
    w.stdin.write(json.dumps({"cmd": "fetch", "repos": REPOS, "lock": read_lock()}) + "\n"); w.stdin.flush()
    r = json.loads(w.stdout.readline())
    w.stdin.write(json.dumps({"cmd": "quit"}) + "\n"); w.stdin.flush(); w.wait()
    if not r.get("ok"):
        sys.exit(f"  下載失敗：{r.get('error')}")
    lock = read_lock()
    for k, v in r["fetched"].items():
        lock[v["repo"]] = v["sha"]
        print(f"  {v['repo']} @ {v['sha'][:12]}   {v['size_mb']} MB   license (card): {v['license']}")
    MODEL_LOCK.write_text("".join(f"{a} {b}\n" for a, b in lock.items()), encoding="utf-8")
    print("  → model.lock\n下一步：./present.sh check")


# ── export ───────────────────────────────────────────────────
def export(rec_path: Path) -> None:
    rec = json.loads(rec_path.read_text(encoding="utf-8"))
    d = json.loads(METRICS_JSON.read_text(encoding="utf-8")) if METRICS_JSON.exists() else {"_note": "量測數字（進版控的出處，speech_ai/CLAUDE.md 4.8）。"}
    keep = ("nfe", "w", "alpha", "sway", "epss", "seed", "passes_per_step", "n_field_calls", "T", "seconds", "acoustic_s", "vocoder_s", "total_s", "step_s", "grid", "cjk", "bytes", "text")
    items = {n: {k: r[k] for k in keep if k in r} for n, r in rec["items"].items() if n.startswith(("nfe_", "cfg_", "sway_", "epss"))}
    for n, r in items.items():
        if "length" in rec["items"][n]:
            r["length"] = {k: v for k, v in rec["items"][n]["length"].items() if k != "ref_text_used"}
    d["demo2"] = {"source": "demos/w08_d2_nfe_cfg_first_packet/runs/rehearsal/rehearsal.json", "time": rec["time"], "tag": rec.get("tag", ""), "texts": rec["texts"],
                  "models": rec["models"], "config": rec["config"], "sweeps": items, "first_packet": rec["first_packet"], "vs_demo1_f5_seed1": rec.get("vs_demo1_f5_seed1"),
                  "definitions": {"w": "the slides' CFG weight (p55): v_u + w (v_c − v_u); F5's cfg_strength α = w − 1; w = 0 is the unconditional field alone",
                                  "grid": "the τ grid actually integrated (uniform, or F5's EPSS table when epss = true, then sway applied)",
                                  "total_s": "wall clock from the call to the waveform (acoustic + vocoder), after a warm-up call",
                                  "first_packet_s": "F5 / Kokoro: time to the whole utterance; Qwen3-TTS: time to the first streamed chunk (ICL prompt encoding included)",
                                  "later_chunks_rtf": "Qwen3-TTS: sum of gaps between consecutive chunk arrivals ÷ sum of their audio lengths (< 1 = play-out never starves)",
                                  "what_we_heard": "filled by hand after the rehearsal (w08.qmd p62, p81)"}}
    METRICS_JSON.parent.mkdir(parents=True, exist_ok=True)
    METRICS_JSON.write_text(json.dumps(d, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    print(f"  → {METRICS_JSON}（demo2）")


# ── selftest ─────────────────────────────────────────────────
def selftest() -> None:
    C.banner("selftest", C.AMBER)
    fails = []

    def ok(cond: bool, msg: str):
        print(f"  {'OK ' if cond else C.RED + 'FAIL' + C.RESET}  {msg}")
        if not cond:
            fails.append(msg)

    # 1. 三句的字數與 F5 的長度（prompt 4.91 s = 460 frames、84 bytes，Demo 1 的彩排）
    ok([cjk(t) for t in FP_TEXTS] == [10, 30, 90], f"第一包的三句：{[cjk(t) for t in FP_TEXTS]} 個漢字（{[len(t.encode('utf-8')) for t in FP_TEXTS]} bytes）")
    ref_samples = 460 * HOP
    Ls = [D1.f5_length(ref_samples, PROMPT_TEXT, t) for t in FP_TEXTS]
    ok(Ls[0]["gen_frames"] < Ls[1]["gen_frames"] < Ls[2]["gen_frames"] and all(L["speed"] == 1.0 for L in Ls),
       f"F5 的長度：{[L['gen_frames'] for L in Ls]} frames = {[round(L['gen_frames'] * HOP / FS24, 1) for L in Ls]} s（含 prompt {[round((L['gen_frames'] + 460) * HOP / FS24, 1) for L in Ls]} s，都 < 30 s）")
    mc = D1.f5_max_chars(D1.f5_ref_text_pre(PROMPT_TEXT), 4.91)
    print(f"  --   F5TTS.infer 切句的上限 {mc} bytes：90 字那句 {len(FP_TEXTS[2].encode('utf-8'))} bytes {'會被切成兩塊' if len(FP_TEXTS[2].encode('utf-8')) > mc else '一塊'}——本 demo 直接走 sample、不切")
    ok(cjk(TEXTS["zh"]) == 13 and D1.f5_length(ref_samples, PROMPT_TEXT, TEXTS["en"])["gen_frames"] == 498, f"掃描句：訂票句 {cjk(TEXTS['zh'])} 字；英文句的長度 498 frames（＝ Demo 1）")
    # 2. 時刻表
    g = time_grid(16, None, False)
    ok(len(g) == 17 and abs(g[1] - 1 / 16) < 1e-12, "等距：NFE 16 → 17 個格點、步長 1/16")
    g = time_grid(16, -1.0, False)
    ok(abs(g[0]) < 1e-12 and abs(g[-1] - 1) < 1e-12 and g[8] < 0.5 and np.all(np.diff(g) > 0), f"sway s = −1：端點不動、中點 τ = {g[8]:.3f} < 0.5、單調")
    ok(np.allclose(time_grid(16, None, True), np.array(EPSS[16]) / 32) and np.allclose(time_grid(32, None, True), np.linspace(0, 1, 33)), "EPSS：16 用表、32 等距")
    ok(np.allclose(time_grid(1, -1.0, False), [0, 1]), "NFE 1：一步從 0 到 1")
    # 3. CFG 的 w ↔ α、前向次數
    ok(alpha_of(3) == 2 and alpha_of(1) == 0 and alpha_of(0) == -1, "w = 3 ↔ α = 2（F5 預設）；w = 1 ↔ α = 0；w = 0 ↔ α = −1")
    ok([passes_per_step(w) for w in F5C["w_sweep"]] == [1, 1, 2, 2, 2], "每步前向次數：w = 0、1 一次，其餘兩次")
    # 4. fake 流程：FakeF5 ＋ worker --fake，全部鍵、表、圖、存檔
    import matplotlib
    matplotlib.use("Agg")
    C.play = lambda *a, **k: None
    out = Out()
    folder = HERE / "runs" / "_selftest"
    shutil.rmtree(folder, ignore_errors=True)
    folder.mkdir(parents=True, exist_ok=True)
    try:
        worker = Worker(folder, fake=True, out=out)
    except Exception as e:
        worker = None
        print(f"  --   worker --fake 起不來（{e!r}），跳過 MLX 那一半")
    d = Demo(out, FakeF5(), worker, folder, tag="selftest FAKE")
    d.make_fig()
    for k in "1e23p456t":
        d.dispatch(k)
    n_expected = 2 * len(F5C["nfe_sweep"]) + len(F5C["w_sweep"]) + 3 + len(FP_TEXTS) * (3 if worker else 1)
    ok(len(d.items) == n_expected, f"fake：{len(d.items)} 個版本（預期 {n_expected}）")
    r = [d.items[nfe_name(SWEEP_KEY, n)] for n in F5C["nfe_sweep"]]
    ok(all(x["T"] == r[0]["T"] for x in r) and r[-1]["total_s"] > r[0]["total_s"], "fake：六個 NFE 長度相同、時間隨 NFE 增")
    ok(d.items[cfg_name(0)]["passes_per_step"] == 1 and d.items[cfg_name(4)]["passes_per_step"] == 2, "fake：CFG 掃描記了前向次數")
    if worker:
        fp = d.log["first_packet"]
        ok(all(fp_name(s, i) in fp for s in ("f5", "qwen3", "kokoro") for i in range(3)), "fake：九格第一包都有數字")
        ok(fp[fp_name("qwen3", 2)]["n_chunks"] > fp[fp_name("qwen3", 0)]["n_chunks"] and fp[fp_name("qwen3", 2)]["later_chunks_rtf"] is not None, "fake：Qwen3 的 chunk 時間表與 RTF")
        worker.close()
    d.rehearse_figs(); d.save()
    ok(all((folder / n).exists() for n in (f"nfe_{SWEEP_KEY}.png", "cfg.png", "sway.png", "first_packet.png", "session.json")), "fake：圖與紀錄存了")
    shutil.rmtree(folder, ignore_errors=True)
    # 5. 有彩排紀錄：對 w08-metrics.json、replay 讀回
    rec_path = REHEARSAL / "rehearsal.json"
    if rec_path.exists() and METRICS_JSON.exists():
        rec = json.loads(rec_path.read_text(encoding="utf-8"))
        mj = json.loads(METRICS_JSON.read_text(encoding="utf-8")).get("demo2", {})
        bad = [n for n, r in rec["items"].items() if n.startswith(("nfe_", "cfg_", "sway", "epss")) and mj.get("sweeps", {}).get(n, {}).get("total_s") != r["total_s"]]
        ok(not bad, f"彩排紀錄對 w08-metrics.json → demo2：{len(bad)} 處不一致")
        ok(mj.get("first_packet") == rec.get("first_packet"), "彩排的第一包對 w08-metrics.json → demo2.first_packet")
        ok(all((HERE / r["wav"]).exists() for r in rec["items"].values()), "彩排的 .wav 都在（replay 用）")
        d = build_replay(Out())
        d.make_fig()
        for k in "1e23p456t":
            d.dispatch(k)
        ok(len(d.items) == len(rec["items"]), f"replay：讀回 {len(d.items)} 個版本、鍵都走得過")
    else:
        print("  --   沒有彩排紀錄或 w08-metrics.json，跳過比對")
    print(f"\n  {'全部通過' if not fails else str(len(fails)) + ' 項失敗'}")
    sys.exit(1 if fails else 0)


# ── main ─────────────────────────────────────────────────────
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["show", "rehearse", "replay", "check", "fetch", "fake", "selftest", "export"])
    ap.add_argument("--device", help="show／rehearse／check：蓋過 demo_config.toml 的 device（cpu／mps／auto）")
    ap.add_argument("--no-plot", action="store_true", help="show／replay／fake：只在終端機，不開圖")
    ap.add_argument("--no-mlx", action="store_true", help="show／rehearse／fake：不開 MLX worker（鍵 5、6 不能用）")
    args = ap.parse_args()

    if args.cmd == "check":
        check(args.device); return
    if args.cmd == "fetch":
        fetch(); return
    if args.cmd == "selftest":
        selftest(); return
    if args.cmd == "export":
        export(REHEARSAL / "rehearsal.json"); return

    if args.cmd == "rehearse" or args.no_plot:
        import matplotlib
        matplotlib.use("Agg")

    out = Out()
    out.banner("W8 Demo 2：NFE, CFG, and the first packet")
    d = build_replay(out) if args.cmd == "replay" else build(args.cmd, out, args.device, args.no_mlx)

    if args.cmd == "rehearse":
        d.make_fig()
        d.rehearse_all()
        d.table()
        d.rehearse_figs()
        d.save()
        if d.worker:
            d.worker.close()
        export(d.folder / "rehearsal.json")
        print("\n下一步：./present.sh 自己按一遍 1 2 3 4 5 6 t；聽到的寫進 p62、p81 的表（what we heard），runs/rehearsal/*.png 看過。")
        return

    d.make_fig()
    d.menu()
    try:
        if args.no_plot:
            C.note("  --no-plot：鍵在終端機按。")
            while d.dispatch(C.getkey()):
                pass
        else:
            C.note("  鍵在視窗裡按（視窗要有焦點）。終端機這裡是紀錄。生成在按鍵時才做，等待是內容的一部分。")
            d.plt.show()
    finally:
        if args.cmd in ("show", "fake"):
            d.save()
        if d.worker:
            d.worker.close()


if __name__ == "__main__":
    main()

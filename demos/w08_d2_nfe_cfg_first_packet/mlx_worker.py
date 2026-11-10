# -*- coding: utf-8 -*-
# 此檔由 bin/publish_demos.py 從 private repo 的 <course>/demos/ 複製而來，請勿在公開 repo 直接編輯。
"""W8 demo 2 的 MLX 那一半：Qwen3-TTS（串流）與 Kokoro（W1 Demo 2 的設定），在共用的 ../.venv（mlx-audio）裡跑。

demo.py 在 ../.venv-tts（torch、F5）裡，兩個環境不能同一個 process，所以這支當 worker：
demo.py 用 subprocess 開它，stdin 一行一個 JSON 指令、stdout 一行一個 JSON 結果（其他輸出都導到 stderr）。

  {"cmd": "versions"}
  {"cmd": "fetch", "repos": {"qwen3": "...", "kokoro": "..."}, "lock": {...}}      → 下載、回 commit（要網路）
  {"cmd": "load", "qwen3": "<repo or path>", "kokoro": "<repo or path>", "revision": {...}}
  {"cmd": "qwen3", "name": "...", "text": "...", "ref_audio": "<wav path>", "ref_text": "...", "interval": 0.32, "seed": 1, ...}
  {"cmd": "kokoro", "name": "...", "text": "...", "voice": "...", "lang_code": "z"}
  {"cmd": "quit"}

每個生成指令把 .wav 存到 --out 底下的 <name>.wav，回傳時間表：
  qwen3：t_first（第一個 chunk 送出的時間，從呼叫起算）、chunks[{t, samples, seconds}]（每個 chunk 送出的時刻與長度）、total、audio_seconds
  kokoro：t_first（= 第一個 result；Kokoro 整句一次出）、n_results、total、audio_seconds

--fake：沒有 mlx 的機器（Linux 容器）演練用，假的時間表＋正弦波；畫面會標 FAKE。
直接跑 `python mlx_worker.py --selfcheck` 只做 import 與版本，不載模型。
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import common as C  # noqa: E402

PROTO = os.fdopen(os.dup(1), "w", encoding="utf-8")       # 協定走原本的 stdout；之後 print 都進 stderr
sys.stdout = sys.stderr


def send(obj: dict) -> None:
    PROTO.write(json.dumps(obj, ensure_ascii=False, default=float) + "\n")
    PROTO.flush()


def versions() -> dict:
    from importlib import metadata
    v = {"python": sys.version.split()[0]}
    for pkg in ("mlx", "mlx-audio", "mlx-lm", "misaki", "numpy", "huggingface_hub"):
        try:
            v[pkg] = metadata.version(pkg)
        except metadata.PackageNotFoundError:
            v[pkg] = None
    return v


class Fake:
    """假的兩個模型：時間表照一個簡單的模型編（Qwen3 第一包固定、之後每 chunk RTF 0.5；Kokoro 隨字數線性），聲音是正弦波。"""
    sr = 24000

    def qwen3(self, text: str, interval: float, **k) -> dict:
        n_tok = max(4, int(len(text) * 0.9 * 12.5 / 3.5))        # 約 3.5 字／秒
        chunk = max(1, int(interval * 12.5))
        t, chunks = 0.25, []
        for i in range(0, n_tok, chunk):
            m = min(chunk, n_tok - i)
            t += (0.5 * m / 12.5) if i else 0
            chunks.append({"t": round(t, 4), "samples": m * 1920, "seconds": m * 1920 / self.sr})
        total = chunks[-1]["t"] + 0.02
        n = sum(c["samples"] for c in chunks)
        wav = 0.1 * np.sin(2 * np.pi * 220 * np.arange(n) / self.sr) * (0.5 + 0.5 * np.sin(2 * np.pi * 3 * np.arange(n) / self.sr))
        return {"t_first": chunks[0]["t"], "chunks": chunks, "total": total, "wav": wav, "sr": self.sr, "n_tokens": n_tok, "ref_encode_s": 0.08}

    def kokoro(self, text: str, **k) -> dict:
        sec = len(text) / 3.5
        t = 0.05 + 0.02 * len(text)
        n = int(sec * self.sr)
        wav = 0.1 * np.sin(2 * np.pi * 330 * np.arange(n) / self.sr)
        return {"t_first": round(t, 4), "n_results": 1, "total": round(t, 4), "wav": wav, "sr": self.sr}


class Real:
    def __init__(self):
        import mlx.core as mx
        from mlx_audio.tts.utils import load_model
        self.mx, self.load_model = mx, load_model
        self.qwen = self.koko = None
        self.load_s = {}

    def load(self, qwen3: str | None, kokoro: str | None, revision: dict | None = None) -> dict:
        """一律傳 repo id ＋ revision（HF_HUB_OFFLINE=1 時 snapshot_download 直接讀快取），**不要傳快照路徑**：
        mlx-audio 的 model_type 從路徑名推（config.json 沒有 model_type 時），快照資料夾名是 commit sha → 「Model type a71e… not supported」
        （2026-11-09 Mac 第一次跑 Kokoro 踩到）。model_type 也明講，不靠它猜。"""
        rev = revision or {}
        out = {}
        if qwen3:
            t0 = time.perf_counter()
            self.qwen = self.load_model(qwen3, revision=rev.get("qwen3"), model_type="qwen3_tts")
            self.load_s["qwen3"] = round(time.perf_counter() - t0, 2)
            cfg = getattr(self.qwen, "config", None)
            out["qwen3"] = {"load_s": self.load_s["qwen3"], "sample_rate": int(getattr(self.qwen, "sample_rate", 24000)),
                            "tts_model_type": getattr(cfg, "tts_model_type", None),
                            "n_params_M": round(sum(v.size for _, v in self._params(self.qwen)) / 1e6, 1)}
        if kokoro:
            t0 = time.perf_counter()
            self.koko = self.load_model(kokoro, revision=rev.get("kokoro"), model_type="kokoro")
            self.load_s["kokoro"] = round(time.perf_counter() - t0, 2)
            out["kokoro"] = {"load_s": self.load_s["kokoro"], "sample_rate": int(getattr(self.koko, "sample_rate", 24000))}
        return out

    @staticmethod
    def _params(model):
        from mlx.utils import tree_flatten
        try:
            return tree_flatten(model.parameters())
        except Exception:
            return []

    def qwen3(self, text: str, interval: float, ref_audio: str, ref_text: str, seed: int = 1, temperature: float = 0.9,
              lang_code: str = "auto", measure_ref_encode: bool = True, **k) -> dict:
        mx = self.mx
        m = self.qwen
        r: dict = {"interval": interval, "chunk_tokens": max(1, int(interval * 12.5))}
        # prompt 的 speech-tokenizer 編碼要多久（ICL 的每一次呼叫都含這一段；單獨量一次當參考）
        if measure_ref_encode:
            try:
                from mlx_audio.utils import load_audio                       # 住在 mlx_audio.utils，不在 tts.utils（11-09 第一次彩排 ImportError）
                a = load_audio(ref_audio, sample_rate=int(getattr(m, "sample_rate", 24000)))
                t0 = time.perf_counter()
                codes = m.speech_tokenizer.encode(a[None, None, :])
                mx.eval(codes)
                r["ref_encode_s"] = round(time.perf_counter() - t0, 4)
                r["ref_tokens"] = int(codes.shape[-1])
            except Exception as e:                                        # 只是參考，量不到不影響主線
                r["ref_encode_error"] = repr(e)
        mx.random.seed(seed)
        t0 = time.perf_counter()
        chunks, pieces = [], []
        gen = m.generate(text=text, ref_audio=ref_audio, ref_text=ref_text, stream=True, streaming_interval=interval,
                         temperature=temperature, lang_code=lang_code, verbose=False)
        n_tok = 0
        for res in gen:
            t = time.perf_counter() - t0
            a = np.asarray(res.audio, dtype=np.float32).reshape(-1)
            pieces.append(a)
            n_tok += int(getattr(res, "token_count", 0) or 0)
            chunks.append({"t": round(t, 4), "samples": int(a.shape[0]), "seconds": a.shape[0] / int(res.sample_rate),
                           "tokens": int(getattr(res, "token_count", 0) or 0), "is_final": bool(getattr(res, "is_final_chunk", False))})
            sr = int(res.sample_rate)
        total = time.perf_counter() - t0
        wav = np.concatenate(pieces) if pieces else np.zeros(0, dtype=np.float32)
        r.update(t_first=chunks[0]["t"] if chunks else None, chunks=chunks, total=round(total, 4), wav=wav, sr=sr if chunks else 24000,
                 n_tokens=n_tok, peak_memory_gb=round(mx.get_peak_memory() / 1e9, 2))
        return r

    def kokoro(self, text: str, voice: str, lang_code: str, seed: int = 1, **k) -> dict:
        mx = self.mx
        mx.random.seed(seed)
        t0 = time.perf_counter()
        pieces, times = [], []
        for res in self.koko.generate(text=text, voice=voice, lang_code=lang_code, verbose=False):
            times.append(round(time.perf_counter() - t0, 4))
            pieces.append(np.asarray(res.audio, dtype=np.float32).reshape(-1))
            sr = int(res.sample_rate)
        total = time.perf_counter() - t0
        wav = np.concatenate(pieces) if pieces else np.zeros(0, dtype=np.float32)
        return {"t_first": times[0] if times else None, "result_times": times, "n_results": len(times), "total": round(total, 4),
                "wav": wav, "sr": sr if pieces else 24000}


def fetch(repos: dict, lock: dict) -> dict:
    from huggingface_hub import HfApi, snapshot_download
    api = HfApi()
    out = {}
    for key, repo in repos.items():
        info = api.model_info(repo)
        sha = lock.get(repo) or info.sha
        path = snapshot_download(repo, revision=sha)
        size = sum(f.stat().st_size for f in Path(path).rglob("*") if f.is_file()) / 1e6
        card = getattr(info, "card_data", None)
        out[key] = {"repo": repo, "sha": sha, "path": path, "size_mb": round(size), "license": card.get("license") if card else None}
    return out


def local_path(repo: str, sha: str | None) -> str | None:
    """已下載的快照路徑，只拿來回報；載入一律傳 repo id（見 Real.load）。"""
    try:
        from huggingface_hub import snapshot_download
        return snapshot_download(repo, revision=sha, local_files_only=True)
    except Exception:
        return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(HERE / "runs" / "_worker"))
    ap.add_argument("--fake", action="store_true")
    ap.add_argument("--selfcheck", action="store_true", help="只 import、回版本")
    args = ap.parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.selfcheck:
        v = versions()
        try:
            import mlx_audio.tts.models.qwen3_tts  # noqa: F401
            import mlx_audio.tts.models.kokoro     # noqa: F401
            v["qwen3_tts_in_mlx_audio"] = True
        except Exception as e:
            v["qwen3_tts_in_mlx_audio"] = repr(e)
        send({"ok": True, "versions": v})
        return
    backend = None
    try:
        backend = Fake() if args.fake else Real()
    except Exception as e:
        send({"ok": False, "error": repr(e), "stage": "import"})
        return
    send({"ok": True, "ready": True, "fake": args.fake, "versions": versions()})
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
            cmd = req.get("cmd")
            if cmd == "quit":
                send({"ok": True, "bye": True}); break
            if cmd == "versions":
                send({"ok": True, "versions": versions()}); continue
            if cmd == "fetch":
                send({"ok": True, "fetched": fetch(req["repos"], req.get("lock", {}))}); continue
            if cmd == "load":
                if args.fake:
                    send({"ok": True, "loaded": {"qwen3": {"load_s": 0.0, "fake": True}, "kokoro": {"load_s": 0.0, "fake": True}}}); continue
                rev = req.get("revision") or {}
                q, kk = req.get("qwen3"), req.get("kokoro")
                send({"ok": True, "loaded": backend.load(q, kk, rev), "paths": {"qwen3": local_path(q, rev.get("qwen3")) if q else None,
                                                                                 "kokoro": local_path(kk, rev.get("kokoro")) if kk else None}}); continue
            if cmd in ("qwen3", "kokoro"):
                name = req["name"]
                r = getattr(backend, cmd)(**{k: v for k, v in req.items() if k not in ("cmd", "name")})
                wav = r.pop("wav")
                p = out_dir / f"{name}.wav"
                C.save_wav(p, np.asarray(wav, dtype=np.float64), r["sr"])
                r.update(wav=str(p), audio_seconds=round(len(wav) / r["sr"], 3), rms=float(np.sqrt(np.mean(np.square(wav)) + 1e-12)) if len(wav) else 0.0)
                send({"ok": True, "cmd": cmd, "name": name, **r}); continue
            send({"ok": False, "error": f"不認得的指令 {cmd!r}"})
        except Exception as e:
            import traceback
            traceback.print_exc()
            send({"ok": False, "error": repr(e), "cmd": req.get("cmd") if isinstance(req, dict) else None, "name": req.get("name") if isinstance(req, dict) else None})


if __name__ == "__main__":
    main()

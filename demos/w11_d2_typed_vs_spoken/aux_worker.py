# -*- coding: utf-8 -*-
# 此檔由 bin/publish_demos.py 從 private repo 的 <course>/demos/ 複製而來，請勿在公開 repo 直接編輯。
"""W11 demo 2 的子程序：把題目念出來的 TTS（Kokoro）與 cascade 那一列的辨識器（Whisper small），在共用的 ../.venv 裡跑。

demo.py 住在 ../.venv-sts（新版的 mlx-audio，Ming-3.0 要它）；Kokoro 與 Whisper 沿用前幾週 demo 釘住的版本
（mlx-audio 0.5.4、transformers 與 torch 照 ../versions.lock），所以分成兩個 process：
demo.py 用 subprocess 開這一支，stdin 一行一個 JSON 指令、stdout 一行一個 JSON 結果（其他輸出都導到 stderr）。

  {"cmd": "versions"}
  {"cmd": "fetch", "repos": {"kokoro": "...", "whisper": "..."}, "lock": {...}}     下載、回 commit（要網路）
  {"cmd": "load", "kokoro": {"repo", "revision"} | null, "whisper": {"repo", "revision"} | null}
  {"cmd": "tts", "text": "...", "voice": "zf_xiaobei", "lang_code": "z", "out": "<wav path>"}
  {"cmd": "asr", "wav": "<wav path>", "language": "zh"}
  {"cmd": "quit"}

--fake：沒有模型的機器演練用（正弦波、把檔名旁的 .txt 當成轉錄）；畫面會標 FAKE。
"""
from __future__ import annotations

import argparse
import json
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
FS_ASR = 16000


def send(obj: dict) -> None:
    PROTO.write(json.dumps(obj, ensure_ascii=False, default=float) + "\n")
    PROTO.flush()


def versions() -> dict:
    from importlib import metadata
    v = {"python": sys.version.split()[0]}
    for pkg in ("mlx", "mlx-audio", "misaki", "torch", "transformers", "numpy", "huggingface_hub"):
        try:
            v[pkg] = metadata.version(pkg)
        except metadata.PackageNotFoundError:
            v[pkg] = None
    return v


def trim_silence(x: np.ndarray, sr: int, rel_db: float = -35.0, keep_s: float = 0.03) -> np.ndarray:
    """修掉頭尾比這一段的峰值低 rel_db 以上的部分，各留 keep_s。"""
    if len(x) == 0:
        return x
    h = max(1, sr // 200)
    n = len(x) // h
    if n == 0:
        return x
    e = np.sqrt(np.mean(np.square(x[:n * h].reshape(n, h)), axis=1))
    on = np.nonzero(e > e.max() * 10 ** (rel_db / 20))[0]
    if len(on) == 0:
        return x
    a = max(0, on[0] * h - int(keep_s * sr))
    b = min(len(x), (on[-1] + 1) * h + int(keep_s * sr))
    return x[a:b]


class Real:
    def __init__(self):
        self.koko = self.whisper = None

    def load(self, kokoro: dict | None, whisper: dict | None) -> dict:
        out = {}
        if kokoro:
            from mlx_audio.tts.utils import load_model
            t0 = time.perf_counter()
            # 傳 repo id ＋ revision，不傳快照路徑；model_type 明講（W8 Demo 2 在 Mac 上踩過：路徑名是 commit sha 時型別猜錯）
            self.koko = load_model(kokoro["repo"], revision=kokoro.get("revision"), model_type="kokoro")
            out["kokoro"] = {"load_s": round(time.perf_counter() - t0, 2)}
        if whisper:
            import torch
            from huggingface_hub import snapshot_download
            from transformers import WhisperForConditionalGeneration, WhisperProcessor
            try:
                from transformers.utils import logging as hf_logging
                hf_logging.set_verbosity_error()
            except Exception:
                pass
            t0 = time.perf_counter()
            path = snapshot_download(whisper["repo"], revision=whisper.get("revision"),
                                     allow_patterns=["*.json", "*.txt", "model.safetensors"])
            self.torch = torch
            self.proc = WhisperProcessor.from_pretrained(path)
            # Whisper 放 CPU：transformers 5.17.0 的 generate 在 MPS 上會掛（W7 Demo 1），small 在 CPU 上一句不到一秒
            self.whisper = WhisperForConditionalGeneration.from_pretrained(path).eval()
            out["whisper"] = {"load_s": round(time.perf_counter() - t0, 2), "device": "cpu"}
        return out

    def _say(self, text: str, voice: str, lang_code: str) -> tuple[list, int]:
        pieces, sr = [], 24000
        for res in self.koko.generate(text=text, voice=voice, lang_code=lang_code, verbose=False):
            pieces.append(np.asarray(res.audio, dtype=np.float32).reshape(-1))
            sr = int(res.sample_rate)
        return pieces, sr

    def tts(self, text: str, voice: str, lang_code: str, seed: int = 1, engine: str = "kokoro", qwen3: dict | None = None,
            **k) -> tuple[np.ndarray, int, dict]:
        import mlx.core as mx
        t0 = time.perf_counter()
        if engine == "qwen3":
            return self._tts_qwen3(text, voice, lang_code, qwen3 or {}, t0)
        mx.random.seed(seed)
        pieces, sr = self._say(text, voice, lang_code)
        wav = np.concatenate(pieces) if pieces else np.zeros(0, dtype=np.float32)
        return wav, sr, {"compute_s": round(time.perf_counter() - t0, 3), "n_pieces": len(pieces), "engine": "kokoro"}

    def _tts_qwen3(self, text: str, voice: str, lang_code: str, q: dict, t0: float) -> tuple[np.ndarray, int, dict]:
        """華語的題目用 Qwen3-TTS 念（W8 Demo 2 的那一份權重；華語句子裡的英文字母它念得出來）。

        Base 版沒有內建的聲音，要給一段參考音與它的文字（ICL cloning）。參考音是 Kokoro 用這個 voice 念的一句
        沒有字母的華語——所以「兩個聲音」仍然是原來那兩個，沒有任何人的錄音。參考音第一次用到時做一次、存檔。
        2026-11-27：Kokoro 的華語聲音不念選項的字母（A–D），切開來念再接也不行（Whisper 寫成「哎、你」）。
        """
        import mlx.core as mx
        if getattr(self, "qwen", None) is None:
            from mlx_audio.tts.utils import load_model
            self.qwen = load_model(q["repo"], revision=q.get("revision"), model_type="qwen3_tts")
        ref = Path(q["ref_dir"]) / f"{voice}.wav"
        if not ref.exists():
            mx.random.seed(1)
            pieces, sr0 = self._say(q["ref_text"], voice, lang_code)
            ref.parent.mkdir(parents=True, exist_ok=True)
            C.save_wav(ref, trim_silence(np.concatenate(pieces), sr0).astype(np.float64), sr0)
        mx.random.seed(int(q.get("seed", 1)))
        pieces, sr = [], 24000
        for res in self.qwen.generate(text=text, ref_audio=str(ref), ref_text=q["ref_text"], stream=True,
                                      streaming_interval=float(q.get("streaming_interval", 2.0)),
                                      temperature=float(q.get("temperature", 0.9)), lang_code=q.get("lang_code", "auto"), verbose=False):
            pieces.append(np.asarray(res.audio, dtype=np.float32).reshape(-1))
            sr = int(res.sample_rate)
        wav = np.concatenate(pieces) if pieces else np.zeros(0, dtype=np.float32)
        return wav, sr, {"compute_s": round(time.perf_counter() - t0, 3), "n_pieces": len(pieces), "engine": "qwen3", "ref": ref.name}

    def asr(self, wav: str, language: str, **k) -> dict:
        torch = self.torch
        x = C.load_wav(Path(wav), FS_ASR).astype(np.float32)
        t0 = time.perf_counter()
        feats = self.proc.feature_extractor(x, sampling_rate=FS_ASR, return_tensors="pt").input_features
        with torch.no_grad():
            ids = self.whisper.generate(input_features=feats, language=language, task="transcribe", return_timestamps=False,
                                        max_new_tokens=220, do_sample=False, num_beams=1)
        text = self.proc.tokenizer.decode(ids[0], skip_special_tokens=True).strip()
        return {"text": text, "compute_s": round(time.perf_counter() - t0, 3), "audio_s": round(len(x) / FS_ASR, 2)}


class Fake:
    """假的：TTS 給一段長度與字數成正比的正弦波，並把文字寫在 wav 旁邊的 .txt；ASR 讀那個 .txt（偶爾掉一個字）。"""

    def load(self, kokoro, whisper) -> dict:
        return {"kokoro": {"load_s": 0.0, "fake": True}, "whisper": {"load_s": 0.0, "fake": True}}

    def tts(self, text: str, voice: str, lang_code: str, out: str = "", **k):
        sr = 24000
        n = int(sr * min(12.0, 0.6 + 0.07 * len(text)))
        f0 = 180.0 if "f" in voice[:2] else 110.0
        wav = (0.1 * np.sin(2 * np.pi * f0 * np.arange(n) / sr)).astype(np.float32)
        if out:
            Path(out).parent.mkdir(parents=True, exist_ok=True)
            Path(out).with_suffix(".txt").write_text(text, encoding="utf-8")
        return wav, sr, {"compute_s": 0.0, "n_pieces": 1}

    def asr(self, wav: str, language: str, **k) -> dict:
        side = Path(wav).with_suffix(".txt")
        text = side.read_text(encoding="utf-8") if side.exists() else ""
        h = sum(ord(c) for c in Path(wav).stem)
        if text and h % 5 == 0:
            cut = len(text) // 3
            text = text[:cut] + text[cut + 2:]
        return {"text": text, "compute_s": 0.0, "audio_s": 0.0}


KOKORO_VOICE_REPO = "prince-canuma/Kokoro-82M"   # mlx-audio 的 Kokoro 從這個 repo 讀 voices/<名字>.safetensors，用到才抓


def fetch_voices(voices: list) -> dict:
    """把設定裡用到的 Kokoro voice 檔先抓進快取。

    mlx-audio 是第一次用到某個 voice 時才下載它；課堂與 check 都是離線模式（HF_HUB_OFFLINE=1），
    沒先抓的 voice 會在那時候報 IncompleteSnapshotError（2026-11-26 的 check：zm_yunxi、af_heart）。
    """
    from huggingface_hub import snapshot_download
    got = {}
    for v in voices:
        path = Path(snapshot_download(KOKORO_VOICE_REPO, allow_patterns=[f"voices/{v}.safetensors"]))
        got[v] = (path / "voices" / f"{v}.safetensors").exists()
    return got


def fetch(repos: dict, lock: dict, voices: list | None = None) -> dict:
    from huggingface_hub import HfApi, snapshot_download
    api = HfApi()
    out = {}
    for key, repo in repos.items():
        info = api.model_info(repo)
        sha = lock.get(repo) or info.sha
        pats = ["*.json", "*.txt", "model.safetensors"] if key == "whisper" else None
        path = snapshot_download(repo, revision=sha, allow_patterns=pats)
        size = sum(f.stat().st_size for f in Path(path).rglob("*") if f.is_file()) / 1e6
        card = getattr(info, "card_data", None)
        out[key] = {"repo": repo, "sha": sha, "size_mb": round(size), "license": card.get("license") if card else None}
    if voices:
        out["kokoro"]["voices"] = fetch_voices(voices)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fake", action="store_true")
    args = ap.parse_args()
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
        req = {}
        try:
            req = json.loads(line)
            cmd = req.get("cmd")
            if cmd == "quit":
                send({"ok": True, "bye": True})
                break
            if cmd == "versions":
                send({"ok": True, "versions": versions()})
            elif cmd == "fetch":
                send({"ok": True, "fetched": fetch(req["repos"], req.get("lock", {}), req.get("voices"))})
            elif cmd == "load":
                send({"ok": True, "loaded": backend.load(req.get("kokoro"), req.get("whisper"))})
            elif cmd == "tts":
                wav, sr, info = backend.tts(**{k: v for k, v in req.items() if k != "cmd"})
                out = Path(req["out"])
                out.parent.mkdir(parents=True, exist_ok=True)
                # 不做各自的峰值正規化：同一個增益存檔（Kokoro 的輸出本來就在 ±1 以內）
                C.save_wav(out, np.asarray(wav, dtype=np.float64), sr)
                rms = float(np.sqrt(np.mean(np.square(wav)) + 1e-12)) if len(wav) else 0.0
                send({"ok": True, "sr": sr, "audio_s": round(len(wav) / sr, 3), "rms": rms, **info})
            elif cmd == "asr":
                send({"ok": True, **backend.asr(**{k: v for k, v in req.items() if k != "cmd"})})
            else:
                send({"ok": False, "error": f"不認得的指令 {cmd!r}"})
        except Exception as e:
            import traceback
            traceback.print_exc()
            send({"ok": False, "error": repr(e), "cmd": req.get("cmd")})


if __name__ == "__main__":
    main()

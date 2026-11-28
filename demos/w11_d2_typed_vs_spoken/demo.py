# -*- coding: utf-8 -*-
# 此檔由 bin/publish_demos.py 從 private repo 的 <course>/demos/ 複製而來，請勿在公開 repo 直接編輯。
"""W11 demo 2：同一題，用打的與用說的——四個數字（只在終端機，單鍵）。

  a  打字 → θ0（原來的文字 LLM，近似）      b  打字 → θ（會聽的模型）
  c  說   → θ                               d  說 → Whisper small → 文字 → θ0（cascade）

子命令：
  fetch      下載 θ、轉出 θ0、確認 Kokoro 與 Whisper 在快取裡（要網路）→ model.lock
  check      課前體檢：每一條路各走一次（θ 讀、聽、轉錄、出聲；θ0；TTS；辨識器）→ runs/check.json
  tts        把題目念成 wav（→ ../_media/w11_d2/tts/，不進版控）
  record     own：自己念 10 題；tone：同一句話錄兩種語氣（→ ../_media/w11_d2/own/，不進版控）
  rehearse   全部條件跑一遍 → runs/rehearsal/rehearsal.json（進版控＝現場的退路）；可中斷、可接續
  show       上課用：讀彩排的數字，另外把模型載起來給現場的鍵
  replay     只讀彩排的紀錄，不載任何模型
  fake       沒有模型也能演練流程（畫面標 FAKE，數字沒有意義）
  selftest   題目、計分規則、統計、fake 整條流程
  export     寫 slides/assets/w11/w11-metrics.json → demo2

主程式在 ../.venv-sts（新版 mlx-audio）；Kokoro 與 Whisper 由 aux_worker.py 在 ../.venv 當子程序。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
import tomllib
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
DEMOS = HERE.parent
sys.path.insert(0, str(DEMOS))
sys.path.insert(0, str(HERE))
import common as C  # noqa: E402
import questions as Q  # noqa: E402
import score as S  # noqa: E402

CFG = tomllib.loads((HERE / "demo_config.toml").read_text(encoding="utf-8"))
RUNS = HERE / "runs"
LOCK = HERE / "model.lock"
METRICS = DEMOS.parent / "slides" / "assets" / "w11" / "w11-metrics.json"
FS = 16000
ITEMS = Q.items()
BY_ID = {q["id"]: q for q in ITEMS}
MC_IDS = [q["id"] for q in ITEMS if q["kind"] != "format"]
FMT_IDS = [q["id"] for q in ITEMS if q["kind"] == "format"]
# 某種語言整題不跑的題目（[run] skip）：念題目的聲音換了 seed 仍念錯的。四個條件一起拿掉，比較仍是同一批題目。
SKIP = {L: set(v) for L, v in CFG["run"].get("skip", {}).items()}


def keep(lang: str, i: str) -> bool:
    return i not in SKIP.get(lang, ())
KINDS = ["knowledge", "arithmetic", "reasoning"]
LANG_NAME = {"zh": "Mandarin", "en": "English"}

# 條件的代號 → 螢幕上的說明。c1／d1 是表上的 c、d（第一個聲音）；其餘是加映。
COND = {
    "a": "text → θ0",
    "b": "text → θ",
    "c1": "speech (TTS voice 1) → θ",
    "d1": "speech (voice 1) → Whisper → text → θ0",
    "e1": "speech (voice 1) → Whisper → text → θ",
    "c2": "speech (TTS voice 2) → θ",
    "d2": "speech (voice 2) → Whisper → text → θ0",
    "co": "speech (my own voice) → θ",
    "do": "speech (my own voice) → Whisper → text → θ0",
    "cn": "speech (voice 1) + babble → θ",
    "dn": "speech (voice 1) + babble → Whisper → text → θ0",
}
ORDER = ["a", "b", "c1", "d1", "e1", "c2", "d2", "co", "do", "cn", "dn"]


def P(rel: str) -> Path:
    return (HERE / rel).resolve()


def digest() -> str:
    """設定與題目的摘要：變了就不接續舊的彩排。"""
    h = hashlib.sha256()
    h.update(json.dumps({k: CFG[k] for k in ("run", "theta", "theta0", "tts", "asr", "noise", "probe", "own")}, sort_keys=True, ensure_ascii=False).encode())
    h.update(json.dumps([q["prompt"] for q in ITEMS], ensure_ascii=False).encode())
    return h.hexdigest()[:16]


# ── 鎖定檔與快取 ─────────────────────────────────────────────────
def read_lock(path: Path) -> dict:
    out = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            parts = line.split()
            if len(parts) == 2 and not line.startswith("#"):
                out[parts[0]] = parts[1]
    return out


def write_lock(d: dict) -> None:
    LOCK.write_text("".join(f"{k} {v}\n" for k, v in sorted(d.items())), encoding="utf-8")


def local_snapshot(repo: str, sha: str | None) -> str:
    """已下載的快照路徑。先問 huggingface_hub；它嫌快照不完整時（新版會檢查）直接到快取裡找。"""
    from huggingface_hub import constants, snapshot_download
    try:
        return snapshot_download(repo, revision=sha, local_files_only=True)
    except Exception:
        root = Path(constants.HF_HUB_CACHE) / ("models--" + repo.replace("/", "--")) / "snapshots"
        cands = [root / sha] if sha else sorted(root.glob("*"))
        for c in cands:
            if (c / "config.json").exists():
                return str(c)
        raise FileNotFoundError(f"快取裡沒有 {repo}（{sha or '任何版本'}）：先跑 ./present.sh fetch")


def versions() -> dict:
    from importlib import metadata
    v = C.versions()
    for pkg in ("mlx", "mlx-audio", "mlx-lm", "transformers", "huggingface_hub"):
        try:
            v[pkg] = metadata.version(pkg)
        except metadata.PackageNotFoundError:
            v[pkg] = None
    return v


# ── 聲音 ─────────────────────────────────────────────────────────
def load16(path: Path) -> np.ndarray:
    return C.load_wav(path, FS).astype(np.float32)


def power(x: np.ndarray) -> float:
    return float(np.mean(np.square(x, dtype=np.float64)) + 1e-20)


def babble(n: int, others: list[np.ndarray], rng: np.random.Generator) -> np.ndarray:
    """幾段別的錄音各調到同一個功率、各從隨機的位置起、循環補到 n 點，相加。做法與 W10 Demo 2 的 babble 相同。"""
    v = np.zeros(n, dtype=np.float64)
    for o in others:
        o = o.astype(np.float64) / np.sqrt(power(o))
        start = int(rng.integers(0, len(o)))
        v += np.resize(np.roll(o, -start), n)
    return v


def mix(s: np.ndarray, v: np.ndarray, snr_db: float) -> np.ndarray:
    """y = s + g v，g 讓整句的 SNR 等於 snr_db；語音本身不動。"""
    g = np.sqrt(power(s) / (power(v) * 10 ** (snr_db / 10)))
    return (s.astype(np.float64) + g * v).astype(np.float32)


def tts_path(lang: str, vi: int, qid: str, root: Path | None = None) -> Path:
    return (root or P(CFG["tts"]["dir"])) / lang / f"v{vi + 1}" / f"{qid}.wav"


def own_path(lang: str, name: str) -> Path:
    return P(CFG["own"]["dir"]) / lang / f"{name}.wav"


# ── 子程序（../.venv：Kokoro、Whisper）──────────────────────────
class Aux:
    def __init__(self, fake: bool = False):
        py = DEMOS / ".venv" / "bin" / "python"
        if fake or not py.exists():
            py = Path(sys.executable)
        cmd = [str(py), str(HERE / "aux_worker.py")] + (["--fake"] if fake else [])
        self.p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, encoding="utf-8", bufsize=1)
        hello = self._read()
        if not hello.get("ok"):
            raise RuntimeError(f"aux_worker 起不來：{hello}")
        self.versions = hello.get("versions", {})
        self.fake = fake
        self.loaded = set()

    def _read(self) -> dict:
        line = self.p.stdout.readline()
        if not line:
            raise RuntimeError("aux_worker 沒有回應（看上面的錯誤訊息）")
        return json.loads(line)

    def call(self, **req) -> dict:
        self.p.stdin.write(json.dumps(req, ensure_ascii=False) + "\n")
        self.p.stdin.flush()
        r = self._read()
        if not r.get("ok"):
            raise RuntimeError(f"aux_worker：{r.get('error')}")
        return r

    def need(self, what: str) -> None:
        if what in self.loaded:
            return
        if what == "kokoro":
            lock = read_lock(P(CFG["tts"]["lock_from"]))
            self.call(cmd="load", kokoro={"repo": CFG["tts"]["repo"], "revision": lock.get(CFG["tts"]["repo"])}, whisper=None)
        else:
            lock = read_lock(P(CFG["asr"]["lock_from"]))
            self.call(cmd="load", kokoro=None, whisper={"repo": CFG["asr"]["repo"], "revision": lock.get(CFG["asr"]["repo"])})
        self.loaded.add(what)

    def tts(self, text: str, voice: dict, out: Path) -> dict:
        self.need("kokoro")
        extra = {}
        if voice.get("engine") == "qwen3":
            q = dict(CFG["tts"]["qwen3"])
            # 個別題目換 seed（念錯的那幾題重念，其餘不動）：鍵是 tts_path 的 <語言>/v<k>/<題號>，例如 "zh/v1/a08"
            by_item = q.pop("seed_by_item", {})
            q["seed"] = int(by_item.get(f"{out.parent.parent.name}/{out.parent.name}/{out.stem}", q.get("seed", 1)))
            extra = {"engine": "qwen3", "qwen3": {**q, "revision": read_lock(P(CFG["tts"]["lock_from"])).get(q["repo"]),
                                                  "ref_dir": str(P(CFG["tts"]["dir"]) / "_ref")}}
        return self.call(cmd="tts", text=text, voice=voice["voice"], lang_code=voice["lang_code"], out=str(out), **extra)

    def asr(self, wav: Path, lang: str) -> dict:
        self.need("whisper")
        return self.call(cmd="asr", wav=str(wav), language=lang)

    def close(self) -> None:
        try:
            self.p.stdin.write('{"cmd": "quit"}\n')
            self.p.stdin.flush()
            self.p.wait(timeout=5)
        except Exception:
            self.p.kill()


# ── 兩個模型 ─────────────────────────────────────────────────────
class Theta:
    """θ：Ming-3.0-audio-tiny（mlx-audio 的 sts 模組）。同一組權重可以讀文字，也可以聽語音。"""

    def __init__(self):
        import mlx.core as mx
        from mlx_audio.sts import load
        self.mx = mx
        repo = CFG["theta"]["repo"]
        t0 = time.perf_counter()
        self.model = load(local_snapshot(repo, read_lock(LOCK).get(repo)), strict=True)
        self.load_s = time.perf_counter() - t0
        r = CFG["run"]
        self.kw = dict(max_new_tokens=int(r["max_new_tokens"]), temperature=float(r["temperature"]), thinking=bool(r["thinking"]),
                       system_prompt=r["system_prompt"] or None)

    def _out(self, res, t0: float) -> dict:
        return {"reply": res.text, "t": round(time.perf_counter() - t0, 3), "n_tok": int(res.generation_tokens),
                "cut": res.finish_reason == "length", "n_in": int(res.prompt_tokens)}

    def text(self, prompt: str) -> dict:
        t0 = time.perf_counter()
        return self._out(self.model.generate(text=prompt, task="text", **self.kw), t0)

    def speech(self, x: np.ndarray, text: str | None = None) -> dict:
        """語音進、文字出。text 給了就與語音放在同一輪（文字在前、語音在後）。"""
        t0 = time.perf_counter()
        return self._out(self.model.generate(np.asarray(x, dtype=np.float32), text=text, task="speech_to_text", sample_rate=FS, **self.kw), t0)

    def asr(self, x: np.ndarray) -> dict:
        t0 = time.perf_counter()
        res = self.model.generate(np.asarray(x, dtype=np.float32), task="asr", sample_rate=FS, max_new_tokens=256)
        return {"text": res.text, "t": round(time.perf_counter() - t0, 3)}

    def speak(self, text: str, voice: str, out: Path) -> dict:
        """Talker 把一段文字念出來（44.1 kHz）。這個移植版的 Talker 只讀文字。"""
        t0 = time.perf_counter()
        res = self.model.generate(text=text, task="tts", voice=voice, seed=int(CFG["run"]["seed"]))
        wav = np.asarray(res.audio, dtype=np.float64).reshape(-1)
        C.save_wav(out, wav, int(res.sample_rate))
        return {"t": round(time.perf_counter() - t0, 3), "audio_s": round(len(wav) / res.sample_rate, 2), "sr": int(res.sample_rate)}

    def peak_gb(self) -> float:
        return round(self.mx.get_peak_memory() / 1e9, 2)


class Theta0:
    """θ0：原來的文字 LLM（mlx-lm）。"""

    def __init__(self):
        from mlx_lm import load, stream_generate
        from mlx_lm.sample_utils import make_sampler
        c = CFG["theta0"]
        path = str(P(c["dir"])) if c["source"] == "convert" else local_snapshot(c["hub_repo"], read_lock(LOCK).get(c["hub_repo"]))
        if not Path(path).exists():
            raise FileNotFoundError(f"沒有 θ0 的權重（{path}）：先跑 ./present.sh fetch")
        t0 = time.perf_counter()
        self.model, self.tok = load(path)
        self.load_s = time.perf_counter() - t0
        self.stream = stream_generate
        self.sampler = make_sampler(temp=float(CFG["run"]["temperature"]))

    def text(self, prompt: str) -> dict:
        r = CFG["run"]
        msgs = ([{"role": "system", "content": r["system_prompt"]}] if r["system_prompt"] else []) + [{"role": "user", "content": prompt}]
        p = self.tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False, enable_thinking=bool(r["thinking"]))
        t0 = time.perf_counter()
        out, last = "", None
        for last in self.stream(self.model, self.tok, prompt=p, max_tokens=int(r["max_new_tokens"]), sampler=self.sampler):
            out += last.text
        return {"reply": S.strip_think(out), "t": round(time.perf_counter() - t0, 3), "n_tok": int(last.generation_tokens) if last else 0,
                "cut": bool(last and last.finish_reason == "length"), "n_in": int(last.prompt_tokens) if last else 0}


def _h(*parts) -> float:
    return int(hashlib.md5("|".join(map(str, parts)).encode()).hexdigest()[:8], 16) / 0xFFFFFFFF


class _FakeLM:
    """假的模型：依題號與條件決定答對與否，回答的寫法輪流換（考計分規則）。數字沒有意義。"""
    acc = 0.85
    tag = "fake"

    def _find(self, text: str) -> tuple[dict, str]:
        best, score = (ITEMS[0], "en"), -1
        for q in ITEMS:
            for lang in ("zh", "en"):
                p = q["prompt"][lang]
                k = 0
                while k < min(len(p), len(text)) and p[k] == text[k]:
                    k += 1
                if k > score:
                    best, score = (q, lang), k
        return best

    def _reply(self, q: dict, lang: str, salt: str) -> str:
        right = _h(q["id"], lang, self.tag, salt) < self.acc
        style = int(_h(q["id"], salt, "style") * 4)
        if q["kind"] == "format":
            good = {"letter": S.labels(lang)[q["answer"]] if q["answer"] is not None else "A", "number": "13" if q["id"] == "f03" else "8",
                    "yes_no": ("是" if lang == "zh" else "Yes") if q["id"] == "f05" else ("不是" if lang == "zh" else "No"),
                    "max_units": "綠色" if lang == "zh" else "Green", "units": "冷" if lang == "zh" else "Cold",
                    "items": "星期一、星期二、星期三" if lang == "zh" else "Monday, Tuesday, Wednesday",
                    "prefix": "答案：巴黎" if lang == "zh" else "Answer: Paris"}[q["fmt"]["rule"]]
            return good if right else (("好的，" + good + "。希望有幫到你。") if lang == "zh" else f"Sure! That would be {good}, I think.")
        i = q["answer"] if right else (q["answer"] + 1 + int(_h(q["id"], salt) * 3)) % 4
        lab, opt = S.labels(lang)[i], q["options"][lang][i]
        if lang == "zh":
            return [f"答案是{lab}。", f"{opt}", f"我們一步一步來。先看題目，再比較選項 A 和 B。\n所以答案是{lab}，{opt}。", f"{lab}，{opt}。"][style]
        return [f"The answer is {lab}.", f"{opt}", f"Let's go step by step. Options A and B look close.\nSo the answer is **{lab}: {opt}**.", f"{lab}. {opt}"][style]

    def text(self, prompt: str) -> dict:
        q, lang = self._find(prompt)
        r = self._reply(q, lang, "text")
        return {"reply": r, "t": 0.0, "n_tok": len(r), "cut": False, "n_in": len(prompt)}


class FakeTheta0(_FakeLM):
    acc, tag, load_s = 0.88, "theta0", 0.0


class FakeTheta(_FakeLM):
    acc, tag, load_s = 0.82, "theta", 0.0

    def speech(self, x: np.ndarray, text: str | None = None, hint: tuple | None = None) -> dict:
        if hint is None:
            r = "抱歉，我沒有聽清楚。" if text and "一" <= text[0] <= "鿿" else "You said you wanted to book a table for two."
            return {"reply": r, "t": 0.0, "n_tok": len(r), "cut": False, "n_in": len(x) // 2560}
        q, lang, salt = hint
        self.acc = 0.70 if salt.startswith("c") and "n" not in salt else 0.5
        r = self._reply(q, lang, salt)
        self.acc = 0.82
        return {"reply": r, "t": 0.0, "n_tok": len(r), "cut": False, "n_in": len(x) // 2560}

    def asr(self, x: np.ndarray, hint: str = "") -> dict:
        return {"text": hint, "t": 0.0}

    def speak(self, text: str, voice: str, out: Path) -> dict:
        C.save_wav(out, 0.1 * np.sin(2 * np.pi * 220 * np.arange(44100) / 44100), 44100)
        return {"t": 0.0, "audio_s": 1.0, "sr": 44100}

    def peak_gb(self) -> float:
        return 0.0


# ── 跑條件 ───────────────────────────────────────────────────────
class Runner:
    def __init__(self, fake: bool = False, need_models: bool = True, tts_root: Path | None = None):
        self.fake = fake
        self.tts_root = tts_root
        self.aux = None
        self.theta = self.theta0 = None
        if need_models:
            self.theta0 = FakeTheta0() if fake else Theta0()
            self.theta = FakeTheta() if fake else Theta()
        self._wav: dict = {}

    def get_aux(self) -> Aux:
        if self.aux is None:
            self.aux = Aux(fake=self.fake)
        return self.aux

    def close(self) -> None:
        if self.aux:
            self.aux.close()

    def wav(self, path: Path) -> np.ndarray:
        if path not in self._wav:
            self._wav[path] = load16(path)
        return self._wav[path]

    # 題目的聲音
    def ensure_tts(self, langs: list[str], force: bool = False, ids: list[str] | None = None) -> dict:
        info = {}
        todo = [(lang, vi, q) for lang in langs for vi in range(len(CFG["tts"][lang])) for q in ITEMS if (ids is None or q["id"] in ids) and keep(lang, q["id"])]
        missing = [(lang, vi, q) for lang, vi, q in todo if force or not tts_path(lang, vi, q["id"], self.tts_root).exists()]
        if missing:
            C.note(f"TTS：要念 {len(missing)} 段（Kokoro）……")
        for k, (lang, vi, q) in enumerate(missing):
            r = self.get_aux().tts(q["prompt"][lang], CFG["tts"][lang][vi], tts_path(lang, vi, q["id"], self.tts_root))
            info[f"{lang}/v{vi + 1}/{q['id']}"] = r["audio_s"]
            if (k + 1) % 25 == 0:
                C.note(f"  {k + 1} / {len(missing)}")
        return info

    def noisy(self, lang: str, qid: str) -> np.ndarray:
        """第一個聲音念的那一題，加上同語言另外 k 段題目錄音（兩個聲音都取）疊成的 babble。每題的抽法由題號決定，重跑相同。"""
        rng = np.random.default_rng(int(hashlib.md5(f"{lang}/{qid}".encode()).hexdigest()[:8], 16))
        pool = [(vi, o) for vi in range(len(CFG["tts"][lang])) for o in MC_IDS + FMT_IDS if o != qid and keep(lang, o)]
        chosen = [pool[i] for i in rng.choice(len(pool), size=int(CFG["noise"]["k"]), replace=False)]
        # babble 的來源是整份題目裡抽的；--n 的預覽只念了一部分題目，抽到還沒念的就在這裡補念
        # （不改抽法：預覽與完整彩排的同一題疊的是同一批聲音）。2026-11-27：預覽在這裡 FileNotFoundError。
        for vi, o in chosen:
            p = tts_path(lang, vi, o, self.tts_root)
            if not p.exists():
                self.get_aux().tts(BY_ID[o]["prompt"][lang], CFG["tts"][lang][vi], p)
        s = self.wav(tts_path(lang, 0, qid, self.tts_root))
        v = babble(len(s), [self.wav(tts_path(lang, vi, o, self.tts_root)) for vi, o in chosen], rng)
        return mix(s, v, float(CFG["noise"]["snr_db"]))

    def source(self, lang: str, key: str, qid: str) -> tuple[np.ndarray | None, Path | None]:
        """說的條件：這一題的聲音（陣列）與它在磁碟上的位置（給辨識器；加噪的那一版先寫成暫存檔）。"""
        tag = key[1:]
        if tag in ("1", "2"):
            p = tts_path(lang, int(tag) - 1, qid, self.tts_root)
            return self.wav(p), p
        if tag == "o":
            p = own_path(lang, qid)
            return (self.wav(p), p) if p.exists() else (None, None)
        if tag == "n":
            y = self.noisy(lang, qid)
            p = RUNS / "_tmp" / f"noisy_{lang}_{qid}.wav"
            C.save_wav(p, y.astype(np.float64), FS)
            if self.fake:
                side = tts_path(lang, 0, qid, self.tts_root).with_suffix(".txt")
                if side.exists():
                    p.with_suffix(".txt").write_text(side.read_text(encoding="utf-8"), encoding="utf-8")
            return y, p
        raise KeyError(key)

    def one(self, lang: str, key: str, qid: str, heard_cache: dict | None = None) -> dict | None:
        """一題、一個條件 → 紀錄（回答、判定、時間；cascade 另有 heard＝辨識器寫出來的字）。"""
        q = BY_ID[qid]
        prompt = q["prompt"][lang]
        if key == "a":
            rec = self.theta0.text(prompt)
        elif key == "b":
            rec = self.theta.text(prompt)
        else:
            x, path = self.source(lang, key, qid)
            if x is None:
                return None
            if key[0] == "c":
                rec = self.theta.speech(x, hint=(q, lang, key)) if self.fake else self.theta.speech(x)
            else:
                ck = f"{lang}/{key[1:]}/{qid}"
                if heard_cache is not None and ck in heard_cache:
                    heard = heard_cache[ck]
                else:
                    heard = self.get_aux().asr(path, lang)["text"]
                    if heard_cache is not None:
                        heard_cache[ck] = heard
                rec = (self.theta0 if key[0] == "d" else self.theta).text(heard)
                rec["heard"] = heard
        rec.update(S.judge(rec["reply"], q, lang))
        return rec

    def probes(self, lang: str) -> dict:
        """兩個探針：靜音與白噪音（只有聲音、聲音加一句打字的問題）；同一句話的兩種語氣（要先 record tone）。"""
        pr = CFG["probe"]
        n = int(float(pr["seconds"]) * FS)
        rng = np.random.default_rng(int(CFG["run"]["seed"]))
        clips = {"silence": np.zeros(n, dtype=np.float32), "white": (float(pr["noise_rms"]) * rng.standard_normal(n)).astype(np.float32)}
        out: dict = {}
        ask = (lambda x, text=None: self.theta.speech(x, text)["reply"])
        for name, x in clips.items():
            out[name] = {"alone": ask(x), "asked": ask(x, pr["question"][lang])}
        tone = {}
        for mood in ("happy", "flat"):
            p = own_path(lang, f"tone_{mood}")
            if p.exists():
                x = self.wav(p)
                tone[mood] = {"alone": ask(x), "followup": ask(x, pr["tone_followup"][lang]), "direct": ask(x, pr["tone_direct"][lang]),
                              "heard": self.theta.asr(x)["text"] if not self.fake else pr["tone_sentence"][lang]}
        if tone:
            out["tone"] = tone
        return out


# ── 彩排的紀錄 → 數字 ────────────────────────────────────────────
def mcs(res: dict, lang: str | None = None) -> list[str]:
    """這份紀錄裡的四選一題號（預覽只跑一部分題目；給了 lang 就再拿掉 [run] skip 的）。"""
    return [i for i in res["meta"].get("ids", MC_IDS) if i in MC_IDS and (lang is None or keep(lang, i))]


def fms(res: dict, lang: str | None = None) -> list[str]:
    return [i for i in res["meta"].get("ids", FMT_IDS) if i in FMT_IDS and (lang is None or keep(lang, i))]


def recs(res: dict, lang: str, key: str) -> dict:
    return res.get("cond", {}).get(f"{lang}/{key}", {})


def ok_list(res: dict, lang: str, key: str, ids: list[str]) -> list[bool] | None:
    r = recs(res, lang, key)
    if any(i not in r for i in ids):
        return None
    return [bool(r[i]["ok"]) for i in ids]


def row(res: dict, lang: str, key: str, ids: list[str] | None = None) -> dict | None:
    ids = ids or mcs(res, lang)
    ok = ok_list(res, lang, key, ids)
    if ok is None or not ids:
        return None
    r = recs(res, lang, key)
    ci = S.bootstrap_ci(ok, int(CFG["run"]["bootstrap"]), int(CFG["run"]["seed"]))
    ci.update(cut=sum(bool(r[i].get("cut")) for i in ids), no_pick=sum(r[i].get("pick") is None for i in ids),
              t_mean=round(float(np.mean([r[i]["t"] for i in ids])), 2), tok_mean=round(float(np.mean([r[i]["n_tok"] for i in ids])), 1),
              by_kind={k: [sum(bool(r[i]["ok"]) for i in ids if BY_ID[i]["kind"] == k), sum(BY_ID[i]["kind"] == k for i in ids)] for k in KINDS})
    return ci


def fmt_row(res: dict, lang: str, key: str) -> dict | None:
    r, ids = recs(res, lang, key), fms(res, lang)
    if not ids or any(i not in r for i in ids):
        return None
    judged = [r[i]["ok"] for i in ids if r[i]["ok"] is not None]
    return {"kept": sum(bool(r[i]["fmt"]) for i in ids), "n": len(ids), "content": sum(bool(v) for v in judged), "n_content": len(judged)}


def pair(res: dict, lang: str, x: str, y: str, ids: list[str] | None = None) -> dict | None:
    ids = ids or mcs(res, lang)
    a, b = ok_list(res, lang, x, ids), ok_list(res, lang, y, ids)
    if a is None or b is None:
        return None
    return S.paired(a, b, int(CFG["run"]["bootstrap"]), int(CFG["run"]["seed"]))


def audit(res: dict, lang: str, who: str = "w1") -> dict | None:
    """聽對了、卻答錯：θ 自己把題目轉錄出來（w1），對題目文字的錯誤率不超過門檻的算「聽對」，再看 c 那一列它答對沒有。"""
    w, c, ids = res.get("heard", {}).get(f"{lang}/{who}", {}), recs(res, lang, "c1"), mcs(res, lang)
    if not ids or any(i not in w or i not in c for i in ids):
        return None
    tol = float(CFG["audit"]["tol"])
    er = {i: S.error_rate(BY_ID[i]["prompt"][lang], w[i], lang) for i in ids}
    right = [i for i in ids if er[i] <= tol]
    return {"heard_right": len(right), "n": len(ids), "exact": sum(er[i] == 0 for i in ids),
            "right_but_wrong": [i for i in right if not c[i]["ok"]], "wrong_c": sum(not c[i]["ok"] for i in ids),
            "mean_er": round(float(np.mean(list(er.values()))), 4)}


def summary(res: dict) -> dict:
    """投影片與 w11-metrics.json 用的數字（不含回答的全文）。"""
    out = {"date": res["meta"].get("finished") or res["meta"].get("started"), "main": CFG["run"]["main"], "models": res["meta"].get("models"),
           "versions": res["meta"].get("versions"), "fake": res["meta"].get("fake", False), "n_mc": len(MC_IDS), "n_format": len(FMT_IDS),
           "skip": {L: sorted(v) for L, v in SKIP.items() if v}, "lang": {}}
    for lang in res["meta"]["langs"]:
        L = {"rows": {}, "format": {}, "pairs": {}}
        for key in ORDER:
            r = row(res, lang, key, [i for i in CFG["own"]["ids"] if i in mcs(res, lang)] if key in ("co", "do") else None)
            if r:
                L["rows"][key] = r
            f = fmt_row(res, lang, key)
            if f:
                L["format"][key] = f
        for x, y in (("a", "b"), ("b", "c1"), ("a", "c1"), ("a", "d1"), ("c1", "d1"), ("b", "e1"), ("e1", "c1"), ("c1", "c2"), ("c1", "cn"), ("d1", "dn")):
            p = pair(res, lang, x, y)
            if p:
                L["pairs"][f"{x}-{y}"] = p
        own = [i for i in CFG["own"]["ids"] if i in mcs(res, lang)]
        if own and row(res, lang, "co", own) and row(res, lang, "c1", own):
            L["own_vs_tts"] = {"own": row(res, lang, "co", own), "tts_same_ids": row(res, lang, "c1", own), "pair": pair(res, lang, "c1", "co", own)}
        a = audit(res, lang)
        if a:
            L["audit"] = a
        if lang in res.get("probe", {}):
            L["probe"] = res["probe"][lang]
        out["lang"][lang] = L
    return out


# ── 螢幕 ─────────────────────────────────────────────────────────
def pct(k: int, n: int) -> str:
    return f"{100 * k / n:4.0f}%" if n else "   —"


def print_row(res: dict, lang: str, key: str, name: str | None = None, ids: list[str] | None = None) -> None:
    r = row(res, lang, key, ids)
    name = name or key
    if r is None:
        print(f"  {C.BOLD}{name:<3}{C.RESET} {COND[key]:<46} {C.GREY}（彩排裡沒有這一列）{C.RESET}")
        return
    kinds = " · ".join(f"{k[:5]} {v[0]}/{v[1]}" for k, v in r["by_kind"].items() if v[1])
    print(f"  {C.BOLD}{name:<3}{C.RESET} {COND[key]:<46} {C.BOLD}{r['k']:>2} / {r['n']}{C.RESET}  {pct(r['k'], r['n'])}  "
          f"95% [{r['lo']:.0f}, {r['hi']:.0f}]   {C.GREY}{kinds}   no pick {r['no_pick']} · cut {r['cut']} · {r['tok_mean']:.0f} tok · {r['t_mean']:.1f} s{C.RESET}")


def print_pair(res: dict, lang: str, x: str, y: str, what: str, ids: list[str] | None = None) -> None:
    p = pair(res, lang, x, y, ids)
    if p is None:
        return
    verdict = f"{C.GREY}區間包含 0：這份題目上不能說有差{C.RESET}" if p["lo"] <= 0 <= p["hi"] else f"{C.AMBER}區間不含 0{C.RESET}"
    print(f"  {what:<34} {x} − {y} = {p['diff']:+d} 題   95% [{p['lo']:+.0f}, {p['hi']:+.0f}]   "
          f"只有 {x} 對 {p['only_x']} 題、只有 {y} 對 {p['only_y']} 題   {verdict}")


def show_table(res: dict, lang: str) -> None:
    C.banner(f"四個數字（{LANG_NAME[lang]}；{len(mcs(res, lang))} 題四選一；格式題 {len(fms(res, lang))} 題）")
    for key, name in (("a", "a"), ("b", "b"), ("c1", "c"), ("d1", "d")):
        print_row(res, lang, key, name)
    own = [i for i in CFG["own"]["ids"] if i in mcs(res, lang)]
    if own and row(res, lang, "co", own):
        print_row(res, lang, "co", "c′", own)
    print()
    for key, name in (("a", "a"), ("b", "b"), ("c1", "c"), ("d1", "d")):
        f = fmt_row(res, lang, key)
        if f:
            print(f"  {C.BOLD}{name:<3}{C.RESET} format kept {C.BOLD}{f['kept']:>2} / {f['n']}{C.RESET}    {C.GREY}content right {f['content']} / {f['n_content']}{C.RESET}")
    C.note("差兩三題不算差距：看區間，不要只看百分比。")


def show_identity(res: dict, lang: str) -> None:
    C.banner("三個落差：a − c = (a − b) + (b − c)；再用 d 與 e 排除「只是聽錯字」")
    print_pair(res, lang, "a", "c1", "understanding gap（Cuervo et al.）")
    print_pair(res, lang, "a", "b", "  = forgetting")
    print_pair(res, lang, "b", "c1", "  + misalignment（Xiang et al.）")
    print()
    print_pair(res, lang, "a", "d1", "recognition alone（θ0）")
    print_pair(res, lang, "b", "e1", "recognition alone（θ）")
    print_pair(res, lang, "e1", "c1", "same words: read vs. heard（θ）")
    print_pair(res, lang, "c1", "d1", "end-to-end − cascade")


def show_format(res: dict, lang: str) -> None:
    C.banner(f"格式題 {len(fms(res, lang))} 題：限制附在題目後面（守住沒有／內容對不對，分開記）")
    for key, name in (("a", "a"), ("b", "b"), ("c1", "c"), ("d1", "d")):
        r = recs(res, lang, key)
        f = fmt_row(res, lang, key)
        if not f:
            continue
        marks = "".join((C.GREEN + "●" if r[i]["fmt"] else C.RED + "○") + C.RESET for i in fms(res, lang))
        print(f"  {C.BOLD}{name}{C.RESET}  kept {f['kept']:>2} / {f['n']}   {marks}   {C.GREY}content right {f['content']} / {f['n_content']}{C.RESET}")
    print(f"     {C.GREY}{' '.join(fms(res, lang))}{C.RESET}")
    worst = next((i for i in fms(res, lang) if recs(res, lang, "c1").get(i) and not recs(res, lang, "c1")[i]["fmt"]), None)
    if worst:
        print(f"\n  例：{worst}  {BY_ID[worst]['prompt'][lang]}")
        for key, name in (("b", "b"), ("c1", "c")):
            if worst in recs(res, lang, key):
                print(f"    {name}: {clip(recs(res, lang, key)[worst]['reply'], 110)}")


def show_own(res: dict, lang: str) -> None:
    own = [i for i in CFG["own"]["ids"] if i in mcs(res, lang)]
    C.banner(f"c′：我自己念的 {len(own)} 題（真人）對同樣 {len(own)} 題的合成聲音")
    if not own or row(res, lang, "co", own) is None:
        print(f"  {C.GREY}彩排裡沒有：先 ./present.sh record own，再 ./present.sh rehearse{C.RESET}")
        return
    print_row(res, lang, "c1", "c", own)
    print_row(res, lang, "co", "c′", own)
    print_row(res, lang, "do", "d′", own)
    print_pair(res, lang, "c1", "co", "TTS − my own voice", own)
    C.note(f"只有 {len(own)} 題：這一列是讓大家聽到差別在哪裡，不是一個估計值。")


def show_noise(res: dict, lang: str) -> None:
    C.banner(f"噪音那一欄：同樣的題目加 babble 到 {CFG['noise']['snr_db']:g} dB（沒有前端）")
    for key, name in (("c1", "c"), ("cn", "c+n"), ("d1", "d"), ("dn", "d+n")):
        print_row(res, lang, key, name)
    print_pair(res, lang, "c1", "cn", "noise costs the model that hears")
    print_pair(res, lang, "d1", "dn", "noise costs the cascade")
    print_pair(res, lang, "cn", "dn", "under noise: end-to-end − cascade")


def show_audit(res: dict, lang: str) -> None:
    C.banner("聽對了，卻答錯？（θ 自己轉錄每一題，再看它聽同一段語音時答對沒有）")
    a = audit(res, lang)
    if a is None:
        print(f"  {C.GREY}彩排裡沒有這一項{C.RESET}")
        return
    print(f"  θ 的轉錄對題目文字：{a['exact']} / {a['n']} 題一字不差；錯誤率 ≤ {CFG['audit']['tol']:g} 的（算「聽對」）{C.BOLD}{a['heard_right']} / {a['n']}{C.RESET}；平均錯誤率 {a['mean_er']:.3f}")
    print(f"  c 那一列答錯 {a['wrong_c']} 題；其中 {C.BOLD}{len(a['right_but_wrong'])} 題是「聽對了、卻答錯」{C.RESET}：{' '.join(a['right_but_wrong']) or '—'}")
    for i in a["right_but_wrong"][:2]:
        print(f"\n  {i}  {BY_ID[i]['prompt'][lang]}")
        print(f"    θ 轉錄：{clip(res['heard'][f'{lang}/w1'][i], 110)}")
        print(f"    θ 聽了之後回答：{clip(recs(res, lang, 'c1')[i]['reply'], 110)}")
        if i in recs(res, lang, "b"):
            print(f"    θ 讀了之後回答：{clip(recs(res, lang, 'b')[i]['reply'], 110)}")
    C.note("轉錄的錯誤率把「17」與「十七」這類寫法的差別也算成錯，所以「聽對」的題數是低估。")


def show_kinds(res: dict, lang: str) -> None:
    C.banner("分三類看：常識／算術／兩步推理")
    print(f"  {'':<4}" + "".join(f"{k:>14}" for k in KINDS))
    for key, name in (("a", "a"), ("b", "b"), ("c1", "c"), ("d1", "d")):
        r = row(res, lang, key)
        if r:
            print(f"  {C.BOLD}{name:<4}{C.RESET}" + "".join(f"{r['by_kind'][k][0]:>10} / {r['by_kind'][k][1]}" for k in KINDS))


def show_voice2(res: dict, lang: str) -> None:
    C.banner("換一個合成的聲音（同一個模型、同一批題目）：這是噪音的底")
    for key, name in (("c1", "c"), ("c2", "c v2"), ("d1", "d"), ("d2", "d v2")):
        print_row(res, lang, key, name)
    print_pair(res, lang, "c1", "c2", "voice 1 − voice 2（θ 聽）")
    C.note("兩個聲音之間差幾題，別的差距就要比它大才值得講。")


def clip(s: str, n: int) -> str:
    s = " ".join((s or "").split())
    return s if len(s) <= n else s[: n - 1] + "…"


def show_question(res: dict, lang: str, qid: str, live: dict | None = None) -> None:
    q = BY_ID[qid]
    C.banner(f"{qid}（{q['kind']}）")
    print(f"  {q['prompt'][lang]}\n")
    for key, name in (("a", "a"), ("b", "b"), ("c1", "c"), ("d1", "d")):
        rec = (live or {}).get(key) or recs(res, lang, key).get(qid)
        if not rec:
            continue
        if q["kind"] == "format":
            mark = (C.GREEN + "format kept" if rec["fmt"] else C.RED + "format broken") + C.RESET
        else:
            lab = S.labels(lang)[rec["pick"]] if rec.get("pick") is not None else "—"
            mark = (C.GREEN + "✓ " if rec["ok"] else C.RED + "✗ ") + lab + C.RESET
        print(f"  {C.BOLD}{name}{C.RESET}  {COND[key]:<40} {mark}  {C.GREY}{rec['n_tok']} tok · {rec['t']:.1f} s{C.RESET}")
        if "heard" in rec:
            print(f"       {C.GREY}Whisper 寫下的：{clip(rec['heard'], 120)}{C.RESET}")
        print(f"       {clip(rec['reply'], 200)}")


def show_probe(res: dict, lang: str, which: str, live: dict | None = None) -> None:
    pr = (live or res.get("probe", {}).get(lang, {}))
    if which in ("silence", "white"):
        d = pr.get(which)
        C.banner(f"探針：{CFG['probe']['seconds']:g} 秒的{'靜音' if which == 'silence' else '白噪音'}")
        if not d:
            print(f"  {C.GREY}彩排裡沒有{C.RESET}")
            return
        print(f"  只有這段聲音，什麼都不問      → {clip(d['alone'], 220)}")
        print(f"  加一句打字的「{CFG['probe']['question'][lang]}」 → {clip(d['asked'], 220)}")
        return
    tone = pr.get("tone")
    C.banner(f"探針：同一句「{CFG['probe']['tone_sentence'][lang]}」，兩種語氣（我自己念的）")
    if not tone:
        print(f"  {C.GREY}沒有錄音：先 ./present.sh record tone，再 ./present.sh rehearse{C.RESET}")
        return
    names = {"happy": "高興", "flat": "沮喪"}
    if which.startswith("tone_reply"):
        for m, d in tone.items():
            if which.endswith(":" + m) or ":" not in which:
                print(f"  {names[m]}　θ 轉錄：{clip(d.get('heard', ''), 60)}")
                print(f"        只有這句話          → {clip(d['alone'], 200)}")
                print(f"        ＋「{CFG['probe']['tone_followup'][lang]}」 → {clip(d['followup'], 200)}\n")
    else:
        for m, d in tone.items():
            print(f"  {names[m]}　＋「{CFG['probe']['tone_direct'][lang]}」 → {clip(d['direct'], 220)}\n")


MENU = """
  q  現場問一題（四種問法）   n  下一題        v  把上一個回答念出來（Talker）
  a b c d  一列一列看          t  整張表        f  格式題        m  c′：我自己念的
  i  三個落差與恆等式          k  分三類        2  換一個聲音    x  e 列：辨識後的字交給 θ
  w  聽對了卻答錯              z  加噪音        s  靜音／白噪音  e  語氣（按三次）
  l  換語言                    ?  這張選單      Q  離開
"""


class Session:
    def __init__(self, res: dict, runner: Runner | None, mode: str):
        self.res, self.runner, self.mode = res, runner, mode
        self.lang = CFG["run"]["main"] if CFG["run"]["main"] in res["meta"]["langs"] else res["meta"]["langs"][0]
        self.qi = -1
        self.s_step = self.e_step = 0
        self.last_reply = ""
        self.live_dir = RUNS / ("fake" if res["meta"].get("fake") else "live")

    def live_ok(self) -> bool:
        return self.runner is not None and self.runner.theta is not None

    def question(self) -> None:
        ids = CFG["live"]["ids"]
        self.qi = (self.qi + 1) % len(ids)
        qid = ids[self.qi]
        live = None
        if self.live_ok():
            live = {}
            p = tts_path(self.lang, 0, qid, self.runner.tts_root)
            for key in ("a", "b", "c1", "d1"):
                if key == "c1" and p.exists() and not self.runner.fake:
                    C.play(p)
                try:
                    live[key] = self.runner.one(self.lang, key, qid)
                except Exception as e:
                    print(f"{C.RED}{key} 現場這一格失敗（{e}）；改印彩排的紀錄{C.RESET}")
            self.last_reply = (live.get("c1") or {}).get("reply", "")
        show_question(self.res, self.lang, qid, live)
        if not self.live_ok():
            self.last_reply = recs(self.res, self.lang, "c1").get(qid, {}).get("reply", "")
            C.note("（彩排的紀錄）")

    def speak(self) -> None:
        if not self.live_ok() or not self.last_reply:
            C.note("沒有可以念的回答，或這個模式不載模型。")
            return
        out = self.live_dir / "spoken_reply.wav"
        r = self.runner.theta.speak(S.clean(self.last_reply)[:400], CFG["theta"]["talker_voice"][self.lang], out)
        print(f"  Talker：{r['audio_s']} 秒的語音，花了 {r['t']} 秒（整段一起出，這個移植版不串流）。它拿到的只有上面那段文字。")
        C.play(out)

    def silence(self) -> None:
        which = ("silence", "white")[self.s_step % 2]
        self.s_step += 1
        live = None
        if self.live_ok() and not self.runner.fake:
            pr = CFG["probe"]
            n = int(float(pr["seconds"]) * FS)
            x = np.zeros(n, dtype=np.float32) if which == "silence" else (float(pr["noise_rms"]) * np.random.default_rng(int(CFG["run"]["seed"])).standard_normal(n)).astype(np.float32)
            p = self.live_dir / f"{which}.wav"
            C.save_wav(p, x.astype(np.float64), FS)
            C.play(p)
            live = {which: {"alone": self.runner.theta.speech(x)["reply"], "asked": self.runner.theta.speech(x, pr["question"][self.lang])["reply"]}}
        show_probe(self.res, self.lang, which, live)

    def tone(self) -> None:
        step = self.e_step % 3
        self.e_step += 1
        mood = ("happy", "flat")[step] if step < 2 else None
        if mood:
            p = own_path(self.lang, f"tone_{mood}")
            if p.exists() and self.mode != "fake":
                C.play(p)
            show_probe(self.res, self.lang, f"tone_reply:{mood}")
        else:
            show_probe(self.res, self.lang, "tone_direct")

    def key(self, k: str) -> bool:
        res, lang = self.res, self.lang
        if k in ("Q", "\x1b", "\x03", "\x04"):
            return False
        if k == "q":
            self.question()
        elif k == "n":
            self.question()
        elif k == "v":
            self.speak()
        elif k in ("a", "b"):
            print_row(res, lang, k)
        elif k == "c":
            print_row(res, lang, "c1", "c")
        elif k == "d":
            print_row(res, lang, "d1", "d")
        elif k == "x":
            C.banner("e 列：同一段語音，先由 Whisper 寫成字，再交給 θ 讀")
            for key, name in (("b", "b"), ("e1", "e"), ("c1", "c")):
                print_row(res, lang, key, name)
        elif k == "t":
            show_table(res, lang)
        elif k == "f":
            show_format(res, lang)
        elif k == "m":
            show_own(res, lang)
        elif k == "i":
            show_identity(res, lang)
        elif k == "k":
            show_kinds(res, lang)
        elif k == "2":
            show_voice2(res, lang)
        elif k == "w":
            show_audit(res, lang)
        elif k == "z":
            show_noise(res, lang)
        elif k == "s":
            self.silence()
        elif k == "e":
            self.tone()
        elif k == "l":
            langs = res["meta"]["langs"]
            self.lang = langs[(langs.index(lang) + 1) % len(langs)]
            print(f"  → {LANG_NAME[self.lang]}")
        elif k == "?":
            print(MENU)
        return True

    def loop(self) -> None:
        m = self.res["meta"]
        tag = f"{C.RED}FAKE — 數字沒有意義{C.RESET}  " if m.get("fake") else ""
        C.banner(f"W11 Demo 2 · typed vs. spoken · {self.mode}")
        print(f"  {tag}θ = {CFG['theta']['label']}   θ0 = {CFG['theta0']['label']} (approx.)   彩排：{m.get('finished') or m.get('started')}   語言：{LANG_NAME[self.lang]}")
        print(MENU)
        while True:
            k = C.getkey()
            if k in ("\n", "\r", " "):
                continue
            if not self.key(k):
                break


# ── 子命令 ───────────────────────────────────────────────────────
def load_res(folder: Path) -> dict:
    p = folder / "rehearsal.json"
    if not p.exists():
        sys.exit(f"沒有 {p.relative_to(HERE)}：先跑 ./present.sh rehearse")
    return json.loads(p.read_text(encoding="utf-8"))


def save_res(folder: Path, res: dict) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    tmp = folder / "rehearsal.json.tmp"
    tmp.write_text(json.dumps(res, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    os.replace(tmp, folder / "rehearsal.json")


def cmd_rehearse(args, fake: bool = False, folder: Path | None = None, quiet: bool = False) -> dict:
    folder = folder or (RUNS / "fake" if fake else RUNS / ("preview" if args.n else "rehearsal"))
    langs = args.langs.split(",") if getattr(args, "langs", None) else list(CFG["run"]["langs"])
    ids = (MC_IDS[:: max(1, len(MC_IDS) // args.n)][: args.n] + FMT_IDS[:3]) if getattr(args, "n", 0) else MC_IDS + FMT_IDS
    tts_root = (folder / "_tts") if fake else None
    res = {"meta": {"started": C.now(), "digest": digest(), "langs": langs, "fake": fake, "ids": ids}, "cond": {}, "heard": {}, "probe": {}}
    p = folder / "rehearsal.json"
    if p.exists() and not getattr(args, "fresh", False):
        old = json.loads(p.read_text(encoding="utf-8"))
        if old["meta"].get("digest") == digest() and old["meta"].get("ids") == ids and old["meta"].get("fake") == fake:
            res = old
            res["meta"]["langs"] = sorted(set(old["meta"]["langs"]) | set(langs), key=["zh", "en"].index)
            if not quiet:
                C.note(f"接續 {old['meta']['started']} 的彩排（設定與題目沒有變）。要從頭來：--fresh")
    runner = Runner(fake=fake, tts_root=tts_root)
    try:
        res["tts_seconds"] = {**res.get("tts_seconds", {}), **runner.ensure_tts(langs, ids=ids)}
        res["meta"]["versions"] = versions()
        res["meta"]["aux_versions"] = runner.get_aux().versions
        lock = read_lock(LOCK)
        res["meta"]["models"] = {"theta": {"repo": CFG["theta"]["repo"], "sha": lock.get(CFG["theta"]["repo"])},
                                 "theta0": {"repo": CFG["theta0"]["hf_repo"] if CFG["theta0"]["source"] == "convert" else CFG["theta0"]["hub_repo"],
                                            "sha": lock.get(CFG["theta0"]["hf_repo"]), "source": CFG["theta0"]["source"], "bits": CFG["theta0"]["bits"]},
                                 "tts": CFG["tts"]["repo"], "asr": CFG["asr"]["repo"]}
        only = args.only.split(",") if getattr(args, "only", None) else ORDER
        heard_cache: dict = {}
        for lang in langs:
            for h in ("1", "2", "o", "n"):
                for i, t in res["heard"].get(f"{lang}/{h}", {}).items():
                    heard_cache[f"{lang}/{h}/{i}"] = t
            for key in [k for k in ORDER if k in only]:
                use = [i for i in (ids if key not in ("co", "do") else [i for i in ids if i in CFG["own"]["ids"]]) if keep(lang, i)]
                done = res["cond"].setdefault(f"{lang}/{key}", {})
                todo = [i for i in use if i not in done]
                if not todo:
                    continue
                t0 = time.perf_counter()
                for i in todo:
                    rec = runner.one(lang, key, i, heard_cache)
                    if rec is None:
                        break
                    done[i] = rec
                    if "heard" in rec:
                        res["heard"].setdefault(f"{lang}/{key[1:]}", {})[i] = rec["heard"]
                if not done:
                    del res["cond"][f"{lang}/{key}"]
                    if not quiet:
                        C.note(f"  {lang}/{key}：沒有錄音，跳過（./present.sh record own）")
                    continue
                save_res(folder, res)
                if not quiet:
                    r = row(res, lang, key, use if key in ("co", "do") else [i for i in use if i in MC_IDS])
                    print(f"  {lang}/{key:<3} {COND[key]:<46} {r['k'] if r else '—'} / {r['n'] if r else '—'}   {time.perf_counter() - t0:6.1f} s")
            # θ 自己的轉錄（第一個聲音）
            w = res["heard"].setdefault(f"{lang}/w1", {})
            for i in [i for i in ids if i in MC_IDS and i not in w and keep(lang, i)]:
                x = runner.wav(tts_path(lang, 0, i, tts_root))
                w[i] = runner.theta.asr(x, hint=BY_ID[i]["prompt"][lang])["text"] if fake else runner.theta.asr(x)["text"]
            if lang not in res["probe"] or getattr(args, "probes", False):
                res["probe"][lang] = runner.probes(lang)
            save_res(folder, res)
        res["meta"]["finished"] = C.now()
        res["meta"]["peak_gb"] = runner.theta.peak_gb()
        save_res(folder, res)
    finally:
        runner.close()
    if not quiet:
        for lang in langs:
            show_table(res, lang)
        if folder == RUNS / "rehearsal":
            cmd_export(args, res)
            print(f"\n→ {folder.relative_to(HERE)}/rehearsal.json；w11-metrics.json → demo2")
    return res


def cmd_show(args, mode: str = "show") -> None:
    if mode == "fake":
        res = cmd_rehearse(argparse.Namespace(n=0, langs=None, only=None, fresh=False, probes=False), fake=True, quiet=True)
        runner = Runner(fake=True, tts_root=RUNS / "fake" / "_tts")
    else:
        res = load_res(RUNS / "rehearsal")
        runner = None
        if mode == "show":
            try:
                C.note("載入模型（θ0、θ）……")
                runner = Runner()
                C.note(f"θ0 {runner.theta0.load_s:.1f} s，θ {runner.theta.load_s:.1f} s。")
            except Exception as e:
                print(f"{C.RED}模型載不起來（{e}）；改成 replay：表都在，現場的鍵印彩排的紀錄。{C.RESET}")
                runner = None
    try:
        Session(res, runner, mode).loop()
    finally:
        if runner:
            runner.close()


def cmd_tts(args) -> None:
    runner = Runner(need_models=False)
    try:
        info = runner.ensure_tts(list(CFG["run"]["langs"]), force=args.force)
        print(f"念了 {len(info)} 段 → {P(CFG['tts']['dir']).relative_to(DEMOS)}（不進版控）")
    finally:
        runner.close()


def cmd_record(args) -> None:
    import sounddevice as sd
    lang = args.lang or CFG["run"]["main"]
    if args.what == "tone":
        todo = [(f"tone_{m}", CFG["probe"]["tone_sentence"][lang], hint) for m, hint in (("happy", "很高興地說"), ("flat", "很沮喪地說"))]
    else:
        todo = [(i, BY_ID[i]["prompt"][lang], "照平常問問題的樣子念") for i in CFG["own"]["ids"]]
    fs, limit = FS, float(CFG["own"]["max_seconds"])

    def key() -> str:
        # 一次按鍵可能送出好幾個 byte（方向鍵、開著中文輸入法時的一個字）；不清掉的話，剩下的 byte 會被
        # 接下來的「開始」「結束」「下一段」各吃掉一個——2026-11-27 的十段錄音裡有七段是 0.1 秒的空檔。
        import termios
        fd = sys.stdin.fileno()
        termios.tcflush(fd, termios.TCIFLUSH)
        k = C.getkey()
        time.sleep(0.05)
        termios.tcflush(fd, termios.TCIFLUSH)
        return k

    def usable(p: Path) -> bool:
        if not p.exists():
            return False
        x = load16(p)
        return len(x) >= fs and float(np.abs(x).max()) >= 0.01

    for name, text, hint in todo:
        p = own_path(lang, name)
        if usable(p) and not args.force:
            C.note(f"{name} 已經有了（--force 重錄）")
            continue
        if p.exists() and not args.force:
            C.note(f"{name} 的檔是空的或不到一秒，重錄。")
        while True:
            print(f"\n{C.BOLD}{name}{C.RESET}（{hint}）\n  {text}\n  按空白鍵開始錄，念完再按空白鍵。（先切到英文輸入法）")
            key()
            buf = sd.rec(int(limit * fs), samplerate=fs, channels=1, dtype="float32")
            t0 = time.perf_counter()
            print(f"  {C.RED}● 錄音中{C.RESET}")
            key()
            sd.stop()
            n = min(int((time.perf_counter() - t0) * fs), len(buf))
            x = buf[:n, 0].astype(np.float64)
            peak = float(np.abs(x).max()) if n else 0.0
            if n < fs or peak < 0.01:
                print(f"  {C.RED}只有 {n / fs:.1f} 秒、峰值 {peak:.2f}——沒有錄到，再來一次。{C.RESET}")
                continue
            C.save_wav(p, x, fs)
            print(f"  {n / fs:.1f} 秒，峰值 {peak:.2f}。按 r 重錄，其他鍵下一段。")
            C.play(p)
            if key() != "r":
                break
    print(f"\n→ {P(CFG['own']['dir']).relative_to(DEMOS)}/{lang}/（不進版控、不上站）")


def cmd_fetch(args) -> None:
    from huggingface_hub import HfApi, snapshot_download
    api, lock, t = HfApi(), read_lock(LOCK), CFG["theta"]["repo"]
    sha = lock.get(t) or api.model_info(t).sha
    C.note(f"θ：{t} @ {sha[:8]}（約 12.7 GB）……")
    snapshot_download(t, revision=sha)
    lock[t] = sha
    c0 = CFG["theta0"]
    if c0["source"] == "convert":
        src, out = c0["hf_repo"], P(c0["dir"])
        sha0 = lock.get(src) or api.model_info(src).sha
        lock[src] = sha0
        mark = out / "conversion.json"
        want = {"source_repo": src, "source_revision": sha0, "bits": c0["bits"], "group_size": c0["group_size"]}
        if mark.exists() and {k: json.loads(mark.read_text())[k] for k in want} == want:
            C.note(f"θ0：{out.name} 已經轉好了。")
        else:
            C.note(f"θ0：下載 {src} @ {sha0[:8]}（約 15.8 GB），再轉成 {c0['bits']}-bit……")
            snap = Path(snapshot_download(src, revision=sha0))
            stage = out.parent / "_theta0_src"
            stage.mkdir(parents=True, exist_ok=True)
            for f in snap.iterdir():
                link = stage / f.name
                if f.name != "config.json" and not link.exists():
                    link.symlink_to(f.resolve())
            cfg = json.loads((snap / "config.json").read_text())
            # 官方 config 裡這一欄寫成整數（-5），新版 transformers 的設定檢查要浮點數；值不變
            if isinstance(cfg.get("kda_lower_bound"), int):
                cfg["kda_lower_bound"] = float(cfg["kda_lower_bound"])
            (stage / "config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")
            if out.exists():
                sys.exit(f"{out} 已經存在但不是這個設定轉的：把它移走（或刪掉）再跑一次 fetch")
            subprocess.run([sys.executable, "-m", "mlx_lm", "convert", "--hf-path", str(stage), "--mlx-path", str(out), "-q",
                            "--q-bits", str(c0["bits"]), "--q-group-size", str(c0["group_size"])], check=True,
                           env={**os.environ, "HF_HUB_OFFLINE": "1"})
            from importlib import metadata
            mark.write_text(json.dumps({**want, "mlx_lm": metadata.version("mlx-lm"), "date": C.now()}, indent=1), encoding="utf-8")
    else:
        sha0 = lock.get(c0["hub_repo"]) or api.model_info(c0["hub_repo"]).sha
        snapshot_download(c0["hub_repo"], revision=sha0)
        lock[c0["hub_repo"]] = sha0
    write_lock(lock)
    aux = Aux()
    try:
        got = aux.call(cmd="fetch", repos={"kokoro": CFG["tts"]["repo"], "whisper": CFG["asr"]["repo"], "qwen3": CFG["tts"]["qwen3"]["repo"]},
                       voices=sorted({v["voice"] for L in ("zh", "en") for v in CFG["tts"].get(L, [])}),
                       lock={**read_lock(P(CFG["tts"]["lock_from"])), **read_lock(P(CFG["asr"]["lock_from"]))})["fetched"]
    finally:
        aux.close()
    for k, v in got.items():
        print(f"  {k}: {v['repo']} @ {v['sha'][:8]}  {v['size_mb']} MB  license {v['license']}")
        for name, ok in (v.get("voices") or {}).items():
            print(f"    voice {name}: {'在快取裡' if ok else C.RED + '沒抓到' + C.RESET}")
    print(f"→ model.lock（θ {sha[:8]}；θ0 的來源 {lock.get(c0['hf_repo'], '')[:8]}）。下一步：./present.sh check")


def cmd_check(args) -> None:
    out = {"date": C.now(), "versions": versions(), "steps": {}}

    def step(name: str, fn):
        t0 = time.perf_counter()
        try:
            r = fn()
            r = dict(r or {})
            if "ok" in r:                       # judge() 的 ok 是「答對了沒有」，不是這一步跑不跑得起來
                r["answer_ok"] = r.pop("ok")
            out["steps"][name] = {"ok": True, "t": round(time.perf_counter() - t0, 2), **r}
            print(f"  {C.GREEN}✓{C.RESET} {name:<34} {time.perf_counter() - t0:6.1f} s  {clip(json.dumps(r, ensure_ascii=False, default=float), 150) if r else ''}")
        except Exception as e:
            out["steps"][name] = {"ok": False, "error": repr(e)}
            print(f"  {C.RED}✗{C.RESET} {name:<34} {e!r}")

    lang, runner = CFG["run"]["main"], Runner(need_models=False)
    q = BY_ID[CFG["live"]["ids"][0]]
    box: dict = {}
    try:
        step("aux worker (../.venv)", lambda: {"versions": runner.get_aux().versions})
        for L in CFG["run"]["langs"]:
            step(f"Kokoro, {L}", lambda L=L: {"audio_s": list(runner.ensure_tts([L], ids=[q["id"]], force=True).values())})
            step(f"Whisper small, {L}", lambda L=L: (lambda t: {"heard": t, "error_rate": round(S.error_rate(q["prompt"][L], t, L), 3)})(
                runner.get_aux().asr(tts_path(L, 0, q["id"]), L)["text"]))
        step("θ0 load", lambda: box.update(t0=Theta0()) or {"load_s": round(box["t0"].load_s, 1)})
        for L in CFG["run"]["langs"]:
            step(f"θ0 reads, {L}", lambda L=L: (lambda r: {**r, **S.judge(r["reply"], q, L)})(box["t0"].text(q["prompt"][L])))
        step("θ load", lambda: box.update(t=Theta()) or {"load_s": round(box["t"].load_s, 1), "peak_gb": box["t"].peak_gb()})
        for L in CFG["run"]["langs"]:
            step(f"θ reads, {L}", lambda L=L: (lambda r: {**r, **S.judge(r["reply"], q, L)})(box["t"].text(q["prompt"][L])))
            step(f"θ hears, {L}", lambda L=L: (lambda r: {**r, **S.judge(r["reply"], q, L)})(box["t"].speech(load16(tts_path(L, 0, q["id"])))))
            step(f"θ transcribes, {L}", lambda L=L: (lambda r: {**r, "error_rate": round(S.error_rate(q["prompt"][L], r["text"], L), 3)})(
                box["t"].asr(load16(tts_path(L, 0, q["id"])))))
        step("θ hears silence + typed question", lambda: box["t"].speech(np.zeros(FS * 2, dtype=np.float32), CFG["probe"]["question"][lang]))
        step("Talker speaks", lambda: box["t"].speak(CFG["probe"]["tone_sentence"][lang], CFG["theta"]["talker_voice"][lang], RUNS / "live" / "check_talker.wav"))
        step("peak memory", lambda: {"peak_gb": box["t"].peak_gb()})
    finally:
        runner.close()
    RUNS.mkdir(exist_ok=True)
    (RUNS / "check.json").write_text(json.dumps(out, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    bad = [k for k, v in out["steps"].items() if not v["ok"]]
    print(f"\n→ runs/check.json；{'全部通過' if not bad else '沒過的：' + '、'.join(bad)}")
    if (RUNS / "live" / "check_talker.wav").exists():
        C.note("Talker 念的那一句在 runs/live/check_talker.wav，聽一下。")


def cmd_export(args, res: dict | None = None) -> None:
    res = res or load_res(RUNS / "rehearsal")
    data = json.loads(METRICS.read_text(encoding="utf-8")) if METRICS.exists() else {}
    data["demo2"] = summary(res)
    METRICS.parent.mkdir(parents=True, exist_ok=True)
    METRICS.write_text(json.dumps(data, ensure_ascii=False, indent=1, default=float) + "\n", encoding="utf-8")
    if args is not None and getattr(args, "cmd", "") == "export":
        print(f"→ {METRICS.relative_to(DEMOS.parent)} → demo2")


def cmd_selftest(args) -> None:
    n_ok = n_bad = 0

    def t(name: str, cond: bool, extra: str = "") -> None:
        nonlocal n_ok, n_bad
        n_ok, n_bad = n_ok + bool(cond), n_bad + (not cond)
        print(f"  {(C.GREEN + '✓') if cond else (C.RED + '✗')}{C.RESET} {name} {C.GREY}{extra}{C.RESET}")

    C.banner("題目")
    t("40 題四選一、10 題格式", len(MC_IDS) == 40 and len(FMT_IDS) == 10)
    t("常識 14、算術 13、推理 13", [sum(BY_ID[i]["kind"] == k for i in MC_IDS) for k in KINDS] == [14, 13, 13])
    t("正解的位置各 10 題", [sum(BY_ID[i]["answer"] == a for i in MC_IDS) for a in range(4)] == [10] * 4)
    t("題號不重複", len({q["id"] for q in ITEMS}) == len(ITEMS))
    t("每題四個選項、兩種語言", all(len(BY_ID[i]["options"][L]) == 4 and len(set(BY_ID[i]["options"][L])) == 4 for i in MC_IDS for L in ("zh", "en")))
    t("華語的選項內容裡沒有單獨的 A–D 字母", all(not S._ZH_LETTER.search(o) for i in MC_IDS for o in BY_ID[i]["options"]["zh"]))
    t("skip 的題號都在題庫裡、現場的題目沒有被 skip", all(i in BY_ID for v in SKIP.values() for i in v)
      and not any(i in v for v in SKIP.values() for i in CFG["live"]["ids"]))
    t("自己念的 10 題與現場的題目都在題庫裡", all(i in BY_ID for i in CFG["own"]["ids"] + CFG["live"]["ids"]) and len(CFG["own"]["ids"]) == 10)
    C.banner("抽選項：每一題，五種寫法的回答")
    bad = []
    for i in MC_IDS:
        q = BY_ID[i]
        for L in ("zh", "en"):
            lab, opt = S.labels(L)[q["answer"]], q["options"][L][q["answer"]]
            forms = [lab, opt, f"{lab}. {opt}" if L == "en" else f"{lab}，{opt}。", f"The answer is {lab}." if L == "en" else f"答案是{lab}。",
                     (f"Let me think. A and B are close.\nSo the answer is {lab}: {opt}." if L == "en" else f"先比較 A 和 B。\n所以答案是{lab}，{opt}。")]
            for f in forms:
                if S.pick(f, L, q["options"][L])[0] != q["answer"]:
                    bad.append((i, L, f))
    t(f"{len(MC_IDS) * 2 * 5} 個回答都抽到正解", not bad, str(bad[:3]) if bad else "")
    wrong = sum(S.pick(BY_ID[i]["options"][L][(BY_ID[i]["answer"] + 1) % 4], L, BY_ID[i]["options"][L])[0] == BY_ID[i]["answer"] for i in MC_IDS for L in ("zh", "en"))
    t("答成隔壁選項的內容，不會被算成對", wrong == 0, f"{wrong}")
    t("簡體的回答對得上繁體的選項", S.pick("鳄鱼不是，海豚才是哺乳类。所以是海豚。", "zh", BY_ID["k06"]["options"]["zh"])[0] == 3)
    t("抄了題目結尾的回答：那一串代號不算答案", S.pick("題目是：哪一顆行星離太陽最近？請問答案是 A、B、C、D 的哪一個？\n水星最近。", "zh", BY_ID["k01"]["options"]["zh"])[0] == 1)
    t("12 不算在 120 裡", S.pick("120", "en", ["12", "20", "24", "18"])[0] is None)
    t("空的回答與拒答：沒有選", S.pick("", "en", BY_ID["k01"]["options"]["en"])[0] is None and S.pick("I could not hear the question.", "en", BY_ID["k01"]["options"]["en"])[0] is None)
    t("推理區塊不算在回答裡", S.pick("<think>maybe Venus, option A</think>Mercury", "en", BY_ID["k01"]["options"]["en"])[0] == 1)
    C.banner("格式")
    cases = [("f01", "en", "C", True), ("f01", "en", "The answer is C.", False), ("f01", "zh", "C", True), ("f01", "zh", "答案是 C", False), ("f01", "zh", "丙", True),
             ("f03", "en", "13", True), ("f03", "en", "8 plus 5 is 13", False), ("f03", "zh", "十三", True), ("f05", "en", "Yes.", True),
             ("f05", "en", "Yes, it is.", False), ("f06", "zh", "不是。", True), ("f06", "zh", "不是，冰比較冷", False), ("f07", "en", "Green", True),
             ("f07", "en", "You get the colour green", False), ("f07", "zh", "綠色", True), ("f08", "zh", "冷", True), ("f08", "zh", "冷的", False),
             ("f08", "en", "Cold", True), ("f09", "en", "Monday, Tuesday, Wednesday", True), ("f09", "en", "Sure! Monday, Tuesday, Wednesday", False),
             ("f09", "zh", "星期一、星期二、星期三", True), ("f09", "zh", "星期一、星期二", False), ("f10", "en", "Answer: Paris", True),
             ("f10", "en", "Paris", False), ("f10", "zh", "答案：巴黎", True), ("f10", "zh", "巴黎", False)]
    bad = [c for c in cases if S.format_kept(c[2], BY_ID[c[0]]["fmt"], c[1]) != c[3]]
    t(f"{len(cases)} 個例子", not bad, str(bad[:3]) if bad else "")
    t("格式與內容分開記", S.judge("Yes, it is.", BY_ID["f05"], "en") == {"fmt": False, "ok": True} and S.judge("No", BY_ID["f05"], "en") == {"fmt": True, "ok": False})
    C.banner("統計")
    ci = S.bootstrap_ci([True] * 30 + [False] * 10)
    t("30 / 40 的區間包含 30、寬度在 8 到 14 題之間", ci["lo"] <= 30 <= ci["hi"] and 8 <= ci["hi"] - ci["lo"] <= 14, f"[{ci['lo']:.0f}, {ci['hi']:.0f}]")
    p3 = S.paired([True] * 30 + [False] * 10, [True] * 27 + [False] * 13)
    t("差 3 題（3 對 0）：符號檢定 p = 0.25，區間碰到 0", p3["diff"] == 3 and abs(p3["p_sign"] - 0.25) < 1e-12 and p3["lo"] <= 0, f"[{p3['lo']:.0f}, {p3['hi']:.0f}]")
    t("同一份答案：差 0、區間 [0, 0]", S.paired([True, False] * 20, [True, False] * 20)["hi"] == 0)
    t("錯誤率：繁簡、標點、空白不算錯", S.error_rate("17 加 26 等於多少？", "17加26等于多少", "zh") == 0 and S.error_rate("What is 9 times 7?", "what is 9 times 7", "en") == 0)
    t("中文數字", [S.zh_number(n) for n in (6, 13, 20, 43, 100, 105, 366)] == ["六", "十三", "二十", "四十三", "一百", "一百零五", "三百六十六"])
    rng = np.random.default_rng(1)
    s = rng.standard_normal(16000).astype(np.float32)
    y = mix(s, babble(16000, [rng.standard_normal(9000), rng.standard_normal(30000)], rng), 5.0)
    snr = 10 * np.log10(power(s) / power(y - s))
    t("加噪音：SNR 是 5 dB、語音沒被動到", abs(snr - 5.0) < 1e-3, f"{snr:.4f}")
    C.banner("fake 整條流程")
    folder = RUNS / "_selftest"
    if folder.exists():
        import shutil
        shutil.rmtree(folder, ignore_errors=True)
    ns = argparse.Namespace(n=0, langs=None, only=None, fresh=True, probes=False)
    res = cmd_rehearse(ns, fake=True, folder=folder, quiet=True)
    t("兩種語言、每個條件都有 50 題減掉 skip 的（自己念的那兩列除外）",
      all(len(recs(res, L, k)) == 50 - len(SKIP.get(L, ())) for L in ("zh", "en") for k in ORDER if k not in ("co", "do")))
    t("skip 的題目哪一個條件都沒有跑", not any(i in recs(res, L, k) for L, v in SKIP.items() for i in v for k in ORDER))
    sm = summary(res)
    t("摘要：a、b、c、d 與格式題都有數字", all(k in sm["lang"]["zh"]["rows"] and k in sm["lang"]["zh"]["format"] for k in ("a", "b", "c1", "d1")))
    t("恆等式：a − c = (a − b) + (b − c)", all(sm["lang"][L]["pairs"]["a-c1"]["diff"] == sm["lang"][L]["pairs"]["a-b"]["diff"] + sm["lang"][L]["pairs"]["b-c1"]["diff"] for L in ("zh", "en")))
    t("每個回答都抽得到選項（fake 的四種寫法）", all(recs(res, L, "a")[i].get("pick") is not None for L in ("zh", "en") for i in MC_IDS if keep(L, i)))
    blob = json.dumps(res, ensure_ascii=False)
    t("紀錄裡沒有絕對路徑", str(Path.home()) not in blob and "/Users/" not in blob and str(DEMOS) not in blob)
    res2 = cmd_rehearse(argparse.Namespace(n=0, langs=None, only=None, fresh=False, probes=False), fake=True, folder=folder, quiet=True)
    t("接續：已經跑過的不重跑、結果相同", res2["cond"] == res["cond"])
    sess = Session(res, None, "replay")
    import contextlib
    import io
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        for k in "qnabcdtfmik2xwzsseeel?":
            sess.key(k)
    t("replay 的每一個鍵都印得出東西", len(buf.getvalue()) > 3000 and "Traceback" not in buf.getvalue(), f"{len(buf.getvalue())} 字元")
    import shutil
    shutil.rmtree(folder, ignore_errors=True)
    print(f"\n{n_ok} 項通過，{n_bad} 項沒過")
    sys.exit(1 if n_bad else 0)


def main() -> None:
    ap = argparse.ArgumentParser(description="W11 demo 2: the same question, typed and spoken")
    sub = ap.add_subparsers(dest="cmd")
    for name in ("show", "replay", "fake", "selftest", "export", "fetch", "check"):
        sub.add_parser(name)
    r = sub.add_parser("rehearse")
    r.add_argument("--n", type=int, default=0, help="只跑 n 題四選一（預覽 → runs/preview/）")
    r.add_argument("--langs", default=None, help="zh,en 或其中一個")
    r.add_argument("--only", default=None, help="只跑這些條件，例如 a,b,c1,d1")
    r.add_argument("--fresh", action="store_true", help="不接續，從頭來")
    r.add_argument("--probes", action="store_true", help="探針重跑（錄了語氣那兩段之後）")
    tt = sub.add_parser("tts")
    tt.add_argument("--force", action="store_true")
    rc = sub.add_parser("record")
    rc.add_argument("what", choices=["own", "tone"])
    rc.add_argument("--lang", default=None)
    rc.add_argument("--force", action="store_true")
    args = ap.parse_args()
    cmd = args.cmd or "show"
    if cmd in ("show", "replay", "fake"):
        cmd_show(args, cmd)
    else:
        {"rehearse": cmd_rehearse, "tts": cmd_tts, "record": cmd_record, "fetch": cmd_fetch, "check": cmd_check,
         "export": cmd_export, "selftest": cmd_selftest}[cmd](args)


if __name__ == "__main__":
    main()

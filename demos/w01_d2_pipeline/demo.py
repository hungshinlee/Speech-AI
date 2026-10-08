#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# 此檔由 bin/publish_demos.py 從 private repo 的 <course>/demos/ 複製而來，請勿在公開 repo 直接編輯。
"""W1 demo 2（投影片 p41「Live: measure your own pipeline」）：同一句話跑 cascade，把延遲預算當場填出來。

子命令（用 speech_ai/demos/setup.sh 建好的共用 .venv 執行；通常經由 present.sh）：

    devices    列出音訊裝置（接投影機之後跑一次，預設裝置常會換）
    record     錄那句話（中間要停頓約 700 ms）→ audio/utterance.wav
    endpoint   只做 endpoint 那一格：對錄音跑串流式能量 VAD，列出各門檻在句中／句尾的行為（Misconception 3 的實證）
    check      課前體檢：套件、模型能不能載入、各跑一次暖機
    rehearse   課前彩排：完整跑 N 遍，存 runs/rehearsal/<name>/（進版控；現場失敗時的退路）
    show       課堂用：先讓學生預測誰是瓶頸 → 跑 → 畫時間軸 → 填預算表
    replay     現場失敗時：把彩排的結果畫出來（畫面標明非現場）
    moshi      啟動 moshi-mlx 的即時全雙工（同一句話對它講一次；沒有數字，只給教室聽）

**每一格量的是使用者停止說話之後的殘餘時間**（投影片 p32 的規則一）：

    endpoint      = 靜音門檻 + 偵測步距        （演算法延遲：門檻是設定，硬體救不了）
    ASR tail      = endpoint 開火 → 文字出來    （非串流 ASR 整句都算在這裡——這本身是教學點）
    LLM           = prompt 進去 → 第一個 token  （prefill + first token）
    TTS first pkt = 第一句文字進去 → 第一段音訊 （不是全部合成完）
    playout       = 設定值，不是量測           （投影片 p36 的 D_buf）

end-to-end 那一半**沒有現場量**：moshi-mlx 的即時模式吃麥克風、不吐時間戳，要量得先錄成雙軌再跑
scripts/analyze_duplex_audio.py。所以時間軸上的 e2e 參考點用 cold open 的 Recording 2（60 ms，
assets/w01/w01-metrics.json），畫面上標明「pre-recorded, different utterance」。

後端（demo_config.toml）：asr = whispercpp | mlx_whisper | fake；llm = mlx_lm | fake；tts = mlx_audio | fake。
真後端只在 Apple silicon 上能跑，而且**寫這支程式的環境沒有 Mac**：mlx_lm 的用法照 nlp_llm/demos 的實測版本，
mlx_audio 與 pywhispercpp 的 API 照各自 README，**第一次彩排要對**（README 有清單）。
fake 後端用合成訊號與 sleep 演練整個流程，任何機器都能跑。
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import tomllib
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import common  # noqa: E402
import dsp  # noqa: E402
from common import AMBER, BLUE, BOLD, DIM, GREEN, GREY, RED, RESET, banner, note  # noqa: E402

CONFIG = HERE / "demo_config.toml"
RUNS = HERE / "runs"
REHEARSAL = RUNS / "rehearsal"
LIVE = RUNS / "live"
METRICS_W01 = HERE.parents[1] / "slides" / "assets" / "w01" / "w01-metrics.json"


def load_config() -> dict:
    with open(CONFIG, "rb") as f:
        return tomllib.load(f)


# ── 能量式 VAD（串流版）─────────────────────────────────────────────
def frame_db(x: np.ndarray, fs: int, win_ms: float, hop_ms: float):
    n, hop = int(round(fs * win_ms / 1000)), int(round(fs * hop_ms / 1000))
    if len(x) < n:
        return np.array([]), hop
    m = 1 + (len(x) - n) // hop
    idx = np.arange(n)[None, :] + hop * np.arange(m)[:, None]
    rms = np.sqrt(np.mean(x[idx] ** 2, axis=1))
    return 20 * np.log10(rms + 1e-10), hop


def speech_mask(db: np.ndarray, cfg: dict) -> np.ndarray:
    """門檻取法與 scripts/analyze_duplex_audio.py 相同：地板往上 30% 的動態範圍、至少 +8 dB。"""
    floor, peak = np.percentile(db, 10), np.percentile(db, 98)
    thr = max(floor + cfg["dynamic_fraction"] * (peak - floor), floor + cfg["min_above_floor_db"])
    return db > thr


def segments(mask: np.ndarray, hop_s: float, min_len_s: float):
    out, start = [], None
    for i, v in enumerate(list(mask) + [False]):
        if v and start is None:
            start = i
        elif not v and start is not None:
            if (i - start) * hop_s >= min_len_s:
                out.append((start * hop_s, i * hop_s))
            start = None
    return out


def simulate_endpoint(mask: np.ndarray, hop_s: float, threshold_s: float):
    """串流式：一個 frame 一個 frame 走，靜音累積到門檻就開火；每次開火都記下來，開火後重置。
    回傳開火時間清單（秒）。"""
    fires, silence, seen_speech = [], 0.0, False
    for i, v in enumerate(mask):
        if v:
            seen_speech, silence = True, 0.0
        elif seen_speech:
            silence += hop_s
            if silence >= threshold_s:
                fires.append((i + 1) * hop_s)
                silence, seen_speech = 0.0, False
    return fires


def endpoint_report(x: np.ndarray, fs: int, cfg: dict) -> dict:
    v = cfg["vad"]
    db, hop = frame_db(x, fs, v["win_ms"], v["hop_ms"])
    hop_s = hop / fs
    mask = speech_mask(db, v)
    # 錄音尾巴之後補 3 秒「虛擬靜音」：現場的計時器本來就會在使用者說完後繼續數，錄音檔在哪裡結束與它無關。
    # 這樣門檻大於尾巴靜音長度時仍然量得到「句尾回應」，不必為了 1000 ms 的門檻硬錄 1 秒安靜。
    mask = np.concatenate([mask, np.zeros(int(3.0 / hop_s), dtype=bool)])
    segs = segments(mask, hop_s, v["min_segment_ms"] / 1000)
    if not segs:
        raise SystemExit(f"{RED}錄音裡偵測不到語音（麥克風權限？）{RESET}")
    # 計時器只看「語音段」，不看原始 mask：開頭的按鍵聲、呼吸這種 < min_segment 的短脈衝
    # 會讓原始 mask 有孤立的 True，計時器就從那裡開始數，在 0.3 s 就誤判一次（2026-09-20 彩排踩到）。
    mask = np.zeros_like(mask)
    for a, b in segs:
        mask[int(round(a / hop_s)):int(round(b / hop_s))] = True
    true_end = segs[-1][1]
    pauses = [(segs[i][1], segs[i + 1][0]) for i in range(len(segs) - 1)]
    rows = []
    for thr_ms in cfg["endpoint"]["thresholds_ms"]:
        fires = simulate_endpoint(mask, hop_s, thr_ms / 1000)
        false_fires = [f for f in fires if f < true_end]
        final = [f for f in fires if f >= true_end]
        rows.append({"threshold_ms": thr_ms,
                     "false_endpoints_s": [round(f, 3) for f in false_fires],
                     "final_after_end_ms": round((final[0] - true_end) * 1000) if final else None})
    return {"hop_ms": v["hop_ms"], "speech_segments_s": [(round(a, 3), round(b, 3)) for a, b in segs],
            "internal_pauses_ms": [round((b - a) * 1000) for a, b in pauses],
            "true_end_s": round(true_end, 3), "rows": rows}


def print_endpoint(rep: dict) -> None:
    banner("endpoint：串流式能量 VAD 對這句話的行為（量級可信、毫秒不可信）", AMBER)
    print(f"語音段：{rep['speech_segments_s']}  句中停頓：{rep['internal_pauses_ms']} ms  "
          f"真正說完：{rep['true_end_s']} s")
    print(f"\n{BOLD}{'threshold':>10}  {'句中誤判':<22}  {'句尾回應':>10}{RESET}")
    for r in rep["rows"]:
        ff = "、".join(f"{t:.2f}s" for t in r["false_endpoints_s"]) or "—"
        fin = f"+{r['final_after_end_ms']} ms" if r["final_after_end_ms"] is not None else "沒開火"
        col = RED if r["false_endpoints_s"] else GREEN
        print(f"{r['threshold_ms']:>7} ms  {col}{ff:<22}{RESET}  {fin:>10}")
    note("句中誤判＝在使用者說完之前就宣告 endpoint（切斷）；句尾回應＝門檻＋一個 hop（遲鈍的那一半）。")


# ── 後端 ─────────────────────────────────────────────────────────
class FakeASR:
    name = "fake"

    def __init__(self, cfg):
        self.text, self.delay = cfg["fake"]["asr_text"], cfg["fake"]["asr_seconds"]

    def transcribe(self, x, fs) -> str:
        time.sleep(self.delay)
        return self.text


class WhisperCppASR:
    """pywhispercpp（whisper.cpp 的 Python 綁定；§6：Apple silicon 走 Metal）。非串流：tail = 整句轉錄時間。"""
    name = "whispercpp"

    def __init__(self, cfg):
        from pywhispercpp.model import Model
        c = cfg["asr"]
        self.model = Model(c["whispercpp_model"], n_threads=c.get("n_threads", 4), print_progress=False)
        self.lang = c.get("language", "zh")

    def transcribe(self, x, fs) -> str:
        if fs != 16000:
            x = dsp.resample(x, fs, 16000)
        segs = self.model.transcribe(x.astype(np.float32), language=self.lang)
        return "".join(s.text for s in segs).strip()


class MLXWhisperASR:
    name = "mlx_whisper"

    def __init__(self, cfg):
        import mlx_whisper  # noqa: F401
        self.repo, self.lang = cfg["asr"]["mlx_whisper_repo"], cfg["asr"].get("language", "zh")

    def transcribe(self, x, fs) -> str:
        import mlx_whisper
        if fs != 16000:
            x = dsp.resample(x, fs, 16000)
        r = mlx_whisper.transcribe(x.astype(np.float32), path_or_hf_repo=self.repo, language=self.lang)
        return r["text"].strip()


class FakeLLM:
    name = "fake"

    def __init__(self, cfg):
        self.reply, self.ttft, self.tps = cfg["fake"]["llm_reply"], cfg["fake"]["llm_ttft_seconds"], 30.0

    def stream(self, user_text: str):
        time.sleep(self.ttft)
        for ch in self.reply:
            yield ch
            time.sleep(1 / self.tps)


class MLXLLM:
    """mlx-lm；用法照 nlp_llm/demos/w01_d1_live_model/demo.py（M5 Max 上跑過的版本）。"""
    name = "mlx_lm"

    def __init__(self, cfg):
        from mlx_lm import load
        c = cfg["llm"]
        self.model, self.tokenizer = load(c["mlx_lm_repo"])
        self.system, self.max_tokens = c["system_prompt"], c["max_tokens"]

    def stream(self, user_text: str):
        from mlx_lm import stream_generate
        from mlx_lm.sample_utils import make_sampler
        msgs = [{"role": "system", "content": self.system}, {"role": "user", "content": user_text}]
        prompt = self.tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                                    enable_thinking=False)
        for r in stream_generate(self.model, self.tokenizer, prompt, max_tokens=self.max_tokens,
                                 sampler=make_sampler(temp=0.0)):
            yield r.text


class FakeTTS:
    name = "fake"

    def __init__(self, cfg):
        self.delay = cfg["fake"]["tts_first_packet_seconds"]

    def first_packet(self, text: str) -> float:
        """回傳第一段音訊出來所花的秒數。"""
        time.sleep(self.delay)
        return self.delay


class MLXAudioTTS:
    """mlx-audio 的 TTS。API 照它的 README（load_model + model.generate(text=…, stream=True)）；
    **沒有在 Mac 上驗過**，第一次彩排若失敗，改成不串流的 generate 並記錄「first packet = 整句」。"""
    name = "mlx_audio"

    def __init__(self, cfg):
        from mlx_audio.tts.utils import load_model
        c = cfg["tts"]
        self.model, self.voice, self.lang = load_model(c["mlx_audio_repo"]), c.get("voice", ""), c.get("lang_code", "")

    def first_packet(self, text: str) -> float:
        t0 = time.perf_counter()
        kw = {"text": text, "stream": True}
        if self.voice:
            kw["voice"] = self.voice
        if self.lang:
            kw["lang_code"] = self.lang        # 不給的話 Kokoro 用 a（美式英語）念中文
        try:
            next(iter(self.model.generate(**kw)))
        except TypeError:                       # 這一版沒有 stream 參數：整句合成完才算第一段
            kw.pop("stream")
            next(iter(self.model.generate(**kw)))
        return time.perf_counter() - t0


ASR = {"fake": FakeASR, "whispercpp": WhisperCppASR, "mlx_whisper": MLXWhisperASR}
LLM = {"fake": FakeLLM, "mlx_lm": MLXLLM}
TTS = {"fake": FakeTTS, "mlx_audio": MLXAudioTTS}


def load_backends(cfg: dict, fake: bool):
    names = ("fake", "fake", "fake") if fake else (cfg["asr"]["backend"], cfg["llm"]["backend"], cfg["tts"]["backend"])
    note(f"載入後端：asr={names[0]}  llm={names[1]}  tts={names[2]}")
    return ASR[names[0]](cfg), LLM[names[1]](cfg), TTS[names[2]](cfg)


# ── 一次完整的 cascade 量測 ─────────────────────────────────────────
def first_sentence(text: str) -> str:
    for i, ch in enumerate(text):
        if ch in "。！？!?.\n":
            return text[: i + 1]
    return text


def run_cascade(x, fs, cfg, asr, llm, tts) -> dict:
    t = time.perf_counter
    t0 = t()
    text = asr.transcribe(x, fs)
    asr_tail = t() - t0

    t0 = t()
    first_tok, reply, n_tok = None, [], 0
    for piece in llm.stream(text):
        if first_tok is None:
            first_tok = t() - t0
        reply.append(piece)
        n_tok += 1
    llm_total = t() - t0
    reply = "".join(reply)

    tts_first = tts.first_packet(first_sentence(reply) or reply)
    return {"transcript": text, "reply": reply, "reply_tokens": n_tok,
            "asr_tail_ms": round(asr_tail * 1000), "llm_first_token_ms": round((first_tok or 0) * 1000),
            "llm_total_ms": round(llm_total * 1000), "tts_first_packet_ms": round(tts_first * 1000)}


def percentiles(values):
    a = np.asarray(values, dtype=float)
    return {"p50": round(float(np.percentile(a, 50))), "p95": round(float(np.percentile(a, 95))),
            "min": round(float(a.min())), "max": round(float(a.max())), "n": int(a.size)}


def measure(x, fs, cfg, asr, llm, tts, repeats: int, quiet=False) -> dict:
    ep = endpoint_report(x, fs, cfg)
    thr = cfg["endpoint"]["threshold_ms"]
    row = next(r for r in ep["rows"] if r["threshold_ms"] == thr)
    if row["final_after_end_ms"] is None:                    # 補了 3 秒虛擬靜音之後不該發生
        raise SystemExit(f"{RED}門檻 {thr} ms 在句尾沒有開火（不該發生，檢查 VAD 的語音段）。{RESET}")
    runs = []
    for i in range(repeats):
        r = run_cascade(x, fs, cfg, asr, llm, tts)
        runs.append(r)
        if not quiet:
            print(f"  run {i + 1:>2}/{repeats}: ASR {r['asr_tail_ms']:>5} ms  LLM ttft {r['llm_first_token_ms']:>5} ms  "
                  f"TTS {r['tts_first_packet_ms']:>5} ms   「{r['transcript'][:24]}」")
    cells = {
        "endpoint": {"ms": row["final_after_end_ms"], "kind": "algorithmic (threshold)",
                     "false_endpoints_s": row["false_endpoints_s"]},
        "asr_tail": {**percentiles([r["asr_tail_ms"] for r in runs]), "kind": "computational"},
        "llm_first_token": {**percentiles([r["llm_first_token_ms"] for r in runs]), "kind": "computational"},
        "tts_first_packet": {**percentiles([r["tts_first_packet_ms"] for r in runs]), "kind": "computational"},
        "playout_buffer": {"ms": cfg["playout"]["buffer_ms"], "kind": "configured, not measured"},
    }
    total_p50 = cells["endpoint"]["ms"] + cells["asr_tail"]["p50"] + cells["llm_first_token"]["p50"] \
        + cells["tts_first_packet"]["p50"] + cells["playout_buffer"]["ms"]
    total_p95 = cells["endpoint"]["ms"] + cells["asr_tail"]["p95"] + cells["llm_first_token"]["p95"] \
        + cells["tts_first_packet"]["p95"] + cells["playout_buffer"]["ms"]
    return {"timestamp": common.now(), "versions": common.versions(),
            "backends": {"asr": asr.name, "llm": llm.name, "tts": tts.name},
            "endpoint_report": ep, "runs": runs, "cells": cells,
            "total_ms": {"p50": total_p50, "p95_sum": total_p95,
                         "_note": "p95_sum 是各格 p95 直接相加，是上界不是整條鏈的 p95（p40：機率相乘）"},
            "e2e_reference": e2e_reference()}


def e2e_reference() -> dict:
    """end-to-end 那一半沒有現場量：用 cold open Recording 2 的 60 ms 當參考點，標明出處。"""
    try:
        m = json.loads(METRICS_W01.read_text(encoding="utf-8"))
        return {"response_latency_ms": m["slides"]["response_latency_good_ms"]["value"],
                "source": "assets/w01/w01-metrics.json → slides.response_latency_good_ms "
                          "(Moshi + Fisher post-training, synthetic_user_interruption/1; pre-recorded, different utterance)"}
    except Exception:
        return {"response_latency_ms": None, "source": "w01-metrics.json 讀不到"}


# ── 輸出：表格與時間軸圖 ─────────────────────────────────────────────
def print_sheet(res: dict, live: bool) -> None:
    c = res["cells"]
    banner(("LIVE　" if live else "REPLAY（彩排結果，非現場）　") + "延遲預算：使用者停止說話之後", GREEN if live else AMBER)
    order = [("endpoint detection", c["endpoint"]["ms"], c["endpoint"]["ms"], c["endpoint"]["kind"]),
             ("ASR tail", c["asr_tail"]["p50"], c["asr_tail"]["p95"], c["asr_tail"]["kind"]),
             ("LLM prefill + first token", c["llm_first_token"]["p50"], c["llm_first_token"]["p95"], c["llm_first_token"]["kind"]),
             ("TTS first packet", c["tts_first_packet"]["p50"], c["tts_first_packet"]["p95"], c["tts_first_packet"]["kind"]),
             ("playout buffer", c["playout_buffer"]["ms"], c["playout_buffer"]["ms"], c["playout_buffer"]["kind"])]
    biggest = max(order, key=lambda r: r[1])[0]
    print(f"{BOLD}{'cell':<28}{'p50':>8}{'p95':>8}   kind{RESET}")
    for name, p50, p95, kind in order:
        mark = f" {RED}◀ largest{RESET}" if name == biggest else ""
        print(f"{name:<28}{p50:>7} {p95:>7}    {DIM}{kind}{RESET}{mark}")
    t = res["total_ms"]
    print(f"{BOLD}{'total':<28}{t['p50']:>7} {t['p95_sum']:>7}{RESET}    {DIM}(p95 欄是各格相加的上界){RESET}")
    if c["endpoint"]["false_endpoints_s"]:
        print(f"\n{RED}門檻在句中誤判了 {len(c['endpoint']['false_endpoints_s'])} 次"
              f"（{c['endpoint']['false_endpoints_s']} s）——那一次的回應會切斷使用者。{RESET}")
    e = res["e2e_reference"]
    print(f"\n{DIM}end-to-end 參考點：{e['response_latency_ms']} ms（{e['source']}）{RESET}")
    print(f"{DIM}ASR「{res['runs'][0]['transcript']}」→ LLM「{res['runs'][0]['reply'][:60]}」{RESET}")


def plot_timeline(res: dict, out: Path, live: bool) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    c = res["cells"]
    names = ["endpoint\ndetection", "ASR tail", "LLM prefill\n+ first token", "TTS\nfirst packet", "playout\nbuffer"]
    p50 = [c["endpoint"]["ms"], c["asr_tail"]["p50"], c["llm_first_token"]["p50"], c["tts_first_packet"]["p50"], c["playout_buffer"]["ms"]]
    p95 = [c["endpoint"]["ms"], c["asr_tail"]["p95"], c["llm_first_token"]["p95"], c["tts_first_packet"]["p95"], c["playout_buffer"]["ms"]]
    cols = ["#0b5cad", "#0b5cad", "#b45309", "#b45309", "#6b7280"]
    fig, ax = plt.subplots(figsize=(12.8, 4.2), dpi=100)
    left = 0
    for n, w, w95, col in zip(names, p50, p95, cols):
        ax.barh(1, w, left=left, color=col, alpha=0.85, edgecolor="white")
        ax.text(left + w / 2, 1, f"{n}\n{w} ms", ha="center", va="center", fontsize=10, color="white" if w > 120 else "#1f2328")
        left += w
    e = res["e2e_reference"]["response_latency_ms"]
    if e is not None:
        ax.barh(0, e, color="#15803d")
        ax.text(e + 15, 0, f"end-to-end reference: {e} ms  (Recording 2, pre-recorded, different utterance)",
                va="center", fontsize=10, color="#15803d")
    ax.set_yticks([1, 0]); ax.set_yticklabels(["cascade (p50)", "e2e ref."])
    ax.set_xlabel("ms after the user stops speaking")
    ax.axvline(0, color="#dc2626", lw=1.2, ls="--"); ax.text(0, 1.45, "user stops", color="#dc2626", fontsize=9)
    ax.set_xlim(0, max(left, e or 0) * 1.12)
    ax.set_ylim(-0.5, 1.7)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    tag = "LIVE" if live else "REPLAY — rehearsal, not live"
    ax.set_title(f"{tag}   total p50 = {res['total_ms']['p50']} ms   (p95 sum = {res['total_ms']['p95_sum']} ms)   "
                 f"backends: {res['backends']['asr']} / {res['backends']['llm']} / {res['backends']['tts']}",
                 loc="left", fontsize=11)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out); plt.close(fig)
    note(f"時間軸圖 → {out}")


def open_file(p: Path) -> None:
    import subprocess
    if sys.platform == "darwin":                            # Linux 的 /usr/bin/open 是別的東西
        subprocess.run(["open", str(p)], check=False)


# ── 素材 ─────────────────────────────────────────────────────────
def fake_utterance(fs: int) -> np.ndarray:
    """合成：1.5 s 母音、停 0.5 s、1.6 s 母音、尾巴 1.5 s 靜音。只用來演練流程。"""
    a = dsp.synth_vowel(fs, 1.5, 120.0, (700, 1200, 2600))
    b = dsp.synth_vowel(fs, 1.6, 110.0, (300, 2300, 3000))
    x = np.concatenate([a, np.zeros(int(0.5 * fs)), b, np.zeros(int(1.5 * fs))])
    return x / (np.abs(x).max() + 1e-9) * 0.5 + np.random.default_rng(0).normal(0, 1e-3, len(x))


def load_utterance(args, cfg) -> tuple[np.ndarray, int, str]:
    fs = cfg["audio"]["fs"]
    if args.fake:
        return fake_utterance(fs), fs, "FAKE"
    wav = Path(args.wav) if args.wav else HERE / cfg["audio"]["default"]
    if not wav.exists():
        raise SystemExit(f"{RED}找不到 {wav}。先 ./present.sh record，或用 --wav 指定。{RESET}")
    return common.load_wav(wav, fs), fs, wav.stem


# ── 子命令 ───────────────────────────────────────────────────────
def cmd_devices(args, cfg):
    import sounddevice as sd
    print(sd.query_devices())
    print(f"\n預設輸入／輸出：{sd.default.device}")


def cmd_record(args, cfg):
    fs = cfg["audio"]["fs"]
    banner("錄那句話。中間要停約 700 ms，說完之後保持安靜到錄音結束。", AMBER)
    print(f"建議：{cfg['audio']['suggested_sentence']}")
    print("按任意鍵開始……"); common.getkey()
    x = common.record(cfg["audio"]["record_seconds"], fs)
    out = HERE / cfg["audio"]["default"]
    common.save_wav(out, x, fs, peak=float(np.abs(x).max()))
    print(f"存到 {out}")
    print_endpoint(endpoint_report(x, fs, cfg))
    common.play(out)


def cmd_endpoint(args, cfg):
    x, fs, _ = load_utterance(args, cfg)
    print_endpoint(endpoint_report(x, fs, cfg))


def cmd_check(args, cfg):
    banner("課前體檢", BLUE)
    print(json.dumps(common.versions(), ensure_ascii=False, indent=1))
    x, fs, name = load_utterance(args, cfg)
    print(f"素材：{name}，{len(x) / fs:.2f} s @ {fs} Hz")
    t0 = time.perf_counter()
    asr, llm, tts = load_backends(cfg, args.fake)
    print(f"載入 {time.perf_counter() - t0:.1f} s；暖機一次（第一次通常特別慢，不算數）……")
    r = run_cascade(x, fs, cfg, asr, llm, tts)
    print(json.dumps(r, ensure_ascii=False, indent=1))


def cmd_rehearse(args, cfg):
    x, fs, name = load_utterance(args, cfg)
    asr, llm, tts = load_backends(cfg, args.fake)
    run_cascade(x, fs, cfg, asr, llm, tts)                  # 暖機，不計
    banner(f"彩排：{cfg['measure']['repeats']} 遍", BLUE)
    res = measure(x, fs, cfg, asr, llm, tts, cfg["measure"]["repeats"])
    res["material"] = name
    out = REHEARSAL / name
    out.mkdir(parents=True, exist_ok=True)
    (out / "result.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print_sheet(res, live=False)
    plot_timeline(res, out / "timeline.png", live=False)
    open_file(out / "timeline.png")                         # 2026-09-21：彩排完直接開圖，不必再跑 replay
    print(f"\n結果 → {out}/（進版控，現場失敗時 ./present.sh replay）")


def cmd_show(args, cfg):
    x, fs, name = load_utterance(args, cfg)
    asr, llm, tts = load_backends(cfg, args.fake)
    run_cascade(x, fs, cfg, asr, llm, tts)                  # 暖機
    banner("先預測：哪一格會最大？", AMBER)
    print("  1 endpoint   2 ASR tail   3 LLM   4 TTS   5 playout")
    print("（等學生喊完，按對應的數字或任意鍵繼續。）")
    guess = common.getkey()
    if not args.fake:
        common.play(HERE / cfg["audio"]["default"])      # 讓教室聽一次那句話
    banner(f"跑 {cfg['measure']['repeats']} 遍", BLUE)
    res = measure(x, fs, cfg, asr, llm, tts, cfg["measure"]["repeats"])
    res["material"], res["class_guess"] = name, guess
    out = LIVE / common.now().replace(":", "")
    out.mkdir(parents=True, exist_ok=True)
    (out / "result.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print_sheet(res, live=True)
    plot_timeline(res, out / "timeline.png", live=True)
    open_file(out / "timeline.png")


def cmd_replay(args, cfg):
    name = args.name or (Path(cfg["audio"]["default"]).stem if not args.fake else "FAKE")
    p = REHEARSAL / name / "result.json"
    if not p.exists():
        raise SystemExit(f"{RED}沒有彩排紀錄 {p}。先 ./present.sh rehearse。{RESET}")
    res = json.loads(p.read_text(encoding="utf-8"))
    print_sheet(res, live=False)
    plot_timeline(res, REHEARSAL / name / "timeline.png", live=False)
    open_file(REHEARSAL / name / "timeline.png")


def cmd_moshi(args, cfg):
    """moshi-mlx 的即時全雙工。沒有時間戳，只是讓教室聽同一句話由 end-to-end 模型回應。
    §6：戴耳機（它沒有回音消除）；-q 與 --hf-repo 要配對；跑久了延遲會漂。"""
    import subprocess
    c = cfg["moshi"]
    banner("moshi-mlx 即時全雙工——戴耳機！Ctrl-C 結束", AMBER)
    # moshi-mlx 釘的 mlx 版本與 mlx-lm 不相容，所以它裝在 uv tool 的獨立環境，不在共用 .venv（見 setup.sh）
    cmd = ["uvx", "--from", f"moshi-mlx=={c['version']}", "python", "-m", "moshi_mlx.local", "-q", str(c["quantization"])]
    if c.get("hf_repo"):
        cmd += ["--hf-repo", c["hf_repo"]]
    print(" ".join(cmd))
    subprocess.run(cmd, check=False)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fake", action="store_true", help="合成訊號＋fake 後端，任何機器都能演練流程")
    ap.add_argument("--wav", help="用這個 WAV 當那句話")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("devices", "record", "endpoint", "check", "rehearse", "show", "moshi"):
        sub.add_parser(name)
    sub.add_parser("replay").add_argument("--name")
    args = ap.parse_args()
    cfg = load_config()
    globals()[f"cmd_{args.cmd}"](args, cfg)


if __name__ == "__main__":
    main()

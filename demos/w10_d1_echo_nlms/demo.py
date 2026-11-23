# -*- coding: utf-8 -*-
# 此檔由 bin/publish_demos.py 從 private repo 的 <course>/demos/ 複製而來，請勿在公開 repo 直接編輯。
"""W10 demo 1：系統聽見自己——回音、NLMS、double-talk（投影片 p18–p19、p33–p34）。

  show       課堂：讀 audio/real_*.wav（三段真的錄音；沒有就只有模擬），算好全部的版本，視窗裡單鍵操作。離開時存 runs/live/<時間>/（不進版控）
  rehearse   課前：不互動，真錄音與模擬各跑一遍（含 VAD 與 Whisper），存 runs/rehearsal/（進版控 = 現場的退路）
  replay     現場出狀況時：從 runs/rehearsal/ 讀回，同樣的鍵、同樣的畫面，不錄音、不算、不碰 torch
  record     錄真的回音：record alone｜talk｜loud（這台筆電的喇叭播、內建麥克風錄）→ audio/real_<名字>.wav；record user 錄一句乾淨的插話給模擬用
  check      課前體檢：版本、音訊裝置、播錄延遲量三次（穩不穩、會不會漂）、麥克風路徑上有沒有系統自帶的處理、
             NLMS 的速度、Silero VAD、Whisper、pyroomacoustics → runs/check.json
  fetch      下載 Whisper small（commit 讀 W7 Demo 1 的 model.lock；要網路）。Silero VAD 的權重在它的 pip 套件裡，不必另外抓
  make       重做模擬的三段（確定性；平常不必跑，show／rehearse 沒有快取時自己會做）
  selftest   不需要模型、麥克風、pyroomacoustics：NLMS 的恆等式與收斂、三種步長、直接相減、Geigel、整條流程走一遍
  export     runs/rehearsal/rehearsal.json 的摘要 → slides/assets/w10/w10-metrics.json（鍵 demo1；投影片數字進版控的出處）

視窗裡的鍵（順序照投影片 p19、p33、p34 講稿的【操作】）：
  1  播麥克風錄到的 y（只有系統在說的那一段）
  2  播 y − x：把播出去的訊號照原樣減掉
  3  播 y − g·x(n − D)：延遲對齊、音量也調到最合之後再減
  4  第一段（只有系統）＋ NLMS 從零開始：ERLE 對時間
  5  第二段（我插話）＋ 固定步長        6  同一段 ＋ Geigel 偵測器凍結        7  同一段 ＋ 軟步長
  8  第三段（喇叭失真）＋ NLMS：ERLE 的天花板
  9  第二段送給 Silero VAD 與 Whisper small：AEC 之前        0  AEC 之後
  y  播目前這一段的麥克風        e  播目前這一版的殘餘        a  只播「我停下來」前後那幾秒的殘餘
  c  印出 NLMS 迴圈的程式碼（與 p23、p25 的式子逐行對得上）
  m  真錄音／模擬 互換        t  表（p19、p33、p34 的三張）        h  選單        q  離開

數字的定義：
  殘餘的位準（鍵 1–3）= 10 log10(Σ 殘餘² ÷ Σ y²)，整段；0 dB = 沒有變小，正的 = 更大聲
  ERLE              = 10 log10(Σ y² ÷ Σ e²)，只在系統有出聲的樣本上算；曲線用過去 500 ms
  插話前／停下來之後 = 使用者開口前最後 2 秒／停止後最先 1 秒「系統有出聲的樣本」上的 ERLE（系統那一句有停頓，數的是有回音的樣本）；
                      「最後」= 這一段最後 3 秒
  使用者插話的區間   = 真錄音：第二段減第一段（同一個 x 錄了兩次）的能量；找不到就用錄音時畫面提示的那幾秒。模擬：已知
  ‖h − ĥ‖（只有模擬）= 10 log10(‖h − ĥ‖² ÷ ‖h‖²)，h 取前 L_h 個數
  VAD 亮的比例       = Silero VAD 判成語音的時間 ÷ 那個區間的時間（「只有系統在說」＝ x 有出聲、而且在使用者插話的區間之外）
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import inspect
import json
import sys
import time
import tomllib
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
import common as C  # noqa: E402
import dsp  # noqa: E402
import aec  # noqa: E402

CFG = tomllib.loads((HERE / "demo_config.toml").read_text(encoding="utf-8"))
FS = int(CFG["audio"]["fs"])
DIG = int(CFG["display"]["digits"])
AUDIO = HERE / "audio"
RUNS = HERE / "runs"
REHEARSAL = RUNS / "rehearsal"
CACHE = RUNS / "cache"
MODEL_LOCK = HERE / "model.lock"
W7 = HERE.parent / "w07_d1_hallucination"
METRICS = (HERE.parent.parent / "slides" / "assets" / "w10" / "w10-metrics.json").resolve()

CLIPS = ["alone", "talk", "loud"]
CLIP_LABEL = {"alone": "the system alone", "talk": "the lecturer talks over it", "loud": "the loudspeaker distorting"}
CLIP_ZH = {"alone": "只有系統在說", "talk": "系統在說、我插話", "loud": "喇叭開到失真"}
VARIANTS = [("alone", "fixed"), ("talk", "fixed"), ("talk", "geigel"), ("talk", "soft"), ("loud", "fixed")]
RULE_LABEL = {"fixed": "fixed step", "geigel": "Geigel freeze", "soft": "soft step"}
RULE_ZH = {"fixed": "固定步長", "geigel": "Geigel 凍結", "soft": "軟步長"}
RULE_COLOR = {"fixed": "#c0392b", "geigel": "#b9770e", "soft": "#1f7a4d"}
SRC_ZH = {"real": "真錄音", "sim": "模擬"}
KEY_VARIANT = {"4": ("alone", "fixed"), "5": ("talk", "fixed"), "6": ("talk", "geigel"), "7": ("talk", "soft"), "8": ("loud", "fixed")}


def f(v, d: int = DIG, unit: str = "") -> str:
    return "—" if v is None else f"{v:.{d}f}{unit}"


# ── 資料 ─────────────────────────────────────────────────────────────────────
def reference() -> np.ndarray:
    """x(n)：系統要播的那一句，16 kHz，前後墊靜音，峰值 = [reference] peak。"""
    r = CFG["reference"]
    p = (HERE / r["file"]).resolve()
    if not p.exists():
        sys.exit(f"找不到參考訊號 {p}（W8 Demo 2 的彩排輸出）。換一個檔：demo_config.toml [reference] file")
    x = C.load_wav(p, FS)
    x = x / (np.max(np.abs(x)) + 1e-12) * float(r["peak"])
    return np.concatenate([np.zeros(int(float(r["lead_s"]) * FS)), x, np.zeros(int(float(r["tail_s"]) * FS))])


def digest(*parts) -> str:
    h = hashlib.sha1()
    for p in parts:
        h.update(p.tobytes() if isinstance(p, np.ndarray) else json.dumps(p, sort_keys=True, default=str).encode())
    return h.hexdigest()[:12]


def x_sha(x: np.ndarray) -> str:
    """參考訊號的指紋：量化成 16-bit 再算（浮點的最後幾位在不同機器、不同版本的 scipy 上會不一樣，不該因此叫人重錄）。"""
    return digest(np.int16(np.clip(x, -1, 1) * 32767))


def synthetic_rir(n: int, delay: int, t60: float, seed: int) -> np.ndarray:
    """沒有 pyroomacoustics 時的替身：直達聲一根，後面接指數衰減的高斯序列（60 dB 衰減在 t60 秒）。"""
    rng = np.random.default_rng(seed)
    h = np.zeros(n)
    h[delay] = 1.0
    k = np.arange(n - delay - 1)
    h[delay + 1:] = 0.35 * rng.standard_normal(len(k)) * 10 ** (-3.0 * k / (t60 * FS))
    return h


def room_rirs() -> tuple[np.ndarray, np.ndarray, str]:
    s = CFG["sim"]
    try:
        import pyroomacoustics as pra
    except Exception:
        n = int(1.2 * float(s["rt60"]) * FS)
        return (synthetic_rir(n, 14, float(s["rt60"]), int(s["seed"])), synthetic_rir(n, 28, float(s["rt60"]), int(s["seed"]) + 1),
                "synthetic exponential-decay path (pyroomacoustics not installed)")
    absorb, order = pra.inverse_sabine(float(s["rt60"]), s["room"])
    room = pra.ShoeBox(s["room"], fs=FS, materials=pra.Material(absorb), max_order=order)
    room.add_source(s["loudspeaker"])
    room.add_source(s["user"])
    room.add_microphone(s["microphone"])
    room.compute_rir()
    return (np.asarray(room.rir[0][0], dtype=np.float64), np.asarray(room.rir[0][1], dtype=np.float64),
            f"pyroomacoustics {pra.__version__} image method, room {s['room']} m, RT60 {s['rt60']} s")


def near_file() -> Path | None:
    for rel in CFG["sim"]["near_files"]:
        p = (HERE / rel).resolve()
        if p.exists():
            return p
    return None


def make_sim(x: np.ndarray, rirs=None, near: np.ndarray | None = None) -> dict:
    """模擬的三段：同一條 h。alone = h∗x + v；talk 多一段使用者；loud 的喇叭輸出是 tanh 削過的 g(x)。"""
    s = CFG["sim"]
    h, hu, how = rirs if rirs is not None else room_rirs()
    rng = np.random.default_rng(int(s["seed"]))
    echo = np.convolve(x, h)[:len(x)]
    e_rms = float(np.sqrt(np.mean(echo ** 2)))
    noise = lambda: rng.standard_normal(len(x)) * e_rms * 10 ** (float(s["noise_db"]) / 20)  # noqa: E731
    if near is None:
        p = near_file()
        near = C.load_wav(p, FS) if p is not None else aec.colour(np.random.default_rng(1).standard_normal(int(float(s["near_len_s"]) * FS)), 0.9)
    near = near[:int(float(s["near_len_s"]) * FS)]
    u = np.convolve(near, hu)[:len(near)]
    u *= e_rms * 10 ** (float(s["near_db"]) / 20) / (float(np.sqrt(np.mean(u ** 2))) + 1e-12)
    a = int(float(s["near_start_s"]) * FS)
    a = min(a, max(len(x) - len(u) - FS, 0))
    sn = np.zeros(len(x))
    sn[a:a + len(u)] = u[:len(x) - a]
    gx = aec.soft_clip(x, float(s["loud_drive"]))
    return {"how": how, "h": h, "near": sn, "dt": (a, a + len(u)),
            "y": {"alone": echo + noise(), "talk": echo + sn + noise(), "loud": np.convolve(gx, h)[:len(x)] + noise()},
            "ceiling_db": aec.linear_ceiling_db(x, gx)}


def sim_cached(x: np.ndarray) -> dict:
    p = near_file()
    key = digest(x, CFG["sim"], str(p), p.stat().st_mtime_ns if p else 0)
    path = CACHE / f"sim_{key}.npz"
    if path.exists():
        z = np.load(path, allow_pickle=False)
        return {"how": str(z["how"]), "h": z["h"], "near": z["near"], "dt": tuple(int(v) for v in z["dt"]),
                "y": {c: z[f"y_{c}"] for c in CLIPS}, "ceiling_db": float(z["ceiling_db"])}
    s = make_sim(x)
    CACHE.mkdir(parents=True, exist_ok=True)
    np.savez(path, how=s["how"], h=s["h"], near=s["near"], dt=np.asarray(s["dt"]), ceiling_db=s["ceiling_db"], **{f"y_{c}": s["y"][c] for c in CLIPS})
    return s


def load_real(x: np.ndarray) -> dict:
    """audio/real_<clip>.wav（record 錄的）＋ 同名 .json。長度與 x 對不上的不要（x 換過了，要重錄）。"""
    out = {}
    for c in CLIPS:
        p = AUDIO / f"real_{c}.wav"
        if not p.exists():
            continue
        y = C.load_wav(p, FS)
        meta = json.loads(p.with_suffix(".json").read_text(encoding="utf-8")) if p.with_suffix(".json").exists() else {}
        if abs(len(y) - len(x)) > 8 or (meta.get("x_sha") and meta["x_sha"] not in (x_sha(x), digest(x))):
            C.note(f"  audio/real_{c}.wav 與目前的參考訊號對不上（x 換過了？），略過——重錄：./present.sh record {c}")
            continue
        y = np.concatenate([y, np.zeros(max(len(x) - len(y), 0))])[:len(x)]
        out[c] = {"y": y, "meta": meta}
    return out


# ── 計算 ─────────────────────────────────────────────────────────────────────
def align(x: np.ndarray, y: np.ndarray) -> tuple[int, np.ndarray]:
    """回音最早到的那一刻 D（GCC-PHAT），與拿掉大部分延遲之後的參考 x(n − D + margin)。"""
    D = aec.first_arrival(x, y, int(float(CFG["record"]["max_delay_s"]) * FS))
    return D, aec.shift(x, max(D - int(CFG["nlms"]["bulk_margin"]), 0))


def step_one(x: np.ndarray, y: np.ndarray, D: int) -> tuple[dict, dict]:
    """第一步：直接減。三個版本的殘餘與它們相對於 y 的位準。"""
    lit = y - x
    g0 = aec.best_gain(x, y)
    xd = aec.shift(x, D)
    g1 = aec.best_gain(xd, y)
    ali = y - g1 * xd
    return ({"delay_samples": D, "delay_ms": 1000.0 * D / FS, "literal_db": aec.level_db(lit, y),
             "gain_only_g": g0, "gain_only_db": aec.level_db(y - g0 * x, y), "aligned_g": g1, "aligned_db": aec.level_db(ali, y)},
            {"sub_literal": lit, "sub_aligned": ali})


def find_user(x_al: np.ndarray, y_talk: np.ndarray, y_alone: np.ndarray | None, D_talk: int, D_alone: int | None, cue: tuple[int, int]) -> tuple[tuple[int, int], str]:
    """真錄音裡使用者插話的區間。同一個 x 錄了兩次，兩段相減回音大致抵消、剩下的主要是使用者。"""
    m = CFG["measure"]
    if y_alone is not None:
        d = y_talk - aec.shift(y_alone, D_talk - D_alone) if D_talk >= D_alone else aec.shift(y_talk, D_alone - D_talk) - y_alone
        p = aec.smooth_power(d, int(0.03 * FS))
        run = aec.longest_run(aec.close_gaps(p > np.max(p) * 10 ** (float(m["user_floor_db"]) / 10), int(float(m["user_gap_ms"]) * FS / 1000)))
        if run is not None:
            a, b = run
            lo = p > np.max(p) * 10 ** (float(m["user_extend_db"]) / 10)      # 兩端往外延伸到能量掉到更低的門檻為止（字頭字尾比較小聲）
            while a > 0 and lo[a - 1]:
                a -= 1
            while b < len(lo) and lo[b]:
                b += 1
            if D_talk < D_alone:
                a, b = a - (D_alone - D_talk), b - (D_alone - D_talk)
            if 0.3 * FS <= b - a <= 10 * FS and a < cue[1] + 2 * FS and b > cue[0] - 2 * FS:
                return (max(a - int(0.03 * FS), 0), min(b, len(y_talk))), "difference of the two recordings"
    return cue, "the on-screen cue (not measured)"


def summarise(clip: str, x_al: np.ndarray, y: np.ndarray, r: dict, dt: tuple[int, int] | None) -> dict:
    m = CFG["measure"]
    act = aec.activity(x_al, FS, float(m["x_floor_db"]))
    e = r["e"]
    N = len(y)
    s = {"erle_final_db": aec.erle_span(y, e, N - int(float(m["final_s"]) * FS) - int(float(CFG["reference"]["tail_s"]) * FS), N, act),
         "erle_all_db": aec.erle_span(y, e, 0, N, act), "mu_mean": float(np.mean(r["mu"][act]))}
    t, curve = aec.erle_curve(y, e, int(float(m["erle_win_ms"]) * FS / 1000), int(float(m["erle_step_ms"]) * FS / 1000), act)
    if s["erle_final_db"] is not None:
        ok = np.where(curve >= s["erle_final_db"] - float(m["recover_db"]))[0]
        s["converge_s"] = float(t[ok[0]] / FS) if len(ok) else None
    if "misalignment" in r:
        s["misalignment_final_db"] = float(10 * np.log10(r["misalignment"][-1] + 1e-12))
    if dt is not None:
        a, b = dt
        s["erle_before_db"] = aec.erle_active(y, e, act, a, int(float(m["before_s"]) * FS), forward=False)
        s["erle_during_db"] = aec.erle_span(y, e, a, b, act)
        s["erle_after_db"] = aec.erle_active(y, e, act, b, int(float(m["after_s"]) * FS), forward=True)
        s["mu_before"] = float(np.mean(r["mu"][max(a - int(float(m["before_s"]) * FS), 0):a]))
        s["mu_during"] = float(np.mean(r["mu"][a:b]))
        w = int(0.25 * FS)
        t2, c2 = aec.erle_curve(y, e, w, int(0.05 * FS), act)
        s["recover_s"] = None
        if s["erle_before_db"] is not None:
            ok = np.where((t2 >= b + w) & (c2 >= s["erle_before_db"] - float(m["recover_db"])))[0]
            s["recover_s"] = float((t2[ok[0]] - b) / FS) if len(ok) else None
        if "misalignment" in r:
            s["misalignment_before_db"] = float(10 * np.log10(r["misalignment"][a // 64] + 1e-12))
            s["misalignment_after_db"] = float(10 * np.log10(r["misalignment"][min(b // 64, len(r["misalignment"]) - 1)] + 1e-12))
    return s


def compute_source(source: str, x: np.ndarray, ys: dict, dts: dict, h_true: np.ndarray | None, out, extra: dict | None = None) -> tuple[dict, dict]:
    """一個來源（real 或 sim）的全部：第一步、五個版本。回傳 (JSON 化的 store, {wav 名: (陣列, 峰值)})。有快取就讀快取。"""
    n, g = CFG["nlms"], CFG["geigel"]
    L_h, mu = int(n["L_h"]), float(n["mu"])
    delta = float(n["delta_rel"]) * L_h * float(np.mean(x ** 2))
    key = digest(x, *[ys[c] for c in sorted(ys)], n, g, CFG["soft"], CFG["measure"], dts, inspect.getsource(aec))
    path = CACHE / f"{source}_{key}.npz"
    meta_path = path.with_suffix(".json")
    if path.exists() and meta_path.exists():
        z = np.load(path, allow_pickle=False)
        store = json.loads(meta_path.read_text(encoding="utf-8"))
        wavs = {k[4:]: (z[k], float(z["pk_" + k[4:]])) for k in z.files if k.startswith("wav_")}
        out(f"  {SRC_ZH[source]}：讀快取（{path.name}）")
        return store, wavs
    store: dict = {"clips": {}, "variants": {}, "L_h": L_h, "mu": mu, "delta": delta, **(extra or {})}
    wavs: dict = {}
    al = {}
    for c in CLIPS:
        if c not in ys:
            continue
        D, x_al = align(x, ys[c])
        al[c] = (D, x_al)
        y = ys[c]
        pk = float(np.max(np.abs(y)) + 1e-12)
        store["clips"][c] = {"seconds": len(y) / FS, "delay_samples": D, "delay_ms": 1000.0 * D / FS, "y_rms_db": float(10 * np.log10(np.mean(y ** 2) + 1e-12)),
                             "y_peak": pk, "clipped_frac": float(np.mean(np.abs(y) > 0.985))}
        wavs[f"{c}_y"] = (y, pk)
    if "alone" in ys:
        s1, arr = step_one(x, ys["alone"], aec.estimate_delay(x, ys["alone"], int(float(CFG["record"]["max_delay_s"]) * FS)))
        s1["first_arrival_samples"] = al["alone"][0]
        store["step1"] = s1
        pk = float(max(np.max(np.abs(arr["sub_literal"])), np.max(np.abs(arr["sub_aligned"])), store["clips"]["alone"]["y_peak"]))
        wavs["alone_sub_literal"] = (arr["sub_literal"], pk)
        wavs["alone_sub_aligned"] = (arr["sub_aligned"], store["clips"]["alone"]["y_peak"])
        wavs["alone_x"] = (x, float(np.max(np.abs(x))))
    # Geigel 的 ϑ：從「只有系統在說」的樣本量（alone 整段；沒有 alone 就用 talk 在使用者開口之前的部分）
    theta = None
    if "talk" in ys:
        if "alone" in ys:
            q = aec.geigel_ratio(al["alone"][1], ys["alone"], L_h)
        else:
            a0 = dts["talk"][0] if dts.get("talk") else len(x) // 3
            q = aec.geigel_ratio(al["talk"][1][:a0], ys["talk"][:a0], L_h)
        theta = float(g["margin"]) * float(np.percentile(q, float(g["percentile"])))
        store["geigel_theta"] = theta
    for c, rule in VARIANTS:
        if c not in ys:
            continue
        D, x_al = al[c]
        y = ys[c]
        t0 = time.perf_counter()
        ht = h_true[max(D - int(n["bulk_margin"]), 0):][:L_h] if h_true is not None else None
        if ht is not None and len(ht) < L_h:
            ht = np.concatenate([ht, np.zeros(L_h - len(ht))])
        kw = dict(L_h=L_h, mu=mu, delta=delta, fs=FS, h_true=ht)
        frozen = None
        if rule == "geigel":
            frozen = aec.geigel(x_al, y, L_h, theta, int(float(g["hold_ms"]) * FS / 1000))
            r = aec.nlms(x_al, y, rule="freeze", frozen=frozen, **kw)
        elif rule == "soft":
            r = aec.nlms(x_al, y, rule="soft", soft=CFG["soft"], **kw)
        else:
            r = aec.nlms(x_al, y, rule="fixed", **kw)
        dt = dts.get(c)
        s = summarise(c, x_al, y, r, dt)
        s["seconds_to_compute"] = time.perf_counter() - t0
        m = CFG["measure"]
        act = aec.activity(x_al, FS, float(m["x_floor_db"]))
        t, curve = aec.erle_curve(y, r["e"], int(float(m["erle_win_ms"]) * FS / 1000), int(float(m["erle_step_ms"]) * FS / 1000), act)
        step = int(0.02 * FS)
        v = {"summary": s, "t": (t / FS).round(3).tolist(), "erle_db": [None if np.isnan(c_) else round(float(c_), 2) for c_ in curve],
             "mu_t": (np.arange(0, len(y), step) / FS).round(3).tolist(), "mu": np.round(r["mu"][::step], 4).tolist()}
        if frozen is not None:
            v["frozen_frac"] = float(np.mean(frozen))
            if dt is not None:
                v["frozen_frac_user"] = float(np.mean(frozen[dt[0]:dt[1]]))
                outside = np.ones(len(y), bool)
                outside[dt[0]:dt[1]] = False
                v["frozen_frac_system_only"] = float(np.mean(frozen[outside & act]))
        if "misalignment" in r:
            v["mis_t"] = (np.arange(len(r["misalignment"])) * 64 / FS)[::8].round(3).tolist()
            v["mis_db"] = np.round(10 * np.log10(r["misalignment"][::8] + 1e-12), 2).tolist()
        store["variants"][f"{c}:{rule}"] = v
        wavs[f"{c}_e_{rule}"] = (r["e"], store["clips"][c]["y_peak"])
        out(f"  {SRC_ZH[source]}  {CLIP_ZH[c]:<9s} {RULE_ZH[rule]:<10s} ERLE 最後 {f(s['erle_final_db'])} dB"
            + (f"；插話前 {f(s.get('erle_before_db'))} → 停下來之後 {f(s.get('erle_after_db'))} dB" if dt is not None else "") + f"   （{s['seconds_to_compute']:.1f} s）")
    for c, dt in dts.items():
        if c in store["clips"] and dt is not None:
            store["clips"][c]["dt_s"] = [dt[0] / FS, dt[1] / FS]
    CACHE.mkdir(parents=True, exist_ok=True)
    np.savez(path, **{f"wav_{k}": v[0].astype(np.float32) for k, v in wavs.items()}, **{f"pk_{k}": v[1] for k, v in wavs.items()})
    meta_path.write_text(json.dumps(store, ensure_ascii=False, default=float), encoding="utf-8")
    return store, wavs


def build_sources(out, want_real: bool = True, sim_override: dict | None = None, x: np.ndarray | None = None) -> tuple[dict, dict]:
    """→ ({source: store}, {source: wavs})。real 只在 audio/real_*.wav 存在時才有。"""
    x = reference() if x is None else x
    stores, wavs = {}, {}
    if want_real:
        real = load_real(x)
        if real:
            ys = {c: real[c]["y"] for c in real}
            dts = {}
            if "talk" in ys:
                rc = CFG["record"]
                lead = int(float(CFG["reference"]["lead_s"]) * FS)
                cue = (lead + int(float(rc["cue_start_s"]) * FS), lead + int(float(rc["cue_stop_s"]) * FS))
                Dt, x_al = align(x, ys["talk"])
                Da = align(x, ys["alone"])[0] if "alone" in ys else None
                dts["talk"], how = find_user(x_al, ys["talk"], ys.get("alone"), Dt, Da, cue)
            extra = {"how": "this laptop's loudspeaker and built-in microphone (sounddevice playrec)", "recorded": {c: real[c]["meta"] for c in real}}
            stores["real"], wavs["real"] = compute_source("real", x, ys, dts, None, out, extra)
            if "talk" in ys:
                stores["real"]["clips"]["talk"]["dt_source"] = how
        else:
            C.note("  沒有 audio/real_*.wav：只有模擬。錄真的：./present.sh record alone → record talk → record loud")
    sim = sim_override if sim_override is not None else sim_cached(x)
    extra = {"how": sim["how"], "ceiling_db": sim["ceiling_db"], "rir_taps": int(len(sim["h"]))}
    stores["sim"], wavs["sim"] = compute_source("sim", x, sim["y"], {"talk": sim["dt"]}, sim["h"], out, extra)
    stores["sim"]["clips"]["talk"]["dt_source"] = "known (simulation)"
    return stores, wavs


# ── 下游：VAD 與 Whisper ─────────────────────────────────────────────────────
def lcs(a: str, b: str) -> int:
    prev = [0] * (len(b) + 1)
    for ca in a:
        cur = [0]
        for j, cb in enumerate(b):
            cur.append(prev[j] + 1 if ca == cb else max(prev[j + 1], cur[j]))
        prev = cur
    return prev[-1]


def only_text(s: str) -> str:
    return "".join(ch for ch in s if ch.isalnum())


def vad_stats(probs: np.ndarray, segments: list, x_active: np.ndarray, dt: tuple[int, int] | None, n: int, hop: int = 512) -> dict:
    on = np.zeros(n, bool)
    for a, b in segments:
        on[int(a * FS):int(b * FS)] = True
    s = {"p_t": (np.arange(len(probs)) * hop / FS).round(3).tolist(), "p": np.round(probs, 3).tolist(),
         "segments": [[round(a, 3), round(b, 3)] for a, b in segments], "active_frac": float(np.mean(on))}
    sys_only = x_active.copy()
    if dt is not None:
        g = int(0.2 * FS)
        sys_only[max(dt[0] - g, 0):dt[1] + g] = False
        s["active_frac_user"] = float(np.mean(on[dt[0]:dt[1]]))
    s["active_frac_system_only"] = float(np.mean(on[sys_only])) if sys_only.any() else None
    return s


class Models:
    """Silero VAD（torch JIT，權重在 pip 套件裡）與 Whisper small（W7 Demo 1 的類別與 commit）。用到才載。"""

    def __init__(self):
        self.vad = self.wh = self.torch = None

    def load_vad(self):
        if self.vad is None:
            import torch
            from silero_vad import load_silero_vad
            self.torch = torch
            self.vad = load_silero_vad()
        return self.vad

    def load_whisper(self):
        if self.wh is None:
            spec = importlib.util.spec_from_file_location("w07_d1_demo", W7 / "demo.py")
            mod = importlib.util.module_from_spec(spec)
            sys.modules["w07_d1_demo"] = mod
            spec.loader.exec_module(mod)
            mod.CFG["whisper"]["max_new_tokens"] = int(CFG["downstream"]["max_new_tokens"])
            self.wh = mod.Whisper()
        return self.wh

    def run_vad(self, v: np.ndarray) -> tuple[np.ndarray, list]:
        from silero_vad import get_speech_timestamps
        m = self.load_vad()
        torch = self.torch
        d = CFG["downstream"]
        a = torch.from_numpy(v.astype(np.float32))
        m.reset_states()
        probs = []
        with torch.no_grad():
            for i in range(0, len(a) - 511, 512):
                probs.append(float(m(a[i:i + 512], FS).item()))
        m.reset_states()
        ts = get_speech_timestamps(a, m, threshold=float(d["vad_threshold"]), sampling_rate=FS,
                                   min_silence_duration_ms=int(d["vad_min_silence_ms"]), return_seconds=True)
        return np.asarray(probs), [(float(t["start"]), float(t["end"])) for t in ts]

    def run_whisper(self, v: np.ndarray) -> str:
        wh = self.load_whisper()
        ids = wh.generate(v, language=CFG["downstream"]["language"])[0]
        return wh.text(ids).strip()


def downstream(store: dict, wavs: dict, x: np.ndarray, models: Models, out) -> dict:
    """talk 與 alone 各兩個訊號（AEC 之前 = y；之後 = [downstream] variant 的殘餘）→ VAD 的機率與區間、Whisper 的文字。"""
    res = {}
    ref = only_text(CFG["reference"]["text"])
    var = CFG["downstream"]["variant"]
    for c in ("talk", "alone"):
        if c not in store["clips"]:
            continue
        rule = var if c == "talk" else "fixed"
        dt = store["clips"][c].get("dt_s")
        dt = (int(dt[0] * FS), int(dt[1] * FS)) if dt else None
        x_al = aec.shift(x, max(store["clips"][c]["delay_samples"] - int(CFG["nlms"]["bulk_margin"]), 0))
        act = aec.activity(x_al, FS, float(CFG["measure"]["x_floor_db"]))
        for when, name in (("before", f"{c}_y"), ("after", f"{c}_e_{rule}")):
            v = wavs[name][0].astype(np.float64)
            t0 = time.perf_counter()
            probs, segs = models.run_vad(v)
            t1 = time.perf_counter()
            text = models.run_whisper(v)
            t2 = time.perf_counter()
            hyp = only_text(text)
            r = {"signal": name, "vad": vad_stats(probs, segs, act, dt, len(v)), "whisper": {"text": text, "n_chars": len(hyp),
                 "system_chars_found": lcs(ref, hyp), "system_chars": len(ref)}, "seconds": {"vad": t1 - t0, "whisper": t2 - t1}}
            res[f"{c}:{when}"] = r
            out(f"  {CLIP_ZH[c]}／AEC {'之前' if when == 'before' else '之後'}：VAD 亮 {100 * r['vad']['active_frac']:.0f}%"
                + (f"（只有系統在說時 {100 * r['vad']['active_frac_system_only']:.0f}%" if r["vad"]["active_frac_system_only"] is not None else "（")
                + (f"、我插話時 {100 * r['vad']['active_frac_user']:.0f}%）" if dt else "）") + f"   Whisper：{text!r}")
    return res


class Out:
    def __init__(self):
        self.lines: list[str] = []

    def __call__(self, s: str = "") -> None:
        print(s)
        self.lines.append(s)

    def banner(self, s: str, color: str = C.BLUE) -> None:
        C.banner(s, color)
        self.lines += ["", "─" * 72, s, "─" * 72]

    def text(self) -> str:
        return "\n".join(self.lines) + "\n"


# ── Demo ─────────────────────────────────────────────────────────────────────
class Demo:
    """stores：{source: store}（JSON 化的數字與曲線）；波形從 wavdir/<source>_<名字>.wav 讀（replay 與 show 走同一條路）。"""

    def __init__(self, out: Out, stores: dict, wavdir: Path, tag: str = "", models: Models | None = None, wavs: dict | None = None, x: np.ndarray | None = None):
        self.out, self.stores, self.wavdir, self.tag = out, stores, wavdir, tag
        self.models, self._wavs, self.x = models, wavs, x
        self.source = "real" if "real" in stores else "sim"
        self.mode = "idle"
        self.clip, self.rule = "alone", "fixed"
        self.shown: list[str] = []
        self.down_after = False
        self.last_play: tuple[str, float, float | None] | None = None
        self.fig = self.plt = None
        self._arr: dict = {}

    # 波形與播放
    @property
    def st(self) -> dict:
        return self.stores[self.source]

    def gains(self) -> dict:
        return self.st.get("wavs", {})

    def arr(self, name: str) -> np.ndarray | None:
        k = f"{self.source}_{name}"
        if k not in self._arr:
            p = self.wavdir / f"{k}.wav"
            if not p.exists():
                return None
            from scipy.io import wavfile
            _, v = wavfile.read(str(p))
            self._arr[k] = v.astype(np.float64) / 32767.0 / self.gains().get(name, 1.0)
        return self._arr[k]

    def play(self, name: str, a_s: float = 0.0, b_s: float | None = None) -> None:
        p = self.wavdir / f"{self.source}_{name}.wav"
        if not p.exists():
            self.out(f"  沒有 {p.name}（這一段在這個來源裡沒有）"); return
        self.last_play = (name, a_s, b_s)
        if a_s <= 0 and b_s is None:
            C.play(p); return
        from scipy.io import wavfile
        fs, v = wavfile.read(str(p))
        tmp = RUNS / "live" / "_play.wav"
        tmp.parent.mkdir(parents=True, exist_ok=True)
        wavfile.write(str(tmp), fs, v[int(a_s * fs):None if b_s is None else int(b_s * fs)])
        C.play(tmp)

    # 面板
    def _need(self, clip: str) -> bool:
        if clip in self.st["clips"]:
            return True
        self.out(f"  {SRC_ZH[self.source]}裡沒有「{CLIP_ZH[clip]}」那一段" + ("：./present.sh record " + clip if self.source == "real" else "") + "；按 m 換來源")
        return False

    def panel_step1(self, which: str) -> None:
        if not self._need("alone") or "step1" not in self.st:
            return
        o, s = self.out, self.st["step1"]
        self.mode, self.clip = "step1", "alone"
        self.step1_shown = max(getattr(self, "step1_shown", 0), int(which))
        T = float(CFG["display"]["step1_play_s"])
        if which == "1":
            o.banner(f"鍵 1：麥克風錄到的 y(n)（{SRC_ZH[self.source]}，只有系統在說）")
            o("  殘餘的位準 0 dB（自己對自己，定義）")
            self.redraw(); self.play("alone_y", 0, T)
        elif which == "2":
            o.banner("鍵 2：y(n) − x(n)，照原樣減")
            o(f"  殘餘的位準 {s['literal_db']:+.1f} dB（相對於 y；正的 = 比原來大聲）")
            o(f"  只准調一個音量 g、不准挪時間：最好的 g = {s['gain_only_g']:+.4f} → {s['gain_only_db']:+.2f} dB")
            C.note("  （播放時音量另外調過：三段用同一個增益，不然這一段會破）")
            self.redraw(); self.play("alone_sub_literal", 0, T)
        else:
            o.banner("鍵 3：y(n) − g·x(n − D)，延遲對齊、音量調到最合")
            o(f"  D = {s['delay_samples']} 個樣本（{s['delay_ms']:.1f} ms），g = {s['aligned_g']:+.4f}")
            o(f"  殘餘的位準 {s['aligned_db']:+.2f} dB")
            self.redraw(); self.play("alone_sub_aligned", 0, T)

    def panel_variant(self, clip: str, rule: str) -> None:
        if not self._need(clip):
            return
        key = f"{clip}:{rule}"
        v = self.st["variants"].get(key)
        if v is None:
            self.out(f"  沒有 {key}"); return
        o, s = self.out, v["summary"]
        if self.mode != "aec" or self.clip != clip:
            self.shown = []
        self.mode, self.clip, self.rule = "aec", clip, rule
        if key not in self.shown:
            self.shown.append(key)
        o.banner(f"{CLIP_ZH[clip]} ＋ {RULE_ZH[rule]}（{SRC_ZH[self.source]}；L_h = {self.st['L_h']}、μ = {self.st['mu']}）")
        if clip == "alone":
            o(f"  ERLE 最後 {f(s['erle_final_db'])} dB；第 {f(s.get('converge_s'))} 秒起到了「最後 − {CFG['measure']['recover_db']:g} dB」以內")
        elif clip == "talk":
            dt = self.st["clips"]["talk"].get("dt_s")
            o(f"  我插話的區間：{dt[0]:.2f}–{dt[1]:.2f} s（{self.st['clips']['talk'].get('dt_source', '')}）")
            if rule == "geigel":
                o(f"  ϑ = {self.st['geigel_theta']:.3f}（從只有系統在說的樣本量的）；凍結的時間：我插話時 {100 * v.get('frozen_frac_user', 0):.0f}%、只有系統在說時 {100 * v.get('frozen_frac_system_only', 0):.0f}%")
            if rule == "soft":
                o(f"  μ 的平均：插話前 {s['mu_before']:.3f} → 插話時 {s['mu_during']:.3f}（上限 {CFG['soft']['mu_max']}）")
            o(f"  ERLE：插話前 {f(s['erle_before_db'])} dB ｜ 插話時 {f(s['erle_during_db'])} dB（殘餘裡有我，這個數字不代表濾波器） ｜ "
              f"停下來之後（系統再出聲的頭 {CFG['measure']['after_s']:g} 秒）{f(s['erle_after_db'])} dB")
            rec = s.get("recover_s")
            o("  回到「插話前 − 3 dB」以內：" + ("這一段結束前沒有回來" if rec is None else f"停下來之後 {rec:.2f} 秒"))
        else:
            ref = self.st["variants"].get("alone:fixed", {}).get("summary", {}).get("erle_final_db")
            o(f"  ERLE 最後 {f(s['erle_final_db'])} dB（第一段、同一個濾波器：{f(ref)} dB）")
            if self.st["clips"]["loud"]["clipped_frac"] > 1e-4:
                o(f"  {C.AMBER}麥克風這一端也削波了（{100 * self.st['clips']['loud']['clipped_frac']:.2f}% 的樣本頂到滿刻度）——那也是一種非線性{C.RESET}")
            if "ceiling_db" in self.st:
                o(f"  模擬裡的失真是 tanh；g(x) 裡一個數乘 x 寫不出來的部分在 −{self.st['ceiling_db']:.1f} dB（粗估的上限）")
        if "misalignment_final_db" in s:
            o(f"  ‖h − ĥ‖：最後 {s['misalignment_final_db']:.1f} dB"
              + (f"；插話前 {s['misalignment_before_db']:.1f} → 停下來那一刻 {s['misalignment_after_db']:.1f} dB" if "misalignment_before_db" in s else ""))
        C.note("  y 播麥克風、e 播這一版的殘餘" + ("、a 只播我停下來前後那幾秒" if clip == "talk" else ""))
        self.redraw()

    def play_after(self) -> None:
        if self.mode != "aec" or self.clip != "talk":
            self.out("  a 只在 5／6／7 之後有用"); return
        dt = self.st["clips"]["talk"]["dt_s"]
        pre, post = CFG["display"]["after_pad_s"]
        self.play(f"talk_e_{self.rule}", max(dt[1] - float(pre), 0), dt[1] + float(post))

    def ensure_downstream(self) -> bool:
        if "downstream" in self.st and self.st["downstream"]:
            return True
        if self.models is None or self._wavs is None:
            self.out("  這份紀錄裡沒有 VAD／Whisper 的結果"); return False
        self.out("  載入 Silero VAD 與 Whisper small，四段各跑一次……")
        try:
            self.st["downstream"] = downstream(self.st, self._wavs[self.source], self.x, self.models, self.out)
        except Exception as e:  # 模型不在快取、套件沒裝：講清楚，不要讓整個 demo 掛掉
            self.out(f"  {C.RED}VAD／Whisper 跑不起來：{type(e).__name__}: {e}{C.RESET}")
            self.out("  先跑 ./setup.sh 與 ./present.sh fetch；現場就 ./present.sh replay")
            return False
        return True

    def panel_down(self, after: bool) -> None:
        if not self._need("talk") or not self.ensure_downstream():
            return
        o, d = self.out, self.st["downstream"]
        self.mode, self.clip, self.down_after = "down", "talk", self.down_after or after
        when = "after" if after else "before"
        r = d[f"talk:{when}"]
        var = CFG["downstream"]["variant"]
        o.banner(f"鍵 {'0' if after else '9'}：同一段錄音，AEC {'之後（' + RULE_ZH[var] + ' 的殘餘 e）' if after else '之前（麥克風 y）'} → Silero VAD、Whisper small")
        v = r["vad"]
        o(f"  VAD 判成語音的時間：整段 {100 * v['active_frac']:.0f}% ｜ 只有系統在說時 {100 * (v['active_frac_system_only'] or 0):.0f}% ｜ 我插話時 {100 * v.get('active_frac_user', 0):.0f}%")
        o(f"  VAD 的區間（秒）：" + ("、".join(f"{a:.1f}–{b:.1f}" for a, b in v["segments"]) or "沒有"))
        w = r["whisper"]
        o(f"  Whisper：{w['text'] or '（空的）'}")
        o(f"  系統那一句的 {w['system_chars']} 個字裡，有 {w['system_chars_found']} 個照順序出現在這份逐字稿裡（最長共同子序列；繁簡不同的字算不到）")
        a = d.get(f"alone:{when}")
        if a:
            o(f"  對照（第一段，沒有人插話）：VAD 亮 {100 * a['vad']['active_frac']:.0f}%；Whisper：{a['whisper']['text'] or '（空的）'}")
        self.redraw()

    def table(self) -> None:
        o, st = self.out, self.st
        o.banner(f"表（{SRC_ZH[self.source]}：{st.get('how', '')}）")
        if "step1" in st:
            s = st["step1"]
            o("  p19  從 y(n) 減掉什麼                          殘餘的位準（相對麥克風）")
            o("       什麼都不減                                 0 dB（定義）")
            o(f"       x(n)，照原樣                             {s['literal_db']:+6.1f} dB      （只調音量、不挪時間的最好結果 {s['gain_only_db']:+.2f} dB）")
            o(f"       g·x(n − D)，D = {s['delay_ms']:.1f} ms                 {s['aligned_db']:+6.2f} dB")
        o("")
        o("  p33  錄音與步長的規則                 ERLE 插話前   插話時   停下來之後   最後    回來要幾秒")
        for c, rule in VARIANTS:
            v = st["variants"].get(f"{c}:{rule}")
            if v is None:
                continue
            s = v["summary"]
            rec = s.get("recover_s")
            o(f"       {CLIP_ZH[c]:<9s} {RULE_ZH[rule]:<11s}     {f(s.get('erle_before_db')):>7s}  {f(s.get('erle_during_db')):>7s}  {f(s.get('erle_after_db')):>9s}  {f(s.get('erle_final_db')):>7s}    "
              + ("" if c != "talk" else "沒回來" if rec is None else f"{rec:.2f}")
              + (f"      （第 {f(s.get('converge_s'))} 秒收斂）" if c == "alone" else ""))
        d = st.get("downstream")
        if d:
            o("")
            o("  p34                      VAD 亮：只有系統在說／我插話時      Whisper")
            for when, zh in (("before", "AEC 之前"), ("after", "AEC 之後")):
                r = d.get(f"talk:{when}")
                if r:
                    o(f"       {zh}            {100 * (r['vad']['active_frac_system_only'] or 0):5.0f}% ／ {100 * r['vad'].get('active_frac_user', 0):3.0f}%                 {r['whisper']['text'] or '（空的）'}")

    def code(self) -> None:
        self.out.banner("aec.py 的 nlms()：迴圈裡的這幾行（(p23)、(p25) 是投影片上的兩條式子）")
        src = inspect.getsource(aec.nlms).splitlines()
        i0 = next(i for i, ln in enumerate(src) if ln.strip().startswith("for n in range(N):"))
        for ln in src[i0:]:
            if "mis is not None" in ln:
                break
            hot = "(p23)" in ln or "(p25)" in ln
            self.out((C.BOLD if hot else C.GREY) + ln + C.RESET)

    # 視窗
    def make_fig(self):
        import matplotlib.pyplot as plt
        self.plt = plt
        d = CFG["display"]
        self.fig = plt.figure(figsize=(d["fig_w"], d["fig_h"]))
        self.fig.canvas.mpl_connect("key_press_event", self.on_key)
        self.draw()
        return self.fig

    def redraw(self) -> None:
        if self.fig is not None:
            self.draw()
            self.fig.canvas.draw_idle()
            self.plt.pause(0.05)

    def _tag(self) -> str:
        return f"   [{self.source}{', ' + self.tag if self.tag else ''}]"

    def _shade(self, ax, clip: str) -> None:
        dt = self.st["clips"].get(clip, {}).get("dt_s")
        if dt:
            ax.axvspan(dt[0], dt[1], color="#7d5fb2", alpha=0.13, lw=0)

    def _wave(self, ax, name: str, title: str, color: str, ylim: float | None = None, back: str | None = None) -> None:
        v = self.arr(name)
        if v is None:
            ax.text(0.5, 0.5, "not available", ha="center", va="center", transform=ax.transAxes, color="#777777"); return
        step = 16
        if back is not None and self.arr(back) is not None:
            b = self.arr(back)
            ax.plot(np.arange(0, len(b), step) / FS, b[::step], color="#c9c9c9", lw=0.5)
        ax.plot(np.arange(0, len(v), step) / FS, v[::step], color=color, lw=0.5)
        if ylim:
            ax.set_ylim(-ylim, ylim)
        ax.set_xlim(0, len(v) / FS)
        ax.set_title(title, fontsize=10, loc="left")
        ax.tick_params(labelsize=8)

    def draw(self) -> None:
        fig = self.fig
        fig.clf()
        if self.mode == "step1":
            self._draw_step1()
        elif self.mode == "aec":
            self._draw_aec()
        elif self.mode == "down":
            self._draw_down()
        else:
            fig.text(0.5, 0.55, "W10 Demo 1 — the system hears itself", ha="center", fontsize=16)
            fig.text(0.5, 0.45, "1 2 3   subtract what we played        4   NLMS        5 6 7   double-talk        8   distortion        9 0   VAD and Whisper",
                     ha="center", fontsize=10, color="#555555")

    def _draw_step1(self) -> None:
        s = self.st["step1"]
        n = getattr(self, "step1_shown", 1)
        pk = self.st["clips"]["alone"]["y_peak"]
        lim = 1.05 * max(pk, float(np.max(np.abs(self.arr("alone_sub_literal")))) if n >= 2 else pk)
        rows = [("alone_y", "the microphone, y(n)   —   0 dB by definition", "#333333"),
                ("alone_sub_literal", f"y(n) − x(n), as played   —   {s['literal_db']:+.1f} dB re. the microphone", "#c0392b"),
                ("alone_sub_aligned", f"y(n) − g·x(n − D),  D = {s['delay_ms']:.1f} ms, best g   —   {s['aligned_db']:+.2f} dB", "#b9770e")]
        self.fig.suptitle("W10 Demo 1, step one — subtract what we played" + self._tag(), fontsize=12)
        for i, (name, title, col) in enumerate(rows):
            ax = self.fig.add_subplot(3, 1, i + 1)
            if i < n:
                self._wave(ax, name, title, col, lim, back="alone_y" if i else None)
            else:
                ax.set_title(f"press {i + 1}", fontsize=10, loc="left", color="#999999"); ax.set_xticks([]); ax.set_yticks([])
            if i == 2:
                ax.set_xlabel("time (s)", fontsize=9)
        self.fig.tight_layout(rect=(0, 0, 1, 0.95))

    def _draw_aec(self) -> None:
        c = self.clip
        has_mis = any("mis_db" in self.st["variants"][k] for k in self.shown)
        gs = self.fig.add_gridspec(4 if has_mis else 3, 1, height_ratios=[1.0, 1.7, 0.9, 1.0][:4 if has_mis else 3], hspace=0.42)
        self.fig.suptitle(f"W10 Demo 1 — {CLIP_LABEL[c]}: the echo canceller learns while it runs" + self._tag(), fontsize=12)
        ax = self.fig.add_subplot(gs[0])
        self._wave(ax, f"{c}_e_{self.rule}", f"grey: the microphone y(n)      colour: the residual e(n), {RULE_LABEL[self.rule]}", RULE_COLOR[self.rule],
                   1.05 * self.st["clips"][c]["y_peak"], back=f"{c}_y")
        self._shade(ax, c)
        ax2 = self.fig.add_subplot(gs[1], sharex=ax)
        for k in self.shown:
            v, rule = self.st["variants"][k], k.split(":")[1]
            cur = rule == self.rule
            y = np.array([np.nan if u is None else u for u in v["erle_db"]], dtype=float)
            ax2.plot(v["t"], y, color=RULE_COLOR[rule], lw=2.0 if cur else 1.1, alpha=1.0 if cur else 0.55, label=RULE_LABEL[rule])
        self._shade(ax2, c)
        ax2.axhline(0, color="#999999", lw=0.6)
        ax2.set_ylabel("ERLE (dB)", fontsize=9); ax2.grid(alpha=0.3); ax2.tick_params(labelsize=8)
        ttl = f"ERLE over the past {CFG['measure']['erle_win_ms']:g} ms, where the system is audible"
        if c == "talk":
            ttl += "   ·   shaded: the lecturer is talking (the number there says nothing about the filter)"
        ax2.set_title(ttl, fontsize=10, loc="left")
        if len(self.shown) > 1:
            ax2.legend(fontsize=8, loc="lower right", ncol=3)
        ax3 = self.fig.add_subplot(gs[2], sharex=ax)
        for k in self.shown:
            v, rule = self.st["variants"][k], k.split(":")[1]
            ax3.plot(v["mu_t"], v["mu"], color=RULE_COLOR[rule], lw=1.2 if rule == self.rule else 0.7, alpha=1.0 if rule == self.rule else 0.45)
        self._shade(ax3, c)
        ax3.set_ylim(-0.03, max(0.6, 1.15 * float(self.st["mu"])))
        ax3.set_ylabel("step size μ", fontsize=9); ax3.grid(alpha=0.3); ax3.tick_params(labelsize=8)
        last = ax3
        if has_mis:
            ax4 = self.fig.add_subplot(gs[3], sharex=ax)
            for k in self.shown:
                v, rule = self.st["variants"][k], k.split(":")[1]
                if "mis_db" in v:
                    ax4.plot(v["mis_t"], v["mis_db"], color=RULE_COLOR[rule], lw=1.6 if rule == self.rule else 0.9, alpha=1.0 if rule == self.rule else 0.55)
            self._shade(ax4, c)
            ax4.set_ylabel("‖h − ĥ‖² / ‖h‖² (dB)", fontsize=9); ax4.grid(alpha=0.3); ax4.tick_params(labelsize=8)
            ax4.set_title("simulation only: distance to the true path (lower is better)", fontsize=10, loc="left")
            last = ax4
        last.set_xlabel("time (s)", fontsize=9)

    def _draw_down(self) -> None:
        d = self.st.get("downstream") or {}
        var = CFG["downstream"]["variant"]
        cols = [("before", "talk_y", "before AEC: the raw microphone y(n)", "#333333")]
        if self.down_after:
            cols.append(("after", f"talk_e_{var}", f"after AEC: the residual e(n), {RULE_LABEL[var]}", RULE_COLOR[var]))
        self.fig.suptitle("W10 Demo 1 — what the next model hears: Silero VAD and Whisper small" + self._tag(), fontsize=12)
        gs = self.fig.add_gridspec(3, 2, height_ratios=[1.0, 1.0, 0.9], hspace=0.5, wspace=0.12)
        lim = 1.05 * self.st["clips"]["talk"]["y_peak"]
        for j, (when, name, title, col) in enumerate(cols):
            r = d.get(f"talk:{when}")
            ax = self.fig.add_subplot(gs[0, j])
            self._wave(ax, name, title, col, lim)
            self._shade(ax, "talk")
            ax2 = self.fig.add_subplot(gs[1, j], sharex=ax)
            if r:
                v = r["vad"]
                for a, b in v["segments"]:
                    ax2.axvspan(a, b, color="#2e86c1", alpha=0.22, lw=0)
                ax2.plot(v["p_t"], v["p"], color="#1b4f72", lw=1.0)
                ax2.axhline(float(CFG["downstream"]["vad_threshold"]), color="#999999", lw=0.6, ls="--")
                ax2.set_title(f"Silero VAD, P(speech). Blue: it says speech — {100 * (v['active_frac_system_only'] or 0):.0f}% of the system-only time",
                              fontsize=9, loc="left")
            self._shade(ax2, "talk")
            ax2.set_ylim(-0.05, 1.05); ax2.tick_params(labelsize=8); ax2.set_xlabel("time (s)", fontsize=9)
            ax3 = self.fig.add_subplot(gs[2, j])
            ax3.axis("off")
            if r:
                txt = r["whisper"]["text"] or "(nothing)"
                wrapped = "\n".join(txt[i:i + 30] for i in range(0, len(txt), 30))
                ax3.text(0.0, 1.0, "Whisper small writes:", fontsize=9, color="#555555", va="top", transform=ax3.transAxes)
                ax3.text(0.0, 0.8, wrapped, fontsize=11, va="top", transform=ax3.transAxes, fontfamily=["PingFang TC", "Heiti TC", "Noto Sans CJK TC", "sans-serif"])
        if not self.down_after:
            ax = self.fig.add_subplot(gs[:, 1]); ax.axis("off")
            ax.text(0.5, 0.5, "press 0: the same recording, after echo cancellation", ha="center", va="center", fontsize=11, color="#999999", transform=ax.transAxes)

    # 鍵
    def menu(self) -> None:
        o = self.out
        o(f"\n  來源：{SRC_ZH[self.source]}（{'、'.join(SRC_ZH[s] for s in self.stores)} 可用；m 互換）")
        o("  1 麥克風 y   2 y − x   3 對齊再減   |   4 NLMS 從零   5 插話＋固定步長   6 ＋Geigel   7 ＋軟步長   8 失真的天花板")
        o("  9 VAD 與 Whisper（AEC 之前）   0（之後）   |   y 播麥克風   e 播殘餘   a 播我停下來前後   c 程式碼   t 表   m 換來源   h 選單   q 離開")

    def dispatch(self, k: str | None) -> bool:
        if k in ("1", "2", "3"): self.panel_step1(k)
        elif k in KEY_VARIANT: self.panel_variant(*KEY_VARIANT[k])
        elif k == "9": self.panel_down(False)
        elif k == "0": self.panel_down(True)
        elif k == "y": self.play(f"{self.clip}_y")
        elif k == "e":
            if self.mode == "aec": self.play(f"{self.clip}_e_{self.rule}")
            elif self.mode == "down": self.play(f"talk_e_{CFG['downstream']['variant']}")
            else: self.out("  e 在 4–8 或 9／0 之後有用")
        elif k == "a": self.play_after()
        elif k == "c": self.code()
        elif k == "t": self.table()
        elif k == "m":
            others = [s for s in self.stores if s != self.source]
            if others:
                self.source, self.shown, self.mode, self.step1_shown, self.down_after = others[0], [], "idle", 0, False
                self.out(f"  來源 → {SRC_ZH[self.source]}（{self.st.get('how', '')}）"); self.redraw()
            else:
                self.out("  只有一個來源")
        elif k == "h": self.menu()
        elif k == "q": return False
        return True

    def on_key(self, ev) -> None:
        if not self.dispatch(ev.key):
            self.plt.close(self.fig)

    # 彩排與存檔
    def run_all(self) -> None:
        C.play = lambda *a, **k: None
        import matplotlib.pyplot as plt
        self.plt = plt
        d = CFG["display"]
        for src in list(self.stores):
            self.source, self.shown, self.mode, self.step1_shown, self.down_after = src, [], "idle", 0, False
            self.out.banner(f"══ {SRC_ZH[src]}：{self.st.get('how', '')} ══", C.AMBER)
            jobs = [("1", None), ("2", None), ("3", "step1"), ("4", "alone"), ("5", None), ("6", None), ("7", "talk"), ("8", "loud"), ("9", None), ("0", "downstream")]
            for k, png in jobs:
                self.fig = None
                self.dispatch(k)
                if png and self.mode != "idle":
                    self.fig = plt.figure(figsize=(d["fig_w"], d["fig_h"]))
                    self.draw()
                    self.fig.savefig(self.wavdir / f"{src}_{png}.png", dpi=100)
                    plt.close(self.fig)
                    self.fig = None
            self.table()

    def versions(self) -> dict:
        v = C.versions()
        from importlib import metadata
        for pkg in ("torch", "transformers", "silero-vad", "pyroomacoustics", "onnxruntime"):
            try:
                v[pkg] = metadata.version(pkg)
            except Exception:
                v[pkg] = None
        v["whisper"] = MODEL_LOCK.read_text(encoding="utf-8").strip() if MODEL_LOCK.exists() else None
        return v

    def save(self, stem: str) -> None:
        self.wavdir.mkdir(parents=True, exist_ok=True)
        log = {"time": C.now(), "tag": self.tag, "versions": self.versions(), "config": CFG, "fs": FS, "sources": self.stores}
        (self.wavdir / f"{stem}.json").write_text(json.dumps(log, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
        (self.wavdir / f"{stem}.txt").write_text(self.out.text(), encoding="utf-8")
        C.note(f"  存在 {self.wavdir}/（{stem}.json、{stem}.txt、*.wav）")


def write_wavs(folder: Path, stores: dict, wavs: dict) -> None:
    """每個來源的每一段存成 16-bit WAV；同一段錄音的 y 與各版殘餘用同一個增益（不然響度差會蓋過差別）。增益記在 store["wavs"]。"""
    folder.mkdir(parents=True, exist_ok=True)
    for src, ws in wavs.items():
        stores[src]["wavs"] = {}
        for name, (v, pk) in ws.items():
            g = 0.89 / pk
            stores[src]["wavs"][name] = g
            C.save_wav(folder / f"{src}_{name}.wav", v, FS, peak=pk)


# ── 錄音 ─────────────────────────────────────────────────────────────────────
def playrec(x: np.ndarray, cue: tuple[float, float] | None = None, say: str = "") -> tuple[np.ndarray, dict]:
    """同步播錄：播 x（16 kHz → 裝置的取樣率）、同時錄內建麥克風，回傳 16 kHz 的 y 與裝置資訊。"""
    import sounddevice as sd
    dfs = int(CFG["record"]["device_fs"])
    out = dsp.resample(x, FS, dfs).astype(np.float32)
    dev_in, dev_out = sd.query_devices(kind="input"), sd.query_devices(kind="output")
    ch = 2 if int(dev_out["max_output_channels"]) >= 2 else 1
    rec = sd.playrec(np.repeat(out[:, None], ch, axis=1), samplerate=dfs, channels=1, dtype="float32")
    t0, lead = time.perf_counter(), float(CFG["reference"]["lead_s"])
    state = ""
    while sd.get_stream().active:
        t = time.perf_counter() - t0 - lead
        if cue is not None:
            new = "wait" if t < cue[0] - 3 else f"{int(np.ceil(cue[0] - t))}" if t < cue[0] else "GO" if t < cue[1] else "stop"
            if new != state:
                state = new
                msg = {"wait": "  （系統在說話……）", "GO": f"{C.BOLD}{C.GREEN}  ▶▶ 現在說：{say}{C.RESET}", "stop": f"{C.BOLD}{C.RED}  ■ 停{C.RESET}"}.get(new, f"  {new} …")
                print(msg, flush=True)
        time.sleep(0.02)
    sd.wait()
    y = dsp.resample(rec[:, 0].astype(np.float64), dfs, FS)
    y = np.concatenate([y, np.zeros(max(len(x) - len(y), 0))])[:len(x)]
    return y, {"input": dev_in["name"], "output": dev_out["name"], "device_fs": dfs, "output_channels": ch,
               "input_default_fs": dev_in["default_samplerate"], "output_default_fs": dev_out["default_samplerate"]}


def record(which: str) -> None:
    rc = CFG["record"]
    AUDIO.mkdir(exist_ok=True)
    if which == "user":
        C.banner(f"錄一句乾淨的插話（{CFG['sim']['near_len_s']:g} 秒，喇叭不出聲）→ audio/user.wav：模擬裡的「使用者」", C.AMBER)
        print(f"  建議的句子：{rc['suggested']}")
        v = C.record(float(CFG["sim"]["near_len_s"]), FS)
        C.save_wav(AUDIO / "user.wav", v, FS)
        print(f"  → audio/user.wav（RMS {np.sqrt(np.mean(v ** 2)):.4f}；不進版控、不上站）")
        return
    x = reference()
    names = CLIPS if which == "all" else [which]
    for c in names:
        C.banner(f"錄「{CLIP_ZH[c]}」→ audio/real_{c}.wav（{len(x) / FS:.1f} 秒）", C.AMBER)
        play = x
        drive = 0.0
        if c == "alone":
            print("  系統音量調到平常說話的大小（教室後排聽得到）。錄的時候不要出聲、不要動筆電。按 Enter 開始。")
        elif c == "talk":
            print(f"  音量、筆電的位置都跟第一段一樣。系統說到第 {rc['cue_start_s']:g} 秒時畫面會叫你說，第 {rc['cue_stop_s']:g} 秒叫你停。")
            print(f"  建議的句子：{rc['suggested']}    按 Enter 開始。")
        else:
            drive = float(rc["software_drive"])
            if drive > 0:
                play = aec.soft_clip(x, drive)
                print(f"  software_drive = {drive:g}：播出去的是 tanh 削過的 g(x)，參考訊號仍是 x。音量不必動。按 Enter 開始。")
            else:
                print(f"  {C.BOLD}把系統音量開到最大{C.RESET}（會很大聲；錄完記得調回來）。不要出聲。按 Enter 開始。")
        try:
            input()
        except EOFError:
            pass
        y, dev = playrec(play, (float(rc["cue_start_s"]), float(rc["cue_stop_s"])) if c == "talk" else None, rc["suggested"])
        D = aec.first_arrival(x, y, int(float(rc["max_delay_s"]) * FS))
        act = aec.activity(aec.shift(x, D), FS, float(CFG["measure"]["x_floor_db"]))
        lead = int(float(CFG["reference"]["lead_s"]) * FS)
        floor = float(np.sqrt(np.mean(y[:lead - 400] ** 2)) + 1e-12) if lead > 1200 else float("nan")   # 開頭那段靜音（系統還沒出聲）
        meta = {"time": C.now(), "clip": c, "x_sha": x_sha(x), "delay_samples": D, "delay_ms": 1000.0 * D / FS, "peak": float(np.max(np.abs(y))),
                "clipped_frac": float(np.mean(np.abs(y) > 0.985)), "echo_rms_db": float(20 * np.log10(np.sqrt(np.mean(y[act] ** 2)) + 1e-12)),
                "floor_rms_db": float(20 * np.log10(floor)), "software_drive": drive, "device": dev, "platform": C.versions()["platform"]}
        C.save_wav(AUDIO / f"real_{c}.wav", y, FS)
        (AUDIO / f"real_{c}.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"  → audio/real_{c}.wav   延遲 {meta['delay_ms']:.1f} ms、峰值 {meta['peak']:.3f}、回音 {meta['echo_rms_db']:.1f} dBFS、開頭的底噪 {meta['floor_rms_db']:.1f} dBFS")
        if meta["peak"] < 0.02:
            print(f"  {C.RED}幾乎沒錄到東西：麥克風權限（系統設定 → 隱私權 → 麥克風 → 這個終端機）或輸入裝置不對（./present.sh check）{C.RESET}")
        if meta["clipped_frac"] > 1e-4 and c != "loud":
            print(f"  {C.AMBER}麥克風削波了（{100 * meta['clipped_frac']:.2f}%）：把系統音量或輸入音量調小再錄一次{C.RESET}")
    print("\n下一步：三段都錄好之後 ./present.sh rehearse；想聽：afplay audio/real_talk.wav")


# ── check ────────────────────────────────────────────────────────────────────
def check(no_audio: bool = False) -> None:
    C.banner("課前體檢", C.AMBER)
    res: dict = {"time": C.now(), "versions": Demo(Out(), {"sim": {}}, RUNS).versions()}
    print("  版本：" + "、".join(f"{k} {v}" for k, v in res["versions"].items() if k not in ("platform",)))
    # 1. NLMS 的速度
    rng = np.random.default_rng(0)
    L_h = int(CFG["nlms"]["L_h"])
    xx = rng.standard_normal(FS)
    t0 = time.perf_counter()
    aec.nlms(xx, xx, L_h, 0.5, 1.0)
    res["nlms_realtime_factor"] = time.perf_counter() - t0
    print(f"  NLMS（L_h = {L_h}，numpy 逐樣本）：1 秒的訊號要 {res['nlms_realtime_factor']:.2f} 秒" + ("（比即時慢——課堂上是先算好再看，不是邊錄邊算）" if res["nlms_realtime_factor"] > 1 else ""))
    # 2. 音訊裝置與播錄
    if no_audio:
        print("  --no-audio：跳過音訊裝置")
    else:
        try:
            import sounddevice as sd
            di, do = sd.query_devices(kind="input"), sd.query_devices(kind="output")
            res["devices"] = {"input": di["name"], "output": do["name"], "input_fs": di["default_samplerate"], "output_fs": do["default_samplerate"],
                              "input_channels": di["max_input_channels"], "output_channels": do["max_output_channels"]}
            print(f"  輸入：{di['name']}（{di['default_samplerate']:g} Hz、{di['max_input_channels']} ch）   輸出：{do['name']}（{do['default_samplerate']:g} Hz、{do['max_output_channels']} ch）")
            if "airpods" in (di["name"] + do["name"]).lower() or "bluetooth" in (di["name"] + do["name"]).lower():
                print(f"  {C.RED}輸入或輸出是藍牙裝置：這個 demo 要內建喇叭與內建麥克風{C.RESET}")
            print("  播三次 3 秒的噪音（小聲），量延遲與線性路徑。不要出聲。")
            probe = np.concatenate([np.zeros(FS // 2), 0.2 * np.clip(rng.standard_normal(3 * FS) / 3, -1, 1), np.zeros(FS // 2)])
            trials = []
            for i in range(3):
                y, _ = playrec(probe)
                D = aec.first_arrival(probe, y, int(float(CFG["record"]["max_delay_s"]) * FS))
                h1, h2 = len(probe) // 2, len(probe)
                D1 = aec.estimate_delay(probe[:h1], y[:h1], int(0.5 * FS))
                D2 = aec.estimate_delay(probe[h1:h2], y[h1:h2], int(0.5 * FS))
                x_al = aec.shift(probe, max(D - int(CFG["nlms"]["bulk_margin"]), 0))
                r = aec.nlms(x_al, y, L_h, 0.5, float(CFG["nlms"]["delta_rel"]) * L_h * float(np.mean(probe ** 2)))
                act = np.abs(x_al) > 0
                enr = 10 * np.log10(np.mean(y[act] ** 2) / (np.mean(y[:FS // 4] ** 2) + 1e-12))
                tr = {"delay_samples": D, "delay_ms": 1000.0 * D / FS, "delay_first_half": D1, "delay_second_half": D2, "echo_to_floor_db": float(enr),
                      "erle_last_second_db": aec.erle_span(y, r["e"], len(y) - FS - FS // 2, len(y) - FS // 2, act), "peak": float(np.max(np.abs(y)))}
                trials.append(tr)
                print(f"    第 {i + 1} 次：延遲 {tr['delay_ms']:.2f} ms（前半 {D1}、後半 {D2} 個樣本）   回音比底噪高 {enr:.1f} dB   NLMS 最後一秒的 ERLE {f(tr['erle_last_second_db'])} dB")
            res["playrec"] = trials
            ds = [t["delay_samples"] for t in trials]
            res["delay_spread_samples"] = int(max(ds) - min(ds))
            res["delay_drift_samples"] = int(max(abs(t["delay_first_half"] - t["delay_second_half"]) for t in trials))
            print(f"  三次延遲差 {res['delay_spread_samples']} 個樣本；同一次前後半差最多 {res['delay_drift_samples']} 個樣本"
                  + ("" if res["delay_drift_samples"] <= 1 else f"  {C.AMBER}← 播與錄的時脈在漂；線性濾波器的 ERLE 會被它壓住（那是 p30 的內容，照實講）{C.RESET}"))
            worst = min(t["echo_to_floor_db"] for t in trials)
            best = max((t["erle_last_second_db"] or -99) for t in trials)
            res["verdict"] = ("no echo reached the microphone" if worst < 6 else "echo present, linear path learnable" if best > 15 else "echo present, but a linear filter removes little")
            msg = {"no echo reached the microphone": f"{C.RED}麥克風裡幾乎沒有回音：音量太小、輸出不是內建喇叭、或輸入路徑上有系統自帶的回音消除（控制中心 → 麥克風模式要是「標準」）{C.RESET}",
                   "echo present, linear path learnable": f"{C.GREEN}麥克風裡有回音，而且線性濾波器學得起來——輸入路徑上沒有看到會妨礙 demo 的處理{C.RESET}",
                   "echo present, but a linear filter removes little": f"{C.AMBER}有回音，但 3 秒內線性濾波器扣得很少：可能是時脈在漂、音量太大已經失真、或輸入路徑上有非線性的處理。先把音量調小再跑一次{C.RESET}"}
            print("  " + msg[res["verdict"]])
        except Exception as e:
            res["audio_error"] = f"{type(e).__name__}: {e}"
            print(f"  {C.RED}音訊裝置：{res['audio_error']}{C.RESET}")
    # 3. 模型與套件
    m = Models()
    try:
        t0 = time.perf_counter()
        probs, segs = m.run_vad(reference())
        res["silero"] = {"seconds_for_reference": time.perf_counter() - t0, "active_frac_on_x": float(np.mean(probs > 0.5)), "segments": len(segs)}
        print(f"  Silero VAD：載得起；x（{len(reference()) / FS:.0f} 秒）跑了 {res['silero']['seconds_for_reference']:.2f} 秒，{100 * res['silero']['active_frac_on_x']:.0f}% 的格子過 0.5、{len(segs)} 段")
    except Exception as e:
        res["silero_error"] = f"{type(e).__name__}: {e}"
        print(f"  {C.RED}Silero VAD：{res['silero_error']}（./setup.sh）{C.RESET}")
    try:
        t0 = time.perf_counter()
        text = m.run_whisper(reference())
        hyp, ref = only_text(text), only_text(CFG["reference"]["text"])
        res["whisper"] = {"seconds_for_reference": time.perf_counter() - t0, "text_on_x": text, "system_chars_found": lcs(ref, hyp), "system_chars": len(ref), "device": str(m.wh.device)}
        print(f"  Whisper small（{m.wh.device}）：乾淨的 x → {text!r}（{res['whisper']['seconds_for_reference']:.1f} 秒；系統那一句 {len(ref)} 個字對到 {res['whisper']['system_chars_found']} 個）")
    except Exception as e:
        res["whisper_error"] = f"{type(e).__name__}: {e}"
        print(f"  {C.RED}Whisper：{res['whisper_error']}（./present.sh fetch）{C.RESET}")
    try:
        _, _, how = room_rirs()
        res["sim"] = how
        print(f"  模擬的路徑：{how}")
    except Exception as e:
        res["sim_error"] = f"{type(e).__name__}: {e}"
        print(f"  {C.RED}pyroomacoustics：{res['sim_error']}{C.RESET}")
    res["real_clips"] = [c for c in CLIPS if (AUDIO / f"real_{c}.wav").exists()]
    print(f"  真錄音：{'、'.join(res['real_clips']) or '還沒有'}（./present.sh record alone｜talk｜loud）")
    RUNS.mkdir(exist_ok=True)
    (RUNS / "check.json").write_text(json.dumps(res, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    print("\n→ runs/check.json。下一步：./present.sh record alone → record talk → record loud → ./present.sh rehearse")


def fetch() -> None:
    from huggingface_hub import HfApi, snapshot_download
    C.banner("下載 Whisper small（W7 Demo 1 的同一個 commit）", C.AMBER)
    repo, sha = "openai/whisper-small", None
    for lk in (W7 / "model.lock", HERE.parent / "w04_d3_attention_heads" / "model.lock"):
        if lk.exists():
            for line in lk.read_text(encoding="utf-8").splitlines():
                parts = line.split()
                if len(parts) == 2 and parts[0] == repo:
                    sha = parts[1]
                elif len(parts) == 1 and lk.parent.name.startswith("w04"):
                    sha = sha or parts[0]
            if sha:
                break
    sha = sha or HfApi().model_info(repo).sha
    path = snapshot_download(repo, revision=sha, allow_patterns=["*.json", "*.txt", "*.safetensors"])
    MODEL_LOCK.write_text(f"{repo} {sha}\n", encoding="utf-8")
    print(f"  {repo} @ {sha[:12]} → {path}\n  commit 寫進 model.lock。Silero VAD 的權重在 silero-vad 的 pip 套件裡（./setup.sh 裝），不必下載。")
    print("\n下一步：./present.sh check")


# ── export ───────────────────────────────────────────────────────────────────
def export() -> None:
    src = REHEARSAL / "rehearsal.json"
    if not src.exists():
        sys.exit("沒有 runs/rehearsal/rehearsal.json，先跑 ./present.sh rehearse")
    log = json.loads(src.read_text(encoding="utf-8"))
    data = json.loads(METRICS.read_text(encoding="utf-8")) if METRICS.exists() else {}
    d1: dict = {"source": "demos/w10_d1_echo_nlms/runs/rehearsal/rehearsal.json", "time": log["time"], "fs": FS,
                "reference": {"file": CFG["reference"]["file"], "text": CFG["reference"]["text"]},
                "nlms": CFG["nlms"], "geigel": CFG["geigel"], "soft": CFG["soft"], "measure": CFG["measure"],
                "versions": {k: log["versions"].get(k) for k in ("python", "numpy", "scipy", "sounddevice", "torch", "transformers", "silero-vad", "pyroomacoustics", "whisper")}}
    for name, st in log["sources"].items():
        e: dict = {"how": st.get("how"), "L_h": st["L_h"], "mu": st["mu"], "delta": st["delta"], "geigel_theta": st.get("geigel_theta"),
                   "clips": st["clips"], "step1": st.get("step1"),
                   "variants": {k: {**v["summary"], **{kk: v[kk] for kk in ("frozen_frac_user", "frozen_frac_system_only") if kk in v}} for k, v in st["variants"].items()}}
        if "ceiling_db" in st:
            e["tanh_ceiling_db"] = st["ceiling_db"]
        if st.get("recorded"):
            e["recorded"] = st["recorded"]
        if st.get("downstream"):
            e["downstream"] = {k: {"signal": r["signal"], "vad_active_frac": r["vad"]["active_frac"], "vad_active_frac_system_only": r["vad"]["active_frac_system_only"],
                                   "vad_active_frac_user": r["vad"].get("active_frac_user"), "vad_segments_s": r["vad"]["segments"],
                                   "whisper_text": r["whisper"]["text"], "system_chars_found": r["whisper"]["system_chars_found"], "system_chars": r["whisper"]["system_chars"]}
                               for k, r in st["downstream"].items()}
        d1[name] = e
    data["demo1"] = d1
    METRICS.parent.mkdir(parents=True, exist_ok=True)
    METRICS.write_text(json.dumps(data, ensure_ascii=False, indent=1, default=float) + "\n", encoding="utf-8")
    print(f"→ {METRICS}（demo1：{'、'.join(log['sources'])}）")


# ── selftest ─────────────────────────────────────────────────────────────────
def selftest() -> None:
    C.banner("selftest（不需要模型、麥克風、pyroomacoustics）", C.AMBER)
    bad = 0

    def chk(name: str, cond: bool, extra: str = "") -> None:
        nonlocal bad
        bad += 0 if cond else 1
        print(f"  {C.GREEN + '✓' if cond else C.RED + '✗'}{C.RESET} {name}" + (f"   {C.GREY}{extra}{C.RESET}" if extra else ""))

    rng = np.random.default_rng(7)
    L = 32
    h = np.zeros(L); h[4:] = rng.standard_normal(L - 4) * np.exp(-np.arange(L - 4) / 7.0); h /= np.linalg.norm(h)
    N = 6000
    x = rng.standard_normal(N)
    echo = np.convolve(x, h)[:N]
    v = 0.01 * rng.standard_normal(N)
    # 1. 一步的恆等式：e⁺ = (1 − μ) e
    for mu in (0.5, 1.0, 1.5):
        hh = rng.standard_normal(L) * 0.1
        xv = x[100:100 + L][::-1]
        y0 = float(h @ xv)
        e0 = y0 - hh @ xv
        h1 = hh + mu * e0 * xv / (xv @ xv)
        chk(f"e⁺ = (1 − μ) e，μ = {mu}", abs((y0 - h1 @ xv) - (1 - mu) * e0) < 1e-9)
    # 2. 收斂到已知的路徑
    r = aec.nlms(x, echo + v, L, 0.5, 1e-3, h_true=h)
    mis = 10 * np.log10(r["misalignment"][-1])
    chk("白噪輸入：‖h − ĥ‖ 掉到 −35 dB 以下", mis < -35, f"{mis:.1f} dB")
    chk("ERLE（最後 1000 個樣本）> 30 dB", aec.erle_span(echo + v, r["e"], N - 1000, N) > 30, f"{aec.erle_span(echo + v, r['e'], N - 1000, N):.1f} dB")
    rc = aec.nlms(aec.colour(x, 0.95), np.convolve(aec.colour(x, 0.95), h)[:N] + v, L, 0.5, 1e-3, h_true=h)
    chk("有色輸入收斂得比白噪慢", rc["misalignment"][len(rc["misalignment"]) // 4] > r["misalignment"][len(r["misalignment"]) // 4])
    chk("μ = 2.5 發散", not np.isfinite(aec.nlms(x, echo + v, L, 2.5, 1e-3)["e"][-1]) or abs(aec.nlms(x, echo + v, L, 2.5, 1e-3)["e"][-1]) > 1e3)
    # 3. 直接減
    D = 37
    y = np.convolve(aec.shift(x, D), h)[:N] + v
    chk("estimate_delay 找回延遲（直達聲在第 4 個數之後的主峰）", abs(aec.estimate_delay(x, y, 200) - (D + int(np.argmax(np.abs(h))))) == 0)
    fa = aec.first_arrival(x, y, 200)
    chk("first_arrival 落在直達聲（第一個非零的數）到主峰之間", D + 4 <= fa <= D + int(np.argmax(np.abs(h))), f"{fa}（直達 {D + 4}、主峰 {D + int(np.argmax(np.abs(h)))}）")
    ysp = np.convolve(aec.shift(aec.colour(x, 0.98), D), np.r_[h, np.zeros(60), 0.9 * h])[:N] + v
    chk("有色輸入加一個強反射：first_arrival 仍在第一條路徑上", D + 4 <= aec.first_arrival(aec.colour(x, 0.98), ysp, 400) <= D + int(np.argmax(np.abs(h))))
    chk("y − x 沒有變小", aec.level_db(y - x, y) > -1.0, f"{aec.level_db(y - x, y):+.1f} dB")
    g = aec.best_gain(aec.shift(x, D + int(np.argmax(np.abs(h)))), y)
    chk("對齊加最好的音量：有變小，但扣不到 6 dB（差的是整條路徑）", -6 < aec.level_db(y - g * aec.shift(x, D + int(np.argmax(np.abs(h)))), y) < -0.2)
    # 4. double-talk：固定、凍結、軟
    dt = (2500, 3500)
    near = v.copy(); near[dt[0]:dt[1]] += rng.standard_normal(1000)
    yd = echo + near
    rf = aec.nlms(x, yd, L, 0.5, 1e-3, h_true=h)
    m_b, m_a = 10 * np.log10(rf["misalignment"][dt[0] // 64]), 10 * np.log10(rf["misalignment"][dt[1] // 64])
    chk("固定步長：double-talk 把 ĥ 帶偏 20 dB 以上", m_a - m_b > 20, f"{m_b:.1f} → {m_a:.1f} dB")
    q = aec.geigel_ratio(x, echo + v, L)
    near4 = v.copy(); near4[dt[0]:dt[1]] += 4.0 * rng.standard_normal(1000)      # Geigel 要近端明顯比回音大聲才看得到：這裡大 12 dB
    fz = aec.geigel(x, echo + near4, L, 1.5 * np.percentile(q, 99.9), int(0.02 * FS))
    rg = aec.nlms(x, echo + near4, L, 0.5, 1e-3, rule="freeze", frozen=fz, h_true=h)
    r4 = aec.nlms(x, echo + near4, L, 0.5, 1e-3, h_true=h)
    m_g, m_4 = 10 * np.log10(rg["misalignment"][dt[1] // 64]), 10 * np.log10(r4["misalignment"][dt[1] // 64])
    chk("Geigel（近端大 12 dB）：區間內大半凍結、之前幾乎不凍結、帶偏得比固定步長少", fz[dt[0]:dt[1]].mean() > 0.6 and fz[:dt[0]].mean() < 0.02 and m_g < m_4 - 6,
        f"停下來時 {m_g:.1f} dB（固定步長 {m_4:.1f}）；凍結 {100 * fz[dt[0]:dt[1]].mean():.0f}%（區間內）／{100 * fz[:dt[0]].mean():.1f}%（之前）")
    rs = aec.nlms(x, yd, L, 0.5, 1e-3, rule="soft", soft={"mu_max": 0.5, "tau_e": 0.005, "tau_lambda": 0.05, "tau_up": 0.5}, fs=FS, h_true=h)
    m_s = 10 * np.log10(rs["misalignment"][dt[1] // 64])
    chk("軟步長：插話時 μ 自己縮到插話前的五分之一以下", rs["mu"][dt[0] + 100:dt[1]].mean() < 0.2 * rs["mu"][dt[0] - 500:dt[0]].mean(),
        f"{rs['mu'][dt[0] - 500:dt[0]].mean():.3f} → {rs['mu'][dt[0] + 100:dt[1]].mean():.3f}")
    chk("軟步長：帶偏得比固定步長少", m_s < m_a - 6, f"停下來時 {m_s:.1f} dB")
    # 5. 小工具
    chk("longest_run／close_gaps", aec.longest_run(aec.close_gaps(np.array([0, 1, 1, 0, 1, 1, 1, 0, 0, 0, 1], bool), 2)) == (1, 7))
    chk("soft_clip：drive 小 → 幾乎線性；drive 大 → 天花板變低", aec.linear_ceiling_db(x, aec.soft_clip(x, 0.1)) > aec.linear_ceiling_db(x, aec.soft_clip(x, 3.0)) + 20)
    chk("lcs", lcs("已幫您訂好票", "幫您訂票了") == 4)
    # 6. 整條流程（小的合成路徑、短的訊號）→ 存檔 → 讀回 → 畫圖
    import matplotlib
    matplotlib.use("Agg")
    global CFG
    keep = json.loads(json.dumps(CFG))
    CFG["nlms"].update(L_h=128, bulk_margin=8)
    CFG["sim"].update(near_start_s=3.0, near_len_s=1.5)
    CFG["reference"].update(lead_s=0.2, tail_s=0.3)
    try:
        n = 7 * FS
        xs = np.concatenate([np.zeros(int(0.2 * FS)), 0.3 * aec.colour(rng.standard_normal(n), 0.8) * (1 + 0.8 * np.sin(2 * np.pi * 3 * np.arange(n) / FS)), np.zeros(int(0.3 * FS))])
        rirs = (synthetic_rir(100, 10, 0.02, 1), synthetic_rir(100, 20, 0.02, 2), "selftest path")
        sim = make_sim(xs, rirs, near=aec.colour(rng.standard_normal(int(1.5 * FS)), 0.9))
        global CACHE
        keep_cache, CACHE = CACHE, RUNS / "_selftest" / "cache"
        out = Out()
        stores, wavs = build_sources(out, want_real=False, sim_override=sim, x=xs)
        folder = RUNS / "_selftest"
        write_wavs(folder, stores, wavs)
        st = stores["sim"]
        chk("流程：五個版本都算出來", len(st["variants"]) == 5)
        sv = {k: v["summary"] for k, v in st["variants"].items()}
        chk("流程：固定步長在停下來之後掉、軟步長守住", sv["talk:fixed"]["erle_after_db"] < sv["talk:fixed"]["erle_before_db"] - 6 and sv["talk:soft"]["erle_after_db"] > sv["talk:fixed"]["erle_after_db"] + 3,
            f"fixed {sv['talk:fixed']['erle_before_db']:.1f} → {sv['talk:fixed']['erle_after_db']:.1f}；soft {sv['talk:soft']['erle_before_db']:.1f} → {sv['talk:soft']['erle_after_db']:.1f}")
        chk("流程：失真那一段的 ERLE 比第一段低", sv["loud:fixed"]["erle_final_db"] < sv["alone:fixed"]["erle_final_db"] - 3,
            f"{sv['alone:fixed']['erle_final_db']:.1f} 對 {sv['loud:fixed']['erle_final_db']:.1f} dB")
        chk("流程：y − x 沒有變小", st["step1"]["literal_db"] > -1 and st["step1"]["aligned_db"] > -10)
        probs = np.r_[np.zeros(40), np.ones(60), np.zeros(120)]
        st["downstream"] = {f"{c}:{w}": {"signal": "fake", "vad": vad_stats(probs, [(1.28, 3.2)], np.ones(len(xs), bool), sim["dt"] if c == "talk" else None, len(xs)),
                                         "whisper": {"text": "selftest", "n_chars": 8, "system_chars_found": 0, "system_chars": 1}} for c in ("talk", "alone") for w in ("before", "after")}
        d = Demo(out, stores, folder, tag="selftest")
        d.run_all()
        d.save("rehearsal")
        log = json.loads((folder / "rehearsal.json").read_text(encoding="utf-8"))
        d2 = Demo(Out(), log["sources"], folder, tag="replay")
        import matplotlib.pyplot as plt
        d2.plt, d2.fig = plt, plt.figure()
        C.play = lambda *a, **k: None
        for k in "123456789 0ytcaem":
            d2.dispatch(k)
        plt.close("all")
        e = d2.arr("talk_e_soft")
        chk("讀回：WAV 還原成原來的刻度（誤差在 16-bit 以內）", e is not None and np.max(np.abs(e - wavs["sim"]["talk_e_soft"][0])) < 2e-4 * wavs["sim"]["talk_e_soft"][1] / 0.89 * 2)
        chk("讀回：圖都畫得出來", all((folder / f"sim_{p}.png").exists() for p in ("step1", "alone", "talk", "loud", "downstream")))
        CACHE = keep_cache
    finally:
        CFG.clear(); CFG.update(keep)
    print(f"\n{'全部通過' if not bad else f'{bad} 項沒過'}")
    sys.exit(1 if bad else 0)


# ── main ─────────────────────────────────────────────────────────────────────
def build_replay(out: Out) -> Demo:
    src = REHEARSAL / "rehearsal.json"
    if not src.exists():
        sys.exit("沒有 runs/rehearsal/rehearsal.json，先跑 ./present.sh rehearse")
    log = json.loads(src.read_text(encoding="utf-8"))
    out.banner(f"REPLAY：{log['time']} 的彩排（{'、'.join(SRC_ZH[s] for s in log['sources'])}）", C.AMBER)
    return Demo(out, log["sources"], REHEARSAL, tag="replay")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["show", "rehearse", "replay", "record", "check", "fetch", "make", "selftest", "export"])
    ap.add_argument("which", nargs="?", help="record 用：alone｜talk｜loud｜all｜user")
    ap.add_argument("--sim", action="store_true", help="show／rehearse：不讀真錄音，只用模擬")
    ap.add_argument("--no-plot", action="store_true", help="不開視窗，鍵在終端機按")
    ap.add_argument("--no-models", action="store_true", help="rehearse：不跑 VAD 與 Whisper（沒有模型的機器上預跑用）")
    ap.add_argument("--no-audio", action="store_true", help="check：跳過音訊裝置")
    args = ap.parse_args()
    if args.cmd == "check": check(args.no_audio); return
    if args.cmd == "fetch": fetch(); return
    if args.cmd == "selftest": selftest(); return
    if args.cmd == "export": export(); return
    if args.cmd == "record":
        if args.which not in CLIPS + ["all", "user"]:
            sys.exit("record 後面要接 alone｜talk｜loud｜all｜user")
        record(args.which); return
    if args.cmd == "make":
        s = make_sim(reference())
        CACHE.mkdir(parents=True, exist_ok=True)
        for p in CACHE.glob("sim_*.npz"):
            p.write_bytes(b"")  # 作廢舊的快取（不刪檔）
        print(f"模擬：{s['how']}；h {len(s['h'])} 個數；使用者在 {s['dt'][0] / FS:.2f}–{s['dt'][1] / FS:.2f} s；tanh 的粗估上限 {s['ceiling_db']:.1f} dB")
        return

    if args.cmd == "rehearse" or args.no_plot:
        import matplotlib
        matplotlib.use("Agg")
    out = Out()
    out.banner("W10 Demo 1：the system hears itself — echo, NLMS, double-talk")
    if args.cmd == "replay":
        d = build_replay(out)
    else:
        x = reference()
        stores, wavs = build_sources(out, want_real=not args.sim, x=x)
        folder = REHEARSAL if args.cmd == "rehearse" else RUNS / "live" / C.now().replace(":", "-")
        write_wavs(folder, stores, wavs)
        d = Demo(out, stores, folder, tag="rehearsal" if args.cmd == "rehearse" else "", models=None if args.no_models else Models(), wavs=wavs, x=x)
    if args.cmd == "rehearse":
        if not args.no_models:
            for src in list(stores):
                d.source = src
                d.ensure_downstream()
        d.run_all()
        d.save("rehearsal")
        if "real" in stores:
            export()
        print("\n下一步：./present.sh 自己按一遍 1 2 3 4 5 a 6 a 7 a 8 9 0 t；runs/rehearsal/*.png 看過；數字回填 p19、p33、p34 的表。"
              + ("" if "real" in stores else "\n（這次只有模擬：真錄音要先 ./present.sh record alone／talk／loud；沒有真錄音時不匯出 w10-metrics.json，要匯就 ./present.sh export）"))
        return
    d.menu()
    if args.no_plot:
        C.note("  --no-plot：鍵在終端機按。")
        while d.dispatch(C.getkey()):
            pass
    else:
        d.make_fig()
        C.note("  鍵在視窗裡按（視窗要有焦點）。終端機這裡是紀錄。")
        d.plt.show()
    if args.cmd == "show":
        d.save("session")


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
# 此檔由 bin/publish_demos.py 從 private repo 的 <course>/demos/ 複製而來，請勿在公開 repo 直接編輯。
"""W11 demo 3 的算術：一串「文字 token／聲音 frame」加上各自的時間戳，能讀出哪些數字。

這個檔案只用標準函式庫與 numpy，不碰任何模型——所以 selftest 可以在沒有 MLX 的機器上把每一條定義驗一遍。

一個事件（event）是一個 dict：
  {"k": "T", "t": 0.412, "id": 1098, "s": "The"}        一個文字 token；t 是「問題的檔案交給模型」之後過了幾秒
  {"k": "A", "t": 0.655, "c": [788, 818, …]}             一個聲音 frame（N_q 個 code）；frame rate 12.5 Hz，一個 frame = 80 ms
t 是這個 token 算完（mx.eval 回來）的時刻，所以相鄰兩個 t 的差就是那一步花的時間。
"""
from __future__ import annotations

import re
import statistics as st

import numpy as np

SPECIAL = re.compile(r"<\|[^|>]*\|>")


# ── 事件串 ───────────────────────────────────────────────────────
def kinds(events: list[dict]) -> str:
    return "".join(e["k"] for e in events)


def schedule(events: list[dict]) -> list[tuple[str, int]]:
    """連續同一種的長度：[("T", 6), ("A", 12), ("T", 6), …]。"""
    out: list[list] = []
    for e in events:
        if out and out[-1][0] == e["k"]:
            out[-1][1] += 1
        else:
            out.append([e["k"], 1])
    return [(k, n) for k, n in out]


def expected_schedule(n_text: int, n_audio: int, n_text_total: int, n_frames_total: int) -> list[tuple[str, int]]:
    """交錯的排法：n_text 個字、n_audio 個 frame 輪流；字寫完之後只剩 frame。selftest 用它對生成出來的順序。"""
    out, t, a = [], n_text_total, n_frames_total
    while t > 0 or a > 0:
        if t > 0:
            k = min(n_text, t)
            out.append(("T", k))
            t -= k
            if a > 0:
                k = a if t == 0 else min(n_audio, a)
                out.append(("A", k))
                a -= k
        else:
            out.append(("A", a))
            a = 0
    return out


def text_of(events: list[dict]) -> str:
    """模型寫的字：文字 token 接起來，拿掉 <|…|> 這類標記。"""
    return SPECIAL.sub("", "".join(e["s"] for e in events if e["k"] == "T")).strip()


def whole_sentences(text: str) -> str:
    """切到最後一個句尾為止（寫到上限被截斷的半句不念）。沒有任何句尾就原樣回傳。"""
    k = max(text.rfind("."), text.rfind("?"), text.rfind("!"))
    return text[: k + 1] if k > 0 else text


def last_sentence(text: str) -> str:
    """最後一個完整的句子（先想再說時，念出來的只有這一句）。"""
    parts = [p for p in re.split(r"(?<=[.?!])\s+", text.strip()) if p]
    return parts[-1] if parts else text


def blank_fraction(events: list[dict]) -> float:
    """文字 token 裡有多少個是「空的」：解出來只有空白，而且後面接的不是一個黏著它的字。
    （tokenizer 會把「 48」寫成一個空白 token 加一個 "48"——那個空白是正常的，不算。）
    正常的回答接近 0；模型在沒見過的排法上寫不出字時，這個數字會先動——所以它跟 WER 分開報。"""
    toks = [SPECIAL.sub("", e["s"]) for e in events if e["k"] == "T"]
    blank = 0
    for i, s in enumerate(toks):
        if s.strip():
            continue
        nxt = toks[i + 1] if i + 1 < len(toks) else ""
        if nxt.strip() and not nxt[:1].isspace():
            continue                                     # 「空白 + 黏著的字」：正常的切法
        blank += 1
    return blank / len(toks) if toks else 0.0


def frames_of(events: list[dict], eos: int) -> list[list[int]]:
    """能解成聲音的 frame：第一層是結束碼的那一格不算。"""
    return [e["c"] for e in events if e["k"] == "A" and e["c"][0] != eos]


# ── 時間 ─────────────────────────────────────────────────────────
def step_times(events: list[dict]) -> dict:
    """每一步花多久。第一個事件的 t 含讀入問題（encoder 與 prefill），不算進任何一種。
    D_text：算一個文字 token 的那一步；D_frame：算一個聲音 frame 的那一步（主幹一次前向加上 depth 的 N_q 小步）。"""
    dT, dA = [], []
    for a, b in zip(events, events[1:]):
        (dT if b["k"] == "T" else dA).append(b["t"] - a["t"])
    med = lambda v: float(st.median(v)) if v else None  # noqa: E731
    return {"D_text": med(dT), "D_frame": med(dA), "n_text_steps": len(dT), "n_frame_steps": len(dA)}


def timeline(events: list[dict], f_r: float, chunk_frames: int, decode_first_s: float, decode_chunk_s: float, eos: int) -> dict:
    """時間軸上的四個記號，加上幾個衍生的數。

    first_text    第一個文字 token 算完的時刻
    first_frame   第一個聲音 frame 算完的時刻
    first_sound   頭 chunk_frames 個 frame 都算完、而且解成波形的時刻（= 最早能開始播的時刻）
    underrun      從 first_sound 開始照實際速度播；第一次「該播的那一塊還沒解出來」是在播了幾秒之後（沒有就是 None）
    min_slack     每一塊「解出來」比「輪到它播」早多少，取最小的（負的 = 斷過）
    播放是用時間戳算的，不是量喇叭：第 c 塊在 first_sound + c·chunk/f_r 輪到它，
    它在「這一塊最後一個 frame 算完 + 解這一塊的時間」的時刻才有。
    """
    T = [e["t"] for e in events if e["k"] == "T"]
    A = [e["t"] for e in events if e["k"] == "A" and e["c"][0] != eos]
    out = {"n_text": len(T), "n_frames": len(A), "audio_s": len(A) / f_r,
           "first_text": T[0] if T else None, "first_frame": A[0] if A else None,
           "first_sound": None, "underrun": None, "min_slack": None, "rtf": None,
           "total": events[-1]["t"] if events else None, "chunk_frames": chunk_frames}
    if not A:
        return out
    n = min(chunk_frames, len(A))
    first_sound = A[n - 1] + decode_first_s
    out["first_sound"] = first_sound
    slacks = []
    for c in range(1, -(-len(A) // chunk_frames)):
        last = min((c + 1) * chunk_frames, len(A)) - 1
        ready = A[last] + decode_chunk_s
        due = first_sound + c * chunk_frames / f_r
        slacks.append(due - ready)
        if ready > due and out["underrun"] is None:
            out["underrun"] = c * chunk_frames / f_r
    out["min_slack"] = min(slacks) if slacks else None
    t0 = T[0] if T else A[0]
    out["rtf"] = (events[-1]["t"] - t0) / out["audio_s"] if out["audio_s"] > 0 else None
    return out


def lead_curve(events: list[dict], f_r: float, eos: int) -> dict:
    """文字領先多少（估計）。每一個聲音 frame 開始生成的時候：
      written = 在它之前已經寫出來的文字 token 數
      spoken  = 到這個 frame 為止「大概念到第幾個 token」= R_txt × (frame 的序號 / f_r)，
                R_txt = 這一段回答的文字 token 總數 ÷ 聲音的總秒數（假設念的速度是均勻的——這是估計，不是對齊）
      lead    = written − spoken
    回傳最小的 lead 與它發生在第幾秒。lead 掉到 0 以下 = 照均勻語速算，嘴巴已經追過了寫出來的字。"""
    n_text = sum(1 for e in events if e["k"] == "T" and SPECIAL.sub("", e["s"]).strip())
    frames = [e for e in events if e["k"] == "A" and e["c"][0] != eos]
    if not frames or not n_text:
        return {"R_txt": None, "min_lead": None, "at_s": None, "curve": []}
    r_txt = n_text / (len(frames) / f_r)
    written, j, curve = 0, 0, []
    for e in events:
        if e["k"] == "T":
            written += 1 if SPECIAL.sub("", e["s"]).strip() else 0
        elif e["c"][0] != eos:
            curve.append(written - r_txt * j / f_r)
            j += 1
    k = int(np.argmin(curve))
    return {"R_txt": r_txt, "min_lead": float(curve[k]), "at_s": k / f_r, "curve": [round(float(v), 2) for v in curve]}


def conditions(n_text: int, n_audio: int, f_r: float, r_txt: float, d_text: float, d_frame: float) -> dict:
    """投影片上的兩個條件，代量到的數字。
      (i)  文字要一直領先：n_text / n_audio ≥ R_txt / f_r
      (ii) 生成要比播放快：一輪花的時間 n_text·D_text + n_audio·D_frame ≤ 一輪的聲音長度 n_audio / f_r
    投影片的 (ii) 寫成 D_token ≤ (n_audio / f_r) / (n_text + n_audio·N_step)、N_step = 1；
    這裡把文字的一步與聲音的一步分開量，同時給「一輪的平均每步」對那個上限。slack = (ii) 剩下的時間。"""
    round_audio = n_audio / f_r
    round_gen = n_text * d_text + n_audio * d_frame
    return {"n_text": n_text, "n_audio": n_audio,
            "i_lhs": n_text / n_audio, "i_rhs": r_txt / f_r, "i_ok": n_text / n_audio >= r_txt / f_r,
            "round_audio_s": round_audio, "round_gen_s": round_gen, "ii_ok": round_gen <= round_audio,
            "slack_s": round_audio - round_gen,
            "mean_step_s": round_gen / (n_text + n_audio), "step_bound_s": round_audio / (n_text + n_audio)}


# ── 字 ───────────────────────────────────────────────────────────
def norm_simple(text: str) -> list[str]:
    """小寫、去標點、切詞。有 Whisper 的 normalizer 時先過它（數字與拼法），這一個是最後一步，也是沒有它時的退路。"""
    return re.findall(r"[a-z0-9]+(?:'[a-z]+)?", text.lower().replace("’", "'"))


def align_words(ref: list[str], hyp: list[str], open_end: bool = False) -> dict:
    """詞級 Levenshtein 加回溯（與 W2 Demo 3 同一支）。ref = 模型自己寫的字，hyp = Whisper 從它的聲音聽到的字。

    open_end：生成被步數上限截斷時用。交錯生成裡文字走在聲音前面，截斷的那一刻有一段字「寫了、還沒念到」；
    把它們算成漏念並不公平。open_end=True 時只比到 ref 的某個開頭為止——取讓錯誤數最少的那個開頭——
    之後還沒念到的字不算（n_ref 也只算到那裡，n_unspoken 記被略過的詞數）。自己停下來的回答不用這個。"""
    R, H = len(ref), len(hyp)
    d = np.zeros((R + 1, H + 1), dtype=np.int32)
    d[:, 0] = np.arange(R + 1)
    d[0, :] = np.arange(H + 1)
    for i in range(1, R + 1):
        for j in range(1, H + 1):
            d[i, j] = min(d[i - 1, j] + 1, d[i, j - 1] + 1, d[i - 1, j - 1] + (ref[i - 1] != hyp[j - 1]))
    end = R
    if open_end and R:
        best = int(d[:, H].min())
        end = max(k for k in range(R + 1) if d[k, H] == best)       # 一樣少的時候取長的
    S = D = I = 0
    i, j = end, H
    while i > 0 or j > 0:
        if i > 0 and j > 0 and d[i, j] == d[i - 1, j - 1] + (ref[i - 1] != hyp[j - 1]):
            S += ref[i - 1] != hyp[j - 1]
            i, j = i - 1, j - 1
        elif i > 0 and d[i, j] == d[i - 1, j] + 1:
            D += 1
            i -= 1
        else:
            I += 1
            j -= 1
    return {"errors": int(d[end, H]), "S": int(S), "D": int(D), "I": int(I), "n_ref": end, "n_hyp": H, "n_unspoken": R - end}


def wer(a: dict) -> float | None:
    return a["errors"] / a["n_ref"] if a["n_ref"] else None


def pooled(aligns: list[dict]) -> dict:
    """幾題合起來算一個 WER：錯的詞數相加 ÷ 參考詞數相加（不是每題的 WER 取平均）。"""
    tot = {k: sum(a.get(k, 0) for a in aligns) for k in ("errors", "S", "D", "I", "n_ref", "n_hyp", "n_unspoken")}
    tot["wer"] = wer(tot)
    return tot


NUM_WORDS = {"zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
             "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18,
             "nineteen": 19, "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90}


def numbers_in(text: str) -> list[int]:
    """字串裡出現的整數，照出現的順序；"thirty-six"、"thirty six"、"36" 都算 36（只處理到 99，這個 demo 的題目夠用）。"""
    out, toks, i = [], re.findall(r"\d+|[a-z]+", text.lower()), 0
    while i < len(toks):
        w = toks[i]
        if w.isdigit():
            out.append(int(w))
        elif w in NUM_WORDS:
            v = NUM_WORDS[w]
            if v >= 20 and i + 1 < len(toks) and toks[i + 1] in NUM_WORDS and 0 < NUM_WORDS[toks[i + 1]] < 10:
                v += NUM_WORDS[toks[i + 1]]
                i += 1
            out.append(v)
        i += 1
    return out


def answer_is(text: str, expected: int) -> dict:
    """答對沒有：看字串裡最後一個數字是不是正解（推理的中間數字不算）。同時回報正解有沒有在任何位置出現過。"""
    nums = numbers_in(text)
    return {"last": nums[-1] if nums else None, "correct": bool(nums) and nums[-1] == expected, "mentioned": expected in nums}


def tokens_before_first_frame(events: list[dict]) -> int:
    n = 0
    for e in events:
        if e["k"] == "A":
            break
        n += 1
    return n

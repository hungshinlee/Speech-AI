# -*- coding: utf-8 -*-
# 此檔由 bin/publish_demos.py 從 private repo 的 <course>/demos/ 複製而來，請勿在公開 repo 直接編輯。
"""W11 demo 2 的計分：從回答裡抽出選項、格式有沒有守住、bootstrap 區間、成對的差、轉錄對不對。

全部是規則，沒有模型當裁判。只用標準函式庫與 numpy；`./present.sh selftest` 對每一條規則有例子。

抽選項的順序（先中先贏）：
  1. 明講的：「the answer is C」「答案是丙」——取最後一次出現的；
  2. 回答裡只出現一個選項代號（英語：後面接標點或結尾的大寫 A–D；華語：前後不接拉丁字母的 A–D，舊的甲乙丙丁也認）；
  3. 回答裡只出現一個選項的內容（數字要整個數字相符；華語的回答先把簡體折成同一種字再比）；
  4. 回答提到不只一個選項（一步一步推、把選項都念過一遍的那種）：只看最後一句，再走一次 2 與 3。
四步都抽不到就是「沒有選」，算答錯——那也是一種結果，表上另外數。
"""
from __future__ import annotations

import math
import re

import numpy as np

import questions as Q
from zh_fold import fold

ZH_LAB = "甲乙丙丁"
_EN_STRONG = re.compile(r"(?:answer|option|choice|correct(?: one)?)\s*(?:is|was|would be|:|=)?\s*[\(\[\"'“*]*\s*([A-D])(?![A-Za-z])", re.I)
_EN_LABEL = re.compile(r"(?<![A-Za-z])([A-D])(?=\s*(?:[:.\)\],;，。]|$))")
_ZH_STRONG = re.compile(r"(?:答案|選|应该是|應該是|正確的是|正确的是)\s*(?:是|為|为|應該是|应该是)?\s*[:：「『（(\"“*\s]*([甲乙丙丁A-Da-d])")
_ZH_LETTER = re.compile(r"(?<![A-Za-z])([A-Da-d])(?![A-Za-z])")   # 中文句子裡單獨的一個字母
# 回答把題目的結尾抄一遍時（「請問答案是 A、B、C、D 的哪一個」）那一串不是它的答案：先拿掉
_ZH_LIST = re.compile(r"(?:答案是\s*)?[A甲]\s*[、,，]\s*[B乙]\s*[、,，]\s*[C丙]\s*(?:[、,，]|或|還是|还是)?\s*[D丁]", re.I)
_THINK = re.compile(r"<think>.*?</think>", re.S)

_EN_NUM = {"zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
           "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18,
           "nineteen": 19, "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90}
_ZH_DIG = "零一二三四五六七八九"


def strip_think(reply: str) -> str:
    """拿掉推理區塊與特殊標記的殘留；換行留著（抽選項的第 4 步要用）。"""
    reply = _THINK.sub(" ", reply or "")
    return re.sub(r"<\|[^|>]*\|>|</?think>|</?role>", " ", reply).strip()


def clean(reply: str) -> str:
    """strip_think 之後再把空白（含換行）壓成一個。"""
    return re.sub(r"\s+", " ", strip_think(reply)).strip()


def zh_number(n: int) -> str:
    """0–999 的中文數字（十三、二十、一百零五）。"""
    if n < 10:
        return _ZH_DIG[n]
    if n < 20:
        return "十" + (_ZH_DIG[n % 10] if n % 10 else "")
    if n < 100:
        return _ZH_DIG[n // 10] + "十" + (_ZH_DIG[n % 10] if n % 10 else "")
    h, r = divmod(n, 100)
    if r == 0:
        return _ZH_DIG[h] + "百"
    if r < 10:
        return _ZH_DIG[h] + "百零" + _ZH_DIG[r]
    return _ZH_DIG[h] + "百" + _ZH_DIG[r // 10] + "十" + (_ZH_DIG[r % 10] if r % 10 else "")


def en_number(n: int) -> str:
    inv = {v: k for k, v in _EN_NUM.items()}
    if n in inv:
        return inv[n]
    if n < 100:
        return f"{inv[n - n % 10]}-{inv[n % 10]}"
    h, r = divmod(n, 100)
    return f"{inv[h]} hundred" + (f" and {en_number(r)}" if r else "")


def _contains_option(reply: str, opt: str, lang: str) -> bool:
    """回答裡有沒有這個選項的內容。數字要整個相符（12 不算在 120 裡）；整數另外接受拼出來的寫法。"""
    r, o = reply, opt
    if lang == "zh":
        r, o = fold(r).replace(" ", ""), fold(o).replace(" ", "")
    else:
        r, o = r.lower(), re.sub(r"^(?:a|an|the|at)\s+", "", o.lower())
    m = re.match(r"^(\d+(?:\.\d+)?)(.*)$", o)
    if m:
        num, unit = m.group(1), m.group(2).strip()
        forms = [num]
        if "." not in num and int(num) < 1000:
            forms.append(fold(zh_number(int(num))) if lang == "zh" else en_number(int(num)))
            if lang == "zh" and int(num) == 2:
                forms.append("两")
        for f in forms:
            if re.search(r"(?<![\d.])" + re.escape(f) + r"(?![\d]|\.\d)", r) if f[0].isdigit() else (f in r if lang == "zh" else re.search(rf"\b{re.escape(f)}\b", r)):
                return True
        return False
    if lang == "zh":
        return o in r
    return re.search(rf"(?<![a-z]){re.escape(o)}(?![a-z])", r) is not None


def pick(reply: str, lang: str, options: list[str], _depth: int = 0) -> tuple[int | None, str]:
    """→ (選項的位置 0–3 或 None, 是哪一步抽到的："stated"／"label"／"content"／"last"／"ambiguous"／"none")。"""
    r = clean(reply)
    if not r:
        return None, "none"
    if lang == "zh":
        r = _ZH_LIST.sub(" ", r)
        m = _ZH_STRONG.findall(r)
        if m:
            c = m[-1]
            return (ZH_LAB.index(c) if c in ZH_LAB else "ABCD".index(c.upper())), "stated"
        labs = {ZH_LAB.index(c) for c in r if c in ZH_LAB} | {"ABCD".index(c.upper()) for c in _ZH_LETTER.findall(r)}
        if len(labs) == 1:
            return labs.pop(), "label"
    else:
        m = _EN_STRONG.findall(r)
        if m:
            return "ABCD".index(m[-1].upper()), "stated"
        labs = set(_EN_LABEL.findall(r))
        if len(labs) == 1:
            return "ABCD".index(labs.pop()), "label"
    hits = [i for i, o in enumerate(options) if _contains_option(r, o, lang)]
    # 一個選項的內容是另一個的一部分時（「6」與「6 個」不會，但「Mimi is an animal」在「Mimi is not an animal」裡不會出現）取最長的
    if len(hits) > 1:
        longest = max(hits, key=lambda i: len(options[i]))
        if all(options[i].lower() in options[longest].lower() for i in hits):
            hits = [longest]
    if len(hits) == 1:
        return hits[0], "content"
    if not (hits or len(labs) > 1):
        return None, "none"
    # 回答把幾個選項都提了一遍（一步一步推的那種）：只看最後一句
    if _depth == 0:
        tail = [t for t in re.split(r"[\n。！？!?]|(?<=[a-z\)])\.\s", strip_think(reply)) if t.strip(" *#-_>|")]
        if len(tail) > 1:
            p, how = pick(tail[-1], lang, options, _depth=1)
            if p is not None:
                return p, "last"
    return None, "ambiguous"


# ── 格式 ─────────────────────────────────────────────────────────
_WRAP = "\"'“”‘’「」『』()（）[]*` \n\t"
_TAIL = ".。!！"


def core(reply: str) -> str:
    """格式的判定只看這一段：去掉推理、頭尾的空白與引號、結尾的句號。"""
    r = clean(reply).strip(_WRAP)
    while r and r[-1] in _TAIL:
        r = r[:-1].rstrip(_WRAP)
    return r


def _han(s: str) -> int:
    return sum(1 for c in s if "一" <= c <= "鿿")


def _is_number(s: str, lang: str) -> bool:
    if re.fullmatch(r"-?\d+(?:\.\d+)?", s):
        return True
    if lang == "zh":
        return re.fullmatch(r"[零一二三四五六七八九十百兩两]+", s) is not None
    return all(w in _EN_NUM or w in ("hundred", "and") for w in re.split(r"[\s-]+", s.lower())) and bool(s)


def format_kept(reply: str, fmt: dict, lang: str) -> bool:
    c = core(reply)
    rule = fmt["rule"]
    if not c:
        return False
    if rule == "letter":
        return c.upper() in ("A", "B", "C", "D") or (lang == "zh" and c in ZH_LAB)
    if rule == "number":
        return _is_number(c, lang)
    if rule == "yes_no":
        return c.lower() in ("yes", "no") if lang == "en" else fold(c) in ("是", "不是", "对", "不对")
    if rule in ("max_units", "units"):
        n = fmt["n"][lang]
        if lang == "zh":
            k = _han(c)
            extra = re.sub(r"[一-鿿，、。\s]", "", c)
            ok_len = k <= n if rule == "max_units" else k == n
            return ok_len and k > 0 and not extra
        k = len(re.findall(r"[A-Za-z0-9'-]+", c))
        return 0 < k <= n if rule == "max_units" else k == n
    if rule == "items":
        parts = [p.strip(_WRAP) for p in re.split(r"[,，、;；]|\band\b|和", c)]
        parts = [p for p in parts if p]
        short = all((_han(p) <= 4 and len(p) <= 6) if lang == "zh" else len(p.split()) <= 2 for p in parts)
        plain = all(re.fullmatch(r"[A-Za-z\u4e00-\u9fff ]+", p) for p in parts)
        return len(parts) == fmt["n"] and short and plain
    if rule == "prefix":
        s = fmt["s"][lang]
        head = clean(reply).lstrip(_WRAP)
        return head.lower().startswith(s) or (lang == "zh" and fold(head).startswith(fold(s).replace("：", ":")))
    raise ValueError(f"unknown format rule {rule!r}")


def content_ok(reply: str, q: dict, lang: str) -> bool | None:
    """格式題的內容對不對（與格式分開記）。沒有正解可對的回 None。"""
    fmt = q["fmt"]
    if q["answer"] is not None:
        return pick(reply, lang, q["options"][lang])[0] == q["answer"]
    key = (fmt.get("key") or {}).get(lang)
    if not key:
        return None
    r = fold(clean(reply)).lower() if lang == "zh" else clean(reply).lower()
    if fmt["rule"] == "yes_no":
        c = core(reply).lower()
        c = fold(c) if lang == "zh" else c
        neg = c.startswith(("no", "不"))
        want_neg = key[0].startswith(("no", "不"))
        return neg == want_neg
    return any((fold(k).lower() if lang == "zh" else k) in r for k in key)


def judge(reply: str, q: dict, lang: str) -> dict:
    """一題的判定。四選一：pick、how、ok。格式題：fmt（守住沒有）、ok（內容）。"""
    if q["kind"] == "format":
        out = {"fmt": format_kept(reply, q["fmt"], lang), "ok": content_ok(reply, q, lang)}
        if q["answer"] is not None:
            out["pick"] = pick(reply, lang, q["options"][lang])[0]
        return out
    p, how = pick(reply, lang, q["options"][lang])
    return {"pick": p, "how": how, "ok": p == q["answer"]}


# ── 統計 ─────────────────────────────────────────────────────────
def bootstrap_ci(ok: list[bool], B: int = 10000, seed: int = 20261126, level: float = 0.95) -> dict:
    """答對題數與 percentile bootstrap 區間（對題目重抽）。回 {"k", "n", "lo", "hi"}，lo、hi 以題數計。"""
    x = np.asarray(ok, dtype=np.float64)
    n = len(x)
    if n == 0:
        return {"k": 0, "n": 0, "lo": 0.0, "hi": 0.0}
    rng = np.random.default_rng(seed)
    sums = x[rng.integers(0, n, size=(B, n))].sum(axis=1)
    a = (1 - level) / 2
    return {"k": int(x.sum()), "n": n, "lo": float(np.quantile(sums, a)), "hi": float(np.quantile(sums, 1 - a))}


def paired(x: list[bool], y: list[bool], B: int = 10000, seed: int = 20261126, level: float = 0.95) -> dict:
    """同一批題目上 x 減 y：差幾題、差的 bootstrap 區間、只有一邊答對的各幾題、對那幾題做的精確雙尾符號檢定。"""
    a, b = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    n = len(a)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(B, n))
    d = (a - b)[idx].sum(axis=1)
    q = (1 - level) / 2
    only_x, only_y = int(((a == 1) & (b == 0)).sum()), int(((a == 0) & (b == 1)).sum())
    m, k = only_x + only_y, min(only_x, only_y)
    p = min(1.0, 2 * sum(math.comb(m, i) for i in range(k + 1)) / 2 ** m) if m else 1.0
    return {"diff": int(a.sum() - b.sum()), "lo": float(np.quantile(d, q)), "hi": float(np.quantile(d, 1 - q)),
            "only_x": only_x, "only_y": only_y, "p_sign": p, "n": n}


# ── 轉錄對不對 ───────────────────────────────────────────────────
def _units(text: str, lang: str) -> list[str]:
    if lang == "zh":
        return [c for c in fold(text).lower() if "一" <= c <= "鿿" or c.isalnum()]
    return re.findall(r"[a-z0-9']+", text.lower().replace("-", " "))


def error_rate(ref: str, hyp: str, lang: str) -> float:
    """華語：字錯誤率（先折成同一種字、去標點）；英語：詞錯誤率（小寫、去標點）。數字寫成阿拉伯數字或拼出來會被算成錯——
    所以這個數字只拿來找出念壞或聽壞的題目，不當成辨識器的成績。"""
    r, h = _units(ref, lang), _units(hyp, lang)
    if not r:
        return 0.0 if not h else 1.0
    prev = list(range(len(h) + 1))
    for i, a in enumerate(r, 1):
        cur = [i] + [0] * len(h)
        for j, b in enumerate(h, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (a != b))
        prev = cur
    return prev[-1] / len(r)


def labels(lang: str) -> list[str]:
    return Q.LABELS[lang]

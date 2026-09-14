#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
每週頁面的公開／不公開過濾。由 scripts/build_weeks.py 呼叫。

設計原則：**預設不公開（deny by default）**。
只有 PUBLIC_SECTIONS 名單內的 `###` 區塊會進入 weeks/*.qmd 上站。
之後在大綱新寫的任何區塊預設留在教師端——忘記標記的後果是「學生看不到」，
不是「教學底牌外流」。

三層控制
--------
1. 區塊層級（主要機制）：`###` 標題名稱比對 PUBLIC_SECTIONS。
   標題尾端的括號會被忽略，所以 `### 課堂骨架（180 min）` 比對的是「課堂骨架」。

2. 區塊層級覆寫：在 `###` 標題的**下一行**放
       <!-- vis: public -->    強制公開這個區塊
       <!-- vis: private -->   強制隱藏這個區塊

3. 段落層級挖空：在**已公開**的區塊裡，用一對註解夾住要拿掉的片段
       <!-- private -->
       這段只有我看得到
       <!-- /private -->

引用可信度
----------
含 `[驗]` 的行（尚未查證的引用）整行不上站：未查證的引用一旦進了學生的
閱讀清單，錯誤會被當成事實轉引出去。查證完拿掉標記，它自然就上站了。
含 `[主題]` 的行保留，只移除標記本身——那本來就是誠實的檢索關鍵字。
"""

import re

# ---------------------------------------------------------------------------
# 想多開一個區塊就加進來（例如 "概念拆解路徑"）。預設保守：
# 只給「這週在幹嘛」「你該學會什麼」「要讀什麼」。
# ---------------------------------------------------------------------------
PUBLIC_SECTIONS = {
    "定位",
    "Learning objectives",
    "參考資料",
}

# 寫檔前的保險絲：頁面若含這些字串，代表過濾邏輯被改壞了，建置直接中止。
LEAK_MARKERS = ("課堂骨架", "常見誤解", "卡點提示", "demo 建議", "[驗]")

_SEC = re.compile(r"^###\s+(.+?)\s*$")
_VIS = re.compile(r"^<!--\s*vis:\s*(public|private)\s*-->\s*$")
_POPEN = re.compile(r"^<!--\s*private\s*-->\s*$")
_PCLOSE = re.compile(r"^<!--\s*/private\s*-->\s*$")
_PAREN = re.compile(r"[（(][^（(]*?[）)]\s*$")
_TOPIC = re.compile(r"`?\[主題\]`?\s*")
_UNVERIFIED = "[驗]"


def canon(heading):
    """`課堂骨架（180 min）` -> `課堂骨架`"""
    return _PAREN.sub("", heading).strip()


def _drop_empty_sections(lines):
    """移除過濾後只剩標題、沒有內文的區塊。"""
    out, i = [], 0
    while i < len(lines):
        if _SEC.match(lines[i]):
            j = i + 1
            while j < len(lines) and not _SEC.match(lines[j]):
                j += 1
            if any(ln.strip() for ln in lines[i + 1:j]):
                out.extend(lines[i:j])
            i = j
        else:
            out.append(lines[i])
            i += 1
    return out


def public_only(body):
    """body 是某一週 `## W<n>｜…` 底下的原始行（尚未 demote）。"""
    out, keep, skip, pending, head = [], False, False, False, None
    for ln in body:
        m = _SEC.match(ln)
        if m:
            keep = canon(m.group(1)) in PUBLIC_SECTIONS
            skip, pending, head = False, True, ln
            if keep:
                out.append(ln)
            continue

        if pending:
            pending = False
            mv = _VIS.match(ln.strip())
            if mv:
                want_public = mv.group(1) == "public"
                if want_public and not keep:
                    keep = True
                    out.append(head)
                elif not want_public and keep:
                    out.pop()          # 收回剛加進去的標題
                    keep = False
                continue

        if not keep:
            continue
        if _POPEN.match(ln.strip()):
            skip = True
            continue
        if _PCLOSE.match(ln.strip()):
            skip = False
            continue
        if skip or _UNVERIFIED in ln:
            continue
        out.append(_TOPIC.sub("", ln))

    return _drop_empty_sections(out)


def assert_no_leak(page_text, where):
    """建置的最後一道保險絲。"""
    hit = [s for s in LEAK_MARKERS if s in page_text]
    if hit:
        raise SystemExit("%s 的網頁版含不該上站的內容 %s" % (where, hit))

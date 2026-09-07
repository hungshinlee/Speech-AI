#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
以 docs/course-outline.md 為 single source of truth，產生 Quarto 網站頁面。

產出：
  weeks/w01.qmd … w14.qmd   每週一頁
  weeks/all.qmd             完整大綱單頁（不含附錄 C：授課者私用備註）
  _includes/*.md            首頁／課程資訊／資源頁引用的片段

用法：python3 scripts/build_weeks.py
改完 docs/course-outline.md 後務必重跑，否則網站與大綱會漂移。
"""

import io
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "docs", "course-outline.md")
WEEKS_DIR = os.path.join(ROOT, "weeks")
INC_DIR = os.path.join(ROOT, "_includes")

BANNER = ("<!-- 此檔由 scripts/build_weeks.py 自動產生，"
          "請勿直接編輯；請改 docs/course-outline.md 後重跑腳本。 -->")

PART_OF = {}
for w in range(1, 7):
    PART_OF[w] = "Part I — 基礎"
for w in range(7, 11):
    PART_OF[w] = "Part II — 模組"
for w in range(11, 15):
    PART_OF[w] = "Part III — 對話系統"


def read_source():
    if not os.path.exists(SRC):
        sys.exit("找不到 %s" % SRC)
    with io.open(SRC, encoding="utf-8") as f:
        return f.read().replace("\r\n", "\n").split("\n")


def demote(lines, levels=1):
    """把標題降級：### -> ##（供每週頁面使用，頁標題由 YAML 提供）。"""
    out = []
    for ln in lines:
        m = re.match(r"^(#{2,6}) (.*)$", ln)
        if m:
            n = len(m.group(1)) - levels
            if n >= 1:
                out.append("#" * n + " " + m.group(2))
                continue
        out.append(ln)
    return out


def strip_hr(lines):
    return [ln for ln in lines if ln.strip() != "---"]


def trim(lines):
    while lines and not lines[0].strip():
        lines.pop(0)
    while lines and not lines[-1].strip():
        lines.pop()
    return lines


def yaml_escape(s):
    return s.replace('"', '\\"')


def slice_section(lines, start_idx):
    """從 start_idx（標題行）之後取到下一個同級或更高級標題為止。"""
    level = len(re.match(r"^(#+)", lines[start_idx]).group(1))
    body = []
    for ln in lines[start_idx + 1:]:
        m = re.match(r"^(#+) ", ln)
        if m and len(m.group(1)) <= level:
            break
        body.append(ln)
    return body


def find_heading(lines, text):
    for i, ln in enumerate(lines):
        m = re.match(r"^#+ (.*)$", ln)
        if m and m.group(1).strip() == text:
            return i
    return None


def write(path, content):
    d = os.path.dirname(path)
    if d and not os.path.isdir(d):
        os.makedirs(d)
    with io.open(path, "w", encoding="utf-8") as f:
        f.write(content)
    return path


def main():
    lines = read_source()

    # ── 1. 每週頁面 ───────────────────────────────────────────
    week_heads = []
    for i, ln in enumerate(lines):
        m = re.match(r"^## W(\d+)｜(.+)$", ln)
        if m:
            week_heads.append((i, int(m.group(1)), m.group(2).strip()))

    if len(week_heads) != 14:
        sys.exit("預期 14 個週次標題，實際找到 %d 個" % len(week_heads))

    written = []
    index_rows = ["| 週次 | 主題 | 區塊 |", "|---|---|---|"]

    for idx, (li, wnum, title) in enumerate(week_heads):
        body = trim(strip_hr(demote(slice_section(lines, li))))
        prev_link = ("weeks/w%02d.qmd" % week_heads[idx - 1][1]) if idx > 0 else None
        next_link = ("weeks/w%02d.qmd" % week_heads[idx + 1][1]) if idx + 1 < len(week_heads) else None
        fm = [
            "---",
            'title: "W%d｜%s"' % (wnum, yaml_escape(title)),
            'subtitle: "%s"' % PART_OF[wnum],
            "toc-depth: 2",
            "---",
            "",
            BANNER,
            "",
        ]
        _ = (prev_link, next_link)  # page-navigation 由 Quarto 依 sidebar 順序處理
        out = "\n".join(fm + body) + "\n"
        written.append(write(os.path.join(WEEKS_DIR, "w%02d.qmd" % wnum), out))
        index_rows.append("| **W%d** | [%s](weeks/w%02d.qmd) | %s |"
                          % (wnum, title, wnum, PART_OF[wnum].split(" — ")[-1]))

    # ── 2. 完整大綱單頁（排除附錄 C）──────────────────────────
    cut = find_heading(lines, "C. 課程設計備註（給授課者自己看）")
    full = lines[:cut] if cut is not None else lines[:]
    # 去掉原始文件的 H1 標題行（頁標題改由 YAML 提供）
    full = [ln for ln in full if not re.match(r"^# 語音處理與人機互動 — 14 週課程大綱$", ln)]
    full = [("***" if ln.strip() == "---" else ln) for ln in full]
    full_out = "\n".join([
        "---",
        'title: "14 週課程大綱（完整單頁）"',
        'subtitle: "適合列印或全文搜尋；分週閱讀請用左側導覽"',
        "toc-depth: 2",
        "---",
        "",
        BANNER,
        "",
        "> 這一頁是把 14 週串起來的完整版，行動裝置上建議改用左側的分週頁面。",
        "> 授課者私用的課程設計備註（附錄 C）不在網站上，僅存於 repo 的 "
        "[`docs/course-outline.md`](https://github.com/hungshinlee/Speech-AI/blob/main/docs/course-outline.md)。",
        "",
    ] + trim(full)) + "\n"
    written.append(write(os.path.join(WEEKS_DIR, "all.qmd"), full_out))

    # ── 3. 首頁／課程資訊／資源頁的引用片段 ────────────────────
    # (檔名, 大綱中的區段標題, 在網站上顯示的標題)
    parts = [
        ("disclaimer.md",    "使用說明與免責聲明",              "使用說明與免責聲明"),
        ("coursemap.md",     "課程地圖（Course Map）",           "課程地圖"),
        ("textbooks.md",     "主要教科書（全課通用）",           "主要教科書"),
        ("reading-table.md", "A. 每週核心文獻速查表（若只讀一篇）", "每週核心文獻速查表（若只讀一篇）"),
        ("toolchain.md",     "B. 工具鏈建議",                    "工具鏈建議"),
    ]
    for fname, heading, display in parts:
        i = find_heading(lines, heading)
        if i is None:
            print("  ! 找不到區段：%s（跳過 %s）" % (heading, fname))
            continue
        body = trim(strip_hr(slice_section(lines, i)))
        out = BANNER + "\n\n## " + display + "\n\n" + "\n".join(body) + "\n"
        written.append(write(os.path.join(INC_DIR, fname), out))

    # 速查表裡的相對連結需指回 weeks/
    rt = os.path.join(INC_DIR, "reading-table.md")
    if os.path.exists(rt):
        with io.open(rt, encoding="utf-8") as f:
            t = f.read()
        t = re.sub(r"^\| W(\d+) \|", lambda m: "| [W%s](weeks/w%02d.qmd) |"
                   % (m.group(1), int(m.group(1))), t, flags=re.M)
        write(rt, t)

    write(os.path.join(INC_DIR, "week-index.md"),
          BANNER + "\n\n" + "\n".join(index_rows) + "\n")

    print("產生 %d 個檔案：" % len(written))
    for p in written:
        print("  " + os.path.relpath(p, ROOT))


if __name__ == "__main__":
    main()

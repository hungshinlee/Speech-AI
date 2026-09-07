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
    index_rows = ["| 週次 | 主題 | 區塊 | 投影片 |", "|---|---|---|---|"]

    def slides_for(n):
        """該週是否已有投影片。存在才給連結，避免死連結。"""
        rel = os.path.join("slides", "w%02d.qmd" % n)
        return rel if os.path.exists(os.path.join(ROOT, rel)) else None

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
        if slides_for(wnum):
            fm += ["::: {.callout-note appearance=\"minimal\"}",
                   "[**▶ 本週投影片（English）**](../slides/w%02d.qmd)" % wnum,
                   ":::",
                   ""]
        _ = (prev_link, next_link)  # page-navigation 由 Quarto 依 sidebar 順序處理
        out = "\n".join(fm + body) + "\n"
        written.append(write(os.path.join(WEEKS_DIR, "w%02d.qmd" % wnum), out))
        sl = ("[▶ 開啟](slides/w%02d.qmd)" % wnum) if slides_for(wnum) else "—"
        index_rows.append("| **W%d** | [%s](weeks/w%02d.qmd) | %s | %s |"
                          % (wnum, title, wnum, PART_OF[wnum].split(" — ")[-1], sl))

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

    # 「課程地圖」區段裡用粗體（而非標題）分隔了三個子塊：ASCII 圖、三條主軸、
    # 延遲預算。三條主軸在 index.qmd 已有手寫版本，會重複；因此在此拆開：
    #   coursemap.md → 只留 ASCII 圖
    #   latency.md   → 延遲預算，掛在 syllabus.qmd
    i = find_heading(lines, "課程地圖（Course Map）")
    if i is not None:
        body = trim(strip_hr(slice_section(lines, i)))

        def marker(prefix):
            for n, ln in enumerate(body):
                if ln.startswith(prefix):
                    return n
            return None

        i_axes = marker("**三條貫穿全課的主軸")
        i_lat = marker("**延遲預算")
        if i_axes is not None:
            write(os.path.join(INC_DIR, "coursemap.md"),
                  BANNER + "\n\n## 課程地圖\n\n"
                  + "\n".join(trim(body[:i_axes])) + "\n")
        if i_lat is not None:
            lat = trim(body[i_lat + 1:])   # 丟掉粗體那行，改用真正的標題
            write(os.path.join(INC_DIR, "latency.md"),
                  BANNER + "\n\n## 延遲預算：全課的共同座標系\n\n"
                  + "\n".join(lat) + "\n")
            written.append(os.path.join(INC_DIR, "latency.md"))

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

    # ── 投影片索引頁：有幾份就列幾份，不必手動維護導覽 ──────────
    rows = ["| 週次 | 主題 | 投影片 |", "|---|---|---|"]
    have = 0
    for _, wnum, title in week_heads:
        if slides_for(wnum):
            have += 1
            rows.append("| **W%d** | [%s](weeks/w%02d.qmd) | [▶ 開啟投影片](slides/w%02d.qmd) |"
                        % (wnum, title, wnum, wnum))
        else:
            rows.append("| W%d | %s | *尚未製作* |" % (wnum, title))
    written.append(write(os.path.join(ROOT, "slides.qmd"), "\n".join([
        "---",
        'title: "投影片"',
        'subtitle: "英文投影片、中文口述；共 %d / 14 週已完成"' % have,
        "toc: false",
        "---",
        "",
        BANNER,
        "",
        "投影片以英文製作，配合中文授課。每份都是 reveal.js 網頁，直接在瀏覽器開啟即可。",
        "",
        "::: {.callout-tip}",
        "## 播放時的快捷鍵",
        "`S` 開啟 presenter view（含中文備註）· `B` 開啟板書層可直接手寫 · "
        "`ESC` 總覽全部投影片 · `F` 全螢幕 · 網址加上 `?print-pdf` 可列印",
        ":::",
        "",
    ] + rows + [
        "",
        "投影片內嵌音訊，請確認裝置聲音已開啟。部分片段為左右聲道分離"
        "（左＝使用者、右＝模型），建議使用耳機或立體聲喇叭。",
        "",
    ])))

    print("產生 %d 個檔案：" % len(written))
    for p in written:
        print("  " + os.path.relpath(p, ROOT))


if __name__ == "__main__":
    main()

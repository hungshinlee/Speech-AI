#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
以 docs/course-outline.md 為 single source of truth，產生 Quarto 網站頁面。

產出：
  weeks/w01.qmd … w14.qmd   每週一頁
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

# 網站骨架（首頁、導覽列、每週索引）一律英文；站名與每週內頁維持中文。
PART_OF = {}
for w in range(1, 7):
    PART_OF[w] = "Part I — Foundations"
for w in range(7, 11):
    PART_OF[w] = "Part II — Modules"
for w in range(11, 15):
    PART_OF[w] = "Part III — Dialogue Systems"

# 首頁課程地圖的英文版（手寫，ASCII 對齊敏感）。
# 中文版仍留在 docs/course-outline.md（離線文件），不上網站。
MAP_EN = os.path.join(ROOT, "docs", "course-map-en.md")

# syllabus.qmd 與 resources.qmd 引用的英文片段（手寫，依 `<!-- file: X -->` 切段）
SITE_EN = os.path.join(ROOT, "docs", "site-en.md")
FILE_RE = re.compile(r"^<!-- file: (\S+) -->$")

# 每週英文標題寫在 docs/course-outline.md 的標題下一行：<!-- en: ... -->
EN_RE = re.compile(r"^<!--\s*en:\s*(.+?)\s*-->$")


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


def strip_en(lines):
    """英文標題註解只給建置腳本讀，不進入網頁內容。"""
    return [ln for ln in lines if not EN_RE.match(ln.strip())]


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


def read_site_en():
    """把 docs/site-en.md 依 `<!-- file: X -->` 切成 {檔名: 內容}。"""
    if not os.path.exists(SITE_EN):
        sys.exit("找不到 %s（syllabus / resources 的英文片段來源）" % SITE_EN)
    with io.open(SITE_EN, encoding="utf-8") as f:
        lines = f.read().replace("\r\n", "\n").split("\n")
    out, cur = {}, None
    for ln in lines:
        m = FILE_RE.match(ln.strip())
        if m:
            cur = m.group(1)
            out[cur] = []
        elif cur is not None:
            out[cur].append(ln)
    return dict((k, "\n".join(trim(v))) for k, v in out.items())


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
            nxt = lines[i + 1].strip() if i + 1 < len(lines) else ""
            me = EN_RE.match(nxt)
            if not me:
                sys.exit("W%s 缺英文標題：請在標題下一行加 <!-- en: ... -->"
                         % m.group(1))
            week_heads.append((i, int(m.group(1)), m.group(2).strip(),
                               me.group(1).strip()))

    if len(week_heads) != 14:
        sys.exit("預期 14 個週次標題，實際找到 %d 個" % len(week_heads))

    written = []
    # 分隔列的破折號長度決定欄寬比例（Topic 欄放雙語標題，需要最多空間）
    index_rows = ["| Week | Topic | Part | Slides |",
                  "|:---|:-----------------------------|:------|:------|"]

    def slides_for(n):
        """該週是否已有投影片。存在才給連結，避免死連結。"""
        rel = os.path.join("slides", "w%02d.qmd" % n)
        return rel if os.path.exists(os.path.join(ROOT, rel)) else None

    for idx, (li, wnum, title, en_title) in enumerate(week_heads):
        body = trim(strip_en(strip_hr(demote(slice_section(lines, li)))))
        prev_link = ("weeks/w%02d.qmd" % week_heads[idx - 1][1]) if idx > 0 else None
        next_link = ("weeks/w%02d.qmd" % week_heads[idx + 1][1]) if idx + 1 < len(week_heads) else None
        fm = [
            "---",
            # 側欄的項目直接取自 title，所以英文標題放 title、中文降為 subtitle
            'title: "W%d｜%s"' % (wnum, yaml_escape(en_title)),
            'subtitle: "%s · %s"' % (PART_OF[wnum], yaml_escape(title)),
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
        sl = ("[▶ Open](slides/w%02d.qmd)" % wnum) if slides_for(wnum) else "—"
        index_rows.append("| **W%d** | [%s](weeks/w%02d.qmd)<br>[%s]{.wk-zh} | %s | %s |"
                          % (wnum, en_title, wnum, title,
                             PART_OF[wnum].split(" — ")[-1], sl))

    # ── 2. syllabus.qmd 與 resources.qmd 引用的片段 ───────────
    # 網站是英文的，這些片段改由手寫的 docs/site-en.md 提供；
    # 中文原文留在 docs/course-outline.md，是離線閱讀用的文件，不上網站。
    en_parts = read_site_en()
    for fname in ("disclaimer.md", "latency.md", "textbooks.md",
                  "reading-table.md", "toolchain.md"):
        if fname not in en_parts:
            sys.exit("docs/site-en.md 缺區段 <!-- file: %s -->" % fname)
        written.append(write(os.path.join(INC_DIR, fname),
                             BANNER + "\n\n" + en_parts[fname] + "\n"))

    # 首頁的課程地圖：ASCII 對齊敏感，單獨一檔手工維護
    if os.path.exists(MAP_EN):
        with io.open(MAP_EN, encoding="utf-8") as f:
            written.append(write(os.path.join(INC_DIR, "coursemap.md"),
                                 BANNER + "\n\n" + f.read().strip() + "\n"))
    else:
        print("  ! 找不到 %s，首頁課程地圖未更新" % MAP_EN)

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
    # 分隔列的破折號長度決定欄寬比例（Slides 欄要放得下 "Not yet available"）
    rows = ["| Week | Topic | Slides |",
            "|:---|:--------------------------|:--------|"]
    have = 0
    for _, wnum, title, en_title in week_heads:
        topic = "%s<br>[%s]{.wk-zh}" % (en_title, title)
        if slides_for(wnum):
            have += 1
            rows.append("| **W%d** | [%s](weeks/w%02d.qmd)<br>[%s]{.wk-zh} | "
                        "[▶ Open](slides/w%02d.qmd) |"
                        % (wnum, en_title, wnum, title, wnum))
        else:
            # 不斷行空格：避免窄欄位把 "Not yet available" 折成兩行
            rows.append("| W%d | %s | *Not\u00a0yet\u00a0available* |" % (wnum, topic))
    written.append(write(os.path.join(ROOT, "slides.qmd"), "\n".join([
        "---",
        'title: "Slides"',
        'subtitle: "English slides, Mandarin delivery — %d of 14 weeks ready"' % have,
        "toc: false",
        "---",
        "",
        BANNER,
        "",
        "The slides are written in English to accompany the Mandarin lectures. "
        "Each deck is a reveal.js page — open it in a browser, nothing to install.",
        "",
        "::: {.callout-tip}",
        "## Keyboard shortcuts",
        "`B` chalkboard layer for writing on a slide · `ESC` slide overview · "
        "`F` full screen · add `?print-pdf` to the URL to print",
        ":::",
        "",
    ] + rows + [
        "",
        "The decks embed audio, so check that sound is on. Some clips are "
        "channel-separated (left = user, right = model); headphones or stereo "
        "speakers are recommended.",
        "",
    ])))

    print("產生 %d 個檔案：" % len(written))
    for p in written:
        print("  " + os.path.relpath(p, ROOT))


if __name__ == "__main__":
    main()

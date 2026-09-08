# Speech-AI

語音處理與人機互動（Speech Processing and Human-Machine Interaction）課程開發資料庫。

**課程網站 → <https://hungshinlee.github.io/Speech-AI/>**

課程以**建構半雙工／全雙工語音對話 AI 系統**為最終目標，涵蓋訊號處理、表徵學習、生成模型與互動建模，內容對齊 2025–2026 的技術脈絡。

## 內容

- [`docs/course-outline.md`](docs/course-outline.md) — 14 週課程大綱（每週含 learning objectives、關鍵數學骨架、常見誤解、課堂時間分配、demo 建議，以及推薦的書籍章節與論文）
- [`CLAUDE.md`](CLAUDE.md) — 課程設定、已定案的設計決策、寫作與引用規範（給 AI 協作與未來的自己看）

## 課程設定

| 項目 | 設定 |
|---|---|
| 授課教師 | [李鴻欣 Hung-Shin Lee](https://web.ntnu.edu.tw/~hslee/) |
| 授課對象 | 碩士班 / 博士班 |
| 時數 | 14 週 × 3 小時，全為 lectures |
| 語言 | 中文授課、英文投影片 |
| 網站語言 | 網站介面與框架為英文（首頁、課程資訊、教材與資源、投影片索引、導覽列、每週標題）；每週內頁與投影片講稿為中文 |
| 學生算力 | Colab 免費版 |
| 評量 | 期中報告 40%、期末論文 60% |
| 修課要求 | 分組，參與 2 次線上討論 |

## 網站建置

網站用 [Quarto](https://quarto.org) 建置，push 到 `main` 後由 GitHub Actions 自動部署。

```bash
brew install --cask quarto        # 首次
python3 scripts/build_weeks.py    # 從大綱重建每週頁面
quarto preview                    # 本機預覽
```

`weeks/`、`_includes/`、`slides.qmd` 與 `notes/` 由腳本自動產生，請勿直接編輯：

- 課程內容改 `docs/course-outline.md`，然後 `python3 scripts/build_weeks.py`
- 每週的英文標題寫在大綱裡標題的下一行（`<!-- en: ... -->`），供首頁索引與投影片使用；缺了腳本會報錯
- 首頁的英文課程地圖是手寫的 `docs/course-map-en.md`（ASCII 對齊敏感，不會自動更新）
- 課程資訊與教材頁的英文片段是手寫的 `docs/site-en.md`；`docs/course-outline.md` 是離線閱讀用的中文完整大綱，不上網站，兩邊各自維護
- 投影片講稿改 `slides/wNN.qmd` 的 `::: {.notes}`，然後 `python3 scripts/extract_notes.py`

## 上課時看中文講稿

| 情境 | 做法 |
|---|---|
| 單螢幕備課 | 投影片網址加 `?showNotes=true` |
| 有第二螢幕 | 按 `S` 開 presenter view |
| 上課（畫面給學生看） | 讀 [`notes/w01.md`](notes/w01.md)，放平板或手機 |

## 待辦

- [ ] 各週投影片（`slides/`）
- [ ] 課堂 demo notebooks（`demos/`）
- [ ] 核對大綱中標記 `[驗]` 的引用

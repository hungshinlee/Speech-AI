# Speech-AI

語音處理與人機互動（Speech Processing and Human-Machine Interaction）課程網站。

**課程網站 → <https://hungshinlee.github.io/Speech-AI/>**

課程以**建構半雙工／全雙工語音對話 AI 系統**為最終目標，涵蓋訊號處理、表徵學習、
生成模型與互動建模，內容對齊 2025–2026 的技術脈絡。

## 這個 repo 放什麼

只放公開內容：Quarto 站台、投影片，以及由課程大綱**過濾後產生**的每週頁面。

課程大綱正本、題目與 rubric、教學設計筆記在另一個 private repo，不在這裡。
每週頁面只公開「定位 / Learning objectives / 參考資料」三個區塊；
課堂時間分配、常見誤解排除、demo 腳本屬於授課用，不上站。

## 課程設定

| 項目 | 設定 |
|---|---|
| 授課教師 | [李鴻欣 Hung-Shin Lee](https://web.ntnu.edu.tw/~hslee/) |
| 授課對象 | 碩士班 / 博士班 |
| 時數 | 14 週 × 3 小時，全為 lectures |
| 語言 | 中文授課、英文投影片 |
| 網站語言 | 介面與框架英文（首頁、課程資訊、教材與資源、投影片索引、導覽列、每週標題）；每週內頁與投影片講稿中文 |
| 學生算力 | Colab 免費版 |
| 評量 | 期中報告 40%、期末論文 60% |
| 修課要求 | 分組，參與 2 次線上討論 |

## 網站建置

Quarto 建置，push 到 `main` 後由 GitHub Actions 部署。CI 只做 `quarto render`，
不重建每週頁面——大綱不在這個 repo 裡。

```bash
brew install --cask quarto                              # 首次
export COURSE_OUTLINE=~/Course-Hub/speech_ai/course-outline.md
python3 scripts/build_weeks.py                          # 從大綱重建（已過濾）
quarto preview                                          # 本機預覽
```

`weeks/`、`_includes/`、`slides.qmd`、`supplements/*.qmd` 由腳本自動產生，**請勿直接編輯**
（手改會在下次 build 被無聲覆蓋，而且可能把不該公開的內容帶上站）：

- 課程內容改大綱正本，然後重跑 `build_weeks.py`
- 補充教材改 `supplements/*.md`（H1 下方需有 `en` / `order` / `summary` 三行註解），同樣重跑
- 每週英文標題寫在大綱標題的下一行（`<!-- en: ... -->`），缺了腳本會報錯
- 課程資訊與教材頁的英文片段是手寫的 `docs/site-en.md`，不會自動更新
- 首頁課程地圖由 `build_weeks.py` 產生 HTML grid（短標籤在腳本的 `SHORT`）；
  `docs/course-map-en.md` 保留備查，已不上站
- 投影片講稿改 `slides/wNN.qmd` 的 `::: {.notes}`，然後 `python3 scripts/extract_notes.py`

哪些區塊會上站由 `scripts/visibility.py` 決定，規則寫在該檔的 docstring。

## 上課時看中文講稿

| 情境 | 做法 |
|---|---|
| 單螢幕備課 | 投影片網址加 `?showNotes=true` |
| 有第二螢幕 | 按 `S` 開 presenter view |
| 上課（畫面給學生看） | 讀 `extract_notes.py` 產出的講稿檔，放平板或手機 |

## 待辦

- [ ] 各週投影片（`slides/`）
- [ ] 課堂 demo notebooks（`demos/`）

# Speech-AI — 語音處理與人機互動

**Speech Processing and Human-Machine Interaction** · English version: [README.md](README.md)

碩士班／博士班課程：14 週 × 3 小時，全為 lectures。中文授課，投影片與課程網站為英文。

**課程網站：<https://hungshinlee.github.io/Speech-AI/>**
授課教師：[李鴻欣 Hung-Shin Lee](https://web.ntnu.edu.tw/~hslee/)

這個 repo 放的是課程**對外公開**的部分——網站原始碼、投影片、課堂 demo 程式與補充教材。產生這些內容的教學材料（完整課程大綱、講稿、題目與 rubric）在另一個 private repo，不在這裡；見〈[刻意不放在這裡的東西](#刻意不放在這裡的東西)〉。

## 一句話說這門課

整門課從一個問題倒推回來：**怎麼建一個能邊說邊聽、能被打斷、知道什麼時候輪到自己說話的語音對話系統？** 以「半雙工／全雙工語音對話 AI 系統」為終點，倒推所需的訊號處理、表徵學習、生成模型與互動建模知識，只教那些，對齊 2025–2026 的技術狀態。深度設定在研究所：含推導骨架、每個想法背後的文獻脈絡與 open problems。

三條主軸貫穿十四週，對每一層都問一次：**表徵**（這一層用什麼表徵——waveform、spectrogram、continuous SSL feature、discrete token——誰決定 frame rate）、**延遲**（這個模組吃掉多少 latency budget，是 lookahead 造成的演算法延遲還是計算延遲）、**監督**（這個能力從哪來——標註資料、自監督、合成資料，還是人類偏好）。

## 課程地圖

| Part I — 基礎 | Part II — 模組 | Part III — 對話系統 |
|---|---|---|
| W1 語音對話系統的系統觀與延遲預算 | W7 語音辨識：三種範式、streaming 與 LLM 化 | W11 Audio-native 語言模型：半雙工語音對話的建成 |
| W2 語音訊號、聽覺前端與表徵的物理基礎 | W8 語音合成 I：生成範式與 streaming 合成 | W12 全雙工 I：turn-taking 的語言學基礎與建模 |
| W3 對齊問題：從 HMM 到 CTC | W9 語音合成 II：可控性、後訓練對齊、評估與濫用防範 | W13 全雙工 II：架構路線、多串流建模與資料問題 |
| W4 序列模型與 streaming 架構：Transformer、Conformer、RNN-T | W10 全雙工的「物理層」：VAD、AEC、增強、分離與說話人 | W14 評估、對齊、落地與 open problems |
| W5 自監督表徵學習（SSL）：語音基礎模型的來源 | | |
| W6 Neural audio codec 與離散化：語音成為「語言」的關鍵一步 | | |

**W6 是全課樞紐。** Part III 的一切都建立在那一週定下的三件事上——frame rate、token budget、語意與聲學資訊的分離——所以這一週不能缺席。W3、W6、W8 是三個數學高峰。

## 目前已公開的內容

週次隨授課進度逐週開放。截至 2026 年 10 月，**W1–W4** 已開放：每週頁面、投影片，以及每一週的 demo 程式。其餘週次在課程地圖上以灰色占位顯示；頁面已產生但不 render，所以沒有死連結。

| | W1 | W2 | W3 | W4 | W5–W14 |
|---|---|---|---|---|---|
| 每週頁面（定位、learning objectives、參考資料、demos） | ✓ | ✓ | ✓ | ✓ | 已產生，尚未公開 |
| 投影片（`slides/wNN.qmd`，reveal.js） | ✓ | ✓ | ✓ | ✓ | — |
| Demo 程式（`demos/wNN_dK_*/`） | ✓ | ✓ | ✓ | ✓ | — |

**Syllabus** 頁有課程設定、三段式結構，以及**延遲預算**——全課共用的座標系，每一週都拿它來量。**Resources** 頁列核心教科書、每週一篇論文、demo 跑在上面的工具鏈，以及示範硬體上做不到的事。**Supplements** 有四份中文指南：以臺灣本土語言為場景、單張 16 GB GPU 為預算的十個 INTERSPEECH 2027 研究選題、期中 PoC 報告、期末論文，以及語音論文的寫作與敘事邏輯。

## 給修課同學

- **你不需要跑任何東西。** 本課全為 lectures，所有 demo 由授課者現場執行；投影片上的每個數字都來自 `demos/*/runs/rehearsal/` 裡進版控的彩排紀錄。程式公開是讓你能讀、能重跑、能改。
- **投影片沒有講稿。** 在 deck 裡按 `S` 打開的講稿欄是空的——那是刻意的，不是壞掉：中文講稿屬於教學材料，不公開。
- **Demo 只在一台機器上測過**——MacBook Pro M5 Max（64 GB 統一記憶體、無 CUDA）。`demos/README.md` 說明哪些 demo 可攜（純標準函式庫，或純 CPU 的 numpy／scipy／PyTorch）、哪些需要 Apple silicon（`mlx`）；所有套件與模型版本釘在 `demos/versions.lock`。授課者自己的錄音（W2 demo 2、W3 demo 1 的素材）不公開，那兩份 README 寫明怎麼自己錄。
- **引用上站前會先過濾。** 大綱裡標為未查證的條目直接不上站，而不是附一句但書放上去；W2–W4 改寫時新增的引用都已對 arXiv 摘要頁、ACL Anthology、ISCA archive、PMLR 或 Crossref 比對過 title 與第一作者。來源若是 blog、model card 或規格文件而非同儕審查論文，句子本身會寫明。
- 網站有**展示模式**（按 `z` 或點 navbar 的按鈕），會收起兩側欄位方便投影。

## Repo 結構

這個 repo 大部分是**產物**。標 ⚙︎ 的檔案在下次發佈時會被覆蓋，不要手改。

| 路徑 | 說明 |
|---|---|
| `index.qmd`、`syllabus.qmd`、`resources.qmd`、`supplements.qmd` | 手寫的站台頁面（英文） |
| `docs/site-en.md` | Syllabus 與 Resources 引用的英文段落來源（延遲預算、教科書、每週一篇論文、工具鏈） |
| `docs/course-map-en.md` | 英文課程地圖，保留備查；首頁上的那份是產生的 |
| `weeks/w01.qmd` … `w14.qmd` | ⚙︎ 每週頁面，由 `scripts/build_weeks.py` 從 private 大綱過濾產生 |
| `_includes/*.md` | ⚙︎ 產生的片段：課程地圖、週次表、閱讀表、延遲預算、教科書、工具鏈 |
| `slides.qmd` | ⚙︎ 投影片索引 |
| `slides/wNN.qmd`、`slides/theme.scss`、`slides/assets/` | ⚙︎ **剝除講稿**後的投影片，從 private 正本發佈 |
| `slides/scripts/` | ⚙︎ 投影片引用的分析腳本（雙聲道錄音的 energy VAD 與 overlap／response／stop latency、雙軌時間軸圖） |
| `demos/` | ⚙︎ 課堂 demo 程式、彩排紀錄與學生版 README，白名單制發佈；入口是 `demos/README.md` |
| `supplements/*.md` | 手寫的補充教材（中文）；旁邊的 `.qmd` 包裝是 ⚙︎ 產物 |
| `scripts/build_weeks.py`、`scripts/visibility.py` | 過濾器：大綱哪些區塊公開（預設不公開、白名單制）、引用標記怎麼處理、哪些週次已開放 |
| `styles.scss`、`_present-mode.html`、`_quarto.yml` | 站台主題、展示模式、Quarto 設定 |
| `.github/workflows/publish.yml` | CI：洩漏檢查加 `quarto render`，部署到 GitHub Pages |
| `CLAUDE.md` | 維護筆記：建置流程、渲染陷阱、決策 |

## 刻意不放在這裡的東西

課程大綱（`course-outline.md`）是這門課唯一的真相來源，只存在於 private repo。站上的每週頁面是從它經**預設不公開的過濾器**產生的：每週只有三個區塊公開——*定位*、*Learning objectives*、*參考資料*——加上連到已公開程式的學生版 *Demos* 區塊。課堂時間分配、學生誤解清單、給授課者的 demo 腳本與教學設計備註都不公開。投影片裡的中文講稿、授課者版的 demo README（彩排筆記與退路）、授課者自己的錄音、所有評量材料（題目與 rubric）亦同。

若這個 repo 裡出現 `docs/course-outline.md`，那是回歸 bug，不是功能。

## 網站怎麼建

正常入口是 private repo 的 `bin/publish.py`。它一次做完：從投影片正本重抽講稿、發佈剝除講稿的投影片、複製白名單上的 demo、跑每週頁面的過濾建置，最後檢查沒有教師端內容外洩——產物裡只要殘留一個 `::: {.notes}` 區塊就中止發佈。它**不會**自動 commit 或 push，那兩步手動做，這樣一定看得到 diff。

```bash
cd ~/Course-Hub && python3 bin/publish.py speech_ai
```

只要從大綱重建每週頁面時（這條路不更新投影片與 demo）：

```bash
export COURSE_OUTLINE=~/Course-Hub/speech_ai/course-outline.md
python3 scripts/build_weeks.py   # 改完大綱後必跑
quarto preview                   # 本機預覽（brew install --cask quarto）
quarto render                    # 產出 _site/
```

Push 到 `main` 觸發 GitHub Actions，但 CI **不重建**每週頁面（它讀不到大綱），只做洩漏檢查與 `quarto render`。這個設計的代價是：忘記在本機重跑建置，網站就會漂移。

開放新的一週要同時改兩處：`scripts/build_weeks.py` 的 `PUBLISHED_WEEKS` 加上該週（控制首頁、課程地圖與 resources 是否給連結），以及 `_quarto.yml` 的 `render` 清單與 sidebar 加上該頁（控制頁面是否存在）。

補充教材放 `supplements/*.md`，H1 下方要有 `en`／`order`／`summary` 三行註解；每週的英文標題寫在大綱週次標題的下一行（`<!-- en: ... -->`），缺了建置會報錯。

## 語言

站台骨架與已公開每週頁的內文都是英文。每週的英文版寫在大綱裡與中文並列，建置時抽出；沒有英文區塊的週次退回中文而不會壞掉，建置結束會印出尚待翻譯的週次清單。兩處刻意保留中文：每週頁的 `subtitle`（中文週次標題，供對照）與 `supplements/` 的指南，後者對齊授課語言。

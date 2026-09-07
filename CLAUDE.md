# CLAUDE.md

給在此 repo 工作的 Claude 讀的常駐記憶。**先讀這份，再動任何檔案。**

---

## 1. 這個 repo 是什麼

`語音處理與人機互動` 這門課的教材開發庫。課程以**建構半雙工／全雙工語音對話 AI 系統**為最終目標，倒推所需的訊號處理、表徵學習、生成模型與互動建模知識。

Remote：`git@github.com:hungshinlee/Speech-AI.git`（branch `main`）

---

## 2. 課程常駐前提（不要重複問）

| 項目 | 設定 |
|---|---|
| 課程名稱 | 語音處理與人機互動 |
| 授課對象 | 碩士班 / 博士班 |
| 先修背景 | **無**（線性代數、機率、Python 僅為軟性要求） |
| 時數 | 14 週 × 3 小時，**全為 lectures** |
| 評量 | **無**（不需要作業、考題、rubric） |
| 學生算力 | **Colab 免費版**（T4 16 GB） |
| 語言 | 中文授課、**英文投影片** |
| 深度分級 | 預設 `研究所`（推導、論文脈絡、open problems） |

「先修無」× 「碩博班」× 「要詳細數學」是刻意接受的張力。處理方式：補課責任集中在 W2 後半與 W3 前半，第一週發兩頁數學前置清單。

---

## 3. 目前狀態

```
├── CLAUDE.md                 ← 本檔
├── README.md                 repo 說明
├── docs/
│   └── course-outline.md     ★ single source of truth（14 週完整大綱，~117 KB）
│
├── _quarto.yml               Quarto 網站設定（導覽、主題、KaTeX）
├── styles.scss               自訂樣式（含手機版規則）
├── index.qmd                 網站首頁
├── syllabus.qmd              課程資訊（設定、閱讀方式、引用規範）
├── resources.qmd             教材與資源（教科書、速查表、工具鏈）
├── weeks/                    ⚙︎ 自動產生，勿手改
│   ├── w01.qmd … w14.qmd     每週一頁
│   └── all.qmd               完整大綱單頁（不含附錄 C）
├── _includes/                ⚙︎ 自動產生的頁面片段
├── scripts/
│   └── build_weeks.py        切頁腳本
└── .github/workflows/
    └── publish.yml           push 到 main 就自動 build + deploy
```

`docs/course-outline.md` 是所有後續產出的**單一真相來源**。投影片、demo notebook 都應該對得回它的週次編號與 learning objectives。

同一份已上傳到 Claude Project「教學｜Speech AI」的 `claude/course-outline-14weeks.md`。**若改動 `docs/course-outline.md`，記得用 `project_write` 同步該 doc**，否則兩邊會漂移。

---

## 4. 已定案的設計決策（不要在沒被要求的情況下推翻）

| 決策 | 內容 | 理由 |
|---|---|---|
| **週數配比** | 基礎 6（W1–W6）+ 模組 4（W7–W10）+ 對話系統 4（W11–W14） | 使用者於 2026-09 明確選定，平衡零先修學生與最終目標 |
| **數學密度** | 只列**關鍵式子與推導骨架**（關鍵步驟、符號定義、卡點提示），不展開完整代數 | 方便直接轉成投影片；完整推導留給白板 |
| **JM3 章節編號** | 固定引用 **2026-08-19 release**（Ch 15 前端／Ch 16 ASR／Ch 17 TTS／Ch 26 對話結構／App A HMM／App K frame-based dialogue） | SLP3 是 draft，章節會重排。**換 release 必須全檔同步改** |
| **W6 不可壓縮** | Neural codec / tokenization 是樞紐週；W11–W13 完全建立其上 | 進度落後時要壓的是 W7（ASR 三範式可講快）與 W10（本來就是掃描式整併週） |
| **三條貫穿主軸** | 表徵軸（用什麼表徵、誰決定 frame rate）／延遲軸（吃掉多少 latency budget）／監督軸（能力從哪來） | 每週投影片都要回扣一次 |
| **延遲預算表** | 同一張投影片母版每週回填，學期末成為完整的系統延遲解剖圖 | 全課的黏著劑，W1、W4、W6、W8、W10、W13 都有回填點 |
| **失敗案例優先** | 六個核心 demo：Griffin-Lim 相位、CTC blank 熱圖、Whisper 幻覺、vocoder artifacts、增強後 WER 反而上升、搶話／遲鈍 | 先播壞的再解釋機制。這六個優先投資製作時間 |
| **不含評量設計** | 大綱不放作業、考題、rubric | 課程設定為無評量 |

---

## 4.5 課程網站（Quarto）

網址：**https://hungshinlee.github.io/Speech-AI/**

### 最重要的一條紀律

`docs/course-outline.md` 是唯一真相來源。`weeks/` 與 `_includes/` **全部由 `scripts/build_weeks.py` 產生**，檔頭都有「勿直接編輯」的註解。

```bash
# 改完 docs/course-outline.md 之後：
python3 scripts/build_weeks.py
```

CI 在 deploy 前也會重跑一次這個腳本，所以就算忘了在本機跑，線上內容也不會漂移；但**手改 `weeks/*.qmd` 的內容會在下次 build 被無聲覆蓋**。

### 已定案的網站決策

| 決策 | 內容 | 理由 |
|---|---|---|
| 工具鏈 | **Quarto** | 原生 KaTeX（大綱有大量行內／display math）；同一套 `.qmd` 之後可 render revealjs 投影片，與網站共用內容 |
| 大綱切頁 | 14 週各一頁 + `all.qmd` 完整單頁 | 1170 行單頁在手機上不可用；完整版留給列印與全文瀏覽 |
| 網址 | `hungshinlee.github.io/Speech-AI/`（子路徑） | Quarto 用相對連結，子路徑可直接運作，不需改設定 |
| KaTeX 版本 | **釘在 0.18.7**（`_quarto.yml` 的 `html-math-method.url`） | Quarto 預設載 `katex@latest`，上游改版會無聲壞掉 |
| 附錄 C | **不上網站**，只存在 `docs/course-outline.md` | 那是授課者私用的課程設計備註 |
| 投影片與 demo | **由授課者自行製作**，AI 不代勞 | 使用者明確指示。之後放 `slides/` 與 `demos/`，並在 `_quarto.yml` 的 sidebar 加入口 |

### 手機閱讀：實際會壞的三個地方

`styles.scss` 針對這三項各有處理，改樣式前先了解為什麼這樣寫：

1. **寬表格**（課堂骨架的時間分配表）→ `≤768px` 時 `table` 改 `display:block; overflow-x:auto`，並給 cell `min-width`（最後一欄 15rem），保留換行可讀性而非壓成窄柱。
2. **ASCII 課程地圖**（`pre`）→ `white-space:pre` + `overflow-x:auto`，字級降到 `.74rem`；靠橫向捲動而非換行，否則圖會散掉。
3. **長行 display math** → `.katex-display{overflow-x:auto}`。

驗證過的結果：390px 寬下 4 個代表頁面的 `document.scrollWidth === clientWidth`，**沒有頁面級橫向溢出**；寬表格與 ASCII 圖各自在容器內捲動（右側有 inset shadow 提示）。改完樣式建議重跑這個檢查。

### 本機預覽

容器與此 VM 都沒裝 Quarto，需要在你的 Mac 上安裝一次：

```bash
brew install --cask quarto
quarto preview          # 本機即時預覽
quarto render           # 產出 _site/
```

### 首次上線需要手動做一次

GitHub repo → **Settings → Pages → Build and deployment → Source 設為 `GitHub Actions`**。workflow 用的是 `upload-pages-artifact` + `deploy-pages`，不走 `gh-pages` 分支，所以 Source 選錯會 deploy 失敗。

---

## 4.6 音訊素材與時間軸圖（W1 起）

### 素材來源與授權

課堂音訊來自 **Kyutai 的 `interactivity-alignment-samples`**（HuggingFace dataset，**CC-BY-4.0**），
即 Ohashi, Zeghidour, Défossez & Kharitonov, *Multi-Faceted Interactivity Alignment in
Full-Duplex Speech Models*, EMNLP 2026（arXiv:2606.11167）——這篇在 W14 的參考文獻裡已經有了，
第一週開場的音訊、最後一週回來讀那篇論文，是刻意安排的閉環。

**每一張用到這些音訊或其衍生圖的投影片都必須標註**（CC-BY 的要求）：

> Audio: Ohashi, Zeghidour, Défossez & Kharitonov, *Multi-Faceted Interactivity Alignment
> in Full-Duplex Speech Models*, EMNLP 2026 (arXiv:2606.11167).
> Samples from `kyutai/interactivity-alignment-samples`, CC-BY-4.0.

檔案格式：stereo WAV，**channel 0 = 輸入者、channel 1 = 模型輸出**。保留 stereo 很重要——
教室播放時使用者在左耳、模型在右耳，重疊與否直接聽得出來。

### 檔案配置

| 位置 | 內容 | 版控 |
|---|---|---|
| `_media/` | 從 HF 下載的原始 WAV（143 MB） | **不進版控**（`.gitignore`）；`_` 開頭 Quarto 也忽略 |
| `slides/assets/w01/` | 裁切後的 `.m4a` + `.opus`（140–220 KB）與時間軸 SVG | 進版控 |

`.gitignore` 擋掉 `*.wav`，所以投影片用的音訊一律轉成 m4a（AAC，相容性最好）+ opus 備援，
在 revealjs 裡用兩個 `<source>`。**HuggingFace 在雲端容器與本機 VM 都被代理擋掉（403）**——
要補下載素材必須在 Mac 上做，指令見對話記錄或 `_media/` 的下載模式。

### 兩支腳本

- `scripts/analyze_duplex_audio.py` — 能量式 VAD 量出互動時序（response latency、overlap、
  stop latency）。**指標命名刻意中性**（`usr_gap` 而非 `barge_in`），因為同一個時間量在不同
  task 代表不同現象；曾經因為命名帶了解釋而得出誤導結論。
- `scripts/make_duplex_timeline.py` — 產生雙軌時間軸 SVG（不需 matplotlib，直接寫 SVG）。

**設計原則：圖只呈現資料，論點寫在投影片文字上。** 所以同一張圖能在 W1 與 W12 用不同論述
重複使用。`no response for X s` 的自動標記有兩個嚴格條件——必須緊接使用者說完之後、且期間
使用者也沒在說話——否則會把「模型在對方講話時正確地保持安靜」誤標成缺陷。這個判定條件
改過兩次才對，不要隨意放寬。

### W1 已選定的素材

| 用途 | 檔案 | 關鍵數字 |
|---|---|---|
| Cold open | `synthetic_user_interruption/1`，Moshi base vs +Fisher | base：蓋過使用者 2.90 s，之後 10.2 s 無回應；Fisher：零重疊、60 ms 回應 |
| 行為分類 | `icc_backchannel/3`，PersonaPlex base vs +Fisher | base：7 s 後沉默 35 s，overlap 0.88 s；Fisher：11 段短發聲，overlap 4.20 s |

這兩組配在一起的教學價值在於：**同一個指標（overlap）在插話情境是壞事，在 backchannel
情境是好事。** 這一刀同時砍掉「backchannel 就是短的 turn」與「overlap 越少越好」兩個誤解。

`synthetic_pause_handling` **不用**：Moshi 是 ps3.0、PersonaPlex 是 ps0.0，兩個模型家族的輸入
條件不同，不是對等比較；且分析結果不可信（多個變體量到零發聲）。要用得先讀清楚論文的 task 定義。

---

## 4.7 投影片（W1 起，`slides/w01.qmd` 是模板）

### 格式決策

| 決策 | 內容 | 理由 |
|---|---|---|
| 格式 | **Quarto revealjs** | 決定性因素是**能嵌入音訊**（PPTX/PDF 做不好，而 cold open 靠聽）；其次是原生數學、與網站同一個 build、純文字可 diff、`chalkboard` 可在投影片上直接手寫（W3/W6/W8 的白板時間） |
| 語言分工 | 投影片英文、`::: {.notes}` 中文 | 對應「中文授課、英文投影片」。學生拿到投影片，拿不到 notes |
| 標題 | **assertion-evidence**：標題寫主張，不寫主題 | 不是 "Cascade Architecture"，而是 "Cascade survives because every module can be debugged separately"。標題就是要學生記住的那句話 |
| **不從大綱自動生成** | 投影片獨立撰寫 | 大綱是閱讀密度（連貫段落、完整論證），投影片是講述密度（一畫面一主張）。自動轉換必然產生 bullet 洪流 |
| 網站連結 | `build_weeks.py` 偵測 `slides/wNN.qmd` 存在才注入連結 | 手動維護 nav 一定會漏；不存在就不給連結，避免死連結 |
| 公開部署 | `slides/*.qmd` 在 `_quarto.yml` 的 render 清單內，一起上線 | 使用者決定。**但投影片應在課後才推上去**——cold open 的效果依賴學生沒有預先看過 |
| `navigation-mode` | **`linear`**（必須） | `#` 章節標題會產生垂直堆疊，預設模式下左右鍵會在章節間跳而不是逐張前進，講課時會出事。踩過了 |
| footer | 只用每張的 `::: {.footer}`，**不要設全域 `footer:`** | 兩者同時存在會疊字。踩過了 |
| KaTeX | 與網站同樣釘 0.18.7 | 一致性 |

### 檔案

```
slides/
├── theme.scss     與網站同色票；字級下限等效 24pt（研究所教室後排看得到才算數）
├── w01.qmd        36 張，對應大綱的 180 分鐘時間分配
└── assets/w01/    音訊（m4a + opus 雙 source）與 SVG 圖
```

`theme.scss` 提供的自訂 class：`.claim`（主張框，另有 `.warn`/`.ok`）、`.metrics`/`.metric`（量測數字卡，`.bad`/`.good`）、`.ask`（提問，留白讓學生先答）、`.audio-label`、`.cite`、`.small`、`.dim`、`.tag`、`.section-title`。

### W1 的敘事結構（後續各週可沿用）

1. **Cold open 先播壞的**（3 張）：兩段音訊 → 時間軸 → 主張。不先解釋要聽什麼。
2. **課程框架**（5 張）：目標、W6 是樞紐、三條主軸、實務說明。
3. **三種架構**（8 張）：同一張圖用兩次——第一次講它為什麼有效，第二次只講瓶頸。中間插一張真正停下來的提問。
4. **延遲**（7 張）：人類是瞄準目標不是最小化 → 誤解破解 → 預算母版 → 兩種延遲的公式 → p95 → live demo 骨架。
5. **四種行為**（6 張）：taxonomy → backchannel 對照 → 「同一個指標在兩個情境相反」。
6. **收尾**（4 張）：各週會填哪一格、下週、出處。

**全 deck 最重要的一張**是「the same measurement is a defect in one task and a requirement in the other」——2.90 s 的重疊是失敗，4.20 s 的重疊是進步。它同時砍掉兩個誤解並推出「沒有可單向最佳化的指標」。

### 產出 PDF

revealjs 的 PDF 走瀏覽器列印：開 `slides/w01.html?print-pdf` 後在 Chrome 列印。音訊與逐步揭示會掉，所以那是補充版而非主要交付。

### 待驗

- W1 從未在真實投影機上跑過。第一次上課前請在教室環境確認：字級、色彩對比、以及**音訊在教室音響上的左右聲道**（左=使用者、右=模型，是 cold open 的關鍵）。
- Stivers et al. 2009 那張投影片刻意**沒有**放常被引用的單一均值（例如「約 200 ms」）——那個數字我沒在論文正文讀到。若要放，請自己從 Fig. 1 / Table 1 讀出來。

---

## 5. 寫作與引用規範（**優先於一切**）

### 語言與格式
- 全程繁體中文；學術名詞、模型名稱、演算法、指標保留英文原文。
- 解釋性內容用連貫段落；比較性內容用表格；步驟才用 bullet。**避免無資訊量的 bullet 洪流。**
- 數學用 GitHub 支援的 `$...$` / `$$...$$`。

### 引用可信度三級標記（大綱中已在用，新增內容必須沿用）
| 標記 | 意義 |
|---|---|
| 無標記 | 領域內廣為引用的標準文獻，對標題／作者／年份有高信心 |
| `[驗]` | 可能記錯年份、會議或副標題，**放進投影片前必須查證** |
| `[主題]` | 刻意不給精確引用，只給檢索關鍵字 |

### 硬規則
1. **不編造引用**。不確定就用 `[主題]` + 關鍵字，不要生成看似合理的 paper 標題。
2. **不虛構數字**。大綱中**沒有任何** WER / MOS / PESQ / DER 數值，這是刻意的。benchmark 表格請直接引原論文，不要填我的印象值。
3. **2026 年的 arXiv 編號**（26xx.xxxxx）在 2026-09 那次已逐筆掃過 arXiv 檢索頁確認標題與編號；再新增時要同樣查證，不要憑印象寫。
4. 比喻要能延伸到後續單元，並**說明它在哪裡會失效**。一次性比喻反而製造混淆。
5. 每個概念的拆解路徑必須包含「**什麼情況下會壞掉**」—— 這項最常被省略，卻最能讓學生真的理解。

---

## 6. Demo 設計約束

- **Colab 免費版（T4 16 GB）跑不動即時語音互動。** 所有 demo 設計成「**離線推論 + 事後對齊時間軸**」，即時性用預錄的官方 demo 呈現。
- 首選工具鏈：`torchaudio` + HF `transformers`（安裝最省事）；ASR 完整流程用 ESPnet / k2-icefall；語音模組用 SpeechBrain；codec 用 `descript-audio-codec` / HF EnCodec / Mimi。
- 標註套件版本、資料集大小、預期執行時間。
- 權重下載時間是實際痛點 → notebook 要能課前預熱。
- 資料建議用 LibriSpeech dev-clean + MUSAN 各取數十句即可，不要動用大資料集。

---

## 7. 待辦

- [ ] **首次上線**：GitHub Pages 的 Source 設為 `GitHub Actions`（見 4.5）
- [ ] 核對大綱中所有 `[驗]` 標記的引用（約 15 處）
- [ ] `slides/` — 各週投影片（授課者自製；之後在 `_quarto.yml` sidebar 加入口）
- [ ] `demos/` — 課堂 demo notebooks（授課者自製）
- [ ] 第一週用的「數學前置清單」兩頁講義
- [ ] 延遲預算表的投影片母版（每週回填用）

## 8. 維護紀律

- **W11–W14 的文獻半衰期約 6 個月。** 每次開課前重掃 arXiv `cs.CL` / `eess.AS` 近三個月，以及 Interspeech / ICASSP / ASRU / SLT 最新議程。
- 動 `docs/course-outline.md` 之後：(a) **重跑 `python3 scripts/build_weeks.py`**；(b) 同步 Project doc（見第 3 節）；(c) 若改了週次結構或標題，回頭檢查附錄 A 速查表、`_quarto.yml` 的 sidebar 清單與 README。
- 新增內容時注意兩個會破渲染的陷阱：**markdown 表格的 cell 裡不能出現裸的 `|`**（行內數學請用 `\lvert … \rvert`），以及 **KaTeX 不支援 `\mathbb{1}`**（用 `\mathbf{1}`）。這兩個都踩過了。
- Commit 訊息用中文或英文皆可，但要說明**改了哪一週、改了什麼層級**（結構／內容／引用）。

---

## 9. 本 Project 的邊界

這個 repo 與其對應的 Claude Project 專用於**課程設計與教學**。若話題轉到論文審稿或投稿寫作，提醒切到學術 Project，但仍然回答。

---

_最後更新：2026-09-07（建立 14 週大綱 v1.0 + Quarto 課程網站）_

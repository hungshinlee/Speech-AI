# CLAUDE.md — Speech-AI 課程網站

這個 repo 只放**學生看得到的東西**：Quarto 站台、投影片，以及由課程大綱
**過濾後產生**的每週頁面。

教師端的正本——課程大綱、題目與 rubric、研究筆記、教學設計決定、素材授權——
在 private repo `hungshinlee/Course-Hub` 的 `speech_ai/`，那裡的 `CLAUDE.md`
是完整的工作手冊。**先讀那一份再動這裡的東西。**

## 鐵則

1. **大綱不在這個 repo，也不得搬回來。** 從 Course-Hub 複製任何內容進來，
   唯一的合法通道是 `scripts/build_weeks.py` + `scripts/visibility.py` 的過濾輸出。
2. **`weeks/*.qmd` 是產物且已過濾。** 手改會在下次 build 被無聲覆蓋，
   更糟的是可能把教師內容帶上站。要改內容請改 Course-Hub 的大綱。
3. **不確定某段內容該不該公開 → 不公開。** 少給的代價是學生看不到，
   多給的代價是教學底牌外流加上未查證引用被轉引。兩者不對稱。

## 0. 誰跑什麼

`device_bash` 跑的是使用者機器上一個**獨立的 Linux VM**，只掛載連進來的資料夾，
**不是** macOS 環境本身——brew 裝的工具、macOS 的 python env、GUI 應用都看不到。

網路（2026-09-14 實測，修正舊版「對外連線全被擋」的敘述）：

| 項目 | 狀況 |
|---|---|
| HTTPS 到允許清單內的主機 | **可以**。`git ls-remote https://github.com/…` 成功 |
| SSH 傳輸層 | **可以**，proxy 有 CONNECT 隧道。host key 從 `api.github.com/meta` 取得寫進 `~/.ssh/known_hosts`（不要用 TOFU） |
| SSH 認證 | **不行**，VM 裡沒有私鑰也沒有 agent forwarding → `git push` 由使用者在 Mac 上執行 |
| 裸 TCP（`/dev/tcp`）、DNS 直查 | 不行 |

刪檔預設被擋，要先 `device_request_delete_permission`（**每個 session 重新授權一次**）。
沒授權就 commit 會留下 `.git/HEAD.lock`，**下一次 commit 會卡住**——授權後
`find .git -name '*.lock' -delete` 清掉即可。

`quarto` 裝在 macOS 上，VM 看不到；要驗證 render 就在雲端容器裡另裝一份。
在雲端容器產生再傳回 repo 的檔案**會被加上 C2PA 標記**（SVG 每張多約 8 KB），
`make_figs_w01.py` 這類決定性腳本在 Mac 上重跑一次就會得到乾淨版本。

## 1. 建置

大綱在 private repo、腳本在這裡，用環境變數接：

```bash
export COURSE_OUTLINE=~/Course-Hub/speech_ai/course-outline.md
python3 scripts/build_weeks.py     # 產生過濾後的 weeks/*.qmd 與 _includes/
git add weeks/ _includes/ && git commit && git push
```

CI **不重建**每週頁面（它讀不到大綱），只跑 `quarto render`。取而代之的是一個
不需要大綱的洩漏檢查：`weeks/` 裡出現「課堂骨架 / 常見誤解 / 卡點提示 / [驗]」
就讓 build 失敗。

過濾規則（白名單、覆寫語法、`[驗]` 處理）定義在 `scripts/visibility.py` 的
docstring，改白名單就是改那裡的 `PUBLIC_SECTIONS`。

---

### 已定案的網站決策

| 決策 | 內容 | 理由 |
|---|---|---|
| 工具鏈 | **Quarto** | 原生 KaTeX（大綱有大量行內／display math）；同一套 `.qmd` 之後可 render revealjs 投影片，與網站共用內容 |
| 大綱切頁 | 網站只放 14 週各一頁 | 1170 行單頁在手機上不可用。曾經有一頁 `weeks/all.qmd` 把全部串起來供列印，**2026-09 移除**（使用者要求）；完整內容看 `docs/course-outline.md`，不要再自動產生那一頁 |
| 網址 | `hungshinlee.github.io/Speech-AI/`（子路徑） | Quarto 用相對連結，子路徑可直接運作，不需改設定 |
| KaTeX 版本 | **釘在 0.18.7**（`_quarto.yml` 的 `html-math-method.url`） | Quarto 預設載 `katex@latest`，上游改版會無聲壞掉 |
| 附錄 C | **不上網站**，只存在 `docs/course-outline.md` | 那是授課者私用的課程設計備註 |
| 網站語言 | **介面與框架英文、課程內文中文**。英文：`index.qmd`、`syllabus.qmd`、`resources.qmd`、`slides.qmd`、`supplements.qmd`、navbar／sidebar／footer、每週索引、每週頁與補充教材頁的 `title`、頁內 TOC 標題、repo-actions 與搜尋（後三項靠 `_quarto.yml` 的 `language:` 覆寫 Quarto 內建中文字串）。**`lang:` 必須寫 `zh-TW` 不能寫 `zh-Hant`**：Quarto 只有 `_language-zh-TW.yml`，給 `zh-Hant` 會退回簡體的 `_language-zh.yml`（畫面上會出現「切换阅读器模式」等 5 處簡體 tooltip）。中文：每週頁內文與 `subtitle`、`supplements/*.md` 的全部內文、投影片的 `::: {.notes}` 講稿 | 修課學生與外部訪客都要照顧。對外的門面一律英文，實際授課用的內文對齊「中文授課」。站名 `語音處理與人機互動` 不翻 |
| 週次索引 | 英文標題為連結、中文標題為次行（`[…]{.wk-zh}`） | 連結文字與點進去的中文頁面標題會對不起來，兩個都給才不會迷路 |
| 首頁課程地圖 | **HTML/CSS grid**，由 `build_weeks.py` 從週次資料產生（樣式在 `styles.scss` 的 `.coursemap`）（2026-09-11 由 ASCII 改成） | 方塊可點進該週頁面、文字進得了站內搜尋、窄螢幕自動疊成一欄——這三件事 ASCII 與 SVG 都做不到（SVG 以 `width:100%` 縮放時，390px 下 1240 寬的圖會縮到 0.31 倍，25px 的字變 8px）。短標籤在腳本的 `SHORT`，缺一週會 `sys.exit`；`docs/course-map-en.md` 保留備查但**不再上站** |
| 表格欄寬 | 用 pipe table 分隔列的**破折號長度**指定比例，不要留 `|---|---|` | Pandoc 依破折號長度分配欄寬；全部等長就是均分，雙語標題那一欄會被擠到不能看。`Not yet available` 這種不該斷行的短語用不斷行空格（U+00A0）釘住 |
| 投影片與 demo | **由授課者自行製作**，AI 不代勞 | 使用者明確指示。之後放 `slides/` 與 `demos/`，並在 `_quarto.yml` 的 sidebar 加入口 |
| 展示模式 | 按 `z` 或點 navbar 的按鈕切換，`localStorage` 持久化、換頁保持（`_present-mode.html` + `styles.scss` 的 `html.present-mode`，2026-09-11 自 NLP-LLM 移植） | 投影時 zoom in 會落在「兩側導覽還沒收起、內容卻被擠掉」的區間（實測 1100px 等效寬時內容只剩 570px，左右佔掉 428px）。**不能用 Quarto 內建的 reader-mode**：那顆按鈕的 inline `onclick` 呼叫 `window.quartoToggleReader`，它由 `quarto.js` 定義、在 `include-in-header` 之後才載入，會吃掉同名覆寫，而且它只調整過場速度、不收側欄。所以用自己的函式名 + 在 document 的 capture 階段攔截點擊。**關鍵**：Quarto 是 named-line grid，只把側欄 `display:none` 不會收回欄寬，要讓 `main.content` 跨到 `page-start / page-end` |

### 手機閱讀：實際會壞的三個地方

`styles.scss` 針對這三項各有處理，改樣式前先了解為什麼這樣寫：

1. **寬表格**（文獻速查表、教科書表；過濾前是課堂骨架的時間分配表）→ `≤768px` 時 `table` 改 `display:block; overflow-x:auto`，並給 cell `min-width`（最後一欄 15rem），保留換行可讀性而非壓成窄柱。
2. **ASCII 課程地圖**（`pre`）→ `white-space:pre` + `overflow-x:auto`，字級降到 `.74rem`；靠橫向捲動而非換行，否則圖會散掉。
3. **長行 display math** → `.katex-display{overflow-x:auto}`。
4. **行內 `code` 的長識別字**（`torch.nn.functional.scaled_dot_product_attention` 這種不可斷詞的單一 token）→ `code:not(pre code)` 先把 `white-space` 收回 `normal` 再給 `overflow-wrap: anywhere`。**只給 `overflow-wrap` 不會生效**，因為上游 `code{white-space:pre}`；`p code` 與 `td code` 上游已是 `pre-wrap`，所以**壞掉的是清單項目、標題、blockquote 裡的行內 code**。實測（Quarto 1.7.32）清單項目內的長識別字在 390px 下把頁面撐寬 143px，加上這兩行後歸零。`pre` 區塊仍維持 `white-space:pre` + 橫向捲動，程式碼與 ASCII 圖不會被折行。

驗證過的結果：390px 寬下 4 個代表頁面的 `document.scrollWidth === clientWidth`，**沒有頁面級橫向溢出**；寬表格與 ASCII 圖各自在容器內捲動（右側有 inset shadow 提示）。改完樣式建議重跑這個檢查。

### 本機預覽

容器與此 VM 都沒裝 Quarto，需要在你的 Mac 上安裝一次：

```bash
brew install --cask quarto
quarto preview          # 本機即時預覽
quarto render           # 產出 _site/
```

### 首次上線的一次性設定（**已完成，2026-09**）

GitHub repo → **Settings → Pages → Build and deployment → Source 設為 `GitHub Actions`**。workflow 用的是 `upload-pages-artifact` + `deploy-pages`，不走 `gh-pages` 分支，所以 Source 選錯會 deploy 失敗。這條留著是備忘：若之後 repo 轉移或重建，要重設一次。

---


## 4.7 投影片（W1 起，`slides/w01.qmd` 是模板）

### 格式決策

| 決策 | 內容 | 理由 |
|---|---|---|
| 格式 | **Quarto revealjs** | 決定性因素是**能嵌入音訊**（PPTX/PDF 做不好，而 cold open 靠聽）；其次是原生數學、與網站同一個 build、純文字可 diff、`chalkboard` 可在投影片上直接手寫（W3/W6/W8 的白板時間） |
| 語言分工 | 投影片英文、`::: {.notes}` 中文 | 對應「中文授課、英文投影片」。**注意：notes 以 `<aside class="notes">` 內嵌在公開的 HTML 裡，只是 CSS 隱藏——並非私密**。2026-09 起網站上**不再提示** `S` 與 `?showNotes=true`（不要再把那句話加回 `slides.qmd`），但那只是不主動告知，講稿仍在公開 HTML 裡；真要藏起來得在 deploy 前用 Lua filter 把 `.notes` div 拿掉 |
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
├── w01.qmd        42 張，對應大綱的 180 分鐘時間分配
└── assets/w01/    音訊（m4a + opus 雙 source）與 SVG 圖
```

`theme.scss` 提供的自訂 class：`.claim`（主張框，另有 `.warn`/`.ok`）、`.metrics`/`.metric`（量測數字卡，`.bad`/`.good`）、`.ask`（提問，留白讓學生先答）、`.audio-label`、`.cite`、`.small`、`.dim`、`.tag`、`.section-title`。

### W1 的敘事結構（後續各週可沿用）

1. **Cold open 先播壞的**（4 張）：兩段音訊 → 時間軸 → 對照 → 主張。不先解釋要聽什麼。
2. **課程框架**（7 張）：終點宣告 → **14 週依賴地圖**（`course-map.svg`）→ W6 是樞紐 → **今日 learning objectives** → 三條主軸 → 實務說明。
   LO 放在地圖之後：學生要先看到十四週長什麼樣，今天的四條目標才有掛的地方。
3. **三種架構**（10 張）：**可延伸的比喻**（對講機→電話→面對面，含失效點）→ 三張架構圖，其中 cascade 那張用兩次（第一次講為什麼有效、第二次只講瓶頸）→ 比較矩陣 → 誤解 1。中間插一張真正停下來的提問。
4. **延遲**（8 張）：人類是瞄準目標不是最小化 → 誤解 2 → 預算母版 → 兩種延遲的公式 → **誤解 3（延遲是研究問題）** → p95 → live demo 骨架。
5. **四種行為**（6 張）：taxonomy → backchannel 對照 → 「同一個指標在兩個情境相反」→ 誤解 4。
6. **收尾**（6 張）：各週會填哪一格 → **本土語言的資料與正字法缺口**（接 W14 open problems；講稿裡註明不要講成愛國動員）→ **讀論文的四個問題** → 下週 → 出處。

**四個誤解依 deck 順序編號**（1 全雙工=支援打斷／2 越快越好／3 延遲是工程問題／4 使用者永遠該贏）。
大綱 `docs/course-outline.md` 的 W1 誤解表已用同一組編號對齊；「end-to-end 一定更好」由比較矩陣那張點名，表裡標成 `—`。

**收尾那兩張的作用**不是補充，是把今天的內容轉成方法：資料缺口那張把「本土語言」從口號變成一個可做的題目；
四個問題那張把 cold open 的方法論（同一使用者音軌 → 差異可歸因）抽成每週讀論文都能用的檢查表。

**全 deck 最重要的一張**是「the same measurement is a defect in one task and a requirement in the other」——2.90 s 的重疊是失敗，4.20 s 的重疊是進步。它同時砍掉兩個誤解並推出「沒有可單向最佳化的指標」。

### 字級與圖內文字（2026-09 調整過一輪）

| 項目 | 值 |
|---|---|
| 正文（`$presentation-font-size-root`） | **34 px**（reveal 內部座標 1280×760） |
| 圖內文字的有效大小 | 22–28 px，即正文的 **66–83%** |

**關鍵觀念**：圖內文字在螢幕上的實際大小 = `SVG 字級 × (顯示寬度 ÷ viewBox 寬度)`。
所以**只放大 SVG 的字級沒有用**——如果為了容納變大的字而把方塊或箭頭也加寬，viewBox 跟著變寬，
縮放比例下降，等於原地踏步。第一次調整就踩了這個坑（架構圖 viewBox 從 1121 漲到 1277，淨增益幾乎為零）。

有效的做法是**把水平空間讓回給字**：

1. **移除線上重複的串流標籤**，底部放一次圖例（`legend()`）。省下每條箭頭約 100px。
2. **移除圖內的頂端標題**。投影片標題已經用 assertion-evidence 寫了同一件主張，圖再說一次是重複。
   這同時省下垂直空間。**新增圖時不要再放圖內標題。**

調整後：時間軸圖內文字從約 16.6 px 提到 25.6 px（**+54%**），架構圖 +34%，延遲預算 +40%。

### 表格字級：一律用絕對值，不要用 em

`theme.scss` 裡表格的 `font-size` 是 **絕對的 26px**（`table` 與 `th, td` 都明確寫），這是刻意的。

原因：投影片常加 `{.smaller}`（Quarto 對整個 section 套 `0.7em`）。若表格用相對字級，
就會與它相乘——W1 原本表格寫 `.82em`，實際算出 **19.5px**（正文 34px 的 57%），
而且**只有加了 `.smaller` 的投影片才會縮**，同一份 deck 裡的表格大小因此不一致。

所以：**新增表格樣式時不要改回 em**。要調整就改那個 px 值，全 deck 一起變、且必然一致。

一個副作用是好的：`.smaller` 現在只影響標題與周邊文字，不再影響表格。表格為主、
標題又短的投影片（W1 的 slide 18、28）可以直接不加 `.smaller`，讓標題回到正常大小。

放大表格後 W1 有兩張溢出（slide 7 −40px、slide 18 −66px），修法同樣是**減字**：
那兩張的儲存格原本是整段散文，改成以 `·` 分隔的關鍵詞。比較矩陣本來就不該放句子。

### 目標顯示尺寸：**只需要考慮 1280×800**

使用者已明確界定唯一需要顧的尺寸。**不必為手機、平板或其他長寬比做任何調整**，
新 deck 也不需要跑多視窗尺寸的驗證。

```yaml
width: 1280
height: 800      # 16:10，剛好填滿目標螢幕
margin: 0.04     # reveal 預設 0.1
```

`margin` 那一行是實際增益：reveal 預設留 10% 白邊，會把 deck 縮到 1152×720 才顯示
（`Reveal.getScale()` = 0.9），白白損失一成字級。改成 0.04 後 scale = 0.96：

| | scale 0.9（預設） | scale 0.96 |
|---|---|---|
| 正文 34px | 30.6 實際 px | **32.6 實際 px** |
| 表格 26px | 23.4 | **25.0** |
| 圖內文字 25.6px | 23.0 | **24.6** |

不要為了「再大一點」把 margin 設到 0：投影機需要安全邊，且文字貼邊在教室後排更難讀。

### footer 與頁碼不隨 deck 縮放（保留這個修正）

即使只顧一個尺寸，這條仍然要留著——它是**正確性**問題而非適配問題：
reveal 用 `transform` 縮放的只有 `.slides`，而 **`.footer` 與 `.slide-number` 畫在縮放容器之外**，
transform 不會作用到它們。用 `em` 設字級時，任何 scale ≠ 1 的情況（現在是 0.96）都會偏大。

`theme.scss` 已就位，新 deck 直接沿用：

```scss
.reveal .footer {
  box-sizing: border-box;          // 少了這行，padding 會加在 100% 寬之外而衝出畫面
  left: 16% !important;            // 讓出左下角控制列
  right: 12% !important;           // 讓出右下角頁碼
  bottom: 1.1em !important;
  padding: 0 !important;
  white-space: normal !important;  // Quarto 預設 nowrap，長引用會溢出盒子外
  font-size: clamp(10px, 1.55vw, 21px) !important;
}
.reveal .slide-number { font-size: clamp(9px, 1.1vw, 15px) !important; }
```

**每張的 footer 要寫短**：完整引用放最後的 Sources 投影片，各張只留一行
（`Audio: Ohashi et al., EMNLP 2026 · CC-BY-4.0 · stereo: left = user, right = model`）。
CC-BY 的標註義務由簡短標註 + Sources 頁共同滿足。

**檢查 footer 要量 `scrollWidth > clientWidth`，不能只量 `getBoundingClientRect`**：
`nowrap` 的文字溢出盒子外時，盒子本身仍在畫面內，只量矩形會漏掉。W1 漏過一次。

### 版面驗證方法（做新的 deck 時照著跑）

字級放大後有兩類問題，肉眼掃不完 36 張，要用量測：

1. **投影片溢出**：用 Playwright 逐張比較 `section.present` 的內容底部與 `clientHeight`。
   單一圖片的投影片 Quarto 會自動加 `.r-stretch`，圖會縮到剩餘空間，所以**純文字的投影片才是風險**
   ——W1 放大後唯一溢出的就是純文字的「W6 is the hinge」，溢出 27px。
2. **圖內元素碰撞**：量測抓不到，只能逐張看圖。W1 這一輪抓到四處：e2e 的 inner monologue 框、
   duplex 的 `speech tokens` 標籤與 `channel 1`、latency 的圖內標題與右上註記。

**溢出的修法是減字，不是縮字。** 會溢出通常代表那張本來就太滿。


### 產出 PDF

revealjs 的 PDF 走瀏覽器列印：開 `slides/w01.html?print-pdf` 後在 Chrome 列印。音訊與逐步揭示會掉，所以那是補充版而非主要交付。

### 待驗

- W1 從未在真實投影機上跑過。第一次上課前請在教室環境確認：字級、色彩對比、以及**音訊在教室音響上的左右聲道**（左=使用者、右=模型，是 cold open 的關鍵）。
- Stivers et al. 2009 那張投影片刻意**沒有**放常被引用的單一均值（例如「約 200 ms」）——那個數字我沒在論文正文讀到。若要放，請自己從 Fig. 1 / Table 1 讀出來。

---


## 4.8 網站字級與一個量測陷阱

### 字級（2026-09 調整）

| | 之前 | 之後 |
|---|---|---|
| 桌機正文 | 16 px | **17 px** |
| 桌機表格 | 15.2 px（正文的 95%） | **17 px（與正文同級）** |
| 手機正文 | 15.5 px | **16.5 px** |
| 手機表格 | 13.44 px | **15.3 px** |

**表格是這個網站的內容本體**（參考資料、文獻速查表、教科書表），不該比正文小。
所以桌機拉到與正文同級。表格不可能比正文大而不顯得怪，因此「表格太小」的正確解法是
**表格 = 正文，再把正文一起微調**。

字級用 **rem**（root 相對）而非 em：表格常出現在 blockquote 等有 `em` 的容器裡，
用 em 會相乘。這與投影片那條規則同源（見 4.7）。

### ⚠️ 這個環境量不到「數學頁面」的真實版面

**雲端沙箱與本機 VM 都連不到 jsdelivr（KaTeX 的 CDN，403）。** 因此在沙箱裡用 Playwright
量含數學的頁面時，看到的是**未渲染的 LaTeX 原始碼**的寬度，不是使用者看到的渲染結果。

實例：曾經量到 `weeks/all.html` 在 390px 下橫向溢出 28px，兇手指向 W10 的 MVDR 公式。
據此加了一條 `main .math.inline { display:inline-block; overflow-x:auto }` 的 CSS，
又發現那會讓 `overflow` 非 visible 的 inline-block 的 baseline 變成底部邊緣——**全站行內數學
都會比周圍文字沉下去**，為了一條公式弄壞三十幾條。接著把那條公式改成 display math。

最後用**使用者機器上的瀏覽器**（KaTeX 正常）在真實 390px 下截圖，發現公式**根本沒有溢出**。
整串都是假警報，兩個改動都撤回了。

**所以：**
- 表格／文字的字級用 `getComputedStyle` 量，可信（與數學無關）。
- **版面溢出的結論，只要頁面含數學，就必須在使用者機器的瀏覽器上用 `resize_window` + 截圖確認**，
  不能只信沙箱的量測。
- 順帶學到：改 CSS 修個案之前先想清楚它對整類元素的副作用。

---


---

## 維護紀律

- **W11–W14 的文獻半衰期約 6 個月。** 每次開課前重掃 arXiv `cs.CL` / `eess.AS`
  近三個月，以及 Interspeech / ICASSP / ASRU / SLT 最新議程。
- 動了 Course-Hub 的大綱之後：(a) 重跑 `build_weeks.py`；(b) 同步 Claude Project
  的副本；(c) 改了週次結構或標題要回頭檢查 README；(d) 改週次標題時**中英文一起改**
  （英文寫在標題下一行的 `<!-- en: -->`）；(e) 改到「使用說明」「延遲預算」
  「主要教科書」「附錄 A」「附錄 B」這五段要同步 `docs/site-en.md`（手寫，不會自動更新）；
  改週次分組或主題則同步 `scripts/build_weeks.py` 的 `SHORT` 與 `PART_OF`。
- 兩個會破渲染的陷阱：**markdown 表格 cell 裡不能出現裸的 `|`**（行內數學用
  `\lvert … \rvert`），以及 **KaTeX 不支援 `\mathbb{1}`**（用 `\mathbf{1}`）。
- Commit 訊息要說明**改了哪一週、改了什麼層級**（結構／內容／引用）。

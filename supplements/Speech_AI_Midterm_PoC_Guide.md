# Speech AI 專案期中報告教戰守策：從問題定義到 Proof of Concept (PoC)
<!-- en: Midterm Report: From Problem Statement to Proof of Concept -->
<!-- order: 2 -->
<!-- summary: How to frame the problem and the failure mode, position the work against prior art, plan the ablations, and what counts as a convincing proof of concept. -->

這份期中報告不是單純的課堂修課作業，而是投稿至 ICASSP、INTERSPEECH、ASRU 或 SLT 的**研究提案（Research Proposal）與概念驗證（Proof of Concept, PoC）**。

頂會審稿人（Reviewer）在評閱短文（4 頁正文）時，最看重的並非模型規模或堆疊複雜度，而是：**問題是否真實存在、動機是否清晰、對比基準（Baseline）是否公平、方法論是否有合理的聲學/訊號假設支撐、實驗設計是否能精準驗證該假設。**

---

## 一、 期中報告結構與撰寫規範

報告建議長度為 3–4 頁（強烈建議直接套用 IEEE ICASSP 或 ISCA INTERSPEECH 雙欄 LaTeX 模板）。各章節應涵蓋的具體內容與標準如下：

### 1. 題目與摘要（Title & Abstract）
* **題目**：精確反映核心貢獻與技術邊界。避免空泛的「基於深度學習之語音增強研究」，改用具體描述：「*Dual-Path Flow-Matching with Spatial Priors for Multi-Channel Speech Separation*」。
* **摘要**（150–200 字）：必須涵蓋四項要素：
  1. 任務背景與既有 SOTA 模型的結構性瓶頸（Pain Point）。
  2. 本研究提出之核心機制與歸納偏置（Proposed Method & Inductive Bias）。
  3. 驗證之資料集與核心指標（Dataset & Key Metrics）。
  4. 預期解決的具體問題與關鍵增益。

### 2. 痛點分析與研究動機（Motivation & Problem Statement）
* **拒絕無效動機**：嚴禁出現「因為語音辨識很重要，所以我們要改進它」這類非技術性贅言。
* **鎖定具體 Failure Mode**：明確指出當前 SOTA 模型在何種具體邊界條件下崩潰。
  * *範例 1*：離散音訊編碼器（Discrete Audio Codec）在極低位元率（Low-bitrate, 如 $\le 1.5$ kbps）下對非平穩背景雜訊（Non-stationary Noise）重構時產生的金屬相位偽影（Phase Artifacts）。
  * *範例 2*：自監督學習表徵（SSL Representations）在跨語種/方言（如台灣台語、客語）語境下，因音素體系差異與聲調特徵缺失導致的聲學表徵退化（Representation Collapse）。
* **確立研究假設（Research Hypothesis）**：你的解法為何在學理上能解決該痛點？必須給出訊號層面、特徵幾何或架構特異性上的因果推論。

### 3. 文獻探討與定位（Related Work & Delta）
* **以解決方案範式分類，切忌流水帳**：依「技術哲學」分類（如：Time-domain vs. Time-frequency approaches、Autoregressive vs. Non-autoregressive generation），而非按年代列舉論文。
* **明確界定「Delta」（技術邊界與貢獻差異）**：
  > 「方法 A 解決了 X 但引入了二次複雜度 $O(T^2)$；方法 B 採用局部視窗降低了延遲，但破壞了全域語義。我們的設計介於兩者之間，透過 Z 機制在維持因果性（Causality）的前提下補償長程相依性。」

### 4. 解決方案（Proposed Methodology）
* **系統架構圖先行**：繪製清晰的張量流向圖（Tensor Flow Diagram），明確標註輸入張量維度（Tensor Shapes，如 $[B, C, T]$）、採樣率（Sampling Rate）、特徵降採樣因子與各模組輸出。
* **數學與演算法定義**：
  * 嚴格定義符號。例如時域訊號 $x[n]$、STFT 複數頻譜 $X(t, f)$、隱藏表徵 $H \in \mathbb{R}^{T \times D}$。
  * 明確寫出損失函數（Loss Function）的加權組成：
    $$\mathcal{L}_{\text{total}} = \lambda_{\text{rec}} \mathcal{L}_{\text{rec}} + \lambda_{\text{adv}} \mathcal{L}_{\text{adv}} + \lambda_{\text{aux}} \mathcal{L}_{\text{aux}}$$
    解釋每個 Loss Term 所約束的聲學物理意義（如 Spectral Convergence 抑制高頻能量誤差、CTC Loss 約束單調對齊）。

### 5. 資料集與評估指標（Data & Metrics）

| 研究領域 | 建議主流公開基準資料集 | 核心評估指標 |
| :--- | :--- | :--- |
| **ASR** | LibriSpeech, Common Voice, AISHELL, Formosa Speech | WER, CER, RTF (Real-Time Factor) |
| **TTS / Voice Cloning** | LJSpeech, VCTK, LibriTTS | UTMOS, NISQA, SECS (Speaker Similarity), CER |
| **Speech Enhancement / Separation** | VoiceBank-DEMAND, WSJ0-2mix, DNS Challenge | PESQ, STOI, SI-SDR (dB) |
| **Audio Codec / Tokenizer** | LibriTTS, AudioCaps | ViSQOL, Mel-Spectral Reconstruction Loss, Bitrate (kbps) |
| **Speaker Verification / Diarization** | VoxCeleb 1 & 2, AMI | EER (Equal Error Rate), minDCF, DER |

* **主客觀指標對齊**：若聲稱改進「自然度」或「聽覺品質」，必須在實驗規劃中納入主觀聽感測試設計（MUSHRA 或 MOS 規範，包含信賴區間計算方式）。

### 6. 實驗規劃與消融設計（Experimental Setup & Ablation Plan）
* **對比基準（Baselines）**：必須包含：
  1. 領域內經典的 Vanilla Baseline。
  2. 近兩年內代碼開源、具有高度代表性的 SOTA 模型。
* **消融實驗表格骨架（Ablation Skeleton）**：在報告中預先畫出期末將填入數值的表格欄位，確保每一個提出的模組、Loss 或超參數調整均有對應的控制組（Control Group）。
* **資源配置**：註明硬體規格（GPU 型號與顯存）、Batch Size、Optimizer、Learning Rate Scheduler 與預計訓練 steps/epochs。

### 7. 初步概念驗證（PoC / Sanity Check）
期中報告必須提供初步可行性證明（以下至少完成一項）：
* **Toy Dataset 驗證**：在小規模子集（如 LibriSpeech-100 或 LJSpeech 1 小時子集）上跑通訓練 Pipeline，證明 Loss 穩定收斂且梯度無爆炸/下溢（NaN）。
* **Pipeline 暢通性**：輸入音訊 $\to$ 聲學特徵提取 $\to$ Backbone $\to$ Loss 計算 $\to$ 反向傳播更新，整條流程打通並產出第一組可聆聽音檔或初步指標。
* **Baseline 重現**：成功重現開源基準模型的 Inference 結果，確認自己編寫的 Evaluation Script 計算出的指標與原論文公開數值量級一致。

---

## 二、 非資工背景同學的執行策略

在 Speech AI 領域，許多突破來自對**語音物理特性（時頻局部性、諧波結構、發音機制）的敏銳洞察**，而非單純的底層系統工程。

* **站在開源巨人的肩膀上**：
  * ASR / TTS / Separation：優先採用 **ESPnet**、**SpeechBrain** 或 **NVIDIA NeMo** 的既有 Recipe，不從零寫訓練迴圈。
  * Foundation Models / SSL 微調：善用 **Hugging Face (`transformers` / `torchaudio`)**。
* **模組化介入（Modular Intervention）**：
  將心力集中於「模組替換」或「表徵轉移」。例如：凍結（Freeze）預訓練好的 Whisper 或 WavLM Encoder，僅針對 Downstream Adapter、Cross-Attention 融合層或輕量化 Head 進行創新。
* **從錯誤分析（Error Analysis）切入**：
  仔細聆聽模型失敗的音檔。分析問題是在清濁音轉換（Voiced/Unvoiced transition）、高頻摩擦音、殘留混響（Reverberation）、抑或重疊語音（Overlapped speech）發生崩潰，以此作為演算法修改的物理支點。

---

## 三、 AI 工具（以 Gemini 為例）協作指引

將 LLM 定位為「研究助理與結構審計員」，嚴守學術誠信底線。

```
                    【推薦的 AI 協作模式】
  
  文獻發想 ───> 獲取關鍵詞與領域概念 ───> 人工手動下載論文確認
  架構設計 ───> 討論優缺點與潛在漏洞 ───> 人工評估訊號物理意義
  程式除錯 ───> 產生 Boilerplate / 查修 ───> 人工跑過 Sanity check
  報告寫作 ───> 潤色語氣與雙欄排版 ───> 逐字確認技術內容無誤
```

### 1. 建議使用情境
* **關鍵字擴展與範疇收斂**：
  * *Prompt 範例*：「*我想在低延遲（Low-latency）條件下改進 Conformer ASR 的串流表現，目前已知機制有 chunk-based attention。請提供近兩年 ICASSP/INTERSPEECH 探討此問題時常用的 5 個關鍵技術概念與其限制。*」
* **PyTorch 樣板代碼生成**：
  * 產生音訊 `Dataset`、動態長度對齊（Collate function with padding）或 STFT/Mel-filterbank 處理樣板。
* **架構缺陷壓力測試**：
  * 輸入架構文字描述，要求 AI 以審稿人視角找出 3 個致命破綻（如因果性洩漏、運算複雜度不符流式標準等）。

### 2. 嚴格禁止的錯誤行為
* **絕對不可採信 AI 生成的論文引用**：
  LLM 極易捏造作者、標題或出處。**報告中出現的每一篇文獻，都必須親自透過 Google Scholar、arXiv 或 IEEE Xplore 下載 PDF 確認。**
* **嚴禁 AI 捏造實驗數據**：
  未經實驗實測的數字填入表格視同偽造數據。尚未完成的數據一律標記為 `[Pending]` 或 `[To be verified]`。

---

## 四、 審稿人（Reviewer）即退紅線（Red Flags）

在繳交期中報告前，請逐條自我核對：

* [ ] **無動機的模組拼接（A + B without insight）**：單純將 Transformer 替換為 Mamba，或任意在 TTS 後端接上 Diffusion，卻無法解釋「該任務的何種語音物理特性需要該架構改進」。
* [ ] **不公平的基準對比（Unfair Comparison）**：拿預訓練巨量資料微調後的模型，去對比未微調的傳統 Baseline；或在不同資料分割（Split）上比對指標。
* [ ] **指標與目標脫節（Metric Misalignment）**：聲稱提升了語音自然度，卻只提供 Mel-spectral L1 Loss 作為唯一佐證。
* [ ] **預先看見未來資訊（Lookahead Violation）**：研究題目宣告為即時串流（Streaming），模組中卻包含雙向注意力（Bidirectional Attention）或全域標準化（Global LayerNorm）。
* [ ] **範疇過大、無法落地**：計畫在學期內從零預訓練 Speech LLM。算力需求明顯超出個人或實驗室負荷者，期中將直接判定為不可行。
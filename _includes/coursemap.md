<!-- 此檔由 scripts/build_weeks.py 自動產生，請勿直接編輯；請改 docs/course-outline.md 後重跑腳本。 -->

## 課程地圖

```
                    ┌─────────────────────────────────────────────┐
   W1  系統觀 ──────▶│  最終目標：全雙工語音對話系統                │
                    │  (listen while speaking, barge-in, <300ms)  │
                    └─────────────────────────────────────────────┘
                                        ▲
        ┌───────────────────────────────┼───────────────────────────────┐
        │                               │                               │
   Part I 基礎 (W2–W6)            Part II 模組 (W7–W10)         Part III 對話系統 (W11–W14)
        │                               │                               │
  W2 訊號與前端                   W7  ASR                        W11 Audio-native LM（半雙工）
  W3 對齊：HMM → CTC              W8  TTS I：生成範式             W12 全雙工 I：turn-taking 建模
  W4 Transformer 與 streaming     W9  TTS II：可控性與評估        W13 全雙工 II：架構與資料
  W5 SSL 表徵學習                 W10 前端：VAD/AEC/分離/說話人   W14 評估、對齊、落地、open problems
  W6 Neural codec 與 tokenization
```

**三條貫穿全課的主軸（每週投影片建議都回扣一次）**

| 主軸 | 問題 |
|---|---|
| **表徵軸** | 這一層用什麼表徵？waveform / spectrogram / continuous SSL feature / discrete token？誰決定 frame rate？ |
| **延遲軸** | 這個模組吃掉多少 latency budget？是演算法延遲（lookahead）還是計算延遲？ |
| **監督軸** | 這個能力從哪來？標註資料 / 自監督 / 合成資料 / 人類偏好（RLHF）？ |

**延遲預算（latency budget）—— 全課的共同座標系**

第一週就把這張表發下去，之後每週回填該模組的貢獻。人類對話中 turn 轉換的中位反應時間約在 200 ms 量級（`[主題]` 檢索 *turn-taking gap distribution, Levinson & Torreira, universals in turn-taking timing*），這是「自然感」的工程目標來源。

```
使用者停止說話 ──▶ [endpoint 偵測] ──▶ [ASR 尾段] ──▶ [LLM prefill + first token]
                                                              ──▶ [TTS first packet] ──▶ [播放緩衝] ──▶ 使用者聽到
```

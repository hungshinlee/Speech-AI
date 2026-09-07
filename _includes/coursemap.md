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

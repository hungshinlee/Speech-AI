<!-- 此檔由 scripts/build_weeks.py 自動產生，請勿直接編輯；請改 docs/course-outline.md 後重跑腳本。 -->

## 工具鏈建議

| 用途 | 首選 | 備註 |
|---|---|---|
| 課堂 demo / 快速實驗 | `torchaudio` + Hugging Face `transformers` | Colab 免費版友善，安裝最省事 |
| ASR 完整流程 | ESPnet 或 k2/icefall | icefall 的 streaming RNN-T 範例最適合 W4 demo |
| 語音任務模組 | SpeechBrain | 分離、說話人、增強的 recipe 齊全，W10 首選 |
| TTS | `F5-TTS`、CosyVoice、Kokoro 類輕量模型 | 注意權重下載時間，課前預熱 notebook |
| Codec | `descript-audio-codec`、`transformers` EnCodec、Mimi | W6 逐層重建 demo |
| VAD / turn-taking | Silero VAD + 開源 turn detection 模型 | W12 demo 核心 |
| 全雙工完整系統 | Moshi 官方實作、以及 W13 各路線的開源 repo | Colab 免費版可能吃不下，建議改播官方 demo |

> **Colab 免費版的現實約束**：T4（16 GB）約可跑 7B 以下的 4-bit 量化推論，但即時語音的吞吐量通常不足。建議所有 demo 設計成**離線推論 + 事後對齊時間軸**，而不是真正的即時互動；即時性用預錄的官方 demo 呈現。

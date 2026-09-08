<!-- 此檔由 scripts/build_weeks.py 自動產生，請勿直接編輯；請改 docs/course-outline.md 後重跑腳本。 -->

## Toolchain

| Purpose | First choice | Notes |
|:------------|:----------------|:--------------|
| In-class demos / quick experiments | `torchaudio` + Hugging Face `transformers` | Friendliest on free Colab; least to install |
| Full ASR pipelines | ESPnet or k2/icefall | icefall's streaming RNN-T recipe is the best fit for the W4 demo |
| Speech task modules | SpeechBrain | Complete recipes for separation, speaker and enhancement; first choice for W10 |
| TTS | `F5-TTS`, CosyVoice, Kokoro-class lightweight models | Mind the weight-download time; warm the notebook up before class |
| Codec | `descript-audio-codec`, `transformers` EnCodec, Mimi | The layer-by-layer reconstruction demo in W6 |
| VAD / turn-taking | Silero VAD plus an open-source turn detection model | Core of the W12 demo |
| Full-duplex systems | The official Moshi implementation, plus the open repos for each W13 route | Free Colab probably cannot hold these; play the official demos instead |

> **What free Colab actually allows.** A T4 (16 GB) will run 4-bit quantized inference up to roughly 7B, but rarely with enough throughput for real-time speech. Design every demo as **offline inference plus a time axis aligned afterwards** rather than genuine live interaction, and show real-time behaviour with pre-recorded official demos.

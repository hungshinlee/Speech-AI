<!--
網站英文片段的來源檔（手寫）。scripts/build_weeks.py 依 `<!-- file: X -->` 切段，
寫成 _includes/X，供 syllabus.qmd 與 resources.qmd 引用。

中文原文仍留在 docs/course-outline.md，供 weeks/all.qmd 使用；兩邊各自維護。
改了大綱的對應區塊（免責聲明、延遲預算、教科書、附錄 A、附錄 B）記得回來同步這裡。
首頁的英文課程地圖另存 docs/course-map-en.md（ASCII 對齊敏感，單獨一檔）。
-->

<!-- file: disclaimer.md -->
## How to Read This Site

1. **Mathematical density.** The "key mathematics" block of each week gives the equations and the skeleton of the derivation — the critical steps, the symbols to define, the point where students get stuck — and not the full algebra. That is a deliberate trade-off: it is meant to transfer straight onto slides.
2. **Citation-confidence tags.**
   - Untagged references are standard, widely cited work; confidence in the title, authors and year is high.
   - **`[驗]`** marks lower confidence — the year, venue or subtitle may be wrong. **Verify on arXiv or DBLP before putting it on a slide.**
   - **`[主題]`** deliberately gives no precise citation, only search keywords, rather than risk fabricating one.
3. **No numbers.** The outline reports no WER / MOS / DER figures. For a benchmark table on a slide, cite the original paper's table rather than a remembered value.
4. **Currency.** Written against arXiv through August–September 2026. Full-duplex spoken dialogue is one of the fastest-moving subfields in speech; rescan `cs.CL` and `eess.AS` for the previous three months before teaching Part III.

<!-- file: latency.md -->
## The Latency Budget: The Course's Shared Coordinate System

Hand this diagram out in week 1, then fill in each module's contribution as the course goes. The median response time at a turn transition in human conversation is on the order of 200 ms (`[主題]` search *turn-taking gap distribution, Levinson & Torreira, universals in turn-taking timing*); that is where the engineering target for "natural" comes from.

```
user stops speaking
      ──▶ [endpoint detection] ──▶ [ASR tail]
      ──▶ [LLM prefill + first token]
      ──▶ [TTS first packet] ──▶ [playout buffer]
      ──▶ user hears
```

<!-- file: textbooks.md -->
## Core Textbooks

| Code | Source | Used for |
|:--|:------------------------------|:------------------|
| **JM3** | Jurafsky & Martin, *Speech and Language Processing*, 3rd ed. **online draft, 2026-08-19 release** (<https://web.stanford.edu/~jurafsky/slp3/>) | The teaching baseline for the speech and dialogue chapters; official slides available to adapt |
| **HAH** | Huang, Acero & Hon, *Spoken Language Processing*, Prentice Hall, 2001 | Solid derivations for signal processing, hearing and HMMs (old, but nothing has replaced it) |
| **YD** | Yu & Deng, *Automatic Speech Recognition: A Deep Learning Approach*, Springer, 2015 | Hybrid DNN-HMM through sequence discriminative training |
| **TAN** | Xu Tan, *Neural Text-to-Speech Synthesis*, Springer, 2023 `[驗]` | Systematic treatment of TTS (check chapter numbers against the printed book) |
| **LOI** | Loizou, *Speech Enhancement: Theory and Practice*, 2nd ed., CRC Press, 2013 | Statistical methods and evaluation metrics for enhancement |
| **VVG** | Vincent, Virtanen & Gannot, *Audio Source Separation and Speech Enhancement*, Wiley, 2018 | Separation and multichannel processing |
| **BB** | Bishop & Bishop, *Deep Learning: Foundations and Concepts*, Springer, 2024 | The mathematics under diffusion, normalizing flows and VAEs |

> ⚠️ **JM3 chapter numbers move between draft releases.** This course fixes the numbering of the **2026-08-19** release: Ch 15 *Phonetics and Speech Feature Extraction*, Ch 16 *Automatic Speech Recognition*, Ch 17 *Text-to-Speech*, Ch 26 *Conversation and its Structure*, App A *Hidden Markov Models*, App K *Frame-based Dialogue Systems*. Cite one release consistently when building slides.

<!-- file: reading-table.md -->
## One Paper per Week

| Week | Topic | If you read only one |
|:--|:----------|:-------------------------|
| W1 | Systems view | Lu et al., *A Survey of Full-Duplex Spoken Dialogue Systems*, arXiv:2606.19453 |
| W2 | Signals and front end | **JM3** Ch 15 |
| W3 | Alignment | Graves et al., *CTC*, ICML 2006 |
| W4 | Sequence models and streaming | Graves, *Sequence Transduction with RNNs*, arXiv:1211.3711 |
| W5 | SSL | Hsu et al., *HuBERT*, TASLP 2021 |
| W6 | Codec / tokenization | Défossez et al., *Moshi* (the Mimi codec section) |
| W7 | ASR | Radford et al., *Whisper*, ICML 2023 |
| W8 | TTS I | Lipman et al., *Flow Matching for Generative Modeling*, ICLR 2023 |
| W9 | TTS II | Rafailov et al., *DPO*, NeurIPS 2023 + ITU-T P.808 |
| W10 | Acoustic front end | Wang & Chen, *Supervised Speech Separation: An Overview*, TASLP 2018 |
| W11 | Audio-native LM | Défossez et al., *Moshi*, arXiv:2410.00037 |
| W12 | Turn-taking | Skantze, *Turn-Taking in Conversational Systems: A Review*, CSL 2021 |
| W13 | Full-duplex architectures | Nguyen et al., *dGSLM*, TACL 2023 |
| W14 | Evaluation and alignment | Lin et al., *Full-Duplex-Bench*, ASRU 2025 |

<!-- file: toolchain.md -->
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

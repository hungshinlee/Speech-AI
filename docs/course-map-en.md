## Course Map

```
                    ┌──────────────────────────────────────────────┐
   W1  Systems ────▶│  Goal: a full-duplex spoken dialogue system  │
       view         │  (listen while speaking, barge-in, <300 ms)  │
                    └──────────────────────────────────────────────┘
                                        ▲
        ┌───────────────────────────────┼───────────────────────────────┐
        │                               │                               │
  Part I — Foundations (W2–W6)    Part II — Modules (W7–W10)      Part III — Dialogue (W11–W14)
        │                               │                               │
  W2 Signals & front-end          W7  ASR                         W11 Audio-native LM (half-duplex)
  W3 Alignment: HMM → CTC         W8  TTS I: generative models    W12 Full-duplex I: turn-taking
  W4 Transformer & streaming      W9  TTS II: control & eval      W13 Full-duplex II: arch. & data
  W5 SSL representations          W10 Front-end: VAD/AEC/sep/spk  W14 Evaluation, alignment, ops
  W6 Neural codec & tokenization
```

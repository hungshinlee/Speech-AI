<!-- 此檔由 bin/publish_demos.py 從 private repo 的 <course>/demos/ 複製而來，請勿在公開 repo 直接編輯。 -->
# W8 · Demo 1 — A regressed mel, a drawn mel, three draws (slides p12–p13)

One line of text, two text-to-speech models that answer the one-to-many problem in opposite ways:

| | Model | How it produces the mel | How the length is decided |
|---|---|---|---|
| regression | **SpeechT5** (`microsoft/speecht5_tts` + `microsoft/speecht5_hifigan`) | autoregressive, two frames per step, each frame a point estimate trained with an L1 loss; the voice comes from an x-vector (the mean of the CMU ARCTIC `bdl` x-vectors — a different speaker from the sentence's) | a learned stop token: generation ends at the first step whose stop probability exceeds 0.5 |
| sampling | **F5-TTS** (`SWivid/F5-TTS`, `F5TTS_v1_Base`, conditional flow matching) + Vocos (`charactr/vocos-mel-24khz`) | starts from Gaussian noise and integrates a learned velocity field (32 Euler steps, classifier-free guidance, sway sampling); the voice comes from a prompt — another sentence by the same speaker | from outside the model: prompt frames × UTF-8 bytes of the text ÷ bytes of the prompt's text |

The sentence is W3's (`1272-128104-0000`, "Mister Quilter is the apostle of the middle classes, and we are glad to welcome his gospel."); the prompt is `1272-128104-0010` from the same chapter. Both are LibriSpeech dev-clean (CC BY 4.0).

Each model uses its own front end, and the real recording is passed through both, so every comparison is made inside one front end:

| Front end | Bins | Sample rate | Hop | Range | Scale |
|---|---|---|---|---|---|
| SpeechT5 | 80, log10 | 16 kHz | 16 ms | 80–7600 Hz | Slaney mel, Slaney-normalised |
| F5 (Vocos) | 100, ln | 24 kHz | 256 samples (10.7 ms) | 0–12 kHz | HTK mel, not normalised |

**Fine detail** (printed on every panel): the log-mel in dB minus its own 3 × 3 local mean, RMS over a band and over frames within 25 dB of the loudest one — how much texture is left once the smooth envelope is removed. It is reported as a percentage of the real recording's fine detail *in the same front end*. It is a reference, not the judge; your ears are.

Above roughly 1.1–1.3 kHz these mel filterbanks are coarser than half the harmonic spacing of this voice (median $F_0$ about 112 Hz), so no mel — real or generated — shows separate harmonics up there; what differs between 2 and 6 kHz is the fine, noise-like texture.

## Keys (in the window)

| Key | What happens | Slide |
|---|---|---|
| `1` | SpeechT5's regressed mel (top left), and its audio | p13 |
| `2` | F5-TTS seed 1's drawn mel (top right), and its audio | p13 |
| `g` | the real recording through both front ends (bottom row), and the original audio | p13 |
| `x` | zoom to 2–6 kHz and back | p13 |
| `3` | F5-TTS seeds 1–3 side by side; seeds 2 and 3 are played; the three lengths are printed | p13 |
| `z` | the three $F_0$ tracks on one time axis (real recording dashed) | p13 |
| `s` | SpeechT5 with the next seed — its decoder pre-net keeps dropout on at inference, so it too varies a little | |
| `t` | the table: lengths and how they were decided, fine detail per band, seed-to-seed differences, run times | |
| `r` `h` `q` | play again, menu, quit | |

The playback files are scaled to the real recording's RMS so that loudness does not stand in for quality; all measurements use the models' raw outputs.

## Running it

```bash
cd speech_ai/demos/w08_d1_mean_vs_sample
./setup.sh                 # builds ../.venv-tts (see below), then runs selftest
./present.sh fetch         # the two models, their vocoders and the x-vectors (commits pinned in model.lock); copies the two sentences
./present.sh check         # versions; CPU vs MPS on the same input; one F5 draw on each device → runs/check/*.wav to compare by ear
./present.sh rehearse      # everything, non-interactive → runs/rehearsal/
./present.sh               # the keys above
./present.sh replay        # the rehearsal's outputs on the same keys, without torch
./present.sh fake          # the whole flow without any model (stand-in mels, marked FAKE)
```

Requirements: a separate environment, `speech_ai/demos/.venv-tts`, built by `./setup.sh` with the same Python, numpy, torch 2.14.0 and transformers 5.17.0 as the shared `.venv`, plus `f5-tts 1.1.22`. It is separate because `f5-tts`'s dependencies (through `cached-path`) need older `rich` and `filelock` than the shared environment pins. `replay`, `fake` and `selftest` fall back to the shared `.venv` if `.venv-tts` is missing.

`f5-tts`'s own inference code is used unchanged (`infer_batch_process`, the function its CLI ends in); the demo passes the prompt waveform directly, after the same preprocessing (edges below −42 dBFS trimmed, 50 ms of silence appended), so no FFmpeg is needed. F5's `cfg_strength` is the $\alpha$ of its Eq. 4, $\mathbf{v}_\theta(\mathbf{X}_\tau, \tau, \mathbf{c}) + \alpha\big(\mathbf{v}_\theta(\mathbf{X}_\tau, \tau, \mathbf{c}) - \mathbf{v}_\theta(\mathbf{X}_\tau, \tau)\big)$ (Ho & Salimans' form); the slides (p55) write CFG with a weight $w$ on the difference added to the *unconditional* field, so $w = \alpha + 1$ and F5's default $\alpha = 2$ is $w = 3$.

Weights: SpeechT5 and its HiFi-GAN are MIT; F5-TTS's weights are CC BY-NC 4.0 (non-commercial); Vocos is MIT; the ARCTIC x-vectors are MIT.

Numbers quoted on the slides come from `runs/rehearsal/rehearsal.json` and are summarised in `slides/assets/w08/w08-metrics.json` (`demo1`). Draws are reproducible only on the same machine, device and seed.

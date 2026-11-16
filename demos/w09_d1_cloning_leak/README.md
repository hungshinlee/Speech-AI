<!-- 此檔由 bin/publish_demos.py 從 private repo 的 <course>/demos/ 複製而來，請勿在公開 repo 直接編輯。 -->
# W9 · Demo 1 — Four prompts, three ways to condition (slides p13–p15)

Voice cloning hands the model a short recording of the target speaker. This demo asks what else in that recording reaches the output. One 4.9 s prompt of one LibriSpeech dev-clean speaker (`1272-128104-0010`, CC BY 4.0) is made into four versions, and each version goes through three systems that take the reference in different ways. Twelve outputs, all reading the same sentence (W3's `1272-128104-0000`, of which the same speaker's real recording exists).

**The four versions of the prompt** (computed by `demo.py`, same length, 16 kHz):

| Version | How it is made |
|---|---|
| clean | the recording, edges below −42 dBFS trimmed |
| + noise | white Gaussian noise added at a signal-to-noise ratio of 10 dB |
| + music | a chord progression synthesised in `demo.py` (C–G–Am–F) added at 10 dB |
| telephone band | zero-phase low-pass at 3.4 kHz, then resampled 16 → 8 → 16 kHz |

**The three ways to condition** (slides p9–p12):

| | System | What it receives from the prompt |
|---|---|---|
| (a) a bottleneck | **SpeechT5** (`microsoft/speecht5_tts`) | 512 numbers: the prompt's x-vector, from the speaker encoder SpeechT5 was trained with (`speechbrain/spkrec-xvect-voxceleb`), unit-normalised |
| (b) no bottleneck, flow matching | **F5-TTS** (`SWivid/F5-TTS`, `F5TTS_v1_Base`) | the prompt's whole mel spectrogram and its text; the model fills in the rest |
| (b) no bottleneck, autoregressive | **Qwen3-TTS** (`mlx-community/Qwen3-TTS-12Hz-0.6B-Base-bf16`, through `mlx-audio`) | the prompt's codec tokens and its text; the model continues them |

**What is measured on every output**

| Quantity | Definition |
|---|---|
| noise floor in the pauses (dB re speech) | the 5th percentile minus the 95th percentile of the 10 ms frame RMS levels, in dB, between the first and the last frame within 30 dB of that 95th percentile (silence before and after the utterance is left out): how far the quietest frames inside the utterance sit below the loudest ones. Closer to 0 means noisier pauses. |
| energy above 4 kHz (dB re total) | after resampling to 16 kHz, the power in 4–8 kHz over the power in 0–8 kHz |
| SIM | the cosine between two embeddings from one speaker-verification encoder (`microsoft/wavlm-base-plus-sv`, WavLM Base+ with an x-vector head): output vs the **clean** prompt, and output vs **that version** of the prompt |
| WER | Whisper small (W7's) against the sentence, both sides through Whisper's English normaliser |

The first two are this demo's own quantities, written to answer two specific questions; they are not standard metrics. SIM is also printed with a second encoder (the SpeechBrain x-vector that conditions SpeechT5): the two encoders' numbers are not comparable with each other, which is the point of slide p18. Two reference points are printed with the SIM table: the same speaker's real recording of the sentence, and last week's SpeechT5 output in a different voice.

Each cell is **one draw** (seed 1 for each system). Before trusting a small difference between two cells, draw again with another seed.

**The extra row (slide p15)** uses two prompts recorded by the lecturer — one sentence spoken in an obvious mood, one Mandarin sentence — and asks the two continuation systems to read a neutral sentence and an English sentence. Those recordings and their outputs are not in this repository; `./present.sh record mood` and `./present.sh record accent` record your own.

## Keys (in the window)

| Key | What happens | Slide |
|---|---|---|
| `p` | the four prompt versions: mel spectrograms, played in order | p14 |
| `1` `2` `3` | the three outputs for the + noise / + music / telephone-band prompt, played in order; the window shows that prompt and the three outputs (dashed line at 4 kHz) | p14 |
| `c` | the three outputs for the clean prompt | p14 |
| `s` `f` `k` | replay only SpeechT5 / F5-TTS / Qwen3-TTS for the current version; `o` replays that version's prompt; `g` plays the speaker's real recording of the sentence | |
| `t` | the numbers: a table in the terminal, bar charts in the window | p14 |
| `e` `a` | the extra row: the mood prompt, the accent prompt | p15 |
| `r` `h` `q` | play again, menu, quit | |

Playback files are scaled to one RMS so that loudness does not stand in for quality; all measurements use the raw outputs. The mel spectrograms in the window all use one display front end (16 kHz, 25 ms / 10 ms, 80 mel bins up to 8 kHz).

## Running it

```bash
cd speech_ai/demos/w08_d1_mean_vs_sample && ./setup.sh && ./present.sh fetch          # the TTS environment, SpeechT5, F5-TTS, the two sentences
cd ../w08_d2_nfe_cfg_first_packet && ./present.sh fetch                               # Qwen3-TTS (MLX)
cd ../w09_d1_cloning_leak
./setup.sh                 # adds speechbrain to ../.venv-tts, then runs selftest
./present.sh fetch         # the x-vector encoder, the SIM encoder, Whisper small (commits pinned in model.lock)
./present.sh check         # both encoders on CPU and MPS, the two SIM reference points, Whisper on the real recording
./present.sh rehearse      # everything, non-interactive → runs/rehearsal/
./present.sh               # the keys above
./present.sh replay        # the rehearsal's outputs on the same keys, without torch
./present.sh fake          # the whole flow without any model (stand-ins, marked FAKE)
```

Requirements: `speech_ai/demos/.venv-tts` (W8 Demo 1's environment: torch 2.14.0, transformers 5.17.0, f5-tts 1.1.22) plus `speechbrain`, and the shared `speech_ai/demos/.venv` with `mlx-audio 0.5.4` for Qwen3-TTS, which runs as a subprocess (W8 Demo 2's `mlx_worker.py`). Qwen3-TTS needs Apple silicon; on other machines `--no-mlx` leaves that column empty. Whisper runs on the CPU.

To use real music instead of the synthesised chords, put a 16-bit WAV at `audio/music.wav`.

Nothing is trained in this demo; all three systems are used as released.

Weights: SpeechT5 and its HiFi-GAN are MIT; F5-TTS's weights are CC BY-NC 4.0 (non-commercial); Vocos is MIT; the SpeechBrain x-vector model and Whisper are Apache-2.0; the Qwen3-TTS conversion's card says Apache-2.0; the model card of `microsoft/wavlm-base-plus-sv` carries no licence field.

Cloning a voice needs the speaker's consent. The prompts here are a LibriSpeech recording (CC BY 4.0, from public-domain LibriVox audiobooks) and the lecturer's own voice.

Numbers quoted on the slides come from `runs/rehearsal/rehearsal.json` and are summarised in `slides/assets/w09/w09-metrics.json` (`demo1`). Draws are reproducible only on the same machine, device and seed.

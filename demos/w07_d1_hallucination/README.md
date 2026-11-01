<!-- 此檔由 bin/publish_demos.py 從 private repo 的 <course>/demos/ 複製而來，請勿在公開 repo 直接編輯。 -->
# W7 · Demo 1 — Silence, noise, and a zeroed encoder: what a prior says when there is nothing to hear (slides p26–p28)

The same "no speech" audio goes into two models that differ only in whether the output side carries a language prior.

| Clip | What it is |
|---|---|
| `silence` | 10 s of digital silence (all zeros) |
| `noise_0`, `noise_20`, `noise_40` | 10 s of white noise at 0, 20 and 40 dB below the RMS of W3's sentence |
| `long` | W3's sentence `1272-128104-0000` followed by 30 s of silence, so that Whisper's second 30 s window contains nothing but silence and the first window's text in its prompt slot |
| `own` | the instructor's own non-speech recording (music, a cough) — not distributed |

| Panel | Model | What is printed |
|---|---|---|
| (1) | **Whisper `small`** (`transformers`, the same weights as W4 demo 3) | the transcript, every token's log-probability $\log P(\ell_u \mid \ell_{<u}, \mathbf{X})$ (teacher-forced after generation, including `<\|endoftext\|>`), the average, `no_speech_prob` (the probability of `<\|nospeech\|>` at the `<\|startoftranscript\|>` position), the gzip compression ratio, and which of the §4.5 fallback rules fired |
| (2) | **wav2vec2-base-960h** (W3 demo 3's CTC model), greedy | $T$, the fraction of frames whose argmax is blank, the number of non-blank frames, and the collapsed string |
| (3) | Whisper with its **encoder output replaced by zeros** (or by the time-average of W3's encoder output), decoder free-running | five beams and five samples — an approximation of the internal language model that p37's ILME estimates |

The fallback loop is written out in `demo.py` rather than taken from `transformers`, so that every attempt is shown: temperature rises from 0 in steps of 0.2 up to 1.0; an attempt fails if the compression ratio exceeds 2.4 or the average log-probability is below $-1$; `no_speech_prob > 0.6` counts as silence only together with the log-probability threshold; the previous window's text is carried into the next window's prompt only if that window's final temperature was below 0.5. Long audio is cut into fixed 30 s windows (the original advances by the last predicted timestamp; this demo does not decode timestamps).

## Keys (terminal, single key)

| Key | Screen | Slide |
|---|---|---|
| `1` | silence → Whisper | p27 |
| `2` | the three noise clips → Whisper | p27 |
| `3` | the long clip in two windows; `c` toggles whether the first window's text is carried into the second | p27, p28 |
| `4` | the instructor's own clip | p27 |
| `w` | every clip (and W3's sentence itself) → wav2vec2 greedy | p27 |
| `z` / `Z` | encoder zeroed / replaced by the mean vector → beams and samples | p27 |
| `f` | toggle the temperature fallback (default off; on, every attempt is printed) | p28 |
| `l` | toggle the language token en / zh — a different prior, the same flat evidence | p24 |
| `s` | W3's sentence through both models (the control for the control) | |
| `h` `q` | menu, quit | |

## Running it

```bash
cd speech_ai/demos/w07_d1_hallucination
./present.sh selftest      # no model needed
./present.sh fetch         # openai/whisper-small and facebook/wav2vec2-base-960h (commits pinned in model.lock)
./present.sh make          # generates audio/ deterministically (seed in demo_config.toml)
./present.sh check         # versions, token ids, the encoder hook, MPS vs CPU → runs/check.json
./present.sh rehearse      # everything, non-interactive → runs/rehearsal/
./present.sh               # the keys above
./present.sh replay        # the rehearsal's outputs on the same keys, without torch
```

Requirements: the shared `speech_ai/demos/.venv` with `torch 2.14.0` and `transformers 5.17.0` (installed by `w03_d3_blank_heatmap/setup.sh`); W3 demo 3's `runs/rehearsal/utterance.wav` for the sentence. Nothing else.

Numbers quoted on the slides come from `runs/rehearsal/rehearsal.json` and are summarised in `slides/assets/w07/w07-metrics.json` (`demo1`). Sampled outputs (temperature $> 0$) are reproducible on the same machine with the same seed only.

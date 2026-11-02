<!-- 此檔由 bin/publish_demos.py 從 private repo 的 <course>/demos/ 複製而來，請勿在公開 repo 直接編輯。 -->
# W2 · Demo 3 — Front-end mismatch: same audio, same recogniser, a different log-mel recipe (slides p64–p66)

Two front-ends can both be called "80-dim log-mel, 25 ms window, 10 ms hop" and still produce different numbers. This demo asks whether a pretrained recogniser notices.

Forty sentences from LibriSpeech dev-clean (one per speaker) go into Whisper `small` (the same weights as W4 demo 3). The **reference** run uses Whisper's own front-end (`transformers`' `WhisperFeatureExtractor`). Every other run recomputes the features with the code in `../dsp.py` and `demo.py`, **changing one knob at a time**, and feeds them to the same model.

| Key | Recipe | What changes |
|---|---|---|
| `r` | reference | Whisper's own front-end |
| `0` | same | the same recipe, our own code — a check that nothing else differs |
| `1` | htk | mel scale: HTK ($2595\log_{10}(1 + f/700)$) instead of Slaney (linear below 1 kHz, `librosa`'s default) |
| `2` | peak | triangular filters normalised to peak 1 (HTK / Kaldi) instead of to unit area (`librosa` `norm="slaney"`) |
| `3` | mag | magnitude $\lvert X \rvert$ instead of power $\lvert X \rvert^2$ |
| `4` | floor | floor before the log $10^{-5}$ instead of $10^{-10}$ |
| `5` | dither | Gaussian noise of one 16-bit step ($1/32768$) added to the waveform — Kaldi's default amount |
| `6` | preemph | pre-emphasis $x[n] - 0.97\,x[n-1]$ |
| `7` | cmvn | output normalisation: per-utterance mean/variance normalisation per dimension instead of Whisper's clamp to $\max - 8$ followed by $(x + 4)/4$ |
| `8` | five at once | HTK + peak + pre-emphasis + natural log + CMVN — roughly "another toolkit's defaults, fed to Whisper" (not an exact Kaldi recipe: window shape, low-frequency cut-off and DC removal are unchanged) |
| `a` `t` `n` `s` `h` `q` | run all in order · table so far · next sentence in the feature plot · play that sentence · menu · quit | |

Whisper's own recipe is written out knob by knob in `demo_config.toml` (`[frontend.reference]`), checked against `openai/whisper`'s `audio.py` and `transformers 5.17.0`'s `feature_extraction_whisper.py`: Slaney scale and area normalisation (the filterbank is `librosa.filters.mel`'s default), power spectrum, `log10` with a $10^{-10}$ floor, no dither, no pre-emphasis, clamp and affine at the end.

## What is printed and plotted

For each recipe: one line per sentence (✓, or ✗ with the number of word errors and the words that changed), the corpus **WER** with substitutions / deletions / insertions, the difference to the reference WER, the **average log-probability per token** (a teacher-forced pass after decoding; it moves before the words do), how far the features are from Whisper's own (**mean |Δ|** over the frames that contain audio, in Whisper's normalised units; and the share of cells that changed at all), how many of the 40 transcripts differ from the reference transcript, and the three sentences that got worst. The window shows the features of the worst sentence — Whisper's, this recipe's on the same colour scale, and the difference — plus bar charts of WER and $-\log P$ for every recipe run so far.

WER is computed after both the LibriSpeech reference and the model output pass through Whisper's own `EnglishTextNormalizer`, so "Mr." against "MISTER" is not an error (but DOUGHNUTS against DONUTS still is — the normaliser does not map spelling variants). With 786 reference words, one error is 0.127 WER points.

## What the rehearsal found (2026-11-02, the lecture laptop, CPU)

| Key | Recipe | mean \|Δ\| | WER % | transcripts changed | avg log P |
|---|---|--:|--:|--:|--:|
| `r` | Whisper's own front-end | 0 | 3.31 | — | −0.111 |
| `0` | same recipe, our own code | 0.000 | 3.31 | 0 / 40 | −0.111 |
| `1` | HTK mel scale | 0.121 | 3.31 | 4 / 40 | −0.113 |
| `2` | peak-normalised filters | 0.460 | 3.18 | 2 / 40 | −0.118 |
| `3` | magnitude | 0.342 | 3.69 | 4 / 40 | −0.115 |
| `4` | floor $10^{-5}$ | 0.112 | 3.05 | 1 / 40 | −0.114 |
| `5` | dither | 0.002 | 3.31 | 0 / 40 | −0.111 |
| `6` | pre-emphasis | 0.193 | 3.31 | 1 / 40 | −0.115 |
| `7` | per-utterance CMVN | 0.564 | **53.56** | 22 / 40 | −0.916 |
| `8` | five knobs at once | 0.602 | **39.69** | 26 / 40 | −0.909 |

The reference's 26 errors are mostly proper names and spellings (VILLEFORT → VILLAFORTE, SANCHO → SANCHEZ, DOUGHNUTS → DONUTS). Six of the seven single knobs stay within ±0.4 WER points of the reference — fewer than three words in 786 — and move the average log-probability by at most 0.007; the knob that moves the *features* most (filter normalisation, mean |Δ| 0.460) makes no word worse. Only the output normalisation breaks the model, and not quietly: 237 inserted words in repetition loops (one sentence becomes "CAN I DO FOR YOU DAVEY" twenty-odd times, another seventy SOs), three sentences come back empty — while 18 of the 40 are still perfect. All five knobs together give *less* damage than CMVN alone; we do not know why. Decoding is greedy, so re-running gives the same numbers; the full record is `runs/rehearsal/rehearsal.txt`.

## Running it

```bash
cd speech_ai/demos/w02_d3_frontend_mismatch
./present.sh selftest      # no model: our front-end against transformers' numpy extractor, WER, sentence selection, a fake run
./present.sh fetch         # openai/whisper-small (commit pinned in model.lock); picks 40 sentences from dev-clean.tar.gz → data/sentences.json
./present.sh check         # versions, features vs. WhisperFeatureExtractor, identical transcripts for the same recipe, seconds per sentence → runs/check.json
./present.sh rehearse      # reference + all recipes, non-interactive → runs/rehearsal/
./present.sh               # the keys above; the reference run happens at start-up
./present.sh replay        # the rehearsal's results on the same keys, without torch
```

Requirements: the shared `speech_ai/demos/.venv` with `torch 2.14.0`, `transformers 5.17.0` and `soundfile 0.14.0` (all installed by the earlier demos' `setup.sh`), and `demos/_media/dev-clean.tar.gz` from W3 demo 3's `fetch` (LibriSpeech, CC BY 4.0). Whisper runs on the CPU; one recipe over 40 sentences takes 34–40 s on the lecture laptop (M5 Max). `data/sentences.json` is not published — `fetch` rebuilds it from the tarball.

The rehearsal record (`runs/rehearsal/rehearsal.json`) is summarised into `slides/assets/w02/w02-metrics.json` (`demo3`) by `./present.sh export`; the numbers on slide p66 come from there. In the published copy of `rehearsal.json` the `model.path` entry (the instructor's local Hugging Face cache directory) is replaced by a placeholder; `replay` does not read it.

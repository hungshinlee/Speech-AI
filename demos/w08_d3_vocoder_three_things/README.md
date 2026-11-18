<!-- 此檔由 bin/publish_demos.py 從 private repo 的 <course>/demos/ 複製而來，請勿在公開 repo 直接編輯。 -->
# W8 · Demo 3 — one mel, three vocoders; then a predicted mel; then no mel at all (slides p72–p74)

The fifth section of W8 says a vocoder does three things: it invents the phase, it undoes the mel compression, and it adds the detail a smoothed spectrogram never had. This demo plays them. The same recording goes through three vocoders; then two *predicted* mels (Demo 1's regressed SpeechT5 mel and Demo 1's sampled F5 mel) go through the same three; and finally a system that has no mel step at all speaks Taiwanese and Hakka from romanised text.

## The grid

| | Griffin-Lim (W2 Demo 2, 64 iterations) | Vocos | BigVGAN v2 |
|---|---|---|---|
| what it is | the 1984 algorithm: mel → pseudo-inverse → magnitude → iterate the phase (numpy, `../dsp.py`) | `charactr/vocos-mel-24khz`: a ConvNeXt stack *at frame rate* that predicts magnitude and phase, then one iSTFT (13.5 M parameters) | `nvidia/bigvgan_v2_24khz_100band_256x`: HiFi-GAN-style transposed convolutions *in the time domain* (×256 upsampling), Snake activations (112 M parameters) |
| its front end | the Vocos one (below) | 100 bins, 24 kHz, hop 256, HTK mel scale, no filter normalisation, `ln(clamp 1e-7)` | 100 bins, 24 kHz, hop 256, **Slaney** mel scale, **area-normalised** filters, `ln(clamp 1e-5)` |

**Rows.** `g` the real recording (W3's sentence `1272-128104-0000`, dev-clean, CC BY 4.0, 16 kHz resampled to 24 kHz — so there is nothing above 8 kHz); `4` Demo 1's SpeechT5 mel (regressed, 80 bins / 16 kHz / log10); `5` Demo 1's F5 mel (sampled by flow matching, natively in Vocos's definition). **Columns.** `1` `2` `3` play the current row through the three vocoders; `4` and `5` play all three in order when you switch to that row.

**Each vocoder gets its own front end** (W2 Demo 3's lesson: two things both called "100-band log-mel at 24 kHz" need not be the same numbers). For the real recording the mel is computed from the waveform with each vocoder's own code. A *predicted* mel exists only in the definition its model was trained with, so to hand it to a different vocoder it has to be converted: in the log domain, each bin's value is divided by that bin's filter gain (assuming a flat spectrum inside the bin), interpolated in Hz onto the other filterbank's bin centres, multiplied by the other gains, and — for SpeechT5 — interpolated in time from 62.5 to 93.75 frames/s and extended above 7.6 kHz with each frame's quietest value. The conversion is checked on the real recording, where both mels exist (`./present.sh check` prints the error, a few dB). Key `m` plays what happens **without** converting: the Vocos-defined mel handed to BigVGAN as-is. Same shape, same name, a different matrix — the Slaney/area-normalised scale sits about 36 dB below the HTK one, so BigVGAN is handed a spectrogram far outside what it was trained on.

**Numbers on screen.** *fit* = the vocoder's output re-analysed with its own front end, against the mel it was given: RMS of the difference in dB (both floored at ln 1e-4). It measures the first of the three jobs — how faithfully the magnitude was honoured — and nothing else: Griffin-Lim optimises exactly this and gets a *lower* fit than the neural vocoders, while sounding worst. *LSD* = log-spectral distance to the original recording (real row only; 25 ms / 10 ms frames at 24 kHz). *time* = wall-clock for one decode after a warm-up call, and its real-time factor. Vocos runs at frame rate, BigVGAN samples the waveform; the time ratio is the point of slide p71's on-device row. All three numbers are references; the judge is your ear.

## No mel at all: MMS-TTS

`6` plays one Taiwanese sentence through `facebook/mms-tts-nan` and `7` one Hakka sentence through `facebook/mms-tts-hak` (Pratap et al., JMLR 2024 §7: one VITS per language, a single speaker, New Testament readings, 16 kHz, straight from characters to waveform, no spectrogram in between). The model cards say nothing about the input; `vocab.json` does — 48 symbols for `nan`, all Latin letters and Pe̍h-ōe-jī marks; 43 for `hak`, Pha̍k-fa-sṳ. The terminal prints what the tokenizer kept and what it dropped (punctuation is not in either vocabulary). `c` feeds the same Taiwanese sentence written in Chinese characters: the tokenizer keeps nothing and there is nothing to synthesise. `e` is the English model on W3's sentence, the fallback.

Input texts, in `demo_config.toml`:

- `nan` (POJ, base tones): *Lí hó, góa sī chi̍t ê ha̍k-seng, góa beh o̍h kóng Tâi-gí.* — 你好，我是一個學生，我欲學講台語。
- `hak` (PFS): *Ngì hó, ngài he ho̍k-sâng, ngài oi ho̍k kóng Hak-fa.* — 你好，我是學生，我愛學講客話。

Whether the `nan` model expects base or sandhi tones in its input, and which Hakka accent the `hak` model speaks, are things to decide by listening.

## Keys

`0` original · `g` `4` `5` choose the row (`4` `5` also play the row's three clips) · `1` `2` `3` Griffin-Lim / Vocos / BigVGAN · `m` the unconverted Vocos mel into BigVGAN · `n` what Demo 1 itself played for this row (SpeechT5's own HiFi-GAN; F5's Vocos) · `6` `7` `c` `e` MMS-TTS · `t` the table · `r` again · `h` menu · `q` quit.

The window shows four panels: the input mel in its own front end, then the three outputs re-analysed in each vocoder's own front end, with fit / LSD / time in the titles.

## Running it

```bash
cd speech_ai/demos/w08_d3_vocoder_three_things
../w08_d1_mean_vs_sample/setup.sh    # once: builds ../.venv-tts (Demo 1's environment; vocos, transformers, librosa live there)
./present.sh fetch         # once: Vocos, BigVGAN (code + weights), MMS-TTS nan/hak/eng, SpeechT5's front-end config; pins commits in model.lock
./present.sh check         # versions; cpu vs mps for each network; the mel conversion checked on the real recording → runs/check.json
./present.sh rehearse      # everything, non-interactive → runs/rehearsal/
./present.sh               # the keys above
./present.sh replay        # the rehearsal's outputs on the same keys, without torch
./present.sh fake          # the whole flow without any model (Griffin-Lim stand-ins, marked FAKE)
```

Requirements: Demo 1's `speech_ai/demos/.venv-tts` (torch 2.14.0, transformers 5.17.0, vocos 0.1.0, librosa 1.0.0; nothing new is installed — BigVGAN's code comes with its weights from the Hugging Face repo and is imported from there). Rows `4` and `5` read Demo 1's rehearsal files (`../w08_d1_mean_vs_sample/runs/rehearsal/*_seed1_mel.npy`), so run Demo 1's rehearsal first.

Weights: Vocos MIT; BigVGAN v2 MIT; MMS-TTS CC BY-NC 4.0 (non-commercial). Numbers quoted on the slides come from `runs/rehearsal/rehearsal.json`, summarised in `slides/assets/w08/w08-metrics.json` (`demo3`); times are from the instructor's machine and are not comparable with the papers' RTFs.

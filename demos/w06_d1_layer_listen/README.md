<!-- 此檔由 bin/publish_demos.py 從 private repo 的 <course>/demos/ 複製而來，請勿在公開 repo 直接編輯。 -->
# W6 · Demo 1 — adding the stages back, one at a time: Mimi, EnCodec, DAC (slides p32–p34)

The same sentence goes through three neural audio codecs, and you listen to the reconstruction with more and more residual-VQ stages — and with **stage 1 alone**.
Every claim in the lecture about what the stages carry (energy first, detail later, content only through distillation) can be heard here.
The window shows the original's log-mel on top, the current reconstruction's log-mel below it with $f_r$, $n_q$, $V$ and $B$, and — on request — the two $F_0$ tracks.

**You are not required to run this.** It is published so that you can read it, re-run it and change it (another sentence, your own language, another codec from `transformers`).
Menus and labels are in Chinese; the keys are listed below.

## Keys

| key | what happens |
|---|---|
| `o` | play the original of the current input |
| `1` `2` `4` `8` | Mimi with the first 1 / 2 / 4 / 8 stages (12.5 Hz; stage 1 is the distilled semantic stage) |
| `e` | EnCodec 24 kHz, one step up per press: 1.5 → 3 → 6 → 12 → 24 kbps (2 / 4 / 8 / 16 / 32 stages) |
| `d` | DAC 24 kHz, one step up per press: 1 → 2 → 4 → 8 stages |
| `m` | switch input (the LibriSpeech sentence ↔ the lecturer's Mandarin sentence — the latter is not published; record your own) |
| `f` | draw the $F_0$ tracks of the original and of the current reconstruction |
| `r` | play the current setting again |
| `t` | print the table of objective references (bitrate, LSD, SNR, $F_0$ RMSE, voicing agreement) |
| `h` menu · `q` quit | |

## Definitions

| | |
|---|---|
| "first $n_q$ stages" | Each codec encodes the input **once** with all its stages, giving an $N_q \times T$ matrix of code indices; "first $n_q$ stages" keeps the first $n_q$ rows and decodes them. The reconstruction is the sum of the first $n_q$ codewords — that is the definition of residual VQ, and the property quantizer dropout trains for. `check` confirms that this gives the same codes as each class's own stage argument (`bandwidth=`, `n_quantizers=`, `num_quantizers=`). |
| $B$ | $f_r \, n_q \log_2 V$ with $f_r$ and $V$ read from the loaded model config; the frame rate is also measured as $T$ / seconds. Mimi: $12.5 \times 11 = 137.5$ bps per stage, 1.1 kbps at eight; EnCodec: $75 \times 10 = 750$ bps per stage. |
| LSD | log-spectral distance in dB (25 ms window, 10 ms hop, 1024-point FFT), from the shared `dsp.py`. |
| SNR | waveform SNR in dB. A GAN-trained codec does not preserve phase, so values near 0 dB are normal — the point of printing it is that it does not predict intelligibility. |
| $F_0$ | the autocorrelation estimator from W2's `dsp.py` (40 ms frames, 10 ms hop, 60–400 Hz), applied to the original and to the reconstruction; RMSE over frames voiced in both, plus the fraction of frames where the voiced/unvoiced decision agrees. **A reference, not the judge**: $F_0$ is a physical quantity, pitch is a percept. |
| gain | within one input, the original and all its reconstructions are written with the **same** gain (the loudest version peaks at 0.89), so loudness differences are real and not normalised away. |

## Requirements

- The shared `speech_ai/demos/.venv` (`../setup.sh`) plus `torch 2.14.0` and `transformers 5.17.0` from `../w03_d3_blank_heatmap/setup.sh`; no new packages.
- Three models from Hugging Face, fetched once and pinned by commit in `model.lock`: `kyutai/mimi`, `facebook/encodec_24khz`, `descript/dac_24khz` (a few hundred MB in total; licences as stated on each model card).
- The LibriSpeech sentence `1272-128104-0000` (dev-clean, CC BY 4.0) is in `audio/`, copied from W3 Demo 3.
- A Mandarin (or any tonal-language) sentence: record your own with `./present.sh record` (6 s, silence trimmed) or import a WAV with `--from`.

## Run it

```bash
cd speech_ai/demos/w06_d1_layer_listen
./present.sh selftest        # bitrate formula, F0 estimator, metric identities, the fake pipeline (≈ 20 s, no models)
./present.sh fetch           # the three codecs (pinned → model.lock) and the LibriSpeech sentence (network)
./present.sh check           # do the three classes load, config f_r / V / N_q vs. the encoded matrix, sliced decode vs. the stage argument, mps vs. cpu → runs/check.json
./present.sh record          # your tonal sentence → ../_media/w06_d1/mandarin.wav (not versioned)
./present.sh rehearse        # all settings for both inputs → runs/rehearsal/ (wavs, metrics, log-mel grids, F0 plots) and w06-metrics.json
./present.sh                 # the window; press o 1 2 4 8 e e e e e d d d d m o 1 8 e f t
./present.sh replay          # the same keys from runs/rehearsal/, no torch
./present.sh fake            # no models: low-pass + noise stands in for the stages (labelled FAKE)
```

## What you should see

- Mimi with stage 1 alone (137.5 bps): the lecture's prediction was that the **words survive and the voice does not**, because stage 1 is distilled from WavLM and carries content rather than identity. **In rehearsal this did not hold**: with stage 1 alone neither the words nor the voice came through (LSD 28.5 dB, $F_0$ RMSE 63 Hz, voicing agreement 0.55 on the LibriSpeech sentence; 62 distinct codes over 74 frames). The sentence comes back as stages are added (stage 2: LSD 17.0 dB, voicing 0.91; stage 8: 13.1 dB, 0.93). "Semantic" names what the codebook was trained towards, not what a decoder gets back from it alone.
- EnCodec at 1.5 kbps (two stages, ten times Mimi's stage-1 bitrate): the prediction was the opposite — **it sounds like the speaker but is harder to understand**. In rehearsal it was clearer than Mimi's stage 1 on *both* counts, and so was DAC with one stage (750 bps): the tenfold bitrate shows before any difference in training target does. A fair acoustic-first vs. semantic-first comparison needs equal bitrates, which this demo does not have.
- The Mandarin sentence with Mimi's stage 1 alone: are the four tones still distinguishable? The lecture's prediction (from the design, not from a measurement) is that distillation carries "tone is not in the units" over from the SSL teacher. In rehearsal it held: at stage 1 the tones were flattened ($F_0$ RMSE 57 Hz, voicing agreement 0.49), at stage 8 they were back (5.4 Hz, 0.86), and EnCodec at 1.5 kbps kept them (23 Hz, 0.86). One sentence, one listener — a classroom listening, not evidence; the $F_0$ plot is only a reference.
- Every number above is from the lecturer's rehearsal on the M5 Max (2026-10-19; `runs/rehearsal/rehearsal.txt` has the full table for both inputs and the model facts read from the configs: all three weights have $N_q = 32$, DAC 24 kHz runs at 75 Hz with $V = 1024$). The Mandarin recording and its reconstructions are not published; `./present.sh record` makes your own.

## Files

| file | |
|---|---|
| `demo.py` | everything: `show` / `rehearse` / `replay` / `record` / `check` / `fetch` / `fake` / `selftest` / `export` |
| `demo_config.toml` | codecs and their stages, inputs, F0 and metric settings |
| `present.sh` | entry point; runs offline (`HF_HUB_OFFLINE=1`) except `fetch` |
| `audio/utterance.wav`, `.txt` | the LibriSpeech sentence (16 kHz; resampled to 24 kHz on load) |
| `model.lock` | the three pinned commits |
| `runs/rehearsal/` | the lecturer's rehearsal: reconstructions of the LibriSpeech sentence, its log-mel grid and $F_0$ plot, and the metrics of both inputs in `rehearsal.json` (the Mandarin recording and its reconstructions stay on the lecturer's machine; `private = true` in the config routes them to `../_media/`) |
| `runs/check.json` | not published |

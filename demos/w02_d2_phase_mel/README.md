<!-- 此檔由 bin/publish_demos.py 從 private repo 的 <course>/demos/ 複製而來，請勿在公開 repo 直接編輯。 -->
# W2 · Demo 2 — phase × magnitude: taking a "spectrogram → waveform" reconstruction apart (slides p34, p40, p42)

One five-second sentence is analysed with the usual ASR front-end (25 ms Hann window, 10 ms hop, `n_fft = 512`) and re-synthesised under a 2 × 3 factorial design, so that the different sources of distortion can be *heard one at a time*:

|  | original phase | Griffin-Lim, $k$ iterations | random phase |
|---|---|---|---|
| **linear magnitude** | `1` (= the original; sanity check) | `2` | `3` |
| **mel-80, inverted back to linear** | `4` | `5` | `6` |

- **Across a row** (`1` → `2` → `3`) only the phase changes. `[` and `]` step the Griffin-Lim iteration count through 0 / 1 / 4 / 16 / 64 / 256; `2` and `5` follow.
- **Down a column** (`1` → `4`) only the mel compression changes — the phase is the original, so whatever you hear in `4` can only come from mel. A three-way "Griffin-Lim vs. original vs. random" comparison cannot separate these.
- `a` `s` `d` `f`: original phase with mel-80 / 40 / 20 / 10 — how the mel loss grows as the number of bands shrinks.
- `c` opens the convergence curve (inconsistency vs. iterations, semi-log); `g` the six spectrograms side by side; `0` the original; `q` quits.

The order used in class is: `3` first (random phase — the broken one), then `[` down to 0 iterations and `]` up one step at a time listening to `2`, then `c`, then `4` and `a s d f`.

## The numbers on screen

- **LSD** (log-spectral distance, dB) compares magnitudes only and is **insensitive to phase**. So the LSD of `3` is *not* zero: a random-phase STFT is not consistent, and after re-analysis the magnitude is no longer the one you started from. That is the consistency argument, audible.
- **inconsistency** $= \lVert \mathbf{C} - P(\mathbf{C})\rVert^2 / \lVert \mathbf{C}\rVert^2$, the quantity Griffin & Lim proved non-increasing. The program checks monotonicity on every run and prints the verdict; after 256 iterations it is still not zero — a local solution.
- `rehearse` also prints the **waveform SNR**. For Griffin-Lim it is negative throughout and does not improve with iterations, while LSD falls steadily — SNR is extremely phase-sensitive and decoupled from what you hear. (W2's key mathematics returns to this as "SNR does not predict WER".)

## Four implementation decisions

1. **ISTFT is Griffin & Lim's least-squares version** (divide by $\sum w^2$). 25 ms Hann at 10 ms hop does not satisfy COLA, but this version reconstructs perfectly for any hop — and it *is* the projection onto the consistent set, so the theory and the code are the same object.
2. **Griffin-Lim is the original 1984 algorithm, no momentum**, random initial phase, fixed seed. `librosa.griffinlim` defaults to a momentum variant that converges faster but has no monotonicity guarantee; this course talks about the original.
3. **mel → linear is the pseudo-inverse followed by clipping negatives.** NNLS is available in `demo_config.toml`; on synthetic vowels it gave sparser, hole-ridden spectra and a worse LSD.
4. **All versions are written with the same gain**, not normalised individually — otherwise loudness differences would mask quality differences.

## Requirements

| | |
|---|---|
| Packages | the shared `.venv` only: `numpy 2.4.4`, `scipy 1.17.1`, `matplotlib 3.10.9`, `sounddevice 0.5.6`. All DSP is in `../dsp.py`; `librosa` is deliberately not used. |
| Input audio | **not in the repository** — `audio/sentence.wav` is the instructor's voice, and so are the reconstructed WAVs under `runs/rehearsal/sentence/`. Record your own 4–5 s sentence with `./present.sh record` (fricatives and stops make the losses easiest to hear: fricatives lose their high band to mel first, stop bursts are smeared by phase error first), or point `--wav` at any WAV. |
| Rehearsal records here | `runs/rehearsal/sentence/metrics.json` (LSD and SNR for all 17 versions), `convergence.png`, `spectrograms.png` |
| Tested on | Apple M5 Max, macOS 26.6, Python 3.12.14 (2026-09-22) |

## Run it

```bash
cd demos && ./setup.sh && cd w02_d2_phase_mel
./present.sh fake             # synthetic signal: flow and keys only, not for listening
./present.sh record           # record 5 s → audio/sentence.wav
./present.sh rehearse         # compute all 17 versions → runs/rehearsal/sentence/ (WAVs, metrics.json, two figures); prints LSD, SNR, inconsistency
./present.sh                  # = show: recompute, then the playback screen (keys above)
./present.sh replay           # the playback screen on the saved rehearsal — only after your own `rehearse`, since the WAVs are not published
./present.sh --wav path.wav rehearse   # any other WAV (16 kHz mono; `afconvert -f WAVE -d LEI16 in.m4a out.wav`)
```

## What you should see

From `runs/rehearsal/sentence/metrics.json` (2026-09-22, the instructor's sentence):

| version | LSD (dB) | SNR (dB) |
|---|---:|---:|
| linear, original phase | 0.00 | 225 (identity) |
| linear, random phase (GL 0) | 8.47 | −1.2 |
| linear, GL 1 / 4 / 16 / 64 | 4.71 / 3.46 / 2.64 / 2.06 | −2.7 / −3.0 / −3.1 / −3.4 |
| linear, GL 256 | **1.71** | **−3.7** |
| mel-80, original phase | 4.15 | 24.9 |
| mel-80, GL 256 | 6.20 | −3.7 |
| mel-40 / 20 / 10, original phase | 5.34 / 7.02 / 9.89 | 16.2 / 7.9 / 6.2 |

Two things to read off: LSD falls monotonically with Griffin-Lim iterations while SNR gets slightly *worse* — the two metrics disagree because one ignores phase and the other is dominated by it; and once mel-80 is in the chain, Griffin-Lim can no longer bring LSD below ≈ 6 dB, because the magnitude it is iterating towards is already wrong.

On the instructor's laptop speakers, 256 iterations of Griffin-Lim were still distinguishable from the original. Whether that holds depends on the room and the loudspeakers — which is itself the point W9 makes about listening tests.

## Files

| File | |
|---|---|
| `demo.py` | `show` / `replay` / `record` / `rehearse` / `fake`; `--wav` |
| `demo_config.toml` | STFT parameters, Griffin-Lim iteration ladder and seed, mel scale (`slaney` / `htk`), bands, inverse method |
| `present.sh` | entry point; uses `../.venv/bin/python` |
| `runs/rehearsal/sentence/metrics.json` | LSD and SNR per version, configuration, package versions |
| `runs/rehearsal/sentence/convergence.png`, `spectrograms.png` | inconsistency vs. iterations; the six spectrograms |

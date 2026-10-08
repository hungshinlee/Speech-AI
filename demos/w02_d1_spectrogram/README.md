<!-- 此檔由 bin/publish_demos.py 從 private repo 的 <course>/demos/ 複製而來，請勿在公開 repo 直接編輯。 -->
# W2 · Demo 1 — live spectrogram, pitch track and vowel chart (slides p12, p23, p52)

A microphone demo with no model in it. The window shows the last four seconds of audio as a spectrogram (wideband or narrowband, switchable), a pitch track, and a vowel chart that places the current $(F_1, F_2)$ as a moving dot. It is used three times in the lecture:

| Slide | What is done | What to watch |
|---|---|---|
| p12 — two formants tell vowels apart | say /a/, /i/, /u/, then glide slowly from /i/ to /a/ | the red dot on the vowel chart; the two bright bands ($F_1$, $F_2$) on the wideband spectrogram moving apart and together |
| p23 — the time–frequency trade-off | hold a long vowel, **freeze with space**, then switch `n` / `w` | the *same* two seconds of sound: `w` (5 ms window) shows vertical striations — glottal pulses — and formants; `n` (30 ms) shows horizontal harmonics. Freezing is the point: otherwise it looks like a different recording |
| p52 — tone and intonation | one syllable on the four Mandarin tones; one sentence as a statement and as a question | the blue pitch track; in `n`, the harmonic spacing changes with $F_0$ |

A failure case worth trying: set `pitch.ceiling_hz = 150` in `demo_config.toml` and have a high voice speak — the pitch track drops to half the frequency. That is an octave error, the classic failure of autocorrelation pitch tracking.

The signal processing is all in `../dsp.py` (STFT, autocorrelation $F_0$, LPC formants), written in plain numpy so that it can be read. Formants are the roots of a 12th-order LPC polynomial on audio down-sampled to 10 kHz; the dot is the median of the last five frames. Known limit: on real voices, especially high-pitched ones, LPC sometimes picks a harmonic instead of a formant, so the dot jumps. The grey reference points /a/ /i/ /u/ on the chart are approximate male-average formant values for American English vowels, used only as landmarks.

`w02_d1.praat` is a Praat script for the moments when a precise reading is needed (formant and $F_0$ values, three plots side by side). It is driven by `./present.sh praat <name>`; it had not been run in Praat at the time of writing, so expect to fix a parameter count on first use.

## Requirements

| | |
|---|---|
| Packages | the shared `.venv`: `numpy 2.4.4`, `scipy 1.17.1`, `matplotlib 3.10.9`, `sounddevice 0.5.6` |
| Hardware | a microphone and a display; no model, no GPU |
| Rehearsal records | **none checked in** — this demo is live capture; the three fallback recordings (`vowels`, `tones`, `question`) are made by `rehearse` and were not recorded at the time of publishing |
| Optional | Praat (`brew install --cask praat`) for `./present.sh praat` |
| Tested on | Apple M5 Max, macOS 26.6, Python 3.12; the window updates every 60 ms recomputing a 4 s spectrogram |

## Run it

```bash
cd demos && ./setup.sh && cd w02_d1_spectrogram
./present.sh fake             # no microphone: synthetic vowels, check that the window opens and the keys respond
./present.sh devices          # list input devices
./present.sh                  # = show: live capture
./present.sh rehearse         # live capture; press space to freeze, then 1 / 2 / 3 to save the 4 s on screen as runs/rehearsal/{vowels,tones,question}.wav + .png
./present.sh replay vowels    # load a saved take; the display is frozen but w / n / p / z still work
./present.sh praat vowels     # open the same WAV in Praat
```

Keys in the window: `w` / `n` wideband (5 ms) / narrowband (30 ms) window; `space` freeze; `p` pitch track on/off; `z` toggle the frequency ceiling (5 / 8 kHz); `c` clear the vowel-chart trail; `s` save; `q` quit. All settings are in `demo_config.toml` (`n_fft = 512` decides only the bin spacing, 31.25 Hz — not the resolution; that is the window length).

## Files

| File | |
|---|---|
| `demo.py` | `show` / `rehearse` / `replay <name>` / `fake` / `devices` |
| `demo_config.toml` | sample rate, display window and tick, the two window lengths, pitch floor/ceiling, vowel-chart smoothing |
| `present.sh` | entry point; uses `../.venv/bin/python`; `praat` subcommand |
| `w02_d1.praat` | Praat script: spectrograms at 5 ms and 30 ms, pitch track, formant readings |

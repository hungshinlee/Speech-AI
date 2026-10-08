<!-- 此檔由 bin/publish_demos.py 從 private repo 的 <course>/demos/ 複製而來，請勿在公開 repo 直接編輯。 -->
# W3 · Demo 1 — the same sentence, twice as fast (slide p6)

A volunteer says one sentence (default: 今天天氣很好, six characters) at normal speed, then again about twice as fast. Both takes are trimmed of leading and trailing silence, aligned at their first voiced frame, and drawn as two spectrograms one above the other with **one column per 10 ms frame**. The terminal prints, for each take, the frame count $T$, the symbol count $U$, $T/U$, and the voiced / unvoiced frame counts.

Until you press `s`, the screen shows only a level meter — no time axis, no durations — because the question comes first: *which of these numbers changed, and which did not?* $T$ changed; $U$ did not; and nothing on the screen says which frames belong to which symbol. That is the alignment problem the rest of W3 is about, made concrete.

| Number | Definition |
|---|---|
| $T$ | length after trimming ÷ 10 ms. Trimming threshold: the higher of (peak − 35 dB) and (noise floor + 15 dB), with the onset and offset required to stay above it for 50 ms; 60 ms of padding is kept at each end |
| $U$ | characters in the configured sentence, spaces and punctuation removed — one symbol per Mandarin character. Change the symbol set (letters, IPA) and $U$ changes: the vocabulary is a decision made *before* the loss |
| voiced / unvoiced | frames with a detected $F_0$ (40 ms window, autocorrelation, the same code as W2 demo 1). A coarse split — reliable in aggregate, not frame by frame — used only to support "vowels compress more than consonants, so $T$ does not halve" |

The spectrograms use a 25 ms window at 10 ms hop; `n` switches to a 30 ms window with the hop unchanged, so the column count stays the same. The shorter take is greyed where it has ended; its end is marked with a red dashed line.

## Requirements

| | |
|---|---|
| Packages | the shared `.venv`: `numpy 2.4.4`, `scipy 1.17.1`, `matplotlib 3.10.9`, `sounddevice 0.5.6`; DSP in `../dsp.py` |
| Hardware | a microphone and a display; nothing else |
| Audio | the two rehearsal takes (`normal*.wav`, `fast*.wav`) are **not in the repository** — they are the instructor's voice. `metrics.json` and `compare.png` are. `replay` therefore needs your own `rehearse` first. |
| Fonts | the sentence is drawn in the figure; the code opens a CJK font file directly (macOS paths) and warns if none has the glyphs — the numbers are unaffected |
| Tested on | Apple M5 Max, macOS 27.0.1, Python 3.12.14 (2026-10-05) |

## Run it

```bash
cd demos && ./setup.sh && cd w03_d1_two_rates
./present.sh fake             # no microphone: a synthetic "sentence", check window and keys
./present.sh devices
./present.sh rehearse         # r → 2 s countdown → 4 s take (normal); r again (fast); s → side by side, saved to runs/rehearsal/
./present.sh                  # = show; s saves to runs/live/<time>/ instead
./present.sh replay           # reload runs/rehearsal/ (re-trims the *_raw.wav with the current thresholds)
./present.sh replay DIR       # reload another folder, e.g. runs/live/2026-…
```

Keys in the window: `r` record the next take; `1` / `2` re-record a take; `s` show side by side; `w` / `n` window length; `p` $F_0$ dots; `z` frequency ceiling; `q` quit. `audio.record_seconds` (4 s) and the trimming thresholds are in `demo_config.toml`; a noisier room wants a larger `trim.floor_margin_db`. Both takes always use the same thresholds — otherwise the comparison is meaningless.

## What you should see

From `runs/rehearsal/metrics.json` (2026-10-05, the instructor's two takes of 今天天氣很好, $U = 6$):

| take | trimmed length | $T$ | $T/U$ | voiced frames | unvoiced frames |
|---|---:|---:|---:|---:|---:|
| normal rate | 1.99 s | **199** | 33.2 | 149 | 47 |
| twice as fast | 1.34 s | **134** | 22.3 | 90 | 41 |

$T$ fell to 0.67 of the original, not 0.5: voiced frames compressed to 0.60, unvoiced only to 0.87. "Twice as fast" is what the speaker was asked for; what came out is a different ratio for vowels and for consonants.

One practical note from the rehearsal log, in case your trims look wrong: a plain "35 dB below peak" threshold did nothing on a MacBook microphone (the noise floor is only 30–38 dB below the peak, so $T$ came out as the full 400 frames). The noise-floor term and the 50 ms run-length requirement were added for that reason.

## Files

| File | |
|---|---|
| `demo.py` | `show` / `rehearse` / `replay [DIR]` / `fake` / `devices` |
| `demo_config.toml` | sentence, recording length and countdown, frame definition, trimming thresholds, pitch settings, display |
| `present.sh` | entry point; uses `../.venv/bin/python` |
| `runs/rehearsal/metrics.json`, `compare.png` | the numbers above and the side-by-side figure; the WAVs are not published |

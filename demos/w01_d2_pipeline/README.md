<!-- 此檔由 bin/publish_demos.py 從 private repo 的 <course>/demos/ 複製而來，請勿在公開 repo 直接編輯。 -->
# W1 · Demo 2 — one sentence through a cascade, and the latency budget filled in live (slide p41)

One recorded sentence with a pause in the middle (a ticket-booking request in Mandarin: "I'd like to book a ticket … er … to Tainan next Wednesday") is pushed through a cascaded pipeline — energy-based endpoint detection → whisper.cpp → an LLM via `mlx-lm` → a TTS model via `mlx-audio` — and every module boundary is time-stamped. The program runs the computational stages ten times, reports **p50 and p95**, draws the timeline, and fills in the latency-budget table from the slides.

Every cell is the time that elapses **after the user stops speaking** (W1's first budget rule), not the module's total processing time:

| Cell | How it is measured | Kind |
|---|---|---|
| endpoint detection | a streaming energy VAD (25 ms / 10 ms) fires when silence has accumulated for `threshold_ms`; the cell is the distance from the true end of speech to that firing | **algorithmic** — the threshold is a setting |
| ASR tail | from the endpoint firing to the transcript. whisper is not a streaming recogniser, so the whole transcription lands in this cell | computational |
| LLM prefill + first token | from prompt in to the first generated token | computational |
| TTS first packet | from the reply text to the first chunk of audio | computational |
| playout buffer | a configured value ($D_{\text{buf}}$ on the slides), not a measurement | setting |

The last row of the table adds the per-cell p95s; the program says on screen that this is an upper bound, not the p95 of the chain.

The end-to-end half of the slide is **not** measured live: the real-time mode of `moshi-mlx` takes the microphone and gives no timestamps. The timeline marks the 60 ms response of the cold-open recording (from `slides/assets/w01/w01-metrics.json`) as a pre-recorded reference on a different utterance, and says so. `./present.sh moshi` lets you talk to Moshi, but produces no numbers.

The `endpoint` subcommand is the measured version of the "pause inside a sentence vs. threshold" slide: it lists what thresholds of 300 / 700 / 1000 / 1500 ms do inside the sentence and at its end.

## Requirements

| | |
|---|---|
| Hardware | **Apple silicon only**; the three models together need well under 32 GB, the LLM alone ≈ 15 GB. Everything else can run `./present.sh fake`. |
| Packages | shared `.venv` (`../setup.sh`) plus this folder's `./setup.sh`: `pywhispercpp 1.5.1` (whisper.cpp, Metal), `mlx 0.32.2`, `mlx-lm` pinned to a git commit, `mlx-audio 0.5.4` (+ `misaki[en,zh]` for the Chinese G2P). `moshi-mlx 0.3.0` pins `mlx<0.27`, which conflicts with `mlx-lm`, so it lives in a separate `uv tool` environment and only the `moshi` subcommand uses it. Versions in `../versions.lock`. |
| Models | downloaded by `./setup.sh` and the first `check` into the package caches, not into this folder: whisper.cpp `small`; `mlx-community/gemma-4-26b-a4b-it-4bit`; `mlx-community/Kokoro-82M-bf16` (voice `zf_xiaobei`, `lang_code = "z"`). |
| Audio input | `audio/utterance.wav` is **not in the repository** (it is the instructor's voice). `./present.sh record` records your own; `./present.sh --wav path.wav …` uses any 16 kHz WAV. `replay` needs no audio. |
| Tested on | Apple M5 Max, macOS 26.6, Python 3.12.14 (2026-09-20 / 2026-09-21) |

## Run it

```bash
cd demos && ./setup.sh && cd w01_d2_pipeline
./setup.sh                    # the four model packages → ../versions.lock (Apple silicon only)
./present.sh fake             # any machine: synthetic signal + fake back-ends, the whole flow and screen
./present.sh devices          # list audio devices
./present.sh record           # record the sentence (pause ≈ 0.7–1 s in the middle, 1.5 s of silence at the end) → audio/utterance.wav
./present.sh endpoint         # what four thresholds do to your recording, inside the sentence and at its end
./present.sh check            # load the three models and warm them up (first run downloads; slow)
./present.sh rehearse         # 10 runs → runs/rehearsal/utterance/ (table + timeline figure)
./present.sh replay           # the instructor's rehearsal record, no models needed
./present.sh                  # = show: predict → play → 10 runs → table → timeline
./present.sh moshi            # talk to moshi-mlx in real time (headphones); no numbers
```

Terminal prompts are in Chinese; Enter advances.

## What you should see

From `runs/rehearsal/utterance/result.json` (2026-09-21, M5 Max; whisper.cpp `small` → Gemma 4 26B-A4B 4-bit → Kokoro):

| Cell | p50 | p95 |
|---|---:|---:|
| endpoint threshold (setting) | 1000 | 1000 |
| ASR tail | 70 | 71 |
| LLM first token | 94 | 95 |
| TTS first packet | 124 | 125 |
| playout buffer (setting) | 100 | 100 |
| **total** | **1388 ms** | 1391 (sum of p95s: an upper bound) |

The transcript was right in all ten runs; the reply was the same 23-token sentence every time. (The slides quote an earlier rehearsal of the same setup, 2026-09-20, with a total of 1383 ms; the record checked in here is the re-run of the next day.)

The pause inside the recorded sentence measured **1040 ms**. The `endpoint` report shows what that does to a threshold: 300, 700 and even 1000 ms all fire *inside* the sentence (false endpoints at 2.59 / 2.99 / 3.29 s); only 1500 ms survives the pause — at the price of 1500 ms after the real end. The run used 1000 ms as the cascade's threshold, which is why that one cell is about seven tenths of the total.

That is the point of the demo: the largest cell in the budget is neither the biggest model nor a compute problem. It is a conservatively set silence threshold — and the three computational cells together are under 300 ms. The slides return to this when W12 introduces semantic endpointing.

Numbers on your machine will differ in the three computational cells (and the first run after a cold start includes Metal shader compilation — `check` warms up first); the shape of the result does not depend on hardware, because the dominant cell is a setting.

## Files

| File | |
|---|---|
| `demo.py` | `show` / `replay` / `record` / `endpoint` / `check` / `rehearse` / `moshi` / `devices`; `--fake` back-ends; `--wav` to change the input |
| `demo_config.toml` | VAD parameters, thresholds, number of repeats, playout buffer, the three back-ends and models, the fake back-end's canned outputs |
| `present.sh` | entry point; uses `../.venv/bin/python` |
| `setup.sh` | installs the model packages into the shared `.venv`; `moshi-mlx` as a separate `uv tool` |
| `runs/rehearsal/utterance/result.json`, `timeline.png` | the ten runs, the endpoint report, package versions, the timeline figure |

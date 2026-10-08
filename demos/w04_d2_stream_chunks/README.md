<!-- 此檔由 bin/publish_demos.py 從 private repo 的 <course>/demos/ 複製而來，請勿在公開 repo 直接編輯。 -->
# W4 · Demo 2 — one streaming transducer, three chunk sizes: when does each word appear? (slide p42; p40–41 lead into it)

The same LibriSpeech sentence as W3 demo 3 is fed **at real-time pace** to a streaming zipformer transducer (`sherpa-onnx`), in chunks of 1280, 640 and 320 ms. Every time a symbol comes out, three clocks are recorded, and each word's clocks are compared with the acoustic end of that word (from W3 demo 3's CTC forced alignment):

| Clock | What it is | Minus the word's acoustic end ≈ |
|---|---|---|
| `t_frame` | the encoder frame on which the model emitted the symbol (`timestamps()`, 40 ms grid) | **emission delay** — the part the model *learned*, which no architectural formula contains. Should not change with the feed chunk |
| `t_fed` | how many seconds of audio had been fed when the symbol appeared | the above plus $m + \alpha C$: the front-end's right padding and the wait for the chunk to fill. **Linear in $C$** |
| `t_wall` | wall-clock time since feeding started | the above plus computation $D_{\text{comp}}$ |

So `t_fed − t_frame` is a realised value of $m + \alpha C$, one per symbol, and `t_wall − t_fed` is $D_{\text{comp}}$ directly. The fourth panel plots the three medians against the feed chunk: the architectural term is a sloped line, the learned term is a constant offset. This is the measured version of the slide's inequality $D_{\text{measured}} \geq (m + \alpha C)\,s\Delta + D_{\text{emit}} + D_{\text{comp}}$, term by term.

## One model, three ways of feeding it — not three models

The published LibriSpeech streaming zipformer exists in `sherpa-onnx` only as a **chunk-16** export (`decode_chunk_len = 32` input frames = 320 ms, right padding 13 frames = 130 ms). Re-exporting at other chunk sizes needs `icefall` and `k2`, which this course does not use. So the model's internal chunk stays at 320 ms and only the *feed* size changes: 320 / 640 / 1280 ms. For teaching this is cleaner — $m$ and the model are identical across the three runs, so "the learned delay does not move, the architectural one grows with $C$" is observed on one model rather than confounded by three trainings. What it cannot show is how emission delay differs between models trained with different chunks. Feed sizes below 320 ms would be meaningless: `sherpa-onnx` waits for a full model chunk anyway.

## Definitions

| | |
|---|---|
| the sentence | LibriSpeech dev-clean `1272-128104-0000`, 17 words, 5.855 s — `../w03_d3_blank_heatmap/runs/rehearsal/utterance.wav`; 0.8 s of silence appended (the last word has to wait for *something*) |
| a word's acoustic start / end | from W3 demo 3's CTC Viterbi (wav2vec2, 20 ms grid): first letter's first spike × 20 ms, (last letter's last spike + 1) × 20 ms. These are spike positions, not segment boundaries — each has ± 1 frame of slack |
| `t_frame` | `recognizer.timestamps(stream)` × 40 ms (zipformer2 outputs at 25 Hz; `check` verifies every timestamp sits on the 40 ms grid) |
| a word's clocks | the clocks of its **last** BPE piece; pieces are grouped into words at the leading space |
| word matching | `difflib.SequenceMatcher` against the reference; deleted or substituted words get no delta and are marked `?` / `≠` |
| $D_{\text{alg}}$ (computed) | a symbol on some frame waits (a) for its model chunk $c = 320$ ms to fill — on average $c/2$ — and (b) for the 13 padding frames, which arrive only with the **next feed boundary**; averaging over where the chunk end falls within a feed of size $F$: **mean $= c + F/2$**, worst $c + F$, best $c$. Predicted 480 / 640 / 960 ms for feeds of 320 / 640 / 1280 |
| median, worst word | medians over matched words; the worst word is the largest wall-clock delta |

A note from the rehearsal log, because it is the kind of mistake worth seeing: the first version computed $m + F/2$ — treating the 130 ms padding as "wait 130 ms more" — and under-predicted by a consistent ≈ 200 ms. Audio arrives in blocks; the 13 frames after a chunk end are not available until the whole next block has arrived. The cost of lookahead is not its length but how many arrival boundaries it crosses — the same point W1 made about the last partial chunk. `demo.py` keeps the naive figure (`naive_mean_ms`) alongside for comparison.

## Requirements

| | |
|---|---|
| Packages | shared `.venv` plus this folder's `./setup.sh`: `sherpa-onnx 1.13.8`, `onnx 1.23.1` (only to read the encoder's metadata; optional). Versions in `../versions.lock` |
| Model | `sherpa-onnx-streaming-zipformer-en-2023-06-26` (Apache-2.0; k2-fsa's export from the icefall LibriSpeech recipe), a few hundred MB, downloaded by `fetch` into `~/.cache/sherpa-onnx`. The same files are on Hugging Face as `csukuangfj/sherpa-onnx-streaming-zipformer-en-2023-06-26` if the GitHub release is slow |
| Audio and alignment | read from `../w03_d3_blank_heatmap/runs/rehearsal/` — keep the folder layout |
| Hardware | CPU; loading 0.8 s, offline decode of the whole sentence 0.21 s on the M5 Max. `sherpa-onnx` ships wheels for several platforms; only Apple silicon was tested |
| Tested on | Apple M5 Max, macOS 27.0.1, Python 3.12.14 (2026-10-06) |

## Run it

```bash
cd demos && ./setup.sh && cd w04_d2_stream_chunks
./setup.sh                    # sherpa-onnx + onnx into the shared .venv
./present.sh selftest         # no model: alignment, word grouping, matching, statistics, D_alg
./present.sh fetch            # the model → ~/.cache/sherpa-onnx (network)
./present.sh check            # versions, model files, encoder metadata (chunk / pad), one offline decode: words and 40 ms grid
./present.sh rehearse         # the three feeds, no audio playback → runs/rehearsal/
./present.sh                  # = show: window; 1 = 1280 ms, 2 = 640 ms, 3 = 320 ms (each fed at real-time pace while the audio plays), a = summary table + fourth panel, p play, m mute, h menu, q quit
./present.sh replay           # same window and keys from runs/rehearsal/, no sherpa-onnx
./present.sh fake             # no model: synthetic emission times, screen marked FAKE
```

Keys are pressed in the plot window. During a run only the red marks (one per symbol) update; the full row is redrawn when the run ends.

## What you should see

From `runs/rehearsal/rehearsal.json` (2026-10-06; all 17 words recognised correctly in all three runs):

| feed chunk | emission delay (`frame − end`, median) | `fed − emit` (median) | $D_{\text{alg}}$ predicted $c + F/2$ | `wall − fed` (median) | word appears after its end (wall, median) | worst word |
|---:|---:|---:|---:|---:|---:|---|
| 320 ms | **220 ms** | 500 | 480 | 21 ms | 742 ms | MIDDLE, 922 ms |
| 640 ms | **220 ms** | 640 | 640 | 33 ms | 1012 ms | MIDDLE, 1249 ms |
| 1280 ms | **220 ms** | 960 | 960 | 54 ms | 1413 ms | WELCOME, 1874 ms |

Reading it: the emission delay is identical in all three runs, frame for frame — it belongs to the model (MISTER 0 … WELCOME 340 ms); the `fed − emit` column tracks the predicted $c + F/2$; computation is tens of milliseconds (a larger feed means more model chunks decoded per call, so a longer call). The worst word is not the last one: it is the word whose last piece sits in a model chunk that ends exactly on a feed boundary, so the padding waits a full block ($\alpha = 1$).

A live `show` adds 20–30 ms to `wall − fed` compared with `rehearse` (playback thread and per-symbol redraw); `t_fed` and `t_frame` are identical between the two.

## Files

| File | |
|---|---|
| `demo.py` | `show` / `rehearse` / `replay` / `check` / `fetch` / `fake` / `selftest` / `export` / `recompute`; `--mute` |
| `demo_config.toml` | audio source and tail silence, model files and cache, the three feed settings, fake parameters, display |
| `present.sh`, `setup.sh` | entry point (offline) and this demo's package install |
| `runs/rehearsal/rehearsal.json` | per run: every symbol's three clocks, every word's deltas, medians, worst word, $D_{\text{alg}}$; encoder metadata; versions; the alignment's source (path replaced with a relative one before publishing) |
| `runs/rehearsal/rehearsal.txt`, `timeline.png` | the terminal transcript; the three timelines and the fourth panel |

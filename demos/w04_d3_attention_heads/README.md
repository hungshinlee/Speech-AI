<!-- 此檔由 bin/publish_demos.py 從 private repo 的 <course>/demos/ 複製而來，請勿在公開 repo 直接編輯。 -->
# W4 · Demo 3 — Whisper's cross-attention heads: attention ≠ alignment (slides p45–p47)

Whisper small (the multilingual model: 244 M parameters, 12 decoder layers × 12 heads = **144 cross-attention heads**) transcribes the same LibriSpeech sentence as W3 demo 3 — correctly — and then every one of the 144 heads' attention matrices ($U$ text tokens × 1500 encoder frames) is laid out to look at. Two things come out of it:

1. **Most heads do not look like an alignment.** Some spread evenly along the whole row, some stare at two or three frames (or at the 24 s of silent padding after the audio), and only a minority form a single monotonic band. The transcript is right regardless.
2. **OpenAI itself trusts only a handful.** Word timestamps in `openai/whisper` come from a fixed list of heads per model (`_ALIGNMENT_HEADS` in `whisper/__init__.py`, a code comment rather than a paper — ten heads for `small`), combined and run through dynamic time warping. The demo boxes those ten, computes timestamps the same way, and overlays W3's CTC forced alignment for comparison.

| Key (in the plot window) | Screen | Slide |
|---|---|---|
| `1` | plays the audio; the terminal prints the reference, Whisper's transcript, word errors (raw and after `tokenizer.normalize`), the number of text tokens $U$, and how many encoder frames the audio occupies | p45 |
| `2` | the current layer's 12 heads as heat-maps ($U$ rows × the frames within the audio, each row scaled to its maximum); titles give class, $\rho$, conc and pad | p45 |
| `3` | next layer (cycles) | |
| `0` | all 144 heads as thumbnails, border colour = class, title = the four class counts | p45 |
| `4` | boxes the ten OpenAI alignment heads (★) and says how many of them are bands | p45 |
| `5` | the ten heads → normalise → median filter → average → DTW: the cost matrix and path; word timestamps printed, with the difference to W3's first spike per word | p45, p47 |
| `a` | overlays W3 demo 3's forced alignment on `5` (grey bars: each word from its first to its last spike) — the seating chart vs. the weights | p46 |
| `h` `q` | menu, quit | |

## The band criterion (it is read out in class)

Only rows of **text tokens** count (not the `<|startoftranscript|><|en|><|transcribe|><|notimestamps|>` prefix or `<|endoftext|>`), and shape is judged only **within the audio** ($5.855\ \text{s} / 20\ \text{ms} = 292$ frames; frames 292–1500 are Whisper's fixed 30 s padding). For each row $u$: $t^*(u)$ is the frame with the largest weight inside the audio; $\text{conc}(u)$ is the weight within $t^*(u) \pm 10$ frames (± 200 ms) as a fraction of the row's in-audio weight; $\text{pad}(u)$ is the fraction of the row's total weight that falls on the padding. Per head: $\rho = $ Spearman$(u, t^*(u))$, conc and pad are medians over rows, distinct is the number of different $t^*(u)$.

| class | condition | reading |
|---|---|---|
| **band** | $\rho \geq 0.9$ and conc $\geq 0.5$ | monotonic and concentrated |
| **flat** | conc $< 0.25$ | spread along the row |
| **stare** | distinct $\leq \max(3,\ 0.2U)$ | fixed on a few frames |
| **other** | everything else | concentrated but not monotonic, or monotonic but diffuse |

**pad-heavy** ($\text{pad} \geq 0.5$) is a separate flag, not a class: a head can be a clean band inside the audio and still put most of its mass on the padding — one of OpenAI's alignment heads does exactly that. The thresholds are in `demo_config.toml` (`[classify]`); `./present.sh recompute` re-classifies from the saved weights without touching the model. This is *a* reproducible criterion, not *the* one; the counts below are "under this criterion".

## How the timestamps are computed (`5`)

Following `openai/whisper`'s `timing.py` (which `transformers` 5.17.0 reimplements as `_extract_token_timestamps`; both were read line by line): take the ten alignment heads' weights within the audio; standardise each frame **along the token axis**; median-filter along time (width 7, mirrored ends); average the ten heads; drop the prefix rows and the final `<|endoftext|>` row; run DTW on the negated matrix (steps: diagonal, down, right); a token's timestamp is the time at which the path moves to its row, × 20 ms; a word starts at its first token and ends where the next word starts.

Two differences from the OpenAI original, both without numerical consequence: the weights here are post-softmax (what `output_attentions` returns) rather than pre-softmax QK re-normalised; and they come from one teacher-forced forward pass over the generated tokens rather than from the step-by-step weights during `generate` — `check` compares the two and reports the largest difference.

**The first word always starts at 0.000 s**: DTW begins in the $(0, 0)$ corner, so the first token's step time is zero. That is a property of the algorithm, not evidence that the model "hears" the sentence start at 0 s; skip the first word when comparing with W3 (the printed median $|\Delta|$ already does).

## Requirements

| | |
|---|---|
| Packages | the shared `.venv` with W3 demo 3's `./setup.sh` already run: `torch 2.14.0`, `transformers 5.17.0`, `soundfile`. Nothing extra for this folder |
| Model | `openai/whisper-small` (Apache-2.0), about 1 GB, downloaded by `fetch` into the Hugging Face cache and pinned in `model.lock`; loaded with `attn_implementation="eager"` — under SDPA, `output_attentions=True` returns `None` for the cross-attentions (verified by `check`) |
| Audio and alignment | read from `../w03_d3_blank_heatmap/runs/rehearsal/` — keep the folder layout |
| Device | CPU by default (deterministic, the rehearsal's numbers, about a second); `check` also runs `mps` |
| Tested on | Apple M5 Max, macOS 27.0.1, Python 3.12.14, model commit `973afd24` (2026-10-06) |

## Run it

```bash
cd demos && ./setup.sh && (cd w03_d3_blank_heatmap && ./setup.sh) && cd w04_d3_attention_heads
./present.sh selftest         # no model: base85 decoding of the head list, DTW, median filter, classifier, word grouping, word errors, the whole flow on a fake
./present.sh fetch            # openai/whisper-small → Hugging Face cache, commit → model.lock (network)
./present.sh check            # eager vs. sdpa, base85 list vs. the model's generation_config.alignment_heads, own DTW vs. transformers' return_token_timestamps, MPS vs. CPU
./present.sh rehearse         # transcribe, teacher-forced forward, all 144 heads → runs/rehearsal/ (weights as float16, numbers, one PNG per layer)
./present.sh                  # = show; keys above
./present.sh replay           # same window and keys from runs/rehearsal/ — no torch
./present.sh recompute        # after changing [classify]: re-classify from the saved weights
./present.sh fake             # no model: 144 synthetic heads, screen marked FAKE
```

## What you should see

From `runs/rehearsal/rehearsal.json` (2026-10-06):

| | |
|---|---|
| transcript | "Mr. Quilter is the apostle of the middle classes, and we are glad to welcome his gospel." — word errors **1** raw (Mr. ≠ MISTER), **0** normalised; $U = 22$ text tokens; audio = 292 encoder frames |
| 144 heads by shape | **band 26**, flat 86, stare 3, other 29. No bands in layers 0–2; the bands cluster in layers 8 (6), 9 (7) and 10 (5) |
| padding | **114 of 144** heads put more than half their weight on the padding after the audio; the median over all heads is **85 %** |
| OpenAI's 10 alignment heads | **7 are bands** (L5H3, L5H9, L8H0, L8H4, L8H7, L8H8, L9H0) and 3 are "other" (L9H7 $\rho = 0.45$, L9H9 $\rho = 0.81$, L10H5 $\rho = 0.49$ — concentrated but not monotonic enough); 3 of the 10 are pad-heavy. **19 bands are not on the list** |
| timestamps | 17 words; start vs. W3's first CTC spike: median **+0.107 s** (Whisper's word starts are systematically ≈ 100 ms later), the largest difference 0.573 s is the first word (DTW starts at 0). Own DTW vs. `transformers`' `return_token_timestamps`: 2 of 22 tokens differ by one frame (0.02 s), the rest identical |
| `check` | eager CPU teacher-forced forward 534 ms, MPS 72 ms, largest weight difference $1.4 \times 10^{-5}$, same transcript; `generate` 0.88 s; under SDPA all `cross_attentions` are `None`; the base85 list decodes to the model's own `alignment_heads` |

The 100 ms offset is not a verdict on who is "more accurate": neither a DTW path through attention weights nor a CTC spike is a segment boundary.

## Files

| File | |
|---|---|
| `demo.py` | `show` / `replay` / `rehearse` / `check` / `fetch` / `fake` / `selftest` / `export` / `recompute` |
| `demo_config.toml` | model, device, language and task; audio and alignment sources; the classifier thresholds; timestamp parameters; display |
| `present.sh` | entry point (offline); uses `../.venv/bin/python` |
| `model.lock` | the model commit `fetch` pinned |
| `runs/rehearsal/cross_audio.npy`, `pad_mass.npy` | the 144 heads' weights within the audio (float16) and each row's padding mass — what `replay` and `recompute` read |
| `runs/rehearsal/rehearsal.json`, `bundle.json`, `rehearsal.txt` | transcript and word errors, every head's statistics and class, the alignment-head list, word timestamps and the comparison with W3; the terminal transcript |
| `runs/rehearsal/layer00.png` … `layer11.png`, `overview.png`, `overview-boxed.png`, `dtw.png`, `last-screen.png` | the figures `2`, `0`, `4`, `5` produce |
| `runs/rehearsal/utterance.wav` | a copy of the sentence |

Sources: `openai/whisper` (`whisper/__init__.py` for `_ALIGNMENT_HEADS`, `timing.py` for the DTW); `transformers` 5.17.0 `models/whisper/generation_whisper.py`; Radford et al., *Robust Speech Recognition via Large-Scale Weak Supervision*, ICML 2023, for the model. The head list is a code comment, not a peer-reviewed result.

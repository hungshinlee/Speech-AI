<!-- 此檔由 bin/publish_demos.py 從 private repo 的 <course>/demos/ 複製而來，請勿在公開 repo 直接編輯。 -->
# W6 · Demo 2 — what the letters look like: token matrix, codebook usage, residual energy, the bill (slides p32–p33, p42)

Demo 1 was about *hearing* what the residual-VQ stages carry. This demo *looks at the tokens themselves* and measures three things the lecture could only draw until now:
the residual-energy curve (slide p20), how many codes each stage actually uses, and how "language-like" the first stage is.
Then it does the arithmetic of slide p42: how many tokens a 30-minute conversation is for three codec settings — and for text.

**You are not required to run this.** It is published so that you can read it, re-run it and change it (another sentence, another subset, another codec from `transformers`, another tokenizer).
Menus and labels are in Chinese; the keys are listed below.

## Keys

| key | what happens |
|---|---|
| `1` | the W3 LibriSpeech sentence through Mimi, EnCodec and DAC: the $N_q \times T$ matrix of code indices. Terminal: the first 8 stages over a 24-frame window (`·` = same code as the previous frame; `[` `]` move the window). Window: the whole sentence, one colour per code, so runs of repeated tokens are visible |
| `2` | on the dev-clean subset (40 speakers × 10 sentences), per stage: distinct codes used / $V$, unigram entropy $H_1 / \log_2 V$, perplexity $2^{H_1}$, repeat rate $P(c_t = c_{t-1})$ |
| `3` | the residual-energy curve $\sum_t \lVert \mathbf{r}_{t,i} \rVert^2 / \sum_t \lVert \mathbf{z}_t \rVert^2$ per stage — measured, not schematic; Mimi's semantic and acoustic paths drawn separately |
| `4` | the bigram conditional entropy $H_2 = H(c_t \mid c_{t-1})$ against $H_1$, per stage, with a shuffled-order baseline |
| `5` | the bill: token rate $f_r N_q$, 30-minute context, steps per second (flattened vs. RQ-Transformer), $1/f_r$ for EnCodec 6 kbps, SpeechTokenizer and Mimi; and the same 30 minutes of speech as text tokens (Gemma 4 tokenizer) |
| `t` | the numbers for the slide tables (p33, p42) |
| `h` menu · `q` quit | |

## Definitions

| | |
|---|---|
| subset | the W5 Demo 1 subset of LibriSpeech dev-clean: 40 speakers × 10 sentences = 396 sentences (CC BY 4.0), resampled to 24 kHz; every codec encodes each sentence once with all its stages |
| codes used / $V$ | the number of distinct code indices that appear at that stage anywhere in the subset, over the codebook size |
| $H_1 / \log_2 V$ | the unigram entropy of the stage's codes over its maximum; 1 = uniform |
| repeat rate, mean run | $P(c_t = c_{t-1})$ over adjacent frames within a sentence; frames ÷ runs |
| $H_2$ | the plug-in estimate of $H(c_t \mid c_{t-1})$ over adjacent pairs within sentences. With a few tens of thousands of frames against $V^2$ possible pairs this estimate is biased low, so the same quantity is also computed after shuffling each sentence's order (unigram unchanged): the gap from $H_1$ to the shuffled value is sampling bias, the gap from the shuffled value to $H_2$ is real temporal structure |
| residual energy | $\mathbf{r}_{t,i} = \mathbf{z}_t - \sum_{j \leq i} \mathbf{q}_{t,j}$ where $\mathbf{q}_{t,j}$ is the codeword selected at stage $j$, taken from the model's own codebooks; the ratio is over the whole subset. The space is the quantizer's own: EnCodec's 128-d encoder output; for Mimi, each path's 256-d projection, computed separately for the semantic VQ and the acoustic RVQ (they run in parallel on the same $\mathbf{z}_t$ and their outputs are summed); for DAC, the 1024-d encoder output with each stage's 8-d lookup mapped back through its `out_proj`. `check` verifies that summing the codewords reproduces the quantizer's own decode exactly, and that the codes can be re-derived from the residuals by nearest neighbour |
| the bill | token rate $= f_r N_q$; context $= f_r N_q \times 1800$ s; a flattened LM takes $f_r N_q$ steps per second, an RQ-Transformer $f_r$; the latency floor is $1/f_r$. The codec rows are arithmetic from the papers' settings. The text row is measured: the tokens of one Mandarin sentence (W1 Demo 2) and of one English sentence (the W3 sentence) divided by their durations, scaled to 30 minutes — an extrapolation from one sentence each, not a corpus statistic |

## Requirements

- The shared `speech_ai/demos/.venv` (`../setup.sh`) plus `torch 2.14.0` and `transformers 5.17.0` from `../w03_d3_blank_heatmap/setup.sh`; no new packages.
- The three codecs and the W3 sentence from **Demo 1** (`../w06_d1_layer_listen`: `./present.sh fetch`); this demo imports Demo 1's `demo.py` for the model classes.
- The dev-clean subset's wav files from **W5 Demo 1** (`../w05_d1_layer_probe`: `./present.sh fetch` downloads dev-clean and writes `../_media/w05_d1/wav/`).
- For key `5`, a text tokenizer in the Hugging Face cache (the course uses the Gemma 4 tokenizer; any `AutoTokenizer` repo works — change `tokenizer_repo` in `demo_config.toml`). Without it the codec rows still print.

## Run it

```bash
cd speech_ai/demos/w06_d2_token_matrix
./present.sh selftest        # entropy / run / bigram functions, a numpy RVQ identity, the bill against w06-data.json, the fake pipeline (≈ 15 s, no models)
./present.sh check           # per codec: Σ codewords == quantizer decode, codes re-derived from residuals, path sizes; tokenizer → runs/check.json
./present.sh extract         # 396 sentences × 3 codecs → ../_media/w06_d2/ (minutes, once)
./present.sh rehearse        # all statistics and the bill → runs/rehearsal/ and w06-metrics.json
./present.sh                 # the window; press 1 ] ] 2 3 4 5 t
./present.sh replay          # the same keys from runs/rehearsal/, no torch
./present.sh fake            # no models: synthetic Markov tokens and a decaying curve (labelled FAKE)
```

## What you should see

- **The matrix (`1`).** The lecture's prediction is that stage 1 has the longest runs of repeated tokens — it follows the content, which changes slowly — while the later stages follow the residual, which looks like noise and changes every frame. Mimi's stage 1 is its distilled semantic VQ; EnCodec's and DAC's stage 1 are reconstruction-trained.
- **Usage and entropy (`2`).** The prediction is that stage 1 is the most skewed (few codes carry most of the mass) and that the later stages approach uniform — the other face of the near-Gaussian residual on p20. Uniform tokens are the hardest for a language model to predict.
- **The residual curve (`3`).** Monotonically decreasing with a shrinking step — the measured version of the schematic on p20. For Mimi the two paths are not one curve: the acoustic RVQ is not the residual of the semantic VQ.
- **$H_2 / H_1$ (`4`).** The lower the ratio, the more temporal structure. Read it together with the shuffled baseline.
- **The bill (`5`).** Speech tokens outnumber text tokens by one to two orders of magnitude; the floor $1/f_r$ rises exactly as the token rate falls.

## What was measured (rehearsal of 2026-10-18, `runs/rehearsal/`)

All three published weights have **32 stages** (Mimi: one semantic VQ plus 31 acoustic stages; the paper's $Q = 8$ is what Moshi uses, not what the weights hold). `check`: summing the codewords reproduces each quantizer's own decode to $< 4 \times 10^{-7}$ relative, and every code is re-derived from the residuals exactly.

| prediction | measured |
|---|---|
| stage 1 has the longest runs | yes: repeat rate EnCodec 0.19 at stage 1 vs. 0.04 at stage 8, DAC 0.07 vs. 0.00; Mimi 0.06 vs. 0.02 (at 80 ms per frame, tokens rarely repeat at all) |
| stage 1 is the most skewed, later stages near uniform | yes for EnCodec ($H_1/\log_2 V$ 0.78 then 0.85–0.86) and DAC (0.78 then 0.99–1.00, every one of its 1024 codes in use from stage 2 on); **not for Mimi's semantic VQ** (0.90 vs. 0.89 for its first acoustic stage) — cosine distillation does not ask for a skewed code distribution |
| (not predicted) | EnCodec's late stages have dead codes: 507 of 1024 used at stage 32 — the stages that quantizer dropout rarely trains |
| the residual decreases monotonically | yes for Mimi's acoustic path (0.17 → 0.009) and EnCodec (0.034 after stage 1 — the first codeword takes 97% of the energy — down to 0.001); **no for DAC**: 0.67 → 0.40 at stage 9 → 0.72 at stage 32. DAC's nearest-neighbour search runs in each stage's 8-d projection, not in the 1024-d space of the residual, so the "nearest neighbour, hence no increase" argument does not apply (our reading of the code) |
| the distilled stage is not the energy-heavy one | yes: Mimi's semantic VQ alone leaves 0.36 of the energy, its acoustic stage 1 leaves 0.17 |
| stage 1 has the most temporal structure | $H_2/H_1$: Mimi semantic 0.36 (acoustic 0.46), EnCodec stage 1 0.67 (later 0.77), DAC 0.80; DAC's stages 2–32 equal their shuffled baseline — no temporal structure at all |
| speech tokens outnumber text tokens | 30 minutes ≈ 5.1 k Gemma 4 tokens from the Mandarin sentence's rate (6.1 k from the English one) against Mimi's 180 k (×35) and EnCodec 6 kbps's 1.08 M (×211) |

The numbers are in `slides/assets/w06/w06-metrics.json`; whatever disagrees with the predictions is reported as measured.

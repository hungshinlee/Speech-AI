<!-- 此檔由 bin/publish_demos.py 從 private repo 的 <course>/demos/ 複製而來，請勿在公開 repo 直接編輯。 -->
# W5 · Demo 1 — three curves across thirteen layers: speaker probe, phone probe, CCA with the CNN output (slide p31)

Two pre-trained (not fine-tuned) encoders with the same architecture and the same training audio but different pre-training targets — `facebook/hubert-base-ls960` and `facebook/wav2vec2-base` — are run over a subset of LibriSpeech dev-clean (all 40 speakers, 10 utterances each), and every one of their thirteen representations (layer 0 = the CNN output, layers 1–12 = the transformer blocks) is scored three ways: how well a linear classifier on it recognises the **speaker**, how well a linear classifier on it recognises the **phone**, and how similar it is to **layer 0** (CCA). The point of the slide is that the three curves peak in different places and that the top layer is not the best one for anything but the pre-training target — "which layer" is a measurement, not a guess.

| Part | Keys | What it does | Slide |
|---|---|---|---|
| 1 | `1` | plays one utterance (`1272-128104-0000`, the sentence of W3 demo 3) and prints the shape of all thirteen layers for it ($T \times 512$ for the CNN, $T \times 768$ for each transformer block) | p31 |
| 2 | `2` | speaker probe, one layer at a time: frame-level 40-way logistic regression (the utterance-level accuracy is printed beside it); the test accuracy of each layer appears as a point on the first panel | p31 purple |
| 3 | `3` | phone probe: frame-level → 39-way logistic regression (ARPAbet, stress stripped); second panel | p31 orange |
| 4 | `4` | PWCCA between each layer and layer 0 on held-out frames; third panel. `l` switches the target to an 80-band log-mel spectrogram, then `4` again | p31 blue |
| — | `w` | switch model (HuBERT ↔ wav2vec 2.0) and repeat `2 3 4`; the second model's curves are drawn dashed on the same panels. `a` does all three for both models at once | p31 table |
| — | `t` | the summary table: for each model the layer where the speaker probe peaks, where the phone probe peaks, and the minimum and top-layer value of the CCA curve — the four cells of the table on p31 | p31 table |

## Definitions

| | |
|---|---|
| layers | layer 0 = the CNN front-end output before the feature projection (`model.feature_extractor`, 512-d); layers 1–12 = the outputs of the twelve transformer blocks (`hidden_states[1:13]`, 768-d). The post-projection input to the first block (`hidden_states[0]`) is not counted as a layer. Frame rate 20 ms, receptive field 25 ms, frame $t$ centred at $(20t + 12.5)$ ms |
| data | LibriSpeech dev-clean: all 40 speakers, the first 10 utterances of each (2–15 s), 400 utterances; the last 3 utterances of every speaker are the test set (120), the rest train (280). Audio and transcripts CC BY 4.0 |
| phone labels | the word/phoneme forced alignments of `gilkeyio/librispeech-alignments` (Hugging Face, CC BY 4.0; produced with the Montreal Forced Aligner, so they are automatic, not hand-labelled). A frame's label is the phone whose interval contains the frame centre; stress digits are dropped (`AH0`, `AH1` → `AH`), giving 39 classes; frames outside any interval (silence) or in `spn` (unknown words) are not used. 60 labelled frames are sampled per utterance (seed 20261011), about 24,000 in all |
| speaker probe | frame-level, on the same sampled frames as the phone probe: multinomial logistic regression over the 40 speakers (numpy, full-batch Adam, 300 steps, L2 $10^{-3}$, inputs standardised on the training set); reported as test-frame accuracy. An utterance-level version (mean over all frames of an utterance) is computed and printed too, but with 40 speakers and 276 training utterances it is ≈ 1.0 at every layer and shows no shape |
| phone probe | the same classifier on the sampled frames; test-frame accuracy |
| CCA | PWCCA (Morcos et al. 2018, the measure used by Pasad et al. 2021): PCA on each side keeping 99 % of the variance, ridge CCA fitted on the training frames, the canonical correlations measured on the test frames and averaged with projection weights. Layer 0 against itself is 1 by definition. The unweighted mean is also printed; it is diluted by the hundreds of uncorrelated directions and does not show the shape |

Only the *shape* of each curve is comparable with the papers (where the peak is, where the minimum is, how much the top layer comes back); the absolute PWCCA values depend on the PCA, the ridge and the data and are not comparable across papers.

## Requirements

| | |
|---|---|
| Packages | the shared `.venv` plus `torch 2.14.0`, `transformers 5.17.0`, `soundfile` (installed by `w03_d3_blank_heatmap/setup.sh`) and `pyarrow` (this folder's `setup.sh`; only `fetch` uses it) |
| Models | `facebook/hubert-base-ls960` and `facebook/wav2vec2-base` from the Hugging Face hub, about 380 MB each, pinned to the commits in `model.lock` by `fetch` |
| Data | `dev-clean.tar.gz` (337 MB, from openslr.org; shared with W3 demo 3) and the `dev_clean` parquet of the alignments dataset — `fetch` reads only its `id` and `phonemes` columns when it can (a few MB) and otherwise downloads the whole file (343 MB) |
| Device | the encoders run on `mps` when available (Apple silicon), else CPU; probes and CCA run in numpy on the CPU. Feature extraction for 400 utterances is cached in `demos/_media/w05_d1/` (about 350 MB per model) |
| Time | extraction ≈ 2 min for the 396 utterances on the instructor's machine (`mps`); each probe curve ≈ 27 s (≈ 2 s per layer), each CCA curve ≈ 3 s |
| Cross-checks | `selftest` runs without the models: the logistic regression recovers planted clusters and not shuffled labels, PWCCA is ≈ 1 for a rotated copy and ≈ 0 for independent data, frame labelling matches the interval rule, and a synthetic feature set with a planted structure gives the expected shapes; it also checks that `runs/rehearsal/` and `slides/assets/w05/w05-metrics.json` agree when both exist |
| Tested on | Apple M5 Max, macOS 27.0.1, Python 3.12.14, torch 2.14.0, transformers 5.17.0, pyarrow 26.0.0 (2026-10-11): `fetch`, `check`, `extract`, `rehearse` and the full key sequence. `selftest` also passes on a 2-core Linux container without torch (the model-dependent parts cannot run there) |

## Run it

```bash
cd demos && ./setup.sh && ./w03_d3_blank_heatmap/setup.sh && cd w05_d1_layer_probe && ./setup.sh
./present.sh selftest         # no models needed, a few minutes
./present.sh fetch            # models, dev-clean subset, alignments (network)
./present.sh check            # cpu / mps for both encoders → runs/check.json
./present.sh extract          # features for 400 utterances → demos/_media/w05_d1/
./present.sh rehearse         # both models, all curves → runs/rehearsal/ and w05-metrics.json
./present.sh                  # = show: single-key menu in the figure window (h menu, q quit)
./present.sh replay           # the rehearsal curves without torch or the cache
```

## What you should see

From `runs/rehearsal/rehearsal.json` (2026-10-11, Apple M5 Max; 396 utterances, 40 speakers, 23,726 labelled frames; all accuracies on held-out utterances):

| | speaker probe: low peak · minimum · top | phone probe: peak · top | PWCCA with layer 0: minimum → top |
|---|---|---|---|
| HuBERT-base | layer 2 **0.94** · layer 8 0.39 · layer 12 0.78 | layer 11 **0.81** · 0.80 | layer 9 0.30 → 0.36 |
| wav2vec 2.0 base | layer 2 0.89 · layer 8 0.54 · layer 12 **0.94** | layer 9 **0.80** · 0.56 | layer 7 0.21 → layer 10 0.44 → layer 12 0.29 |

- `2`: speaker information is highest in the low layers for both models and falls by half in the middle — then it comes back at the top, and for wav2vec 2.0 the top layer is the best speaker layer of all. The utterance-level probe printed beside it is 0.96–1.00 at every layer and shows none of this.
- `3`: phone accuracy climbs all the way to layer 11 for HuBERT and stays there; for wav2vec 2.0 it is flat at 0.76–0.80 over layers 6–9 and drops to 0.56 at the top.
- `4`: the CCA curve with layer 0 bottoms out in the middle and turns back toward the top — clearly for wav2vec 2.0 (0.21 → 0.44), slightly for HuBERT (0.30 → 0.36), as Pasad et al. 2023 report for the two pre-training targets. The speaker comeback at the top goes with it: the turned-back layers look like the CNN output again, and the CNN output carries the speaker (this reading is the course's, not the papers').
- `check`: on the instructor's machine the 13-layer forward pass of a 5.9 s utterance takes 39 ms (HuBERT) / 21 ms (wav2vec 2.0) on `mps` against 146 / 137 ms on the CPU, with a largest difference of $8 \times 10^{-5}$; feature extraction for the 396 utterances takes about two minutes, each probe curve about 27 s, each CCA curve about 3 s.

## Files

| File | |
|---|---|
| `demo.py` | `show` / `rehearse` / `replay` / `extract` / `check` / `fetch` / `fake` / `selftest` / `export` |
| `demo_config.toml` | the two models, device, subset size and split, phone-label rules, frames per utterance and seed, log-mel settings, probe hyper-parameters |
| `present.sh`, `setup.sh` | entry point (offline) and the extra package |
| `model.lock` | the two model commits pinned by `fetch` |
| `data/subset.json`, `data/alignments.json` | the utterance list with split, and the phone intervals of the subset (CC BY 4.0) |
| `runs/rehearsal/rehearsal.json` | all four curves for both models (per-layer accuracy and CCA, timings), the summary table, versions and date |
| `runs/rehearsal/rehearsal.txt`, `curves.png` | the terminal output without colour codes, and the figure |

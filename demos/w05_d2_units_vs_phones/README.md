<!-- 此檔由 bin/publish_demos.py 從 private repo 的 <course>/demos/ 複製而來，請勿在公開 repo 直接編輯。 -->
# W5 · Demo 2 — units are not phones: three teachers' purities, and one sentence aligned (slides p44–p47)

HuBERT's pre-training target is the id of a k-means cluster, and the model is trained to predict it at masked frames. This demo asks two questions about those ids. **How good is the teacher?** k-means with $K = 100$ is run on three feature sets of the same LibriSpeech dev-clean frames — 39-d MFCC (the teacher of HuBERT's first iteration), and layers 6 and 9 of `facebook/hubert-base-ls960` (the layers the paper clusters for later iterations) — and each clustering is scored against phone labels with the three measures of the HuBERT paper: phone purity, cluster purity and PNMI. **Are the units phones?** The sentence of W3 (`1272-128104-0000`, "Mister Quilter is the apostle…") is drawn with its words and phones from a forced alignment on top and its k-means units underneath, so you can see how many units one phone gets, whether unit boundaries sit on phone boundaries, which unit is silence, and what the same phone becomes in another context; then the word *mister* as said by three other speakers is lined up against the same word in the sentence, unit by unit.

| Part | Keys | What it does | Slide |
|---|---|---|---|
| 1 | `1` | the purity table: for MFCC, layer 6 and layer 9 — phone purity, cluster purity, PNMI on held-out frames (and on all frames), with the paper's MFCC numbers as the last row | p45 table |
| 2 | `2` | the sentence, zoomed to its first two words: a row of words and a row of phones (both from the MFA alignment), and a row of layer-6 units (one cell per 20 ms frame, runs of the same id merged, dotted lines = phone boundaries); the terminal prints runs and distinct units per phone, how often a unit boundary falls on a phone boundary (against chance), the silence unit, and the unit runs of every occurrence of the sentence's most frequent phone. `[` / `]` move the window one word, `f` shows the whole sentence, `u` adds the layer-9 and MFCC unit rows | p46 |
| 3 | `3` | the word *mister* in the sentence and in three other speakers' utterances: each row is the run-length-collapsed unit string of that word; cells not in the longest common subsequence with the sentence's row are outlined in red; the terminal prints the LCS fraction and the Jaccard overlap of the unit sets for each feature set | p46 third row |
| 4 | `4` | optional: your own recordings (e.g. one Mandarin syllable in four tones) as unit strings side by side, with pairwise LCS — the instructor's recordings are not published, record your own | p46 notes |

## Definitions

| | |
|---|---|
| frames | the 23,726 phone-labelled frames that Demo 1 sampled (60 per utterance, 40 speakers × 10 utterances, seed 20261011), with Demo 1's split: the last 3 utterances of every speaker are the test set. k-means is fitted on the training frames; purities are reported on the test frames (the paper fits and scores on the same data; that number is printed too) |
| features | **MFCC**: 13 cepstra (including $c_0$) from the 80-band Slaney log-mel of Demo 1 (25 ms window, 20 ms hop, 512-point FFT) by an orthonormal DCT-II, plus Δ and ΔΔ over ±2 frames — 39-d, not standardised. **Layer 6 / layer 9**: the outputs of transformer blocks 6 and 9 of `hubert-base-ls960` (768-d). Note that this checkpoint is the *second*-iteration BASE model: the paper's second iteration was taught by the *first* model's layer 6, and its LARGE model by the second model's layer 9 |
| k-means | $K = 100$, k-means++ initialisation, Lloyd iterations until the assignment stops changing (at most 100), best of 3 starts, seed 20261011; plain numpy |
| phone purity | $\sum_k \max_y p(y,k)$ — the frame accuracy you get by calling every cluster its most frequent phone |
| cluster purity | $\sum_y \max_k p(y,k)$ — how much of each phone falls in its single most frequent cluster; with 100 clusters and 39 phones this is necessarily low |
| PNMI | $I(y;k)/H(y)$ — the fraction of the phone entropy explained by the cluster id. All three follow §IV-E of the HuBERT paper |
| phone and word labels | the Montreal Forced Aligner alignments of `gilkeyio/librispeech-alignments` (Hugging Face, CC BY 4.0; automatic, not hand-labelled); stress digits dropped, 39 ARPAbet classes; silence has no interval; `spn` (unknown words) is excluded |
| the sentence | units are the nearest-centroid assignment of the whole utterance's layer-6 / layer-9 / MFCC frames; frame $t$ is drawn centred at $(20t + 12.5)$ ms |

## Requirements

| | |
|---|---|
| Packages | the shared `.venv` plus `torch`, `transformers`, `soundfile` (installed by `w03_d3_blank_heatmap/setup.sh`) and `pyarrow` (`w05_d1_layer_probe/setup.sh`; only `fetch` uses it). Nothing else |
| Demo 1 first | this demo reuses Demo 1's subset, alignments and cached layer features: run `../w05_d1_layer_probe/present.sh fetch` and `extract` before `cluster` here |
| Data | `fetch` reads only the `words` column of the alignment parquet (a few MB) into `data/words.json`; `cluster` computes MFCC for the 396 utterances (seconds) and the three k-means (about 25 s on a 4-core CPU); `extract` runs the encoder over the four utterances it needs (and your own recordings) |
| Cross-checks | `selftest` runs without the models: the three purities give 1/1/1 for a perfect clustering and the right values for a degenerate one, k-means recovers planted clusters and is reproducible from its seed, the DCT rows are orthonormal and the Δ of a ramp is constant, run-length encoding and the LCS do what they should, a synthetic run of the whole key sequence works, and `replay` reproduces it; with the caches present it also checks that the MFCC frames are the same frames as Demo 1's |
| Tested on | Apple M5 Max, macOS 27.0.1, Python 3.12.14, torch 2.14.0, transformers 5.17.0, pyarrow 26.0.0 (2026-10-11): `fetch`, `cluster`, `extract`, `rehearse` and the full key sequence; `selftest` and `cluster` also on a 4-core Linux VM with numpy 2.2.6, where the purities came out bit-identical |

## Run it

```bash
cd demos && ./setup.sh && ./w03_d3_blank_heatmap/setup.sh && ./w05_d1_layer_probe/setup.sh
cd w05_d1_layer_probe && ./present.sh fetch && ./present.sh extract && cd ..
cd w05_d2_units_vs_phones
./present.sh selftest         # no models needed, about 20 s
./present.sh fetch            # word intervals → data/words.json (network)
./present.sh cluster          # MFCC + three k-means → demos/_media/w05_d2/
./present.sh extract          # whole-utterance features for the sentence and the other speakers (needs the model)
./present.sh rehearse         # everything → runs/rehearsal/ and w05-metrics.json
./present.sh                  # = show: single-key menu in the figure window (h menu, q quit)
./present.sh replay           # the rehearsal numbers without torch or the caches
```

## What you should see

From `runs/rehearsal/rehearsal.json` (2026-10-11, Apple M5 Max; the Demo 1 frames, held-out test frames, $K = 100$):

| teacher features | phone purity | cluster purity | PNMI |
|---|--:|--:|--:|
| MFCC (39-d) | 0.257 | 0.101 | 0.304 |
| HuBERT-base layer 6 | **0.549** | 0.275 | **0.602** |
| HuBERT-base layer 9 | 0.542 | 0.278 | 0.583 |
| paper: MFCC, $K = 100$, 960 h (arXiv 2106.07447v1 §V-C) | 0.335 | 0.099 | 0.255 |

- `1`: the model's own middle layers cluster into phones twice as well as the MFCC that taught the first iteration — the student overtakes the teacher. Layer 9 is not better than layer 6 on this subset (the paper's Fig. 2 has the second-iteration model improving with depth; 7,200 test frames and $K = 100$ do not resolve a difference that small). Cluster purity is low for all three because 39 phones are spread over 100 clusters. The paper's absolute numbers are not comparable (different data size, different k-means), only the direction.
- `2`: on layer 6 the sentence's 63 phone segments contain 165 unit runs — 2.6 runs and 2.6 distinct units per phone, and only 11 % of the phones get a single unit from start to end. Of the 157 unit boundaries, 40 (25 %) fall exactly on a phone boundary, against 22 % for a random position: the units are not segmenting the signal the way the phones do. Silence (44 frames outside any interval) is shared by units 90, 43 and 83, each of which occurs almost nowhere else. The phone AH occurs eight times and gets eight different run strings (`20 57 65 | 38 30 14 | 3 | 57 | 87 85 14 | 63 93 9 23 | 50 46 | 70 14`). Layer 9: 2.5 units per phone, 19 % single-unit phones, boundaries 25 %. MFCC: 2.8 units per phone, boundaries 29 % — the frame-level features follow the phone edges a little more closely than the contextual layers.
- `3`: *mister* by the sentence's speaker (15 frames) and speakers 2412, 3752 and 5338 (13 / 16 / 17 frames): of the sentence's 11 collapsed layer-6 runs, 6 / 6 / 5 survive in the other speakers' strings (LCS 55 / 55 / 45 %; Jaccard of unit sets 0.40 / 0.35 / 0.31); on MFCC units only 3 / 6 / 2 of 12 (25 / 50 / 17 %). The units share more across speakers than MFCC does, but about half of the string still changes with the speaker — reduced, not gone. The slide (p46) shows a schematic with hand-written ids, not these numbers.

## Files

| File | |
|---|---|
| `demo.py` | `show` / `rehearse` / `replay` / `cluster` / `extract` / `fetch` / `fake` / `selftest` / `export` |
| `demo_config.toml` | the three feature sets, k-means settings, the sentence and zoom, the word to compare, optional own recordings, figure size |
| `present.sh` | entry point (offline except `fetch`) |
| `data/words.json` | word intervals of the subset from the alignment dataset (CC BY 4.0), written by `fetch` |
| `runs/rehearsal/rehearsal.json` | purities, k-means info, the sentence's phones, words, unit sequences and statistics, the word comparison, versions and date |
| `runs/rehearsal/rehearsal.txt`, `sentence_zoom.png`, `sentence_full.png`, `sentence_word.png` | the terminal output without colour codes, and the figures |

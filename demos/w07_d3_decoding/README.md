<!-- 此檔由 bin/publish_demos.py 從 private repo 的 <course>/demos/ 複製而來，請勿在公開 repo 直接編輯。 -->
# W7 · Demo 3 — One knob, λ: how heavy should the external prior be? (slides p32–p41)

The same CTC model decoded with an external word-level language model whose weight $\lambda$ is swept from 0 upward, everything else held fixed; then the same CTC model on a Mandarin sentence, where right-sound-wrong-character errors can be counted.

| Half | What runs | What to look at |
|---|---|---|
| (1) the $\lambda$ sweep, keys `1` `2` `3` | **wav2vec2-base-960h** (W3 demo 3's model — character-level CTC, almost no internal language model) on **30 LibriSpeech dev-clean sentences** (one per speaker from W5 demo 1's subset, selected deterministically, with a disjoint 10-sentence tuning set; `fetch` writes the list and the reference transcripts to `data/sentences.json`, which is not in this repository). Its log-softmax goes through the **CTC prefix beam search** of W7 demo 2 (`../w07_d2_biasing/ctc_prefix_beam.py`: Hannun et al. 2014, Algorithm 1; numpy, 32 beams) with an **n-gram bonus** (`arpa_lm.py`): score $= \ln P_{\text{ctc}} + \lambda \sum_w \ln P_{\text{LM}}(w \mid w_{-2}, w_{-1}) + \eta \cdot n_{\text{words}}$, the LM applied at each word boundary and at the end of the sentence. The LM is the **official LibriSpeech 3-gram, pruned 1e-7** (openslr 11), read from its ARPA file by our own code — Katz back-off, out-of-vocabulary words mapped to `<unk>` plus a penalty of $-5 - 3 \cdot (\text{characters})$ nats (`oov_ln_penalty`, `oov_ln_per_char`: this ARPA's `<UNK>` is cheaper than most real words, and at a large $\lambda$ the beam otherwise deletes the spaces and pays for one `<unk>` instead of many words — a fixed penalty alone did not stop a 27-word sentence from fusing into one token, so the penalty grows with length, like a uniform spelling model; and a word still being spelled is scored as an OOV of its current length while the beam is pruned, so that a hypothesis that skips a space cannot look cheap mid-search — the partial-token score of `pyctcdecode`), no `kenlm`. $\eta$ (the per-word insertion bonus) is tuned once on the tuning set at $\lambda = 0.5$ and then frozen; the beam is frozen; only $\lambda$ moves: 0, 0.1, 0.2, 0.3, 0.5, 0.7, 1, 1.5, 2, 3, 4. | the **shape of WER against $\lambda$**; where greedy decoding sits on it; at a too-large $\lambda$, *which* words change and whether the errors turn into deletions or insertions (key `3` prints three decodings of one sentence with the score split into its three terms) |
| key `0` | slide p32's two-frame example run through the same prefix beam search: nine paths enumerated, merged by label, greedy vs. prefix beam — checked against `slides/assets/w07/w07-data.json` | greedy picks the single best path (**b**, 0.16); summing paths per label makes **a** (0.35) the answer |
| (2) Mandarin homophones, key `m` | one Mandarin sentence (the instructor's recording; the W1 demo 2 ticket-booking sentence's text is public, other recordings are not distributed) through **MMS-1B-all**'s CTC with the `cmn-script_simplified` adapter (greedy, no LM; its simplified-character output is mapped to traditional before display), **Whisper small** and **Qwen3-ASR-1.7B**. Each output is aligned to the reference character by character (both sides mapped to simplified first — the script is not counted as an error) and every substituted character is classified with `pypinyin`: same reading, same reading but different tone, or a different reading | whether the CTC model's substitutions are **homophones** — right sound, wrong character — and whether the two decoders with a text prior make the same kind of error |

The mapping to traditional characters is itself a normalisation step that can be wrong: one simplified character can correspond to several traditional ones (周 → 周 or 週), and the table built into the code is a few hundred common pairs, not OpenCC.

## Keys (terminal, single key)

| Key | Screen |
|---|---|
| `0` | p32's example: paths, labels, greedy vs. prefix beam |
| `1` | the test set: duration, frames, reference words; greedy WER; prefix-beam $\lambda = 0$ WER |
| `2` | the $\lambda$ sweep — one line per $\lambda$ (WER, S/D/I, seconds) and a live plot |
| `3` `n` | one sentence at $\lambda = 0$, the best $\lambda$ and a too-large $\lambda$, errors coloured, score terms printed; `n` moves to the next sentence |
| `m` | the Mandarin half |
| `e` | the $\eta$ tuning table |
| `l` | the LM itself: n-gram counts, `<unk>`, a few example sentences' $\ln P$, the test references' perplexity |
| `t` | the slide's table |
| `h` `q` | menu, quit |

## Running it

```bash
cd speech_ai/demos/w07_d3_decoding
./present.sh selftest      # no model needed: the ARPA reader and its back-off against hand-computed values, the LM bonus, p32's example, WER, the alignment and homophone judges, the fake flow
./present.sh fetch         # models (wav2vec2, whisper-small, Qwen3-ASR as in demo 2; facebook/mms-1b-all ≈ 3.9 GB plus one adapter), the 3-gram ARPA (34 MB), the reference transcripts
./present.sh check         # versions, models in cache, the ARPA parsed once and cached, back-off sanity, beam λ=0 against greedy, a timing estimate for the sweep, MMS on mps or cpu → runs/check.json
./present.sh rehearse      # tunes η, then everything, non-interactive → runs/rehearsal/ (rehearsal.json, rehearsal.txt, eta.json, lambda_sweep.png)
./present.sh               # the live run
./present.sh replay        # the rehearsal record, same keys, no model
./present.sh fake          # synthetic log-probabilities and a six-sentence LM, to walk the flow
```

Requirements: the shared `.venv` (`../setup.sh`) plus `torch 2.14.0`, `transformers 5.17.0` (installed by `../w03_d3_blank_heatmap/setup.sh`) and `pypinyin 0.55.0` (in `../requirements.lock.txt`). `arpa_lm.py` needs numpy only, so the LM half can be read and re-run without torch once the log-probabilities exist. wav2vec2 and MMS use `mps` when it is available; Whisper runs on the CPU (W7 demo 1 found `generate` on `mps` fails in transformers 5.17.0). Tested on the instructor's MacBook Pro (M5 Max, 64 GB); `check` records what ran where.

`runs/rehearsal/` holds the full English half and, for the Mandarin half, the counts (CER, S/D/I, homophone counts) plus the text of the ticket-booking sentence only.

What the rehearsal showed (2026-11-02, Apple M5 Max): the LibriSpeech 3-gram (200k words, 2.45 M bigrams, 1.13 M trigrams) parses in 3.4 s; the 30 sentences are 199 s of speech and 563 reference words. Greedy decoding and prefix beam search without an LM both give **2.66% WER** (the same 30 outputs — this model's posteriors are too peaked for merging to matter without an LM). $\eta$ tuned to 0. With the LM, WER falls to **1.60% at $\lambda = 0.2$–$0.3$** (every fix is a substitution: MEDLE → MEDDLE, LEONITAS → LEONIDAS, JIK → JIG, ANY ONE → ANYONE), then rises in two stages: at $\lambda = 1.5$–$2$ substitutions the LM prefers (MEDDLE → MEAL, EXHORT → EXERT; 2.84%), and from $\lambda = 3$ deletions, up to whole sentences decoded as nothing at $\lambda = 4$ (51.7%, 219 deletions). The whole sweep takes 9 s. On the Mandarin ticket sentence MMS-1B-all (on `mps`) substitutes two characters — 訂→定, a homophone, and 三→山, which pinyin counts as a different reading — while Whisper small and Qwen3-ASR make no error; on a second sentence with three names, seven of MMS's nine substitutions are homophones, and the two decoders with a text prior err once and twice, both on the same name and both homophones.

*Three rehearsals before this one all failed in the same way — at a large $\lambda$ the beam deleted the spaces and fused whole sentences into one out-of-vocabulary token. The fix is the OOV penalty that grows with length plus scoring partial words during pruning, described above; the instructor's notes in `../README.md` keep the full record.*

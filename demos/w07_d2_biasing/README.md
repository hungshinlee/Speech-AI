<!-- 此檔由 bin/publish_demos.py 從 private repo 的 <course>/demos/ 複製而來，請勿在公開 repo 直接編輯。 -->
# W7 · Demo 2 — The same audio, with and without the list: three ways to add a name at run time (slides p45–p51)

Two recordings that contain proper names go through three ASR systems; a list of those names is handed to each system at the layer where it can take one.

| Clip | What it is |
|---|---|
| English | one LibriSpeech dev-clean sentence with proper names in it (`demo_config.toml` → `[english]`; the default is `2803-154320-0011`, "think of Lady Glenarvan, think of Mary Grant") |
| Mandarin | one sentence recorded by the instructor with three or four proper names — a person, a road, a product. **Not distributed**: the recording, its transcript and the names live outside the repository, and the published rehearsal record only reports *how many* names were recognised, the script (traditional / simplified) and whether the prompt leaked |

| Panel | Layer (slide) | Model | How the list goes in | What is printed |
|---|---|---|---|---|
| (a) key `7` | the **score** (p46) | wav2vec2-base-960h, W3 demo 3's CTC model | `ctc_prefix_beam.py`: a CTC prefix beam search (Hannun et al. 2014, Algorithm 1; numpy, character level, 32 beams) with the names in a trie — every character on a path of the list adds $\delta$ to the score, a path that is abandoned gives the bonus back (the failure arc of Zhao et al. 2019), $\delta \in \{1, 6, 24\}$; greedy and $\delta = 0$ as controls | each hypothesis with the names marked ✓/✗, the word errors, $\log P$ + bonus, and the **word-level diff against $\delta = 0$** — which ordinary words turned into names |
| (c) keys `1`–`6`, `e` | the **prompt** (p48) | Whisper `small` (`transformers`, the same weights as W4 demo 3 and W7 demo 1) | `prompt_ids`, the previous-text slot: empty; the list of names; a sentence written in traditional characters (no names — only to see whether the output script flips); a full line of dialogue (to see whether it is read into the transcript) | the transcript, names ✓/✗, the teacher-forced average log-probability, word errors; for Mandarin the script judgement; and **prompt leakage**: any run of ≥ 3 English words / ≥ 4 Chinese characters of the prompt that appears in the transcript |
| (c′) keys `8`, `9` | the **context slot of an LLM-based ASR** (p48, p54) | Qwen3-ASR (`Qwen/Qwen3-ASR-1.7B-hf`, the Transformers-native release) | `apply_transcription_request(audio, language, prompt=…)` — the model card's "free-form context … to bias the transcription", the same slot as a system message in its chat template; the same list (`Vocabulary: …`), the same traditional sentence, the same line of dialogue | the same fields, side by side with Whisper's |

Deep biasing (CLAS, p44–p45) has no ready-made model and is only drawn on the slides.

Definitions: a name counts as recognised when it appears as a word sequence in the lower-cased, de-punctuated transcript (English) or as a substring after both sides are mapped to simplified characters (Mandarin — the script is not an error here). The script judgement counts characters that exist only in simplified or only in traditional usage, from a table of about six hundred common pairs built into `demo.py` (not OpenCC). The $\delta$ bonus is added to the log score once per character; wav2vec2's character log-probabilities are near 0 on a confident character and $-5$ to $-15$ on a wrong one, so $\delta = 1$ roughly outweighs one hesitant character and $\delta = 6$ can rewrite a whole name.

Differences from Zhao et al. 2019: the bonus is applied at the character level (that is wav2vec2's vocabulary; the paper argues for word pieces because character matches fire too often — exactly the side effect the large $\delta$ shows); the space inside a multi-word phrase is part of the trie path; when one phrase is a prefix of another ("Mary" and "Mary Grant"), abandoning the longer one also refunds the shorter — keep such pairs off the list.

## Keys (terminal, single key)

| Key | Screen | Slide |
|---|---|---|
| `1` `2` `e` | English → Whisper: no prompt / the list / a line of dialogue | p51 |
| `3` `4` `5` `6` | Mandarin → Whisper: no prompt / the list / a traditional-script sentence / a line of dialogue | p51 |
| `7` | English → wav2vec2 log-probabilities → greedy, beam $\delta = 0$, $\delta$ = 1 / 6 / 24 | p46, p51 |
| `8` | Qwen3-ASR, English and Mandarin: no context / the list | p51 |
| `9` | Qwen3-ASR, Mandarin: the traditional sentence / the line of dialogue | p51 |
| `0` | 8 s of near-silence (synthetic low-level white noise) with the dialogue prompts → Whisper and Qwen3-ASR: what each writes when the prompt is all it has | p48, p51 |
| `p` | print every prompt | |
| `t` | the slide's table: names recognised / script / leakage per cell | p51 |
| `h` `q` | menu, quit | |

## Running it

```bash
cd speech_ai/demos/w07_d2_biasing
./present.sh selftest      # no model needed: trie and failure arc, prefix beam search against enumeration, the judges, the fake flow
./present.sh fetch         # openai/whisper-small, facebook/wav2vec2-base-960h (commits pinned in model.lock), Qwen/Qwen3-ASR-1.7B-hf (≈ 3.4 GB); extracts the English candidates' wav from dev-clean
./present.sh pick          # runs all candidate sentences through Whisper and CTC greedy and marks the ones with a missed name
./present.sh check         # versions, models in cache, Qwen3-ASR on mps or cpu, the prompt argument, beam δ=0 against greedy → runs/check.json
./present.sh rehearse      # everything, non-interactive → runs/rehearsal/ (English only)
./present.sh               # the live run
./present.sh replay        # the rehearsal record, same keys, no model
./present.sh fake          # canned output to walk the flow
```

Requirements: the shared `.venv` (`../setup.sh`) plus `torch 2.14.0` and `transformers 5.17.0` (installed by `../w03_d3_blank_heatmap/setup.sh`; Qwen3-ASR's native Transformers support needs ≥ 5.13.0). `record` (the Mandarin clip) needs a microphone and `sounddevice`. Whisper runs on the CPU (W7 demo 1 found that `generate` on `mps` fails in transformers 5.17.0); wav2vec2 and Qwen3-ASR use `mps` when it is available. Rehearsed on the instructor's MacBook Pro (M5 Max, 64 GB); `check` records what ran where.

`runs/rehearsal/` holds the English half of the rehearsal (the slide's first row, the CTC column, and the near-silence panel). The Mandarin half is the instructor's own voice and is not distributed; the slide shows only counts for it.

What the English half showed (2026-11-01, Apple M5 Max): Whisper `small` and wav2vec2 both spell *Glenarvan* as *Glenarvin*; the list fixes it in both (Whisper's mean log-probability rises from $-0.34$ to $-0.12$; CTC needs $\delta = 1$, up to $\delta = 6$ no other word changes, and at $\delta = 24$ *LADY* becomes GLENARVAN — the bonus outweighs the acoustics). A dialogue-style prompt did **not** leak into the transcript over speech, in either language — only the punctuation style changed — but over 8 s of near-silence (key `0`) Whisper transcribed the English prompt verbatim and most of the Chinese one, and Qwen3-ASR returned every context line verbatim. Qwen3-ASR 1.7B spelled *Glenarvan* correctly with no list at all. On the Mandarin sentence both models missed one of three names and the list fixed it; Whisper's output was already in traditional characters (a traditional prompt only changed 台 to 臺), while Qwen3-ASR wrote simplified characters until a traditional-script context flipped the whole sentence.

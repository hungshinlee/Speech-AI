<!-- 此檔由 bin/publish_demos.py 從 private repo 的 <course>/demos/ 複製而來，請勿在公開 repo 直接編輯。 -->
# W11 · Demo 2 — The same question, typed and spoken (slides p33–p35, p46)

A text language model has been taught to hear. Did its ability to *answer* come across? One number cannot say where a loss comes from, so this demo measures four. Nothing is trained; every model is used for inference only.

**The four conditions**

| | the question arrives as | answered by |
|---|---|---|
| **a** | text | $\theta_0$ — the original text LLM (see the note on "approx." below) |
| **b** | text | $\theta$ — the model that hears |
| **c** | speech (a TTS voice reads the same string) | $\theta$ |
| **d** | speech → Whisper small → text | $\theta_0$ — the cascade |

In the notation of the lecture these are $\mathrm{Acc}(\theta_0, \ell^{\text{usr}})$, $\mathrm{Acc}(\theta, \ell^{\text{usr}})$, $\mathrm{Acc}(\theta, \mathbf{X}^{\text{usr}})$ and $\mathrm{Acc}(\theta_0, \hat{\ell}^{\text{usr}})$, and

$$\underbrace{a - c}_{\text{understanding gap}} = \underbrace{a - b}_{\text{forgetting}} + \underbrace{b - c}_{\text{misalignment}}$$

is an identity. Row **d** is there to rule out "it simply misheard": $a - d$ is what recognition errors alone cost.

**What goes in**

| | |
|---|---|
| questions | `questions.py`: 40 four-option questions (14 general knowledge, 13 arithmetic, 13 two-step reasoning) and 10 questions with a format constraint appended ("Reply with the letter only."), each in English and in Mandarin. In Mandarin two of them (`k10`, `f01`) are left out in every condition — the TTS misread their options and re-sampling did not fix it (`[run] skip` in `demo_config.toml`) — so Mandarin has 39 + 9. They were written for this course and are deliberately easy: the point is not whether a model can answer, but whether the answer changes with the way the question gets in. The correct option is A, B, C, D ten times each |
| one string, four ways | every condition uses the *same string* (`prompt`). The typed conditions send it as text; the spoken conditions hand it to the TTS. No system prompt, reasoning mode off, greedy decoding |
| $\theta$ | Ming-3.0-audio-tiny (`mlx-community/Ming-3.0-audio-tiny-8bit`, through `mlx_audio.sts`): a Whisper-style encoder and a convolutional adapter feed continuous vectors (6.25 per second) into a mixture-of-experts language model. The same weights accept text (`task="text"`) or speech (`task="speech_to_text"`) |
| $\theta_0$ | Ling-3.0-tiny (`inclusionAI/Ling-3.0-tiny`), converted locally with `mlx-lm` to the same 8-bit quantisation |
| TTS | English: Kokoro, two voices. Mandarin: Qwen3-TTS (`mlx-community/Qwen3-TTS-12Hz-0.6B-Base-bf16`), cloning two Kokoro voices from a short reference sentence — Kokoro's Mandarin voices do not say the letters A–D. Qwen3-TTS samples, so a misread question can be re-read with another seed (`[tts.qwen3.seed_by_item]`). The table uses the first voice; the second (key `2`) shows how many answers flip when *only the voice* changes |
| recogniser | Whisper small, on CPU |

**Why "approx."** The language-model part of Ming-3.0-audio-tiny's configuration is identical, field by field, to Ling-3.0-tiny's (24 layers, width 1536, 128 experts, the same vocabulary). Neither model card states that the audio model was initialised from that checkpoint, so $\theta_0$ is the closest public stand-in, not a certified starting point. If it is not the true starting point, $a - b$ mixes "forgetting" with "a different model".

**Scoring** (`score.py`; rules only, no model acts as judge)

1. an explicit statement — "the answer is C", 「答案是 C」 — the last one wins;
2. otherwise, a single option label in the reply;
3. otherwise, a single option's content in the reply (numbers must match as whole numbers; simplified and traditional characters are folded together first);
4. if the reply mentions several options — a step-by-step answer — only its last sentence is examined.

No option found counts as wrong and is reported separately (`no pick`). A reply that ran out of tokens is counted under `cut`. For the format questions, *whether the format was kept* and *whether the content is right* are recorded separately.

**Statistics.** Forty questions is a small sample. Every row is printed as a count with a 95 % percentile bootstrap interval (questions resampled, 10 000 draws), and every difference as a paired bootstrap interval together with the number of questions only one side got right. **Two or three questions apart is not a gap.**

**What the keys show**

| Key | What happens | Slide |
|---|---|---|
| `q` `n` | one question, live, asked four ways; the replies and what Whisper wrote down | p35 |
| `a` `b` `c` `d` | one row at a time from the rehearsal run; `t` the whole table | p35 |
| `f` | the ten format questions | p35 |
| `m` | row c′: questions read by the lecturer (ten, nine in Mandarin after `k10` was left out), against the same ones in the TTS voice | p35 |
| `i` | the three differences and the identity, with intervals | p36 |
| `x` | an extra row **e**: the same speech, written down by Whisper, then *read* by $\theta$ — so $b - c = (b - e) + (e - c)$ | p37 |
| `w` | "heard correctly, answered wrongly": $\theta$ transcribes each question itself (`task="asr"`); among the questions it transcribed correctly, how many did it get wrong when it heard them? | p37 |
| `z` | rows c and d again with babble added at 5 dB (six other question recordings summed) | p46 |
| `s` | probe: five seconds of silence, then of white noise — alone, and with the typed question "What did I just say?" | p46 |
| `e` | probe: one sentence said happily and said flatly (recordings are not distributed) — the reply to each, then "How do I sound?" | p46 |
| `v` | the Talker reads the last reply aloud | |
| `k` `2` `l` | by category; the second TTS voice; switch language | |

**Reading the results honestly**

- One audio model, one stand-in for its text ancestor, fifty home-made questions, synthetic voices. The outcome is a measurement of this setup, not of audio language models in general.
- The transcript error rate used by `w` counts "17" against "seventeen" as an error, so "heard correctly" is an underestimate.
- In this MLX port the Talker receives only the text of the reply — not the Thinker's hidden states and not the user's audio — and speaks after the whole reply has been written. Streaming is not implemented in the port.
- The probes are read, not scored.
- **What the rehearsal found** (2026-11-28, M5 Max; `runs/rehearsal/rehearsal.json`): correct answers, Mandarin (of 39) / English (of 40) — a 33 / 37, b 34 / 36, c 36 / 39, d 33 / 34. **No gap on these questions**: spoken was answered as well as typed, in English better; the prediction a ≥ b > c did not hold. The gap in the papers is real, but it is a finding about particular models and benchmarks, not a law — and one run cannot tell whether it is missing here because the questions are easy, the model is recent, or $\theta_0$ is only a stand-in. Two numbers were larger than any of these differences: changing only the TTS voice cost 6 questions in Mandarin (row c with the second voice: 30), and reading Whisper's transcript instead of hearing cost the same model 9 (Mandarin) and 6 (English). Asked directly how the speaker sounds, the model tells the two tones apart; its advice afterwards is the same for both — heard, not used.
- In English, $\theta$ answers the question instead of transcribing it, so key `w` is meaningful only in Mandarin.

**Running it** (Apple silicon; about 37 GB of downloads and converted weights)

```bash
cd demos && ./setup.sh                   # the shared environment (Kokoro, Whisper)
cd w11_d2_typed_vs_spoken
./setup.sh                               # a second environment, .venv-sts, with a newer mlx-audio
./present.sh fetch && ./present.sh check
./present.sh rehearse --n 12 --langs en  # a 12-question preview
./present.sh rehearse                    # everything → runs/rehearsal/rehearsal.json
./present.sh                             # the keys above
./present.sh fake                        # walk through the keys with no models (numbers are meaningless)
```

Row c′ and the tone probe need your own recordings: `./present.sh record own`, `./present.sh record tone`.

**Licences.** Ming-3.0-audio-tiny and Ling-3.0-tiny: MIT (the speaker encoder bundled with the Talker: Apache-2.0). Kokoro-82M and Qwen3-TTS: Apache-2.0, as declared on their model cards. Whisper: MIT. The character table in `zh_fold.py` is derived from OpenCC (Apache-2.0).

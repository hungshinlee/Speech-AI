<!-- 此檔由 bin/publish_demos.py 從 private repo 的 <course>/demos/ 複製而來，請勿在公開 repo 直接編輯。 -->
# W11 · Demo 3 — How far ahead is the text? (slides p63–p65, p73)

Every speech-to-speech model in this week's survey still writes text; what differs is *where* the text sits relative to the speech. In an interleaved model the text and the speech tokens share one output sequence, with the text a little ahead. "A little ahead" has been a symbol all week, $U_{\text{lead}}$. This demo puts a stopwatch on it. Nothing is trained; every model is used for inference only.

**What runs**

| step | what | with |
|---|---|---|
| 1 | a spoken question in English — a recorded file, so there is no "wait for the user to finish" cell in any time below | the files in `audio/` were synthesised by the model's own TTS mode; no person's voice is in this folder |
| 2 | the model answers, writing text tokens and speech frames into one sequence. Each token is printed as it is generated, coloured by kind, with its time | LFM2.5-Audio-1.5B (Liquid AI), through `mlx-audio` 0.5.4 |
| 3 | the speech frames (8 codes each, 12.5 frames per second) are turned into a 24 kHz waveform | the model's own detokenizer |
| 4 | the generated speech is transcribed and compared with the text the model itself wrote | Whisper small (the W7 demo's weights) |

**About the model.** One language model (1.2 B parameters) reads continuous speech through a FastConformer encoder and writes text tokens and, for each speech frame, 8 codes produced by a small depth transformer. In a conversation it generates in a fixed pattern: $n_{\text{txt}}$ text tokens, then $n_{\text{aud}}$ speech frames, and again. The released configuration is $6 : 12$, and — unusually — the two numbers are arguments of the generation call, so they can be changed at inference time. That is why this demo uses this model. Two things follow from the model card and must be said: it supports **English only**, and its licence is the **LFM Open License v1.0** (Apache-style terms with a commercial-use limitation; read the `LICENSE` file in the weights repository), not Apache or MIT.

**First half — the ratio as a knob (keys `1`, `2`, `t`, `3`, `4`)**

The same question is answered under five settings: the default $6 : 12$; $2 : 12$, $12 : 12$ and $24 : 12$; and "all the text first, then speech". **Only $6 : 12$ was seen in training.** Every other row is out of distribution, so when a row goes wrong there are two causes on top of each other — the arrangement itself, and the fact that the model never practised it — and this demo cannot separate them.

"All the text first" cannot be had by raising $n_{\text{txt}}$ (see what key `2` does at $24 : 12$). It is run the way this model *can* do it, in two passes on one clock: a text-only pass (no system prompt; the model then writes like a text LLM), followed by the model's TTS mode reading that text aloud.

**Second half — thinking before speaking (keys `5`, `6`)**

One two-step arithmetic question, answered two ways: as the model normally does (it starts speaking after six text tokens and works through the problem aloud), and "reason first": a text-only pass in which it reasons step by step — printed in a third colour, never turned into sound — followed by TTS of the last sentence only. This is again two passes of one model; it is not a built-in thinking mode.

**What is measured** (definitions are in `trace.py`, which uses only numpy and can be read on its own)

| | definition |
|---|---|
| first text token, first speech frame | when each has been computed, counted from the moment the question file is handed to the model (reading the question is included) |
| first sound | when the first 6 frames (0.48 s; `chunk_frames` in the config) have been generated **and** decoded to a waveform — the earliest moment playback could start. The detokenizer has no streaming interface in this library; the chunk size is this demo's choice |
| playback catches up | playback is simulated from the timestamps: chunk $c$ is due at first sound $+\, c \cdot 0.48$ s and is ready when its last frame has been generated and decoded. "Never" means generation stayed ahead |
| real-time factor | generation time ÷ seconds of audio generated |
| WER | Whisper small's transcript of the generated speech against the model's own text (both through Whisper's English normaliser), pooled over the three questions: errors summed ÷ reference words summed |
| blank | the share of text tokens that decode to nothing but a space. It is reported next to WER because WER compares speech with text and says nothing about whether the text itself is still an answer |
| the two conditions (key `3`) | (i) $n_{\text{txt}} / n_{\text{aud}} \ge R_{\text{txt}} / f_r$, with $R_{\text{txt}}$ = text tokens per second of speech — measured on the $6 : 12$ answers when one of them finishes within the step limit, otherwise the 3.42 per second of W6 Demo 2 (that is what the rehearsal used: none finished); (ii) one round's generation time $n_{\text{txt}} D_{\text{text}} + n_{\text{aud}} D_{\text{frame}}$ against the $n_{\text{aud}} / f_r$ seconds it plays for. One "frame step" is one pass of the backbone plus 8 small steps of the depth model, so it is measured separately from a text step |
| the cascade (key `4`) | the same question through the W1 Demo 2 pipeline (whisper.cpp → a text LLM → Kokoro), using that demo's own code with its settings switched to English: ASR tail + LLM first token + TTS first packet, medians of 5 runs |
| second half | text tokens written before the first sound ($U_{\text{th}}$), the first sound, $U_{\text{th}} \cdot D_{\text{text}}$, and whether the last number in the text — and in Whisper's transcript of the speech — is the right answer |

Three questions make a small sample. Read the WER column as "a few words more or fewer", not to the decimal.

**Sampling.** Text is decoded greedily (temperature 0), so differences in the text between settings come from the arrangement and not from sampling; speech codes are sampled as in the model card's example (temperature 1.0, top-k 4) with a fixed seed.

**Keys**

| Key | What happens |
|---|---|
| `p` | play the question |
| `1` | the same question twice: all the text first (two passes), then the default $6 : 12$ |
| `2` | sweep the ratio: $2 : 12$, $6 : 12$, $12 : 12$, $24 : 12$ |
| `t` | the table from the rehearsal (three questions pooled) |
| `3` | the two conditions with the measured numbers |
| `4` | the cascade's three cells beside this model's (`$` runs the cascade now) |
| `5` | the arithmetic question: first press, answer as you go; second press, reason in text first |
| `6` | the second table |
| `w` | the last run: what the model wrote, what Whisper heard |
| `l` | play the last run again |
| `?` `q` | menu, quit |

**Run it** (Apple silicon; the shared environment of this repository's `demos/`)

```bash
./present.sh selftest      # no model needed: every definition in trace.py, and the whole flow with a stand-in
./present.sh fetch         # downloads LFM2.5-Audio (about 2.4 GB), synthesises any question missing from audio/
./present.sh check         # schedule, speed against real time, chunked decoding, the cascade's back ends
./present.sh rehearse      # 3 questions × 5 settings, the arithmetic question, the cascade → runs/rehearsal/
./present.sh               # the classroom session
./present.sh replay        # no model: re-prints the rehearsal's tokens at their recorded times
./present.sh fake          # no model at all: walks through the keys with a stand-in (marked FAKE)
```

`./present.sh record q1` records the question in your own voice (kept outside the repository) and uses it instead of the synthetic one.

**Files**

| file | |
|---|---|
| `trace.py` | the arithmetic: schedules, the four marks on the timeline, the lead estimate, the two conditions, WER |
| `demo.py` | the model wrapper, the token printer, rehearsal, the classroom session |
| `demo_config.toml` | questions, settings, sampling, chunk size |
| `audio/` | the four questions as synthesised speech |
| `runs/rehearsal/` | the rehearsal's audio and numbers |

<!-- 此檔由 bin/publish_demos.py 從 private repo 的 <course>/demos/ 複製而來，請勿在公開 repo 直接編輯。 -->
# W11 · Demo 1 — A language model with no text in its sequence (slides p21–p23)

A cascade passes only text between its modules, so everything that is not text — the voice, the tone, the timing — is lost at the first arrow. One way out is to drop the text and build the language model directly on speech tokens. This demo asks what such a model actually produces. Nothing is trained; every model is used for inference only.

**What runs**

| step | what | with |
|---|---|---|
| 1 | the first 3.2 s of one real sentence (LibriSpeech dev-clean `1272-128104-0000`, the sentence used since W3) is turned into tokens | Mimi, first 4 levels: 12.5 frames/s × 4 = 50 tokens/s, so 40 frames = **160 tokens** |
| 2 | a single Transformer continues the flattened token sequence, one token at a time, for up to 12 s (600 tokens) | `llm-jp/Llama-Mimi-1.3B` (Sugiura et al., 2025, arXiv:2509.14882) |
| 3 | the whole sequence — prompt and continuation — is decoded back to audio | Mimi decoder |
| 4 | **only after listening**: each continuation is transcribed | Whisper small (the W7 demo's weights) |
| 5 | a text LLM continues *the prompt's words* with the same seed and temperature, for as many text tokens as the continuation's seconds are worth | the text LLM of W1 Demo 2, through `mlx-lm` |

The same prompt is continued four times: three seeds at the model card's settings (temperature 0.8, top-k 30) and one at temperature 0.5.

The cut is at 3.2 s rather than 3.0 s for two reasons: 3.2 s is a whole number of Mimi frames (3.0 s is 37.5 frames, which Mimi pads to 38), and it falls in the pause after "…the middle classes", so the prompt is a complete clause.

**About the model.** There is no text token anywhere in the sequence it models. Its *weights*, however, did not start from nothing: the model card lists `meta-llama/Llama-3.2-1B` as the base model (hence the Llama 3.2 Community License), its vocabulary is Llama 3.2's with 4 × 2048 audio tokens and two tags appended, and the paper initialises the backbone from a pretrained checkpoint, following TWIST. The card says it was trained on about 240 k hours of English audio. `check` prints how much probability the model puts on audio tokens for its next token.

**What is measured**

| | definition |
|---|---|
| tokens per second, real-time factor | generated tokens ÷ generation time (Mimi decoding excluded); generation time ÷ seconds of audio produced. Real time needs 50 tokens/s |
| SIM | cosine between WavLM-SV embeddings (the W9 encoder) of the original prompt and of the continuation. Four reference values give the number a scale: the real remainder of the same sentence, another sentence by the same speaker, a sentence by a different speaker, and the prompt itself after Mimi at 4 levels |
| how sure Whisper was | the average log-probability Whisper assigns to its own transcript and the transcript's compression ratio (both as in the W7 demo), next to the same two numbers for real speech |
| token counts | the prompt's 160 speech tokens against the number of text tokens for the same nine words |

No "meaning score" is computed. Judging continuations with another language model is what the papers do, and it is the subject of W14; here you read the transcripts yourself.

**One caution.** The middle column of the transcript view is *Whisper's reading* of the generated audio. Whisper has a strong language prior and turns unclear speech into fluent words (W7), which is why the demo plays the audio first. Key `w` shows how confident Whisper was, so you can tell how much of the fluency is Whisper's.

**Keys**

| Key | What happens |
|---|---|
| `p` | play the 3.2 s prompt |
| `1` `2` `3` `4` | play a continuation: the prompt followed by the model, from one Mimi decode (`c` toggles "continuation only") |
| `t` | the words: the prompt, each continuation as Whisper heard it, and the text LLM's continuation |
| `n` | token counts, tokens per second, real-time factor, SIM with its reference values |
| `w` | Whisper's confidence, for each continuation and for real speech |
| `r` | what the speaker actually said next (text and audio) |
| `m` | the prompt through Mimi at 4 levels — the best this token stream can sound |
| `g` | draw one more continuation live (loads the models; a new seed each time) |
| `?` `q` | menu; quit |

**Running it**

```bash
cd demos && ./setup.sh                     # the shared environment (once)
cd w11_d1_speech_continuation
./present.sh fetch                         # Llama-Mimi (about 5 GB), Mimi, Whisper small, WavLM-SV, the text LLM
./present.sh check                         # token layout, cpu vs mps, speed, the two judges, the text LLM
./present.sh rehearse                      # four draws → runs/rehearsal/
./present.sh                               # the keys above, reading runs/rehearsal/
./present.sh replay                        # the same without loading any model
```

`runs/rehearsal/` holds the audio and numbers behind the slide. Sampling is seeded, but a different machine, device or precision gives different continuations — that is expected, and it is part of the point: every draw is different, and each of them sounds like speech.

The text LLM is a 26B model (about 15 GB at 4-bit); `./present.sh rehearse --no-text` skips it, and `--text-repo <repo>` substitutes another `mlx-lm` model. In `demo_config.toml`, `[text_lm] mode = "chat"` (the setting used for the rehearsal) asks the model for a continuation with one instruction; `"raw"` feeds the prompt's words as the start of the sequence with no chat template — the same thing the speech model is asked to do, but this text model is instruction-tuned and, without its template, produced only repeated fragments (rehearsal of 2026-11-26). So the comparison is not like for like: the speech model continues, the text model is *asked* to continue.

**Licences.** Llama-Mimi-1.3B: Llama 3.2 Community License. Mimi: CC BY 4.0. Whisper small: Apache-2.0. The LibriSpeech audio (`audio/other_speaker.wav`, and the prompt under `w03_d3_blank_heatmap/`): CC BY 4.0.

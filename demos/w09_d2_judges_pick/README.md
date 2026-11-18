<!-- 此檔由 bin/publish_demos.py 從 private repo 的 <course>/demos/ 複製而來，請勿在公開 repo 直接編輯。 -->
# W9 · Demo 2 — One sentence, twelve draws, three judges (slides p55–p57)

Post-training with a reward pushes a model towards whatever its judge prefers. This demo shows the judges' preferences without training anything: one sentence is drawn twelve times from the same model with the same prompt, every draw is scored by three judges, and each judge names its favourite. It is **not** reinforcement learning. It is best-of-12 — the judge kept in inference (slide p33).

**The generator.** Qwen3-TTS 0.6B Base (`mlx-community/Qwen3-TTS-12Hz-0.6B-Base-bf16`, through `mlx-audio`), an autoregressive codec language model that samples every token, so each seed reads the sentence differently. Seeds 1–12, temperature 0.9. As a control, F5-TTS draws the English sentence twelve times too (seeds 1–12): F5 fixes the output length from the character ratio of text to prompt, so its twelve draws all have the same length.

**The sentences** are chosen to be hard for an autoregressive model — the same few words repeated in a different order:

| | Sentence | Prompt |
|---|---|---|
| English | the "Peter Piper" tongue-twister, four clauses (50 syllables) | LibriSpeech dev-clean `1272-128104-0010` (CC BY 4.0), 4.9 s |
| Mandarin | two common tongue-twisters joined: 吃葡萄不吐葡萄皮… and 媽媽騎馬，馬慢，媽媽罵馬 (27 syllables; the second differs only in tone) | a Mandarin sentence recorded by the lecturer (not in this repository) |

There are no digit strings: Whisper's normaliser treats "five five five" and "555" inconsistently, which would add errors that belong to the judge rather than to the speech.

**The three judges**

| Judge | What it is | Reward |
|---|---|---|
| WER | Whisper small (W7's) transcribes the draw; compared with the sentence. English: words, both sides through Whisper's English normaliser. Mandarin: tone-numbered pinyin syllables, so a wrong tone is an error and a homophone is not. S, D, I come from a Levenshtein alignment. | −WER |
| SIM | cosine between the embeddings of the draw and of the prompt from one speaker-verification encoder (`microsoft/wavlm-base-plus-sv`) | the cosine |
| predicted MOS | UTMOS22 strong learner (Saeki et al., Interspeech 2022): wav2vec 2.0 followed by a BLSTM, trained on the BVCC listening-test data (English; systems up to 2022). The code is the `tarepan/SpeechMOS` v1.2.0 re-implementation with weights converted from the official checkpoint. Input is scaled to a fixed RMS first, because the score moves slightly with level. | the predicted score, 1–5 |

**Two descriptors that are not judges.** Nobody uses them as a reward; they describe the draw. Both are this demo's own definitions, not standard metrics.

| Descriptor | Definition |
|---|---|
| syllables per second | syllables of the sentence to be read, divided by the time between the first and the last 10 ms frame within 30 dB of the 95th-percentile level (pauses inside the utterance count; silence before and after does not). The numerator is the reference sentence, so a draw that drops words looks fast. |
| F0 spread | standard deviation of 12·log2 F0 over voiced frames, in semitones (autocorrelation F0 from W2's `dsp.py`, 40 ms frames every 10 ms; frames more than a factor 1.8 from the median are dropped as octave errors) |

**What the screen computes**

- **The long tail**: draws with more word errors than the group's best draw (so an error every draw shares is not counted), or whose speech runs longer than 1.5 × the group median.
- **The group-relative advantage** of slide p41, per judge: (reward − group mean) / group standard deviation. When all twelve rewards are equal the standard deviation is 0 and the advantage is undefined — that group carries no signal for that judge.
- **Each judge's first choice.** Ties are reported as ties.
- **Rank against descriptor**: Spearman correlation between a judge's ranking and each descriptor. With twelve draws, |ρ| has to reach about 0.59 to pass a two-sided 5 % test. Selecting among twelve draws is also not the same as training: selection can only pick among readings the model already produces.

## Keys (in the window)

| Key | What happens | Slide |
|---|---|---|
| `g` | the table: twelve draws × (WER, SIM, predicted MOS \| syllables / s, F0 spread) | p56 |
| `l` | marks the long tail and plays its worst draw | p56 |
| `a` | adds one advantage column per judge | p56 |
| `w` `s` `m` | plays the first choice of the WER / SIM / MOS judge | p57 |
| `h` | plays the draw the lecturer picked by ear (blind: `./present.sh listen`) | p57 |
| `p` | scatter plots: a judge's rank against the two descriptors; press again for the next judge | p57 |
| `z` | the other sentence | |
| `f` | the F5-TTS control group ↔ Qwen3-TTS | |
| `n` | draws a fresh group now, with the next twelve seeds (`show` only) | |
| `1`–`9` `0` `-` `=` | plays draw 1–12; `o` plays the prompt | |
| `t` `r` `?` `q` | summary lines, play again, menu, quit | |

Playback files are scaled to one RMS; all measurements use the raw outputs.

## Running it

```bash
cd speech_ai/demos/w08_d1_mean_vs_sample && ./setup.sh && ./present.sh fetch          # the TTS environment, F5-TTS, the prompt
cd ../w08_d2_nfe_cfg_first_packet && ./present.sh fetch                               # Qwen3-TTS (MLX)
cd ../w09_d1_cloning_leak && ./setup.sh && ./present.sh fetch                         # the SIM encoder and Whisper small
cd ../w09_d2_judges_pick
./present.sh fetch         # the MOS predictor: clones tarepan/SpeechMOS at v1.2.0 and downloads the weights (411 MB); commit and SHA-256 are checked
./present.sh check         # the MOS predictor on CPU and MPS; the three judges on a few known recordings
./present.sh rehearse      # both sentences, twelve draws each, plus the F5 control → runs/rehearsal/
./present.sh listen en     # pick your own favourite without seeing numbers
./present.sh               # the keys above, on the rehearsal's draws; n draws a new group
./present.sh replay        # the same without torch or MLX
./present.sh fake          # the whole flow without any model (stand-ins, marked FAKE)
```

Requirements: `speech_ai/demos/.venv-tts` (W8 Demo 1's environment: torch 2.14.0, torchaudio 2.11.0, transformers 5.17.0, f5-tts 1.1.22) and the shared `speech_ai/demos/.venv` with `mlx-audio 0.5.4` for Qwen3-TTS, which runs as a subprocess (W8 Demo 2's `mlx_worker.py`). No new package is installed; the MOS predictor's code is cloned to `speech_ai/demos/_media/w09_d2/` and imported from there, and its weights are loaded with `weights_only=True`. Qwen3-TTS needs Apple silicon; `git` is needed for `fetch`.

To use your own sentences, edit `[sentences.*]` in `demo_config.toml` (for English, also set `syllables`). Without a Mandarin recording of your own, the Mandarin sentence falls back to the English prompt.

Nothing is trained in this demo.

Licences: `tarepan/SpeechMOS` is MIT, and its UTMOS weights derive from the MIT-licensed `sarulab-speech/UTMOS22` checkpoint; F5-TTS's weights are CC BY-NC 4.0 (non-commercial); Whisper is Apache-2.0; the Qwen3-TTS conversion's card says Apache-2.0; the model card of `microsoft/wavlm-base-plus-sv` carries no licence field.

Cloning a voice needs the speaker's consent. The prompts here are a LibriSpeech recording (CC BY 4.0, from public-domain LibriVox audiobooks) and the lecturer's own voice.

Numbers quoted on the slides come from `runs/rehearsal/rehearsal.json` and are summarised in `slides/assets/w09/w09-metrics.json` (`demo2`). Draws are reproducible only on the same machine, device and seed.

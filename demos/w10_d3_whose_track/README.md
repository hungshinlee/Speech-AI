<!-- 此檔由 bin/publish_demos.py 從 private repo 的 <course>/demos/ 複製而來，請勿在公開 repo 直接編輯。 -->
# W10 · Demo 3 — Two people at once: which track is the user? (slides p76–p78, p87)

A separation model returns two tracks. Nothing in it says which track is the user, or that the user stays in the same track from one chunk to the next. This demo builds a mixture whose answer is known, separates it, lets two speaker encoders pick the user's track against an enrolment, and then hands the same audio to a diarization pipeline and scores its one output three ways. Nothing is trained; every model is used for inference. Only voices from a public corpus are used.

**What goes in**

| | |
|---|---|
| the user | LibriSpeech dev-clean speaker 1272 (the speaker used since W3). The enrolment is another sentence of theirs, `1272-128104-0010` |
| the other talker | speaker 174 (a man); key `g` switches to speaker 2277 (a woman). The user's sentences and the timing are identical in both |
| fully overlapped mixture | two sentences, trimmed, equal RMS, summed — what a two-speaker benchmark looks like |
| the "conversation" | ten sentences placed on a timeline (`mix.py`, `build`): each starts a little after the previous one ends, or a little *before* it ends. About 40 s, with roughly a tenth of the speech time overlapped |

Because the mixture is made here, each talker's clean track still exists. That gives two things a deployed system never has: which separated track the user is *really* in (the one with the higher SI-SDR against the user's clean track), and an exact reference for who spoke when (an energy gate on each clean track, 10 ms frames, pauses under 0.2 s bridged).

**What runs**

| | |
|---|---|
| separation | SepFormer, SpeechBrain's `sepformer-libri2mix` weights (two speakers, 8 kHz, trained with PIT). The mixture is resampled to 8 kHz, separated, and resampled back, so the tracks are empty above 4 kHz. The conversation is cut into 4 s chunks with no overlap, each separated on its own (`c`: 2 s or 8 s; `x`: the whole 40 s in one pass) |
| output level | a separation model's output scale is arbitrary — and in these weights one track comes out with inverted polarity. The two tracks are rescaled by the pair of gains that makes their sum closest to the mixture (least squares; a gain may be negative). SI-SDR and the speaker encoders ignore scale and sign |
| encoder A | ECAPA-TDNN, SpeechBrain's `spkrec-ecapa-voxceleb` |
| encoder B | WavLM Base+ with an x-vector head, `microsoft/wavlm-base-plus-sv` (W9 Demo 1's SIM judge) |
| SIM | the cosine between the embedding of a separated track and the embedding of the enrolment (W9 p18) |
| recogniser | Whisper small (W7), on the tracks stitched back together; WER is counted against the user's words only, after Whisper's `EnglishTextNormalizer` |
| diarization | `pyannote/speaker-diarization-community-1` through `pyannote.audio`, on the conversation as one file. Key `k` switches to a deliberately simple clustering baseline written in `demo.py` (`naive_diarize`): 1.5 s windows, one ECAPA embedding each, agglomerative clustering, one speaker per frame |
| scoring | `der.py`: 10 ms frames, optimal one-to-one mapping between system labels and reference speakers, then the three scorings of slide p86. `check` compares it with `pyannote.metrics` on the same input |

**What the keys show**

| Key | What happens | Slide |
|---|---|---|
| `1` | plays the fully overlapped mixture, then the two separated tracks; SI-SDR of each track against each talker | p78 |
| `p` | plays the "conversation" | p78 |
| `2` `3` | one row per chunk: the SIM of track 1 and of track 2 to the enrolment, and the track picked — encoder A, then encoder B | p78 |
| `o` | reveals which track the user is really in, chunk by chunk, and how often the index changes | p78–p79 |
| `n` | plays "track 1 of every chunk", stitched | p79 |
| `4` `l` | several ways of stitching, each sent to Whisper; plays the stitch picked by SIM | p78 |
| `5` | the chunks where the user says nothing: two tracks still come out, and one SIM is still the higher one — plays the track that gets picked | p75, p78 |
| `a` | a threshold on SIM: does it reject the chunks where the user is absent, for each encoder? | p81 |
| `c` `x` `g` | chunk length; no chunking; the other talker | |
| `6` | reference, system output and the three kinds of error on one timeline | p87 |
| `7` | the same output scored three ways | p87 |
| `k` `t` | the clustering baseline instead of pyannote; the slide tables | |

**The definitions to keep in mind**

$$\mathrm{SIM} = \cos\!\big(E_{\text{spk}}(\hat{\mathbf{s}}_j),\, E_{\text{spk}}(\mathbf{x}^{\text{enr}})\big), \qquad \mathrm{DER} = \frac{T_{\text{FA}} + T_{\text{miss}} + T_{\text{conf}}}{T_{\text{speech}}}$$

Per 10 ms frame, with $N_{\text{ref}}$ speakers active in the reference, $N_{\text{hyp}}$ in the output and $N_{\text{ok}}$ of them matched: missed $= \max(0, N_{\text{ref}} - N_{\text{hyp}})$, false alarm $= \max(0, N_{\text{hyp}} - N_{\text{ref}})$, wrong speaker $= \min(N_{\text{ref}}, N_{\text{hyp}}) - N_{\text{ok}}$, and $T_{\text{speech}} = \sum N_{\text{ref}}$ (an overlapped frame counts once per speaker). The two switches: a **collar** of 0.25 s on each side of every reference boundary is not scored; frames where the reference has **overlap** are not scored.

**Reading the results honestly**

- This is blind separation followed by picking a track with SIM. It is **not** target speaker extraction (p74): no model here is conditioned on the enrolment.
- Whether the user's track index changes between chunks depends on the model, the voices, the chunk length, and small numerical details. It is not guaranteed to change — and nothing guarantees that it will not. The second pair of voices (`g`) is there to show a different outcome.
- The two SIM numbers for the same pair of voices come from two different embedding spaces. Their thresholds (0.25 from SpeechBrain's verification interface, 0.86 from the WavLM model card) are the publishers' examples, not tuned on this data, and are not interchangeable.
- The enrolment and the test sentences come from the same speaker reading audiobooks, in some cases the same recording session. Real enrolments are older, shorter and recorded elsewhere.
- The "conversation" is spliced from read sentences. Its reference is exact, which a hand-labelled recording never is; its turn-taking is not real (W12).
- The reference depends on a choice too: what counts as speech inside a sentence (here, pauses shorter than 0.2 s are bridged). Change that and every DER changes.

**Run it yourself**

You need LibriSpeech dev-clean (`../_media/dev-clean.tar.gz`, as fetched by W3 Demo 3) and the shared environments of W8–W9. `pyannote.audio` lives in its own environment (`./setup.sh`): version 4.1.1 requires a newer matplotlib than the one pinned for the rest of the course. Its model is gated on Hugging Face: accept the conditions on the model page and log in before `fetch`. Without it the first half runs unchanged and the second half shows the clustering baseline only.

```bash
./setup.sh              # builds ../.venv-diar
./present.sh fetch      # sentences, SepFormer, ECAPA, the pyannote pipeline
./present.sh check
./present.sh rehearse   # writes runs/rehearsal/
./present.sh            # or: ./present.sh replay
```

`pyannote.audio` 4 reports usage telemetry by default; `present.sh` and `diar_worker.py` switch it off (`PYANNOTE_METRICS_ENABLED=0`).

**Sources**: LibriSpeech (Panayotov et al. 2015, CC BY 4.0); SepFormer (Subakan et al. 2021) and ECAPA-TDNN (Desplanques et al. 2020) weights from SpeechBrain (Apache-2.0); WavLM (Chen et al. 2022); pyannote.audio (Bredin et al. 2020; Plaquet & Bredin 2023); SI-SDR (Le Roux et al. 2019).

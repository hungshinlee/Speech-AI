<!-- 此檔由 bin/publish_demos.py 從 private repo 的 <course>/demos/ 複製而來，請勿在公開 repo 直接編輯。 -->
# W9 Demo 3 · Two lines of defence, and where each one breaks

Slides p87–p90. Run by the lecturer on a MacBook Pro (M5 Max, no CUDA); you are not asked to run it.

## What it does

**First half — a detector.** An anti-spoofing model, AASIST, with the weights its authors released: trained on the ASVspoof 2019 LA training set,
that is, on speech synthesis and voice conversion systems from 2019 or earlier. It scores, one by one, sounds this course has already produced:

- real recordings — one LibriSpeech dev-clean utterance, plus one utterance from each of 40 dev-clean speakers as a reference set, plus a sentence spoken live into the microphone;
- W2's Griffin-Lim reconstruction;
- W8 and W9 outputs: SpeechT5 + HiFi-GAN, F5-TTS + Vocos, Kokoro, Qwen3-TTS, MMS-TTS (`nan`);
- **real speech that has been resynthesised**: through Mimi and EnCodec (W6 Demo 1), and through Vocos and BigVGAN (W8 Demo 3).

The window shows one score axis and one threshold.

**Second half — a watermark.** AudioSeal marks one F5-TTS utterance. We listen to it A / B, look at the detector's per-sample curve,
splice the marked audio into a real recording, and then process it: resampling, added noise, MP3, and **regeneration** —
EnCodec keeping fewer and fewer codebooks, Mimi, and Vocos (waveform → mel → waveform).
Every condition has a control: the same audio *without* the watermark, sent through the same processing (the grey curve).

## How the numbers are defined

| quantity | definition |
|---|---|
| score $s(\mathbf{x})$ | AASIST's bona fide logit minus its spoof logit, averaged over windows of 64 600 samples (≈ 4 s; hop 32 300; the last window ends with the utterance; shorter audio is repeated, as in the model's own data loader). The input is first scaled to a fixed RMS, because the score moves with level. |
| threshold | $0$: the model's own decision boundary between its two outputs. It was written into the config before any result was seen. It is **not** the equal-error-rate threshold on ASVspoof 2019 LA — that would need the dataset. |
| mean of Det | the mean over samples of the AudioSeal detector's probability that a sample is watermarked |
| detected | more than half of the samples have probability above 0.5. (The 0.5 per sample is the package default; "more than half" is this demo's choice.) |
| bits | how many of the 16 message bits are decoded correctly; 8 of 16 is chance |
| regeneration | 16 kHz → 24 kHz → codec (encode once, decode only the first $n_q$ codebooks) → 16 kHz. The row "16 → 24 → 16 kHz" is the same path with the resampling only. |

A different threshold, a different detection rule, or a different resampling order can change a verdict; published results on watermark robustness differ partly in exactly these choices.

## Files

| file | |
|---|---|
| `demo.py` | everything; `python demo.py --help` lists the sub-commands |
| `demo_config.toml` | which sounds, which model revisions, the threshold, the processing steps |
| `present.sh`, `setup.sh` | entry points; `setup.sh` adds the `audioseal` package to the TTS environment |
| `model.lock` | the Hugging Face revision of the AudioSeal weights |
| `runs/rehearsal/` | the rehearsal: audio, figures, `rehearsal.json` (all numbers, including the per-sample curves at 100 Hz) |

## To run it yourself

You need the environments of W6 Demo 1 and W8 Demo 1, and the rehearsal outputs of the earlier demos (they are in this repository).

```bash
cd demos/w09_d3_two_defences
./setup.sh               # audioseal → ../.venv-tts
./present.sh fetch       # clones clovaai/aasist (checked against a commit and a SHA-256), downloads two AudioSeal checkpoints (94 MB)
./present.sh check
./present.sh rehearse    # or: ./present.sh replay   (reads runs/rehearsal/, needs no model)
```

`./present.sh fake` runs the whole flow with stand-ins and no models; the window says FAKE.

## Models and licences

- AASIST: Jung et al., *AASIST: Audio Anti-Spoofing using Integrated Spectro-Temporal Graph Attention Networks*, ICASSP 2022. Code and weights: `github.com/clovaai/aasist`, MIT licence.
- AudioSeal: San Roman et al., *Proactive Detection of Voice Cloning with Localized Watermarking*, ICML 2024. Package `audioseal`; weights `facebook/audioseal` on Hugging Face, MIT licence. The checkpoints are Python pickles: `fetch` verifies their SHA-256 before anything loads them.
- Mimi (`kyutai/mimi`), EnCodec (`facebook/encodec_24khz`), Vocos (`charactr/vocos-mel-24khz`): as in W6 Demo 1 and W8 Demo 3.
- Audio: LibriSpeech dev-clean (CC BY 4.0) and outputs of earlier demos in this repository.

## What this demo does not do

It trains nothing. It does not attack the detector or the watermark detector adversarially: every processing step here is something that happens to audio in ordinary use.

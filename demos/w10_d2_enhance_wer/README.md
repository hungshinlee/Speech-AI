<!-- 此檔由 bin/publish_demos.py 從 private repo 的 <course>/demos/ 複製而來，請勿在公開 repo 直接編輯。 -->
# W10 · Demo 2 — After enhancement, where does WER go? (slides p48–p49, p55)

Speech enhancement is trained and scored against judges built for human ears, or judges that count energy. The model that comes next in the pipeline is a different judge. This demo takes the same sentences, adds noise at five levels, sends them through several front ends, and asks two recognisers — and one spoofing detector — what they think. Nothing is trained; every model is used for inference only.

**What goes in**

| | |
|---|---|
| speech $s$ | 40 sentences of LibriSpeech dev-clean, one per speaker, about 7 s each (the same 40 as W2 Demo 3 and W9 Demo 3's reference set) |
| noise $v$ | synthetic, three kinds: **white**; **pink** (power $\propto 1/f$); **babble** (six *other* talkers from the same 40 sentences, summed). One draw per sentence and kind, reused at every SNR |
| microphone $y = s + g\,v$ | $g$ sets the SNR over the whole utterance: $20$, $10$, $5$, $0$, $-5$ dB. The speech itself is never rescaled |

Because the mixture is made here, both $s$ and $v$ are known — which is what the signal-level judges need, and what a deployed system never has.

**Front ends**

| | |
|---|---|
| none | $\hat{s} = y$ |
| statistical | a decision-directed Wiener filter in `numpy` (`enhance.py`, `wiener_dd`): $G = \xi/(1+\xi)$ per time–frequency cell, noisy phase. The noise power per bin is the median of $\lvert Y\rvert^2$ over the utterance, corrected for the exponential distribution — a simplification of this demo that assumes stationary noise |
| MetricGAN+ | SpeechBrain's `metricgan-plus-voicebank`: a BLSTM predicts a real-valued mask on the magnitude; its training target was a learned predictor of PESQ. Used exactly as published, except that the final peak normalisation is skipped so that $\hat{s}$ and $y$ stay on the same scale |
| SepFormer (extra) | SpeechBrain's `sepformer-dns4-16k-enhancement`, a time-domain model. Its output level is arbitrary, so it is rescaled by the least-squares gain against $y$ |

**Judges**

| Judge | What it is |
|---|---|
| WER, CTC model | `facebook/wav2vec2-base-960h`, greedy decoding, no language model. It was fine-tuned on clean read speech only (W3) |
| WER, Whisper small | weakly supervised on a large and noisy collection, with a strong prior on its output side (W7) |
| SDR, SNR, SAR | two projections of $\hat{s}$: onto $s$, then onto the plane spanned by $s$ and $v$ (`enhance.py`, `decompose`; slides p50–p51). What is left outside the plane is the artifact term |
| SI-SDR, STOI, PESQ | SI-SDR as defined in W2; STOI and wide-band PESQ from the `pystoi` and `pesq` packages (left empty if not installed) |
| AASIST | W9 Demo 3's spoofing detector, used exactly as configured there: is this *real* speech still called real after enhancement? |

WER is corpus-level; references and both recognisers' outputs go through Whisper's `EnglishTextNormalizer`.

**What the keys show**

| Key | What happens | Slide |
|---|---|---|
| `0` `1` | plays the clean sentence; then the same sentence noisy, after the statistical front end, after MetricGAN+ | p49 |
| `u` | the statistical front end with no smoothing ($\alpha = 0$) — listen for musical noise | p38 |
| `2` `3` | WER against SNR, one line per front end: the CTC model, then Whisper | p49 |
| `4` `w` | the two recognisers' transcripts of the same enhanced sentence, side by side; error types (substitutions / deletions / insertions) | p49 |
| `5` | one point per (front end, SNR): SAR against WER | p55 |
| `6` | observation adding, $\hat{s} + \omega\,y$: SNR, SAR and both WERs as $\omega$ grows | p53–p55 |
| `7` | the fraction of real sentences AASIST calls spoof, per front end and SNR | p55 |
| `m` `e` | every judge in one table; the front ends applied to clean speech | p47 |
| `b` `s` `n` | change the noise kind, the listening SNR, the listening sentence | |
| `l` | live: a fresh noise draw for the current sentence, all front ends, both recognisers | |

**The two definitions to keep in mind**

$$\hat{\mathbf{s}} = \underbrace{\mathbf{P}_{s}\hat{\mathbf{s}}}_{\mathbf{s}_{\text{target}}} + \underbrace{\mathbf{P}_{s,v}\hat{\mathbf{s}} - \mathbf{P}_{s}\hat{\mathbf{s}}}_{\mathbf{e}_{\text{noise}}} + \underbrace{\hat{\mathbf{s}} - \mathbf{P}_{s,v}\hat{\mathbf{s}}}_{\mathbf{e}_{\text{artif}}}, \qquad \mathrm{SNR} = \frac{\lVert\mathbf{s}_{\text{target}}\rVert^2}{\lVert\mathbf{e}_{\text{noise}}\rVert^2}, \quad \mathrm{SAR} = \frac{\lVert\mathbf{s}_{\text{target}} + \mathbf{e}_{\text{noise}}\rVert^2}{\lVert\mathbf{e}_{\text{artif}}\rVert^2}$$

The subspaces here are spanned by $s$ and $v$ themselves, with **no delayed copies** (the published definition allows short filters of each). The consequence: any time-varying gain applied to the noise counts as artifact, so SAR values here are lower than they would be with the published definition, and the SNR term only counts the part of the residual that is still a scaled copy of $v$. With no front end, $\mathbf{e}_{\text{artif}} = 0$ and SAR is infinite. Adding $\omega\,y$ back leaves $\mathbf{e}_{\text{artif}}$ untouched — `./present.sh selftest` checks this to rounding error.

**Reading the results honestly**

- These are two recognisers, one learned front end trained on one corpus, 40 read sentences and synthetic noise. The outcome depends on all four; it is not a statement about enhancement in general.
- MetricGAN+ was trained on VoiceBank-DEMAND: 28 speakers, mixed at 0, 5, 10 and 15 dB with eight recorded environmental noises plus babble and speech-shaped noise. An SNR of $-5$ dB, and white or pink noise, are outside what it has seen; LibriSpeech is a different recording condition as well.
- AASIST was trained on ASVspoof 2019 LA; LibriSpeech is already out of distribution for it (W9 Demo 3).

**Running it**

```bash
cd demos
./setup.sh                              # shared environment
./w03_d3_blank_heatmap/setup.sh         # torch, transformers
./w08_d1_mean_vs_sample/setup.sh        # the second environment, .venv-tts
./w09_d1_cloning_leak/setup.sh          # speechbrain
cd w10_d2_enhance_wer
./setup.sh                              # pystoi, pesq
./present.sh fetch                      # model weights (network)
./present.sh replay                     # browse the recorded run, no models needed
./present.sh rehearse                   # recompute everything (tens of minutes; resumable)
```

`fetch` also expects the 40 sentences (`../w02_d3_frontend_mismatch/present.sh fetch`) and AASIST (`../w09_d3_two_defences/present.sh fetch`). `runs/rehearsal/` holds the numbers, the figures and the audio of the recorded run.

**Sources.** LibriSpeech dev-clean (CC BY 4.0). `speechbrain/metricgan-plus-voicebank` and `speechbrain/sepformer-dns4-16k-enhancement` (both Apache-2.0 on the Hugging Face Hub). `facebook/wav2vec2-base-960h` and `openai/whisper-small` as in W3 Demo 3 and W2 Demo 3. AASIST: `clovaai/aasist` (MIT), as in W9 Demo 3. The error decomposition follows Vincent et al. (2006) as used by Iwamoto et al. (2022); observation adding is from the same line of work.

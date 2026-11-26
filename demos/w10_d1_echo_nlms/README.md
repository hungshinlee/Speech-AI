<!-- 此檔由 bin/publish_demos.py 從 private repo 的 <course>/demos/ 複製而來，請勿在公開 repo 直接編輯。 -->
# W10 · Demo 1 — The system hears itself: echo, NLMS, double-talk (slides p18–p20, p34–p35)

A system that listens while it speaks hears its own voice come back through the room. This demo plays one sentence of text-to-speech through a laptop's loudspeaker, records the laptop's own microphone at the same time, and asks how much of that echo can be removed when the played signal is known. Everything that removes echo is a few dozen lines of `numpy` in `aec.py`; no neural network is trained, and the adaptive filter has no training phase at all — it learns while it runs.

**The signals** (16 kHz):

| | |
|---|---|
| $x(n)$, the reference | a 21 s Mandarin sentence spoken by Kokoro (an output of W8 Demo 2), with 0.5 s of silence before and 1 s after |
| $y(n)$, the microphone | three recordings of the same $x$: **alone** (nobody talks), **talk** (the lecturer talks over the system for about 3 s), **loud** (the loudspeaker turned up until it stops being linear) |
| a simulated copy | the same three clips through a simulated room (`pyroomacoustics`, image method), where the true echo path $\mathbf{h}$ is known. Key `m` switches between the recordings and the simulation |

The lecturer's own recordings are not in this repository; the simulation and its outputs are. `./present.sh record alone|talk|loud` records your own.

**What the keys show**

| Key | What happens | Slide |
|---|---|---|
| `1` | plays the microphone signal $y(n)$ of the *alone* clip | p20 |
| `2` | plays $y(n) - x(n)$: the played signal subtracted as it is | p20 |
| `3` | plays $y(n) - g\,x(n - D)$: delayed to line up with the echo, with the best single gain $g$ | p20 |
| `4` | *alone* + NLMS starting from zero: ERLE against time | p34 |
| `5` `6` `7` | *talk* + a fixed step size / a Geigel double-talk detector that freezes the filter / a soft step size | p34 |
| `8` | *loud* + NLMS: where the ERLE stops | p34 |
| `9` `0` | the *talk* clip goes to Silero VAD and Whisper small, before and after echo cancellation | p35 |
| `y` `e` `a` | play the microphone / the residual / only the seconds around the moment the lecturer stops | |
| `c` | prints the NLMS loop; the two lines marked (p24) and (p26) are the two equations on the slides | |
| `t` `m` `h` `q` | the tables, recordings ↔ simulation, menu, quit | |

**The algorithm** (`aec.py`, function `nlms`). With $\mathbf{x}(n) = [x(n), \dots, x(n - L_h + 1)]$:

$$e(n) = y(n) - \hat{\mathbf{h}}^{\top}(n)\,\mathbf{x}(n), \qquad \hat{\mathbf{h}}(n+1) = \hat{\mathbf{h}}(n) + \mu\,\frac{e(n)\,\mathbf{x}(n)}{\lVert\mathbf{x}(n)\rVert^2 + \delta}$$

$L_h = 2048$ (128 ms), $\mu = 0.5$, $\delta = 0.01\,L_h\,\overline{x^2}$. The fixed playback-to-capture delay $D$ is found by cross-correlation and removed first, so that the $L_h$ coefficients are spent on the room. The three step-size rules:

| Rule | What it does |
|---|---|
| fixed | $\mu = 0.5$ always |
| Geigel freeze | $\mu = 0$ while $\lvert y(n)\rvert > \vartheta \max_{0 \le i < L_h} \lvert x(n-i)\rvert$, and for 200 ms after. $\vartheta$ is set from the *alone* clip: 1.5 times the 99.9th percentile of that ratio when only the system speaks |
| soft | $\mu(n) = \min\big(0.5,\ \Lambda(n)\,p_x(n) / P_e(n)\big)$, the residual-echo share of the error (slide p29). $p_x(n) = \lVert\mathbf{x}(n)\rVert^2 / L_h$ is known; $P_e$ is a 20 ms average of $e^2$; $\Lambda$ tracks $e^2 / p_x$ with a 1 s average and may at most double every 2 s. **This estimator is this demo's own simplification, not the leakage estimator of Valin (2007)**; its three time constants were chosen on the simulated clip |

**What is measured**

| Quantity | Definition |
|---|---|
| residual level (keys `1`–`3`) | $10\log_{10}(\sum \text{residual}^2 / \sum y^2)$ over the clip. 0 dB means nothing was removed; a positive number means it got louder |
| ERLE | $10\log_{10}(\sum y^2 / \sum e^2)$ over samples where the system is audible. The curve uses the past 500 ms |
| before / after the interruption | ERLE over the last 2 s of system-audible samples before the lecturer starts, and over the first 1 s of system-audible samples after the lecturer stops. While the lecturer talks, the residual contains the lecturer, so the number there says nothing about the filter |
| when the lecturer talks | recordings: where the *talk* clip minus the *alone* clip (the same $x$, recorded twice) has energy. Simulation: known |
| $\lVert\mathbf{h} - \hat{\mathbf{h}}\rVert^2 / \lVert\mathbf{h}\rVert^2$ | simulation only, over the first $L_h$ coefficients of the true path |
| VAD active | the share of time Silero VAD (threshold 0.5, its defaults) reports speech, split into "only the system is speaking" and "the lecturer is talking" |

"Loud" in the simulation is $\tanh$ soft clipping of $x$ before the room; the filter still receives the undistorted $x$.

## Running it

```bash
cd speech_ai/demos && ./setup.sh && ./w03_d3_blank_heatmap/setup.sh     # the shared environment, torch and transformers
cd w10_d1_echo_nlms
./setup.sh                  # adds silero-vad and pyroomacoustics, then runs selftest
./present.sh fetch          # Whisper small (the commit W7 Demo 1 uses)
./present.sh check          # plays 3 s of noise three times: delay, drift, whether a linear filter can learn the path
./present.sh record alone   # then: record talk, record loud (follow the prompts)
./present.sh rehearse       # everything, non-interactive → runs/rehearsal/
./present.sh                # the keys above, in the window
./present.sh --sim          # the simulation only: no microphone needed
./present.sh replay         # read runs/rehearsal/ back: no recording, no computation
```

`./present.sh selftest` needs neither models nor a microphone: it checks $e^{+} = (1-\mu)e$, convergence to a known path, that $\mu > 2$ diverges, that $y - x$ removes nothing, and the three step-size rules on a toy path.

## Things to keep in mind

- Laptop loudspeakers and microphones differ, and so do rooms: the numbers in `runs/rehearsal/` belong to one laptop in one room on one day.
- Use the built-in loudspeaker and microphone. Bluetooth headsets change the sampling rate and add their own processing; an external display may silently become the output device.
- The NLMS here runs sample by sample in the time domain so that it matches the slides line for line. Real echo cancellers work in blocks in the frequency domain; the structure is the same.

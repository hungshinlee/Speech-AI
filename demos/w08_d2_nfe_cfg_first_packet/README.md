<!-- 此檔由 bin/publish_demos.py 從 private repo 的 <course>/demos/ 複製而來，請勿在公開 repo 直接編輯。 -->
# W8 · Demo 2 — NFE, CFG, and the first packet (slides p62–p64, p84)

Two halves, one program. The first half (slide p64) takes the flow-matching model of Demo 1 and changes one thing at a time; the second half (p84) measures the *first packet* — how long after the request the first sound is ready — for three systems with three different shapes.

## First half: one sentence, one prompt, one seed

F5-TTS (`SWivid/F5-TTS`, `F5TTS_v1_Base`, conditional flow matching) with Vocos, the same prompt as Demo 1 (`1272-128104-0010`, LibriSpeech dev-clean, CC BY 4.0 — an English prompt reading a Mandarin sentence, i.e. cross-lingual cloning), seed 1.

| Key | Sweep | What stays fixed | What to listen for |
|---|---|---|---|
| `1` | $N_{\text{FE}} \in \{1, 2, 4, 8, 16, 32\}$ Euler steps | $w = 3$, sway $s = -1$, uniform time grid | the knee: the smallest number of steps that sounds like 32; the time printed after each clip grows linearly with the steps |
| `e` | the same sweep on the English sentence of Demo 1 | | the 32-step clip is identical to Demo 1's seed 1 |
| `2` | CFG weight $w \in \{0, 1, 2, 3, 4\}$ | $N_{\text{FE}} = 16$ | $w = 0$ is the unconditional field alone (no text, no prompt); $w = 1$ the plain conditional model; $w = 3$ F5's default; $w = 0$ and $1$ need one forward pass per step, the others two |
| `3` | sway sampling on ($s = -1$) and off ($s = 0$) | $N_{\text{FE}} = 16$, uniform grid before sway | same number of steps, spent at different $\tau$; the lower panel shows where the 16 steps land |
| `p` | F5's own 16-step schedule (its EPSS table, then sway) | | what `f5-tts` actually uses when you ask it for 16 steps |

**About $w$.** The slides write classifier-free guidance as $\tilde{\mathbf{v}} = \mathbf{v}_\theta(\mathbf{X}_\tau, \tau) + w\,\big(\mathbf{v}_\theta(\mathbf{X}_\tau, \tau, \mathbf{c}) - \mathbf{v}_\theta(\mathbf{X}_\tau, \tau)\big)$ — the weight sits on the difference added to the *unconditional* field. F5's `cfg_strength` is the $\alpha$ of its Eq. 4, $\mathbf{v}_c + \alpha(\mathbf{v}_c - \mathbf{v}_u)$, so $w = \alpha + 1$ and F5's default $\alpha = 2$ is $w = 3$. The sweep uses the slides' $w$.

**About the time grid.** `f5-tts` 1.1.22 uses a fixed, non-uniform table of time steps for 5, 6, 7, 10, 12 and 16 steps (`get_epss_timesteps`) and a uniform grid otherwise, then applies sway. To change one thing at a time, this demo always starts from a uniform grid and applies sway on top; key `p` plays F5's own 16-step table for comparison. For that reason the sampling loop is written in `demo.py` (`F5Sampler.sample`) line-for-line after `f5_tts/model/cfm.py`'s `sample()`, with the grid, the CFG weight and per-step timing exposed; `./present.sh check` verifies that it reproduces the library's output for the same seed ($w = 3$: max abs mel difference $\sim 10^{-5}$; $w = 1$ against `cfg_strength = 0`: exactly 0).

## Second half: the first packet

Three Mandarin texts of 10, 30 and 90 characters (a ticket-booking assistant's replies — W1 Demo 2's scenario), three systems:

| Key | System | Shape | What is timed |
|---|---|---|---|
| `4` | F5-TTS, whole utterance, 32 steps | non-autoregressive, the whole utterance in one pass | from the call to the whole waveform: the first packet *is* the last packet, and it grows with the text |
| `5` | Qwen3-TTS (`mlx-community/Qwen3-TTS-12Hz-0.6B-Base-bf16` through `mlx-audio`), streaming, cloning from the same prompt | autoregressive codec LM at 12.5 tokens/s + a streaming decoder | the first chunk (4 tokens = 0.32 s of audio, the $C = 4$ of the paper's 97 ms figure), then the gap between consecutive chunks against each chunk's audio length (RTF, which must stay below 1 for play-out not to starve); the right-hand panel plots when each chunk arrived against its play-out deadline |
| `6` | Kokoro (`mlx-community/Kokoro-82M-bf16`, the voice and language of W1 Demo 2) | non-autoregressive, whole utterance, iSTFT output | from the call to the whole waveform — W1's 124 ms was this row |

All times are wall-clock on the instructor's machine (F5 on PyTorch/MPS, the other two on MLX/Metal), measured after one warm-up call per model; the Qwen3-TTS first packet includes encoding the prompt with its speech tokenizer, as cloning requires. They are not comparable with the RTFs quoted in the papers (different hardware, batch sizes, definitions).

## Keys

`1` `e` `2` `3` `p` — the sweeps above; `4` `5` `6` — the first packet; `t` the table of numbers; `r` play again; `h` menu; `q` quit. Generation happens when a key is pressed (the waiting is part of the point); the terminal prints each clip's time as it is made.

## Running it

```bash
cd speech_ai/demos/w08_d2_nfe_cfg_first_packet
../w08_d1_mean_vs_sample/setup.sh     # once: builds ../.venv-tts (Demo 1's environment; F5 lives there)
../w08_d1_mean_vs_sample/present.sh fetch   # once: F5, Vocos, the prompt
./present.sh fetch         # Qwen3-TTS and Kokoro (commits pinned in model.lock)
./present.sh check         # versions; the re-written sampler against f5-tts's own; ms per step → runs/check.json
./present.sh rehearse      # everything, non-interactive → runs/rehearsal/
./present.sh               # the keys above
./present.sh replay        # the rehearsal's outputs on the same keys, without torch or MLX
./present.sh fake          # the whole flow without any model (stand-ins, marked FAKE)
```

Requirements: Demo 1's `speech_ai/demos/.venv-tts` for F5 (torch 2.14.0, f5-tts 1.1.22) and the shared `speech_ai/demos/.venv` for the two MLX models (`mlx-audio 0.5.4`, which already includes `qwen3_tts`; Apple silicon only). The two environments cannot share a process, so `demo.py` runs `mlx_worker.py` as a subprocess in the shared environment and talks to it over JSON lines. Without MLX (`--no-mlx`, or a non-Apple machine) keys `5` and `6` are unavailable and the rest works.

Weights: F5-TTS is CC BY-NC 4.0 (non-commercial); Vocos MIT; Qwen3-TTS (`mlx-community/Qwen3-TTS-12Hz-0.6B-Base-bf16`) and Kokoro (`mlx-community/Kokoro-82M-bf16`) Apache-2.0, as declared on their model cards (checked 2026-11-10; `./present.sh fetch` prints the license field). Numbers quoted on the slides come from `runs/rehearsal/rehearsal.json`, summarised in `slides/assets/w08/w08-metrics.json` (`demo2`). Draws are reproducible only on the same machine, device and seed.

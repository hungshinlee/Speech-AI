<!-- 此檔由 bin/publish_demos.py 從 private repo 的 <course>/demos/ 複製而來，請勿在公開 repo 直接編輯。 -->
# W6 · Demo 3 — sixteen codes, four clusters, five ways to train: dead codes you can see (slides p15–p16)

`numpy` only (matplotlib draws the figure), CPU, a terminal and one plot window — no audio, no model. Slide p14 names two causes of codebook collapse — a codebook initialised outside the data's support, and an update rule that only moves the code that was chosen — and p13 says the EMA codebook update is online $k$-means. In two dimensions all of this is visible. Four Gaussian clusters sit inside $[3, 6]^2$; a codebook of $V = 16$ is initialised from $\mathcal{N}(\mathbf{0}, \mathbf{I})$, deliberately far from the data; five ways of training it run for 200 steps each, and every one prints how many of the sixteen codes are used and the quantisation error. With gradient updates on the codebook loss only 2 codes are ever chosen; with EMA it is **still 2** (the question on p16 — EMA fixes noisy codebook updates, not collapse); $k$-means initialisation and dead-code reset use all 16; FSQ has no codebook at all, so nothing can die, but its equally spaced grid pays in error. One more key starts the same sixteen codes *inside* the data: all sixteen live under both update rules, so the collapse here is not tuned in — it is the direct consequence of where the codebook starts.

| Key | What it does | Slide |
|---|---|---|
| `1` | random init + gradient: $\mathbf{e}_v \mathrel{+}= \eta\,\overline{(\mathbf{z} - \mathbf{e}_v)}$ for the chosen $v$ only. The animation shows two codes pulled to the edge of the data and fourteen red crosses that never move; the terminal prints codes used and batch MSE at steps 1, 5, 10, 25, 50, 100, 200, then the assignment of all 2000 points | p16 panel 1 |
| `2` | the same $\mathbf{E}_0$, EMA update ($c_v$, $\mathbf{m}_v$, $\mu = 0.99$) — the answer to the p16 question | p16 panel 2 |
| `3` | $k$-means initialisation on the first batch (10 Lloyd rounds), then EMA — SoundStream's recipe | p16 panel 3 |
| `4` | random init + EMA + dead-code reset: a code unchosen for 5 consecutive steps is replaced by a random point of the current batch (Jukebox, SoundStream). The terminal prints at which step resets happened | p16 panel 4 |
| `5` | FSQ with $\Lambda = (4, 4)$: per dimension $\tanh((z - 4.5)\cdot 0.9)$ snapped to four equally spaced levels; prints the $4 \times 4$ table of how many points land on each grid point | p16 panel 5, p18 |
| `i` | the same settings with $\mathbf{E}_0$ drawn uniformly from $[3, 6]^2$ (a separate random stream), gradient and EMA each once | p16 notes |
| `t` | the five lines printed under the panels on p16 (plus two for `i` once it has run) | p16 |
| `r` | a new seed from the clock; all five settings are recomputed, then `1`–`5` again — the numbers are not pre-arranged; the slide uses seed 20261017 | |

## Definitions

| | |
|---|---|
| data | four Gaussians with centres $(3.6, 3.6)$, $(5.4, 3.6)$, $(3.6, 5.4)$, $(5.4, 5.4)$ and std 0.3; 2000 points |
| codebook | $V = 16$; keys `1`, `2`, `4` share one $\mathbf{E}_0 \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$ |
| a step | 128 points drawn without replacement → nearest-neighbour assignment $k = \arg\min_v \lVert \mathbf{z} - \mathbf{e}_v \rVert$ → update; 200 steps |
| gradient | $\mathbf{e}_v \mathrel{+}= 0.2 \cdot \mathrm{mean}_{k = v}(\mathbf{z} - \mathbf{e}_v)$, the negative gradient of the codebook loss $\lVert \mathrm{sg}[\mathbf{z}] - \mathbf{e}_v \rVert^2$; zero for codes that were not chosen |
| EMA | $c_v \leftarrow \mu c_v + (1 - \mu) n_v$, $\mathbf{m}_v \leftarrow \mu \mathbf{m}_v + (1 - \mu) \sum_{k = v} \mathbf{z}$, $\mathbf{e}_v = \mathbf{m}_v / c_v$; for an unchosen code $c_v$ and $\mathbf{m}_v$ only decay and $\mathbf{e}_v$ does not move |
| reset | a code with $n_v = 0$ for 5 consecutive steps is replaced by a random point of the current batch |
| FSQ | per dimension $\tanh((z - 4.5)\cdot 0.9)$ rounded to $\{\pm 0.25, \pm 0.75\}$ (the half-step offset FSQ uses for even $\Lambda$; the levels avoid $\pm 1$ so they can be mapped back through $\operatorname{arctanh}$ to compute the error in the data plane) |
| codes used | after training, assign all 2000 points to their nearest code and count the codes that receive at least one point |
| MSE | the mean squared distance of that assignment (for FSQ, computed in the data plane) |

Why all four codebook settings are computed at start-up: the figure script that produced the slide's panels runs gradient → EMA → $k$-means → reset on one random stream, each continuing where the previous one stopped. The demo reproduces that order once at start (0.3 s), records a snapshot every four steps, and each key replays its trajectory — so whatever order the keys are pressed in, the numbers match the slide. `selftest` checks this against `slides/assets/w06/w06-data.json`.

Why the fourteen dead codes revive together in `4`: all of them have been unchosen since step 1, so their counters reach the threshold at the same step and they are all moved into the data at once; in a real codec, codes die at different times and revive one by one.

## Requirements

| | |
|---|---|
| Packages | `numpy`, `matplotlib` (the shared `.venv`; `../common.py` also imports scipy) |
| Time | all five settings compute in well under a second; each animation plays in two to four seconds (`frame_every`, `frame_pause` in `demo_config.toml`, no effect on the numbers) |
| Terminal | ≈ 70 columns for `t`; `./present.sh show --no-plot` runs without the figure window |
| Cross-checks | `selftest` recomputes the five settings with the slide's seed and compares the 2000 data points, $\mathbf{E}_0$, and for every setting the number of codes used, the MSE (tolerance $10^{-9}$), the final codebook and the assignment of all 2000 points with `w06-data.json`; then checks the properties the slide relies on: `1` and `2` use the same few codes and their unchosen codes never moved; `3` and `4` use all sixteen and at least one reset happened; FSQ uses all sixteen grid points with a higher MSE than $k$-means; `1`/`2` are an order of magnitude worse than `3`/`4`; and codes initialised inside the data stay alive under both update rules |
| Tested on | Apple M5 Max, macOS 27.0.1, Python 3.12.14, numpy 2.4.4 (2026-10-18): `selftest` passes and all five settings are bit-identical to the pre-computed panels; `runs/rehearsal/` is this run. On a 2-core Linux container (Python 3.13, numpy 2.5.3) `selftest` also passes, likewise bit-identical — 200 steps without Adam do not amplify BLAS differences |

## Run it

```bash
cd demos && ./setup.sh && cd w06_d3_dead_codes
./present.sh selftest
./present.sh rehearse         # non-interactive, class order 1 2 3 4 5 i t → runs/rehearsal/ (json, txt, panels.png)
./present.sh                  # = show: single-key menu (h menu, q quit)
./present.sh replay           # print the rehearsal transcript; the figure is runs/rehearsal/panels.png
```

## What you should see

From `runs/rehearsal/rehearsal.json` (2026-10-18, Apple M5 Max; identical to the slide's panels):

- `1` random init + gradient: **2 / 16** codes used, MSE 0.997. `2` the same with EMA: **2 / 16**, MSE 0.993 — the fourteen unchosen codes stay exactly where they started.
- `3` $k$-means init + EMA: **16 / 16**, MSE 0.065. `4` random init + EMA + reset: **16 / 16**, MSE 0.069; one reset event at step 5 moving all fourteen dead codes at once.
- `5` FSQ $\Lambda = (4, 4)$: **16 / 16**, MSE 0.123 — the four corner grid points take about 350 points each, the four central ones 12–16; equally spaced levels waste capacity on four clusters.
- `i` codes initialised inside $[3, 6]^2$: gradient **16 / 16** (MSE 0.065), EMA **16 / 16** (MSE 0.066).

## Files

| File | |
|---|---|
| `demo.py` | `show` / `rehearse` / `replay` / `selftest` |
| `demo_config.toml` | seed, $V$, number of points, steps, batch, step size, EMA decay, reset threshold, cluster centres and std, FSQ levels and affine; the range for `i`; display settings |
| `present.sh` | entry point; uses `../.venv/bin/python` |
| `runs/rehearsal/rehearsal.json` | per setting: codes used, MSE, the trace table, the final codebook, reset events; the FSQ grid counts; the `i` results; the comparison with `w06-data.json`; versions and date |
| `runs/rehearsal/rehearsal.txt` | the terminal output without colour codes |
| `runs/rehearsal/panels.png` | the 2 × 3 figure |

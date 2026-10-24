<!-- 此檔由 bin/publish_demos.py 從 private repo 的 <course>/demos/ 複製而來，請勿在公開 repo 直接編輯。 -->
# W5 · Demo 3 — the ceiling at $\ln N$: InfoNCE on a Gaussian pair with known mutual information (slides p14–p16)

`numpy` only (matplotlib draws the figures), CPU, a terminal and one or two plot windows — no audio. The last line of the InfoNCE derivation on p13 says that for **any** critic, $I(\mathbf{z};\mathbf{c}) \geq \ln N - \mathcal{L}_N$; since $\mathcal{L}_N \geq 0$, the estimate can never exceed $\ln N$. This demo makes that visible on a pair of random vectors whose mutual information is known in closed form: two 8-dimensional Gaussians with per-dimension correlation $\rho = 0.7$, so $I = -\frac{d}{2}\ln(1 - \rho^2) = 2.693$ nats. A critic is trained with InfoNCE at each $N = 2, 4, \dots, 1024$ ($N$ = one positive + $N - 1$ negatives). At $N = 8$ the ceiling $\ln 8 = 2.08$ sits below the truth, so the estimate cannot reach it however long you train; from $N = 16$ on the ceiling clears the truth and the estimate creeps up — but the ceiling rises only as $\ln N$, and sixteen times more negatives (64 → 1024) buy 0.31 nat and a smaller variance, not new information.

| Part | Keys | What it does | Slide |
|---|---|---|---|
| 1 | `1` | the truth $I = -\frac{d}{2}\ln(1-\rho^2)$ and $\ln N$ for every $N$; rows whose ceiling is below the truth are marked; the smallest $N$ whose ceiling clears the truth ($N > e^{I} = 14.8$) | p15 question |
| 2 | `2` | for each $N$ in turn: train the critic (Adam, 3000 steps, a fresh batch every step), evaluate $\ln N - \mathcal{L}_N$ on 200 held-out batches, print the estimate, $\mathcal{L}_N$, distance to the ceiling and to the truth, the std over batches and the seconds taken; the figure gains a point (error bar = std). Ends with the 64 → 1024 comparison and the two invariants (every estimate $\leq \ln N$ and $<$ truth) and the difference to the pre-computed curve in `slides/assets/w05/w05-data.json` | p15 curve, p16 table |
| 3 | `3` | with the critics from `2`: histograms of the 200 per-batch estimates at $N = 2$, 32, 1024 (same $x$ axis and bins), with mean, std, min, max and how many single batches exceed the truth — a single batch can, the bound is on the expectation | p15, p16 row 2 |
| 4 | `b` | a purely bilinear critic $f = \exp(\mathbf{x}^{\top}\mathbf{W}\mathbf{y})$ at $N = 8$, 128, 1024 next to the full critic from `2`; the optimal value of the extra coefficient, $a^{*} = -\rho^2 / (2(1-\rho^2))$, is printed for comparison with the learned $a$ | p15 notes |
| — | `r` | new seed (from the clock), then `2` again — the curve is not pre-arranged; the slide numbers are seed 20261011 | |

## Definitions

| | |
|---|---|
| data | per dimension $y = \rho x + \sqrt{1-\rho^2}\,\epsilon$ with $x, \epsilon \sim N(0,1)$, $d = 8$ independent dimensions; a batch is $N$ such pairs |
| truth | $I(\mathbf{x};\mathbf{y}) = -\frac{d}{2}\ln(1-\rho^2)$ (sum over dimensions of $h(y) - h(y \mid x)$) |
| critic | $f(\mathbf{x},\mathbf{y}) = \exp(\mathbf{x}^{\top}\mathbf{W}\mathbf{y} + a\lVert\mathbf{y}\rVert^2)$, $\mathbf{W} \in \mathbb{R}^{d \times d}$ and the scalar $a$ both trained from 0 |
| loss | $S_{ij} = \ln f(\mathbf{x}_i, \mathbf{y}_j)$ over the $N \times N$ pairs of a batch; the diagonal is the positive, the other $N - 1$ entries of the row are the negatives; $\mathcal{L}_N = -\frac{1}{N}\sum_i \ln \mathrm{softmax}(S_{i\cdot})_i$ |
| estimate | $\ln N - \mathcal{L}_N$, with $\mathcal{L}_N$ averaged over 200 held-out batches; the std is over those 200 values |
| training | Adam ($\beta_1 = 0.9$, $\beta_2 = 0.999$, lr 0.01) for 3000 steps, written out in numpy; $\partial\mathcal{L}/\partial S = (P - \mathbf{I})/N$ |

Why the critic has the extra term $a\lVert\mathbf{y}\rVert^2$: the optimal critic is the density ratio $p(\mathbf{y}\mid\mathbf{x})/p(\mathbf{y})$, whose logarithm for a Gaussian pair is quadratic in $\mathbf{x}$ and $\mathbf{y}$ — besides $\mathbf{x}^{\top}\mathbf{y}$ it contains $\lVert\mathbf{x}\rVert^2$ and $\lVert\mathbf{y}\rVert^2$. InfoNCE is a softmax within a row (fixed $\mathbf{x}_i$, over the $\mathbf{y}_j$), so the $\lVert\mathbf{x}\rVert^2$ term cancels but the $\lVert\mathbf{y}\rVert^2$ term does not. A purely bilinear family does not contain the optimal critic, and its bound stays looser (`b`). Both are valid lower bounds; they differ in tightness.

Why $d = 8$, $\rho = 0.7$: the truth then falls between $\ln 8$ and $\ln 16$, so "can $N = 8$ reach it?" has a clean answer, while the ceiling at $N = 1024$ (6.93) is far above the truth, so the right half of the curve shows what extra negatives buy once the ceiling is cleared.

## Requirements

| | |
|---|---|
| Packages | `numpy`, `matplotlib` (the shared `.venv`; `../common.py` also imports scipy) |
| Time | the full sweep in `2` takes 30 s on the instructor's Apple M5 Max ($N = 1024$ alone 22 s) and about 70 s on a 2-core Linux container; `rehearse` runs `1 2 3 b` in about one to two minutes |
| Terminal | ≈ 90 columns for the rows of `2`; lower `display.digits` in `demo_config.toml` if it wraps. `./present.sh show --no-plot` runs without figure windows |
| Cross-checks | `selftest` re-runs the whole sweep with the same seed and compares the estimate, $\mathcal{L}_N$ and std at every $N$ with `slides/assets/w05/w05-data.json` (tolerance $10^{-9}$; it also reports how many of the 30 numbers are bit-identical — 3000 Adam steps amplify BLAS summation-order differences of $10^{-16}$ between machines, so bit-identity is expected on one machine, not across machines), then checks the properties the inequality implies: every estimate $\leq \ln N$, every estimate $<$ truth, non-decreasing in $N$, std falling from $N = 2$ to 1024, the ceiling first clearing the truth at $N = 16$, and the bilinear critic looser than the full one at $N = 128$ |
| Tested on | Apple M5 Max, macOS 27.0.1, Python 3.12.14, numpy 2.4.4 (2026-10-11): `selftest` passes and all ten estimates are bit-identical to the pre-computed curve (same machine, same numpy). On a 2-core Linux container (Python 3.13, numpy 2.5.3) `selftest` also passes, with 18 of 30 numbers bit-identical and a largest difference of $8.9 \times 10^{-16}$ |

## Run it

```bash
cd demos && ./setup.sh && cd w05_d3_infonce_bound
./present.sh selftest
./present.sh rehearse         # non-interactive, class order 1 2 3 b → runs/rehearsal/ (json, txt, two png)
./present.sh                  # = show: single-key menu (h menu, q quit)
./present.sh replay           # print the rehearsal transcript; the figures are runs/rehearsal/*.png
```

## What you should see

From `runs/rehearsal/rehearsal.json` (2026-10-11, Apple M5 Max; truth 2.6934 nats):

- `2`: estimates $N$=2 **0.561**, 4 1.093, 8 **1.543** (ceiling 2.079), 16 1.846, 32 2.134, 64 2.339, 128 2.482, 256 2.563, 512 2.602, 1024 **2.652**. Every estimate is below its $\ln N$ and below the truth; 64 → 1024 adds 0.31 nat.
- `3`: std over 200 batches 0.346 at $N = 2$, 0.237 at 32, 0.061 at 1024; at $N = 1024$ a quarter of the single batches land above the truth while their mean stays below it.
- `b`: bilinear-only estimates 1.294 / 1.869 / 1.944 at $N = 8$ / 128 / 1024 against 1.543 / 2.482 / 2.652 with the extra term; the learned $a \approx -0.48$ to $-0.53$ against $a^{*} = -0.480$.

## Files

| File | |
|---|---|
| `demo.py` | `show` / `rehearse` / `replay` / `selftest` |
| `demo_config.toml` | seed, $d$, $\rho$, the $N$ list, steps, learning rate, held-out batches; which $N$ the histograms and the bilinear comparison use; display digits |
| `present.sh` | entry point; uses `../.venv/bin/python` |
| `runs/rehearsal/rehearsal.json` | the truth table; per-$N$ estimate, $\mathcal{L}_N$, std, gaps, learned $a$, seconds, and all 200 per-batch estimates; the variance and bilinear tables; versions and date |
| `runs/rehearsal/rehearsal.txt` | the terminal output without colour codes |
| `runs/rehearsal/ceiling.png`, `variance.png` | the two figures |

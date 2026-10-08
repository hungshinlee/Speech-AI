<!-- 此檔由 bin/publish_demos.py 從 private repo 的 <course>/demos/ 複製而來，請勿在公開 repo 直接編輯。 -->
# W3 · Demo 2 — CTC paths by brute force vs. the forward table (slides p24, p31, p36)

Pure `numpy`, CPU, terminal only — no window, no audio, no microphone. One program with three parts, each a single key:

| Part | Keys | What it does | Slide |
|---|---|---|---|
| 1 | `1` `2` `3` `0` | lists every path that $\mathcal{B}$ collapses to a label: "aa" at $T = 3$ (1 path), "aba" at $T = 3$ (1), "aa" at $T = 4$ (5), "aa" at $T = 2$ (**none** — the empty set, since two identical labels need a blank between them). In class you write them on paper first, then the program lists them | p24 |
| 1 | `x` | all $K^T$ candidate strings for the last question and what each collapses to, ending with the ones that *look* right but collapse wrong ("aa∅" → "a", "∅aa" → "a") — the "collapse repeats first, then delete blanks" order matters | p24 |
| 2 | `4` | a random $\mathbf{y}^t$ (one softmax of $N(0, 1.5^2)$ logits per frame) for "ab", $T = 4$: prints the matrix, the 15 valid paths out of 81 candidates, each path's product, and the sum | p31 |
| 2 | `5` | the same $\mathbf{y}^t$ through the forward recursion, **revealed one column at a time** (any key for the next column, `a` for all, `q` to stop); every cell shows which $\alpha_{t-1}$ it adds and marks the blocked third term `[×]`; at the end $\alpha_T(S) + \alpha_T(S-1)$ against the brute-force sum, and their difference | p31 |
| 2 | `r` | draw a new random $\mathbf{y}^t$ (seed from the clock) and repeat `4` `5` — proof that nothing was arranged | |
| 3 | `6` | $T = 500$, vocabulary $K = 32$, label "hello": the forward recursion **in the linear domain**. Lists $\max_s \alpha_t(s)$ at checkpoints, marks the first $t$ where it becomes subnormal and where the whole column is 0; ends with $P = 0.0$, loss $= \infty$ | p31, p36 |
| 3 | `l` | the same in the **log domain** (`np.logaddexp`): $\ln P$ is finite, the loss is a few thousand nats; converting $\ln P$ back to linear still gives 0.0 | p36 |

What the three parts add up to: $P(\ell \mid \mathbf{X})$ is one number, obtainable either by summing over every path in $\mathcal{B}^{-1}(\ell)$ (exponentially many) or by recursion on a $T \times S$ lattice ($O(TS)$), and the two agree to floating-point error; the zero-probability condition has a concrete shape; and linear-domain forward underflows at a few hundred frames while the log domain does not.

Why part 3 uses 32 symbols rather than 3: with $K = 3$ and a uniform $\mathbf{y}$, $3^{-500} = 10^{-238.6}$ is still a float64 (the smallest subnormal is $5 \times 10^{-324}$), so nothing underflows. To see a column go to zero, each frame's probability has to be of order 0.1, i.e. a vocabulary of a few dozen — 32 is the vocabulary of the model in Demo 3. `6` prints this comparison on its last line.

## Definitions

| | |
|---|---|
| path $\pi$ | an element of $\mathcal{V}'^{\,T}$, in `itertools.product` order (the same order as the slide figures, so the lists match the slides line by line) |
| $\mathcal{B}$ | collapse adjacent repeats, then delete blanks |
| forward | $\alpha_t(s) = y^t_{\ell'_s}\bigl(\alpha_{t-1}(s) + \alpha_{t-1}(s-1) + [\ell'_s \neq \varnothing \wedge \ell'_s \neq \ell'_{s-2}]\,\alpha_{t-1}(s-2)\bigr)$, with $\alpha_1(1) = y^1_\varnothing$, $\alpha_1(2) = y^1_{\ell_1}$, and $P = \alpha_T(S) + \alpha_T(S-1)$ — exactly the slide's recursion |
| random $\mathbf{y}^t$ | `np.random.default_rng(seed)`, logits $\sim N(0, 1.5^2)$, softmax; seed 20260926 in `demo_config.toml`, so `rehearse` and `show` give the same numbers |
| $K = 32$ vocabulary | the label's letters first, filler symbols `s1`, `s2`, …, blank last |

## Requirements

| | |
|---|---|
| Packages | `numpy 2.4.4` (the shared `.venv`; nothing else is imported by the program) |
| Hardware | anything; part 3 is a pure-Python double loop over $500 \times 11$ cells and takes well under a second |
| Terminal | at least ≈ 110 columns for the per-cell detail of `5`; lower `display.digits` in `demo_config.toml` if it wraps. Column-by-column reveal uses `termios` single-key reads and falls back to printing everything when piped |
| Cross-check | `selftest` compares the path lists and forward values against `slides/assets/w03/w03-data.json` at the repository root |
| Tested on | Apple M5 Max, macOS 26.6, Python 3.12.14 (2026-09-26, 2026-10-05) |

## Run it

```bash
cd demos && ./setup.sh && cd w03_d2_paths_vs_forward
./present.sh selftest         # paths and forward values vs. w03-data.json; random y: enumeration = forward = log domain; T = 500 underflows / does not
./present.sh rehearse         # non-interactive: everything in class order → runs/rehearsal/
./present.sh                  # = show: single-key menu (keys above; h menu, q quit)
./present.sh replay           # print the rehearsal transcript
```

Terminal labels are in Chinese; blank is printed as ∅.

## What you should see

From `runs/rehearsal/rehearsal.json` (2026-09-26, M5 Max; identical to the last digit on Linux, since the seed is fixed and `default_rng` is platform-independent):

- `4` / `5`: brute-force sum **0.1201518366618488**, forward $P = 0.12015183666184881$, absolute difference $1.4 \times 10^{-17}$ — one unit in the last place.
- `6`: linear-domain forward on $T = 500$, $K = 32$: the largest $\alpha_t(s)$ becomes subnormal at $t = 174$ and the whole column is zero from $t = 182$; $P = 0.0$.
- `l`: $\ln P = -2127.73$, i.e. a loss of about 2128 nats — finite.

`rehearsal.txt` is the full terminal output and is what `replay` prints.

## Files

| File | |
|---|---|
| `demo.py` | `show` / `rehearse` / `replay` / `selftest` |
| `demo_config.toml` | the four paper questions, the random-$\mathbf{y}$ settings and seed, the $T = 500$ settings and checkpoints, display digits |
| `present.sh` | entry point; uses `../.venv/bin/python` |
| `runs/rehearsal/rehearsal.json` | path lists, the random matrix, the 15 path products and their sum, the full forward table, $P$, the difference, the $T = 500$ checkpoints, $\ln P$, loss; versions and date |
| `runs/rehearsal/rehearsal.txt` | the terminal output without colour codes |

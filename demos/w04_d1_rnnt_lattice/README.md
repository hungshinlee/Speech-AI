<!-- 此檔由 bin/publish_demos.py 從 private repo 的 <course>/demos/ 複製而來，請勿在公開 repo 直接編輯。 -->
# W4 · Demo 1 — the RNN-T lattice: paths by brute force vs. forward, and `max_symbols_per_step` (slides p29, p31, p33, p34)

Standard library only, CPU, terminal only — no window, no audio. The W3 demo again, one week later: $P(\ell \mid \mathbf{X})$ for a transducer is still one number that can be reached either by summing every path or by a recursion on a lattice, and the two agree. What changed is the lattice — a $T \times (U + 1)$ grid where **moving up costs no frame** — so $T = 1$ already has a path, repeated labels need no blank between them, and CTC's floor $T \geq U + r$ is gone. The third part separates two things students merge: the training sum has no per-frame cap on symbols; the decoder's `max_symbols_per_step` is an engineering limit.

| Part | Keys | What it does | Slide |
|---|---|---|---|
| 1 | `1` `2` `3` `0` | every RNN-T path for "ab" at $T = 2$ (3 paths), "aa" at $T = 1$ (1 path; CTC has none), "ab" at $T = 3$ (6; CTC 5), "aa" at $T = 2$ (3; CTC 0). Each path as its action string (E = emit, go up; R = blank, go right), the nodes $(t, u)$ it visits, and what each step means; several emits on one frame are highlighted. Counts are checked against $\binom{T+U-1}{U}$ and the CTC count is printed beside each (read from `w03-data.json` where available, enumerated otherwise) | p29 |
| 2 | `4` | a random $\mathbf{y}^{t,u}$ — one softmax over $\{a, b, \varnothing\}$ per node — for "ab", $T = 2$: the six nodes' probabilities, the four factors of each of the 3 paths, their products, and the sum | p33 |
| 2 | `5` | the same $\mathbf{y}^{t,u}$ through the forward recursion, **revealed one cell at a time** (any key for the next, `a` for all, `q` to stop); each cell shows which two $\alpha$'s times which numbers it adds and marks the missing term (`[t=1: nothing to the left]`, `[u=0: nothing below]`); then $\alpha(T,U)\,\varnothing(T,U)$ against the brute-force sum; then the backward table and the four anti-diagonals of $\sum \alpha\beta$ | p33, p34 |
| 2 | `r` | a new random $\mathbf{y}^{t,u}$ (seed from the clock), then `4` `5` again | |
| 3 | `6` | greedy decoding on a **synthetic** joint network with `max_symbols_per_step = 5`: frame 3 has a badly calibrated $\varnothing$; the decoder emits five symbols there and is forced to a blank. The last line connects the cap back to part 2: the share of $P$ carried by paths that emit more than once on a frame — all of them are in the training sum | p31 |
| 3 | `u` | the same joint with the cap removed: on frame 3, $\varnothing$ never wins and the decoder alternates a / b until the program's safety limit (40) | p31 |

## Definitions

| | |
|---|---|
| path | from $(1, 0)$ to $(T, U)$: an interleaving of $T - 1$ R's and $U$ E's, closed with one more R; `itertools.combinations` order, the same as the slide figures and `w04-data.json` |
| forward | $\alpha(t,u) = \alpha(t-1,u)\,\varnothing(t-1,u) + \alpha(t,u-1)\,y(t,u-1)$, $\alpha(1,0) = 1$, $P = \alpha(T,U)\,\varnothing(T,U)$ (Graves 2012, eqs. 16–17) |
| backward, diagonals | $\beta(t,u) = \beta(t+1,u)\,\varnothing(t,u) + \beta(t,u+1)\,y(t,u)$, $\beta(T,U) = \varnothing(T,U)$; $\sum_{t+u=n}\alpha\beta = P$ for every $n$ (eqs. 18–19) |
| random $\mathbf{y}^{t,u}$ | three `random.gauss(0, 1.5)` logits per node, softmax, in the order $(a, b, \varnothing)$; emit uses $\ell_{u+1}$'s entry. With seed 20261004 the draws follow the slide-figure script's order, so the numbers match the slide table to the last digit |
| the part-3 joint | **synthetic, not a model**: $\mathbf{z}^{t,u} = \mathbf{f}_t + \mathbf{g}(\text{previous symbol})$, symbol logits $\sim N(0, 1.5^2)$, $\varnothing$'s logit set to (second-highest symbol + 0.5), $\mathbf{g}$ subtracts 3 from the symbol just emitted; on frame 3 only, $\varnothing$'s logit is (lowest symbol − 8). It isolates the one `if` that the cap is — `6` and `u` differ in that line alone |

Why part 3 is synthetic: the point is that the cap is a property of the decoder, not of the model, and a joint built by hand makes the cap's effect visible without the noise of a real network. The screen says it is synthetic. Real transducers do hit this — the cap exists because $\varnothing$ is poorly calibrated early in training and on out-of-domain audio — but how often is not something this demo measures.

## Requirements

| | |
|---|---|
| Packages | none beyond the standard library for `demo.py` itself; `../common.py` imports numpy and scipy, so use the shared `.venv` |
| Terminal | ≈ 100 columns for the factor lines of `4` and the per-cell detail of `5`; lower `display.digits` if it wraps. Cell-by-cell reveal uses `termios` and falls back to printing everything when piped |
| Cross-checks | `selftest` compares path lists, counts, CTC counts, uniform and random forward values and the diagonals against `slides/assets/w04/w04-data.json` (and `w03-data.json`) at the repository root, to $10^{-15}$ |
| Tested on | Apple M5 Max, macOS 27.0.1, Python 3.12.14 (2026-10-06); identical digits to a Linux pre-run |

## Run it

```bash
cd demos && ./setup.sh && cd w04_d1_rnnt_lattice
./present.sh selftest
./present.sh rehearse         # non-interactive, class order → runs/rehearsal/
./present.sh                  # = show: single-key menu (h menu, q quit)
./present.sh replay           # print the rehearsal transcript
```

## What you should see

From `runs/rehearsal/rehearsal.json` (2026-10-06):

- `4` / `5`: brute-force sum $= P = $ **0.008173862970686905**, difference $0.0$; all four anti-diagonals $\sum\alpha\beta$ equal $P$.
- `6` (cap 5): hypothesis `"bababab"`, emits per frame $[1, 1, 5, 0, 0, 0]$ — five on frame 3, then the forced blank.
- `u` (no cap): frame 3 runs to the safety limit of 40 emits; the terminal shows the first six and a summary.

## Files

| File | |
|---|---|
| `demo.py` | `show` / `rehearse` / `replay` / `selftest` |
| `demo_config.toml` | the four paper questions, random-$\mathbf{y}$ seed, the synthetic joint's parameters and cap, display digits |
| `present.sh` | entry point; uses `../.venv/bin/python` |
| `runs/rehearsal/rehearsal.json` | path lists with CTC counts, the six nodes' probabilities, path factors and products, $\alpha$ and $\beta$ tables, $P$, difference, diagonals, both decodes step by step; versions and date |
| `runs/rehearsal/rehearsal.txt` | the terminal output without colour codes |

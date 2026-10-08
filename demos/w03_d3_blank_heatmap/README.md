<!-- 此檔由 bin/publish_demos.py 從 private repo 的 <course>/demos/ 複製而來，請勿在公開 repo 直接編輯。 -->
# W3 · Demo 3 — a blank heat-map, and `ctc_loss` going to `inf` (slides p39, p42, p45)

One LibriSpeech sentence through a character-level CTC model (`facebook/wav2vec2-base-960h`, 94.4 M parameters, 32-symbol vocabulary), run on the CPU with `transformers`. Three parts, keys pressed in the plot window:

| Part | Key | What it shows | Slide |
|---|---|---|---|
| 1 | `1` | plays the sentence | p39 |
| 1 | `2` | the posterior heat-map: the 31 non-blank symbols on top (one column per 20 ms frame), the argmax row in the middle, and **the blank row enlarged** at the bottom, $P(\varnothing \mid t)$; non-speech frames shaded grey (energy threshold) | p39 |
| 1 | `3` | four numbers, printed and written across the top of the window: $T$; $U$ (word boundaries included), $r$, $U + r$; frames where blank is the argmax; frames **inside speech** where blank is the argmax. Also the greedy decode against the reference | p39 |
| 2 | `4` | `torch.nn.functional.ctc_loss` of the correct transcript against these logits (batch of one, `reduction="sum"`, so the value is $-\ln P(\ell \mid \mathbf{X})$), next to a numpy log-domain forward on $\ell'$ — Demo 2's recursion, the same number; finite gradient | p45 |
| 2 | `5` | the transcript repeated after itself and lengthened character by character until $U + r = T$ (loss finite but huge) and then $U + r = T + 1$: loss `inf`, gradient `nan` (`zero_infinity=False`). The condition is exact | p45 |
| 2 | `6` | the same with `zero_infinity=True`: loss `0`, gradient all zero — hidden, not fixed | p45 |
| extra | `7` | Viterbi on $\ell'$ (the forward recursion with $\sum$ replaced by $\max$, written in numpy): the spike frame of every character, each word from its first to its last spike, how many frames wide a spike is; also calls `torchaudio.functional.forced_align` and compares frame by frame | p42, p43 |
| extra | `a` | overlays the Viterbi path on the heat-map (green boxes) | |
| | `h` `q` | menu, quit | |

The window opens empty: nothing is shown until `2`, because the question comes first — *in a sentence with no pauses, on what fraction of frames will blank win?*

What the three parts establish: in a sentence spoken without pauses, blank is the argmax on about half of all frames and on **40 % of the frames inside speech** — blank is not silence, it is "no new symbol on this frame"; the same logits give a finite loss for the right transcript and `inf` the moment $U + r$ exceeds $T$ by one; and `zero_infinity` turns that `inf` into a silent zero with a zero gradient.

## Definitions

| | |
|---|---|
| $T$ | frames out of the model's convolutional front-end (stride 320 samples = 20 ms, receptive field 25 ms); `check` verifies $(L - 400)/320 + 1$ against the actual $T$ |
| $U$ | label length **including the word-boundary symbol `\|`** — in this vocabulary the space is a symbol, so "MISTER QUILTER" has one more label than it has letters. The letters-only count is printed too: one more example of "the vocabulary is decided before the loss" |
| $r$ | adjacent repeats in $\ell$ (DD, SS, …); $U + r$ is the shortest valid path |
| blank | `config.pad_token_id` (the Hugging Face convention: `<pad>` = id 0 = blank), read from the model, not hard-coded |
| speech frames | frame energy above (peak − 35 dB), on the model's own 20 ms grid. An energy threshold, not a hand annotation; its exact value does not change the conclusion |
| loss | `ctc_loss(log_softmax(logits), targets, [T], [U], blank, reduction="sum")`. The default `reduction="mean"` divides by $U$; the slides quote the sum |
| spike | the frame where the Viterbi path sits on label $u$'s state ($2u + 1$); a frame's time is its centre, $(20t + 12.5)$ ms |

## Requirements

| | |
|---|---|
| Packages | shared `.venv` plus this folder's `./setup.sh`: `torch 2.14.0`, `torchaudio 2.11.0`, `transformers 5.17.0`, `soundfile 0.14.0` (versions in `../versions.lock`) |
| Model | `facebook/wav2vec2-base-960h`, downloaded by `fetch` into the Hugging Face cache and pinned to the commit in `model.lock`; `present.sh` then runs with `HF_HUB_OFFLINE=1` |
| Audio | **checked in**: `audio/utterance.wav` + `.txt` — LibriSpeech dev-clean `1272-128104-0000` (CC BY 4.0), 5.855 s, "MISTER QUILTER IS THE APOSTLE OF THE MIDDLE CLASSES …". `fetch` would otherwise download the 337 MB `dev-clean.tar.gz` and extract this one file. Another sentence: change `utterance` in `demo_config.toml`, or `--wav path.wav` with the text in a `.txt` of the same name |
| Device | CPU (`demo_config.toml`): sub-second inference, deterministic, the same numbers as the rehearsal. `check` also measures `mps` (inference and `ctc_loss` forward/backward against CPU) |
| Tested on | Apple M5 Max, macOS 26.6, Python 3.12.14 (2026-09-26; keys walked through again 2026-10-05) |

## Run it

```bash
cd demos && ./setup.sh && cd w03_d3_blank_heatmap
./setup.sh                    # torch / torchaudio / transformers / soundfile into the shared .venv
./present.sh fetch            # the model → Hugging Face cache, commit → model.lock (needs network; the audio is already here)
./present.sh check            # versions, parameter count, T from stride vs. actual, MPS vs. CPU, forced_align present
./present.sh rehearse         # everything non-interactively → runs/rehearsal/
./present.sh                  # = show: load, compute logits, open the empty window; keys above
./present.sh replay           # same window and keys from runs/rehearsal/logits.npy — no torch
./present.sh fake             # no model: a synthetic posterior, screen marked FAKE
```

Menus and labels are in Chinese. The keys are pressed in the plot window (it needs focus); the terminal only logs.

## What you should see

From `runs/rehearsal/rehearsal.json` (2026-09-26):

| | |
|---|---|
| $T$ | **292** frames (5.855 s at 20 ms) |
| $U$ / letters only / $r$ / $U + r$ | **89** / 73 / 2 / **91** → 3.3 frames per label |
| blank is the argmax | **149 / 292 frames = 51 %** |
| speech frames (energy) | 232 |
| blank is the argmax inside speech | **94 / 232 = 41 %** (median blank probability inside speech 0.014 — i.e. on the frames where it does not win, it loses decisively) |
| greedy decode | equals the reference |
| `4` loss, correct transcript | **53.950** (torch) vs. 53.950 (numpy forward on $\ell'$); gradient finite |
| `5` loss at $U + r = T = 292$ | 4626.98 — finite, gradient finite |
| `5` loss at $U + r = 293$ | **`inf`**, gradient contains `nan` |
| `6` same, `zero_infinity=True` | **0.0**, gradient all zero |
| `7` | Viterbi spikes 1–2 frames wide; `torchaudio.functional.forced_align` is present in 2.11.0 and returns the identical path (0 frames differ) |

`check.json` (not published) also recorded: 94,396,320 parameters; $T$ predicted from the stride = 292 = actual; MPS inference 9.9 ms vs. CPU 73 ms with a maximum logit difference of 0.004; `ctc_loss` on MPS matches CPU in forward and backward.

## Files

| File | |
|---|---|
| `demo.py` | `show` / `replay` / `rehearse` / `check` / `fetch` / `fake`; `--wav` |
| `demo_config.toml` | model and device, utterance id, speech-energy threshold, how the transcript is lengthened for `5`, optional hand-marked word boundaries for `7`, display |
| `present.sh`, `setup.sh` | entry point (offline) and this demo's package install |
| `model.lock` | the model commit `fetch` pinned |
| `audio/utterance.wav`, `audio/utterance.txt` | the LibriSpeech sentence |
| `runs/rehearsal/logits.npy` | $T \times 32$ logits (float32) — what `replay` reads instead of running the model |
| `runs/rehearsal/utterance.wav`, `utterance.txt`, `vocab.json` | the sentence, text and vocabulary as used at rehearsal (W4 demos 2 and 3 read these) |
| `runs/rehearsal/rehearsal.json`, `rehearsal.txt`, `heatmap.png` | the four numbers, the three losses, the alignment (every character's first/last spike — W4 reuses it), versions, model commit; the terminal transcript; the figure with the Viterbi overlay |

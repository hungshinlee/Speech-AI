<!-- 此檔由 bin/publish_demos.py 從 private repo 的 <course>/demos/ 複製而來，請勿在公開 repo 直接編輯。 -->
# Classroom demos — code and rehearsal records

These are the programs the instructor runs live in class. **You are not required to run anything**: the course is lectures only, and every number you see on the slides comes from the rehearsal records checked in here (`runs/rehearsal/` in each folder). The code is published so that you can read it, re-run it, and change it if you want to.

Each week's page on the course site has a *Demos* section that says what each demo shows and links to its folder here. Each folder has its own `README.md` with requirements, setup, how to run it, and what you should see.

## Layout

```
demos/
  common.py                 shared helpers: terminal colours, WAV read/write, microphone capture, playback, single-key input
  dsp.py                    the signal processing used by W2 and W3: STFT, least-squares ISTFT, the original Griffin-Lim, two mel scales, F0, LPC formants
  selftest.py               checks dsp.py against synthetic signals — run it once after changing package versions
  setup.sh                  builds the shared .venv (Python 3.12) and pins versions (see below)
  versions.lock             pinned versions of every package the demos import, and the mlx-lm git commit
  requirements.lock.txt     full `pip freeze` of the instructor's environment, for reference
  wNN_dK_<name>/            one demo per folder: week NN, demo K (the "Demo K" in the slide footers)
    demo.py                 the program
    present.sh              entry point; model demos run offline (HF_HUB_OFFLINE=1)
    demo_config.toml        parameters, sentences, thresholds — change these, not the code
    setup.sh                (some demos) installs that demo's extra packages into the shared .venv
    model.lock              (model demos) the exact model revision `fetch` pinned
    audio/                  (some demos) the input audio, when it is public material
    runs/rehearsal/         the instructor's rehearsal record: the numbers on the slides
    README.md               what it shows, requirements, how to run, what to expect
```

## The machine these were tested on

**Everything here was run on one machine only: a MacBook Pro with an Apple M5 Max (64 GB unified memory, no CUDA), macOS 26.6 / 27.0, Python 3.12.** The per-demo READMEs say which demos are portable and which are not:

| Needs | Demos | Runs on |
|---|---|---|
| standard library only | W4 demo 1 | anything with Python 3.11+ (`common.py` still imports numpy/scipy) |
| `numpy` + `scipy` + `matplotlib`, CPU | W3 demo 2 (terminal only) | any laptop |
| the above + `sounddevice` + a microphone + a window | W2 demo 1, W3 demo 1 (live capture); W2 demo 2 (`record`) | any laptop with a working audio input; playback uses macOS `afplay` |
| `torch 2.14.0` + `torchaudio` + `transformers 5.17.0` + a small Hugging Face model | W3 demo 3 (wav2vec2-base-960h, CPU), W4 demo 3 (whisper-small, CPU) | any laptop with a few GB free RAM and a network connection for the first download; `mps` is measured by `check` but not used in class |
| `sherpa-onnx` + an ONNX model | W4 demo 2 | any platform sherpa-onnx ships wheels for; tested on Apple silicon only |
| `mlx` + `mlx-lm` + `mlx-audio` + `pywhispercpp`, ≈ 15 GB of models | W1 demo 2 | **Apple silicon only**, 32 GB or more |

Cross-folder dependencies to know about: W4 demos 2 and 3 read the LibriSpeech sentence and its CTC forced alignment from `w03_d3_blank_heatmap/runs/rehearsal/`; W3 demo 2 and W4 demo 1 check themselves against `slides/assets/w03/w03-data.json` and `slides/assets/w04/w04-data.json` at the repository root; W1 demo 2 reads `slides/assets/w01/w01-metrics.json`. Keep the folder layout as it is.

Nothing has been tested on Linux, Windows, or a CUDA GPU. Where a README says "should also work on …", read that as a statement of what the code *does not* depend on, not as something that was verified.

## Setting up

### Apple-silicon Mac (same path as the instructor)

```bash
cd demos
./setup.sh                 # creates demos/.venv with numpy, scipy, matplotlib, sounddevice pinned from versions.lock; runs selftest.py (needs `uv`: brew install uv)
```

Then follow the per-demo README. Demos that need a model have their own `./setup.sh` (installs that demo's packages into the shared `.venv`) and a `./present.sh fetch` step.

### Anything else (Linux, Windows/WSL, Intel Mac)

The shared `setup.sh` itself only installs numpy, scipy, matplotlib and sounddevice, so it should work anywhere `uv` does — but it has not been run anywhere but the Mac. By hand:

```bash
cd demos
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python numpy==2.4.4 scipy==1.17.1 matplotlib==3.10.9 sounddevice==0.5.6
```

(`pip` works too: `python3.12 -m venv .venv && .venv/bin/pip install …`.) The `present.sh` scripts look for `../.venv/bin/python`, so the folder has to be called `.venv` and live directly under `demos/`. W1 demo 2 needs `mlx`, which only exists for Apple silicon; its `./present.sh fake` walks through the flow with synthetic signals on any machine.

Playback in `common.py` uses macOS `afplay` when it exists and falls back to `sounddevice` elsewhere; the fallback has not been exercised. The WAVs are in the folders — any player opens them.

The versions are the ones in `versions.lock`; the code was not tested with any other.

## Models and audio

Models are **not** in this repository. They are downloaded by each demo's `fetch` command (or `setup.sh`) into the Hugging Face cache (`~/.cache/huggingface/hub`, or wherever `HF_HOME` points), the whisper.cpp cache, or `~/.cache/sherpa-onnx`, and the exact revision is pinned in that demo's `model.lock` or in `versions.lock`. `present.sh` runs everything with `HF_HUB_OFFLINE=1`, so after the first `fetch` nothing touches the network.

Audio: the one LibriSpeech sentence the W3 and W4 model demos use (dev-clean `1272-128104-0000`, CC BY 4.0) is checked in under `w03_d3_blank_heatmap/`. **The instructor's own recordings are not published** — the five-second sentence behind W2 demo 2 and the two takes behind W3 demo 1. Those two demos keep their rehearsal *numbers* and figures here; to hear them you record your own input with `./present.sh record` (W2 demo 2) or `./present.sh rehearse` (W3 demo 1) — each README says how.

## Two things to know about the output

- Prompts and labels printed to the terminal, and the menus in the plot windows, are in Chinese (the course is taught in Chinese; slides are in English). The keys are the same everywhere: a demo's README lists them; **q** quits, **h** prints the menu where there is one.
- Every `replay` command shows the rehearsal record, not a live run, and the screen says so. Numbers from a `replay` are the ones on the slides; numbers from a live run on your machine will differ in timing — and for the microphone demos, in everything, because the input is your voice.

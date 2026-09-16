<!-- 此檔由 scripts/build_weeks.py 從課程大綱過濾產生，請勿直接編輯。大綱正本在 private repo（$COURSE_OUTLINE），改完請重跑腳本。 -->

## Toolchain

**You need no compute for this course.** It is lectures throughout, and every demo is run live in class on an Apple-silicon laptop with 64 GB of unified memory and **no CUDA**. That last constraint is what decides most of this table — and it decides it more sharply for speech than it would for a text-only course, because a good deal of the classical ASR training stack is written against CUDA kernels.

| Purpose | First choice | Notes |
|:------------|:----------------|:--------------|
| ASR, most mature path | **`whisper.cpp`** | Apple silicon is a first-class target: inference runs on the GPU through Metal, and Core ML can put the encoder on the Neural Engine. The first run after install is slow while Core ML compiles |
| The MLX speech stack | **`mlx-audio`** | The actively developed one — text-to-speech, speech-to-text, separation, enhancement and VAD in a single package |
| NVIDIA's ASR models, on a Mac | `parakeet-mlx` | Parakeet with a streaming API. Worth noting that the models run here even though the framework they came from does not |
| Signal processing and metrics | `librosa`, `torchaudio`, `jiwer`, `pesq`, `pystoi` | Hardware-independent. Note that `torchaudio` is in maintenance mode: `load()` and `save()` are now TorchCodec, and `sox_effects` is gone |
| Speech modules (VAD, separation, speaker) | SpeechBrain, pyannote.audio, Silero VAD | Silero is fast enough on CPU that the question does not arise. pyannote 4 needs `ffmpeg` installed |
| Codec | Mimi | EnCodec (2022) and Descript Audio Codec (2023) are both frozen; Mimi is the one still moving, and it is what the full-duplex models in W13 use |
| TTS | `F5-TTS`, or the TTS models in `mlx-audio` | Weight download is the real delay — warm it up before class |
| **Full duplex** | **`moshi-mlx`** | This is the reason W12 and W13 can be shown live rather than played from a recording. The PyTorch build of the same model needs a 24 GB GPU and has no quantization, so on this machine MLX is not the fallback — it is the only path |

### What cannot be shown on this hardware

| Not available | Why |
|:-----------------|:--------|
| `bitsandbytes`, `vLLM`, FlashAttention, custom Triton kernels | CUDA only |
| NeMo | Built on NVIDIA's stack; its requirements begin with an NVIDIA GPU |
| GPU inference in `faster-whisper` | It is built on CTranslate2, whose GPU backends are CUDA and ROCm only. It runs here, but on the CPU — the "Apple Accelerate" it advertises is a CPU math library, not GPU acceleration. That distinction is worth holding on to: **"Apple silicon acceleration" means at least four different things** — Metal, the Neural Engine through Core ML, PyTorch's MPS backend, and Accelerate — and a tool that supports one of them supports none of the others by default |
| CosyVoice | No macOS or Apple-silicon support in its documentation |

> **The limitation is part of the material.** Saying in class that a demo cannot run here, and naming which kernel is missing, makes the point better than a slide can: in speech, what a system can do is bound to what it runs on. That is the latency axis, stated concretely.

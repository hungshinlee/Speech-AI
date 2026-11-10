#!/usr/bin/env bash
# 此檔由 bin/publish_demos.py 從 private repo 的 <course>/demos/ 複製而來，請勿在公開 repo 直接編輯。
# W8 的 TTS demo 用的環境：speech_ai/demos/.venv-tts（與共用的 ../.venv 分開）。W8 的 d2、d3 也用這一個。
#
#   ./setup.sh            建（或更新）../.venv-tts，裝釘住的版本，寫 ../versions-tts.lock 與 ../requirements-tts.lock.txt
#
# 為什麼另開一個環境：f5-tts 1.1.22 照它的文件裝（pip install f5-tts）會帶進 cached-path，而 cached-path 要 rich<14、filelock<4；
# 共用的 ../.venv 釘的是 rich 15.0.0、filelock 4.0.1，十幾個 demo 靠它，不為一個 demo 降版。W1 d2 的 moshi-mlx（釘 mlx<0.27）
# 也是同一個理由另裝在 uv tool 的環境裡。代價是 torch 多裝一份（約 1 GB）。
#
# 版本政策：
#   - 與共用環境相同的釘從 ../versions.lock 讀：Python、numpy、scipy、matplotlib、sounddevice、soundfile、torch、torchaudio、transformers——兩個環境同一版。
#   - f5-tts 釘 1.1.22（speech_ai/CLAUDE.md §6 查證過、本 demo 建置時讀過原始碼的版本）。
#   - datasets 給下限 5：不給的話 uv 會為了配合 fsspec 2026.9 選到 2020 年的 datasets 1.1.1，而 f5_tts.model 在 import 時就讀它。
#   - sentencepiece（SpeechT5 的 tokenizer）、vocos、x-transformers、torchdiffeq、datasets：第一次跑時取解出來的版本並鎖進 ../versions-tts.lock，
#     之後都用鎖定的那一版；要升級就刪掉那一行再跑。完整清單在 ../requirements-tts.lock.txt。
set -euo pipefail
cd "$(dirname "$0")"
DEMOS=".."
ENV="${DEMOS}/.venv-tts"
LOCK="${DEMOS}/versions-tts.lock"
command -v uv >/dev/null || { echo "需要 uv：brew install uv" >&2; exit 1; }
[[ -f "${DEMOS}/versions.lock" ]] || { echo "找不到 ${DEMOS}/versions.lock：先跑 ${DEMOS}/setup.sh 與 ${DEMOS}/w03_d3_blank_heatmap/setup.sh" >&2; exit 1; }

base() { grep -E "^$1=" "${DEMOS}/versions.lock" | cut -d= -f2; }
own()  { grep -E "^$1=" "${LOCK}" 2>/dev/null | cut -d= -f2 || true; }
spec() { local v; v="$(own "$2")"; [[ -n "$v" ]] && echo "$1==${v}" || echo "$3"; }

F5_TTS_VERSION="1.1.22"
for k in PYTHON_VERSION NUMPY_VERSION SCIPY_VERSION MATPLOTLIB_VERSION SOUNDDEVICE_VERSION SOUNDFILE_VERSION TORCH_VERSION TORCHAUDIO_VERSION TRANSFORMERS_VERSION; do
  [[ -n "$(base "$k")" ]] || { echo "${DEMOS}/versions.lock 裡沒有 ${k}" >&2; exit 1; }
done

[[ -x "${ENV}/bin/python" ]] || uv venv --python "$(base PYTHON_VERSION)" "${ENV}"
uv pip install --python "${ENV}/bin/python" \
  "numpy==$(base NUMPY_VERSION)" \
  "scipy==$(base SCIPY_VERSION)" \
  "matplotlib==$(base MATPLOTLIB_VERSION)" \
  "sounddevice==$(base SOUNDDEVICE_VERSION)" \
  "soundfile==$(base SOUNDFILE_VERSION)" \
  "torch==$(base TORCH_VERSION)" \
  "torchaudio==$(base TORCHAUDIO_VERSION)" \
  "transformers==$(base TRANSFORMERS_VERSION)" \
  "f5-tts==${F5_TTS_VERSION}" \
  "$(spec datasets DATASETS_VERSION 'datasets>=5')" \
  "$(spec sentencepiece SENTENCEPIECE_VERSION sentencepiece)" \
  "$(spec vocos VOCOS_VERSION vocos)" \
  "$(spec x-transformers X_TRANSFORMERS_VERSION x-transformers)" \
  "$(spec torchdiffeq TORCHDIFFEQ_VERSION torchdiffeq)"

ver() { "${ENV}/bin/python" -c "from importlib.metadata import version; print(version('$1'))"; }
{
  echo "# 由 demos/w08_d1_mean_vs_sample/setup.sh 產生（../.venv-tts）。與 versions.lock 相同的釘從那裡讀；要換版本就刪掉對應那一行再重跑。"
  {
    for k in PYTHON_VERSION NUMPY_VERSION SCIPY_VERSION MATPLOTLIB_VERSION SOUNDDEVICE_VERSION SOUNDFILE_VERSION TORCH_VERSION TORCHAUDIO_VERSION TRANSFORMERS_VERSION; do
      echo "${k}=$(base "$k")"
    done
    echo "F5_TTS_VERSION=${F5_TTS_VERSION}"
    echo "DATASETS_VERSION=$(ver datasets)"
    echo "SENTENCEPIECE_VERSION=$(ver sentencepiece)"
    echo "VOCOS_VERSION=$(ver vocos)"
    echo "X_TRANSFORMERS_VERSION=$(ver x-transformers)"
    echo "TORCHDIFFEQ_VERSION=$(ver torchdiffeq)"
  } | LC_ALL=C sort
} > "${LOCK}.tmp" && mv "${LOCK}.tmp" "${LOCK}"
uv pip freeze --python "${ENV}/bin/python" > "${DEMOS}/requirements-tts.lock.txt"
echo
echo "→ ${LOCK}、${DEMOS}/requirements-tts.lock.txt"
"${ENV}/bin/python" demo.py selftest
echo
echo "下一步：./present.sh fetch（下載模型，要網路）→ ./present.sh check（聽 runs/check/ 的兩個 .wav）→ ./present.sh rehearse"

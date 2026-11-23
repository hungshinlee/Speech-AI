#!/usr/bin/env bash
# 此檔由 bin/publish_demos.py 從 private repo 的 <course>/demos/ 複製而來，請勿在公開 repo 直接編輯。
# W10 demo 3 多需要的一個環境：speech_ai/demos/.venv-diar，只裝 pyannote.audio（後半的 diarization）。
#
#   ./setup.sh            建（或更新）../.venv-diar，鎖版本到 ../versions-diar.lock 與 ../requirements-diar.lock.txt，跑 selftest
#
# 為什麼另開一個環境、不裝進 ../.venv-tts：
#   pyannote.audio 4.1.1 要 matplotlib ≥ 3.11，而 ../.venv 與 ../.venv-tts 釘的是 3.10.9（投影片的圖是用那一版產的）；
#   它另外還帶進 lightning、torchcodec、opentelemetry 等二十幾個套件。分開住，前幾週的 demo 一個版本都不會動。
#   demo.py 用子程序叫它（diar_worker.py），兩邊只交換一個 wav 與一份 json。
#
# 版本政策：Python、numpy、scipy、torch、torchaudio 從 ../versions.lock 讀（與另外兩個環境同一版）；
# pyannote.audio 第一次跑時取解出來的版本並鎖進 ../versions-diar.lock，之後都用鎖定的那一版；要升級就刪掉那一行再跑。
# 寫這支腳本時在 Linux 容器裡裝過的是 pyannote.audio 4.1.1（torch 2.14.0、torchaudio 2.11.0）；macOS arm64 上還沒有裝過。
#
# 這支腳本不處理的一件事（要人做）：模型在 Hugging Face 上要先同意使用條款、登入。見 README〈課前〉。
set -euo pipefail
cd "$(dirname "$0")"
DEMOS=".."
ENV="${DEMOS}/.venv-diar"
PY="${ENV}/bin/python"
BASE="${DEMOS}/versions.lock"
LOCK="${DEMOS}/versions-diar.lock"
HEADER="# 由 demos/w10_d3_whose_track/setup.sh 產生（../.venv-diar）。與 versions.lock 相同的釘從那裡讀；要換版本就刪掉對應那一行再重跑。"
command -v uv >/dev/null || { echo "需要 uv：brew install uv" >&2; exit 1; }
[[ -f "${BASE}" ]] || { echo "找不到 ${BASE}：先跑 ${DEMOS}/setup.sh" >&2; exit 1; }

base() { grep -E "^$1=" "${BASE}" | cut -d= -f2; }
lock() { grep -E "^$1=" "${LOCK}" 2>/dev/null | cut -d= -f2 || true; }
ver() { "${PY}" -c "from importlib.metadata import version; print(version('$1'))" 2>/dev/null || true; }

[[ -x "${PY}" ]] || uv venv --python "$(base PYTHON_VERSION)" "${ENV}"
PA="$(lock PYANNOTE_AUDIO_VERSION)"
if [[ -n "${PA}" ]]; then PA_SPEC="pyannote.audio==${PA}"; else PA_SPEC="pyannote.audio"; fi
uv pip install --python "${PY}" \
  "numpy==$(base NUMPY_VERSION)" "scipy==$(base SCIPY_VERSION)" \
  "torch==$(base TORCH_VERSION)" "torchaudio==$(base TORCHAUDIO_VERSION)" \
  "${PA_SPEC}"

{
  echo "${HEADER}"
  {
    echo "PYTHON_VERSION=$(base PYTHON_VERSION)"
    echo "NUMPY_VERSION=$(ver numpy)"
    echo "SCIPY_VERSION=$(ver scipy)"
    echo "TORCH_VERSION=$(ver torch)"
    echo "TORCHAUDIO_VERSION=$(ver torchaudio)"
    echo "MATPLOTLIB_VERSION=$(ver matplotlib)"
    echo "PYANNOTE_AUDIO_VERSION=$(ver pyannote.audio)"
    echo "PYANNOTE_METRICS_VERSION=$(ver pyannote.metrics)"
    echo "PYANNOTE_CORE_VERSION=$(ver pyannote.core)"
  } | LC_ALL=C sort
} > "${LOCK}.tmp" && mv "${LOCK}.tmp" "${LOCK}"
uv pip freeze --python "${PY}" > "${DEMOS}/requirements-diar.lock.txt"
echo
echo "→ pyannote.audio $(ver pyannote.audio)（matplotlib $(ver matplotlib)，只在這個環境裡）；${LOCK}、${DEMOS}/requirements-diar.lock.txt"
SELF="${DEMOS}/.venv-tts/bin/python"
[[ -x "${SELF}" ]] || SELF="${DEMOS}/.venv/bin/python"
"${SELF}" demo.py selftest
echo
echo "下一步（要人做，一次）：瀏覽器開 https://huggingface.co/pyannote/speaker-diarization-community-1 同意使用條款，"
echo "然後 ../.venv-tts/bin/hf auth login（貼一個 read 權限的 token）。再跑 ./present.sh fetch → ./present.sh check → ./present.sh rehearse"

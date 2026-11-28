#!/usr/bin/env bash
# 此檔由 bin/publish_demos.py 從 private repo 的 <course>/demos/ 複製而來，請勿在公開 repo 直接編輯。
# W11 的 speech-to-speech demo 多需要的一個環境：speech_ai/demos/.venv-sts（新版的 mlx-audio 與 mlx-lm）。
#
#   ./setup.sh            建（或更新）../.venv-sts，鎖版本到 ../versions-sts.lock 與 ../requirements-sts.lock.txt，跑 selftest
#
# 為什麼另開一個環境、不升級共用的 ../.venv：
#   Ming-3.0-audio-tiny 的 MLX 版要有 ming_v3_audio 模組的 mlx-audio，共用環境釘的 0.5.4 沒有；
#   新版另外要 mlx ≥ 0.32.4、較新的 transformers 與 huggingface_hub。前十週的 demo 都釘在共用環境的版本上，
#   分開住，那些 demo 一個版本都不會動。Kokoro 與 Whisper 仍由共用環境跑（aux_worker.py，子程序）。
#
# 版本政策：Python、numpy、scipy、sounddevice 從 ../versions.lock 讀（與其他環境同一版）；
# mlx-audio、mlx-lm、transformers 第一次裝下面寫的版本（寫這個 demo 時讀過原始碼、在 Linux 容器裡試過的那一組），
# mlx 與其餘相依取解出來的版本；全部鎖進 ../versions-sts.lock，之後都用鎖定的。要升級就刪掉那一行再跑。
set -euo pipefail
cd "$(dirname "$0")"
DEMOS=".."
ENV="${DEMOS}/.venv-sts"
PY="${ENV}/bin/python"
BASE="${DEMOS}/versions.lock"
LOCK="${DEMOS}/versions-sts.lock"
HEADER="# 由 demos/w11_d2_typed_vs_spoken/setup.sh 產生（../.venv-sts）。與 versions.lock 相同的釘從那裡讀；要換版本就刪掉對應那一行再重跑。"
command -v uv >/dev/null || { echo "需要 uv：brew install uv" >&2; exit 1; }
[[ -f "${BASE}" ]] || { echo "找不到 ${BASE}：先跑 ${DEMOS}/setup.sh" >&2; exit 1; }

base() { grep -E "^$1=" "${BASE}" | cut -d= -f2; }
lock() { grep -E "^$1=" "${LOCK}" 2>/dev/null | cut -d= -f2 || true; }
ver() { "${PY}" -c "from importlib.metadata import version; print(version('$1'))" 2>/dev/null || true; }
pin() { local v; v="$(lock "$2")"; v="${v:-${3:-}}"; if [[ -n "${v}" ]]; then echo "$1==${v}"; else echo "$1"; fi; }

[[ -x "${PY}" ]] || uv venv --python "$(base PYTHON_VERSION)" "${ENV}"
MA="$(lock MLX_AUDIO_VERSION)"; MA="${MA:-0.6.8}"
uv pip install --python "${PY}" \
  "numpy==$(base NUMPY_VERSION)" "scipy==$(base SCIPY_VERSION)" "sounddevice==$(base SOUNDDEVICE_VERSION)" \
  "mlx-audio[sts]==${MA}" "$(pin mlx-lm MLX_LM_VERSION 0.32.2)" "$(pin transformers TRANSFORMERS_VERSION 5.24.1)" \
  "$(pin mlx MLX_VERSION)" "$(pin huggingface_hub HUGGINGFACE_HUB_VERSION)"

{
  echo "${HEADER}"
  {
    echo "PYTHON_VERSION=$(base PYTHON_VERSION)"
    echo "NUMPY_VERSION=$(ver numpy)"
    echo "SCIPY_VERSION=$(ver scipy)"
    echo "SOUNDDEVICE_VERSION=$(ver sounddevice)"
    echo "MLX_VERSION=$(ver mlx)"
    echo "MLX_AUDIO_VERSION=$(ver mlx-audio)"
    echo "MLX_LM_VERSION=$(ver mlx-lm)"
    echo "TRANSFORMERS_VERSION=$(ver transformers)"
    echo "HUGGINGFACE_HUB_VERSION=$(ver huggingface_hub)"
  } | LC_ALL=C sort
} > "${LOCK}.tmp" && mv "${LOCK}.tmp" "${LOCK}"
uv pip freeze --python "${PY}" > "${DEMOS}/requirements-sts.lock.txt"
echo
echo "→ mlx-audio $(ver mlx-audio)、mlx-lm $(ver mlx-lm)、mlx $(ver mlx)、transformers $(ver transformers)；${LOCK}、${DEMOS}/requirements-sts.lock.txt"
"${PY}" demo.py selftest
echo
echo "下一步：./present.sh fetch（要網路；下載約 28 GB、轉出 θ0 約 8.5 GB）→ ./present.sh check → ./present.sh record own → ./present.sh rehearse"

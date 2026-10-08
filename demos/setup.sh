#!/usr/bin/env bash
# 此檔由 bin/publish_demos.py 從 private repo 的 <course>/demos/ 複製而來，請勿在公開 repo 直接編輯。
# speech_ai/demos 所有 demo 共用的環境建置。
#
#   ./setup.sh          建（或更新）共用的 .venv，裝釘住的版本
#
# 共用基底（W2 的 d1、d2）只需要 numpy、scipy、matplotlib、sounddevice，全部跑 CPU；
# 需要模型的 demo 各自有 setup.sh 往同一個 .venv 加套件並寫進 versions.lock（W1 d2 起）。
# **刻意不裝 librosa**：mel 的兩種尺度、最小平方 ISTFT、原始版 Griffin-Lim 都寫在 dsp.py 裡，
# 因為「前端的定義不只一種」本身就是 W2 要教的事；也省掉 numba 對 Python／numpy 版本的牽制。
# 之後需要模型的 demo（W2 d3 起）再往這裡加，工具的選擇見 speech_ai/CLAUDE.md §6。
#
# 版本政策：
#   - numpy、scipy、matplotlib 釘在 2026-09-17 實際測過 dsp.py 與兩個 demo 的版本。
#   - sounddevice 第一次跑時取當時的最新版並寫進 versions.lock，之後都用鎖定的那一版。
#     （寫這支腳本的環境連不到 PyPI，查不到它的最新版號，所以不憑印象釘。）
#     要升級就刪掉 versions.lock 裡對應那一行再跑。
set -euo pipefail
cd "$(dirname "$0")"

PYTHON_VERSION="3.12"
NUMPY_VERSION="2.4.4"
SCIPY_VERSION="1.17.1"
MATPLOTLIB_VERSION="3.10.9"

command -v uv >/dev/null || { echo "需要 uv：brew install uv" >&2; exit 1; }

SOUNDDEVICE_VERSION=""
if [[ -f versions.lock ]]; then
  SOUNDDEVICE_VERSION="$(grep -E '^SOUNDDEVICE_VERSION=' versions.lock | cut -d= -f2 || true)"
fi
SD_SPEC="sounddevice"
[[ -n "$SOUNDDEVICE_VERSION" ]] && SD_SPEC="sounddevice==${SOUNDDEVICE_VERSION}"

[[ -x .venv/bin/python ]] || uv venv --python "$PYTHON_VERSION" .venv
uv pip install --python .venv/bin/python \
  "numpy==${NUMPY_VERSION}" \
  "scipy==${SCIPY_VERSION}" \
  "matplotlib==${MATPLOTLIB_VERSION}" \
  "$SD_SPEC"

SOUNDDEVICE_VERSION="$(.venv/bin/python -c 'from importlib.metadata import version; print(version("sounddevice"))')"
{
  echo "# 由 demos/setup.sh 產生。要換版本就刪掉對應那一行再重跑。"
  {
    echo "MATPLOTLIB_VERSION=${MATPLOTLIB_VERSION}"
    echo "NUMPY_VERSION=${NUMPY_VERSION}"
    echo "PYTHON_VERSION=${PYTHON_VERSION}"
    echo "SCIPY_VERSION=${SCIPY_VERSION}"
    echo "SOUNDDEVICE_VERSION=${SOUNDDEVICE_VERSION}"
  } | LC_ALL=C sort
} > versions.lock
uv pip freeze --python .venv/bin/python > requirements.lock.txt
echo "完整的套件清單 → requirements.lock.txt"

echo
echo "自我檢查（合成訊號，不需要麥克風與喇叭）："
.venv/bin/python selftest.py
echo
echo "下一步：cd w02_d1_spectrogram && ./present.sh fake"

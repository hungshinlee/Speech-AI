#!/usr/bin/env bash
# 此檔由 bin/publish_demos.py 從 private repo 的 <course>/demos/ 複製而來，請勿在公開 repo 直接編輯。
# W10 demo 1 多需要的兩個套件，裝進 speech_ai/demos 共用的 .venv（先跑 ../setup.sh 與 ../w03_d3_blank_heatmap/setup.sh）。
#
#   ./setup.sh            裝 silero-vad、pyroomacoustics（不動環境裡其他套件的版本），更新 ../versions.lock 與 ../requirements.lock.txt，跑 selftest
#
# 為什麼要它們：
#   silero-vad        鍵 9／0 的 VAD（投影片 p15 介紹的那一個）。權重在 pip 套件裡，不必另外下載；
#                     這個 demo 用的是 torch JIT 那一份權重，不走 ONNX（容器裡裝的時候沒有帶進 onnxruntime）。
#   pyroomacoustics   模擬的回音路徑（image method）。真錄音出狀況時的對照，也是唯一畫得出 ‖h − ĥ‖ 的版本。沒裝時 demo 會退到一條合成的路徑並在畫面上標明。
#
# 版本政策：第一次跑時取解出來的版本並鎖進 ../versions.lock（SILERO_VAD_VERSION、PYROOMACOUSTICS_VERSION），之後都用鎖定的那一版；
# 要升級就刪掉那一行再跑。安裝時把環境裡現有的每個套件都當成 constraint，所以帶進來的只會是新套件，不會動 torch、transformers、numpy。
# 寫這支腳本時在 Linux 容器裡跑過的是 silero-vad 6.2.3、pyroomacoustics 0.10.1（torch 2.14.0、numpy 2.4.4）；macOS arm64 上還沒有裝過。
set -euo pipefail
cd "$(dirname "$0")"
DEMOS=".."
PY="${DEMOS}/.venv/bin/python"
LOCK="${DEMOS}/versions.lock"
command -v uv >/dev/null || { echo "需要 uv：brew install uv" >&2; exit 1; }
[[ -x "${PY}" ]] || { echo "先跑 ${DEMOS}/setup.sh 建共用環境" >&2; exit 1; }
"${PY}" -c "import torch, transformers" 2>/dev/null || { echo "先跑 ${DEMOS}/w03_d3_blank_heatmap/setup.sh 裝 torch／transformers" >&2; exit 1; }

lock() { grep -E "^$1=" "${LOCK}" 2>/dev/null | cut -d= -f2 || true; }
spec() { local v; v="$(lock "$2")"; [[ -n "${v}" ]] && echo "$1==${v}" || echo "$1"; }

cons="$(mktemp)"
trap 'rm -f "${cons}"' EXIT
uv pip freeze --python "${PY}" | grep -v -i -E '^(silero[-_]vad|pyroomacoustics|onnxruntime)==' > "${cons}"
uv pip install --python "${PY}" "$(spec silero-vad SILERO_VAD_VERSION)" "$(spec pyroomacoustics PYROOMACOUSTICS_VERSION)" -c "${cons}"

ver() { "${PY}" -c "from importlib.metadata import version; print(version('$1'))"; }
SV="$(ver silero-vad)"
PRA="$(ver pyroomacoustics)"
{
  grep -E '^#' "${LOCK}" || true
  {
    grep -E '^[A-Z0-9_]+=' "${LOCK}" | grep -vE '^(SILERO_VAD_VERSION|PYROOMACOUSTICS_VERSION)=' || true
    echo "SILERO_VAD_VERSION=${SV}"
    echo "PYROOMACOUSTICS_VERSION=${PRA}"
  } | LC_ALL=C sort
} > "${LOCK}.tmp" && mv "${LOCK}.tmp" "${LOCK}"
uv pip freeze --python "${PY}" > "${DEMOS}/requirements.lock.txt"
echo
echo "→ silero-vad ${SV}、pyroomacoustics ${PRA}；${LOCK}、${DEMOS}/requirements.lock.txt"
"${PY}" demo.py selftest
echo
echo "下一步：./present.sh fetch（要網路）→ ./present.sh check → ./present.sh record alone／talk／loud → ./present.sh rehearse"

#!/usr/bin/env bash
# 此檔由 bin/publish_demos.py 從 private repo 的 <course>/demos/ 複製而來，請勿在公開 repo 直接編輯。
# W10 demo 2 多需要的兩個套件：pystoi 與 pesq，裝進 W8 的 TTS 環境 speech_ai/demos/.venv-tts
# （W8 Demo 1 的 setup.sh 建的；W9 Demo 1 的 setup.sh 裝了這個 demo 也要用的 speechbrain）。
#
#   ./setup.sh            裝 pystoi、試著裝 pesq（不動環境裡其他套件的版本），更新 ../versions-tts.lock 與 ../requirements-tts.lock.txt，跑 selftest
#
# 為什麼要它們：投影片 p48「每個指標都是一位裁判」裡給人耳設計的那兩位。
#   pystoi   STOI，純 Python（只要 numpy、scipy）。
#   pesq     PESQ（ITU-T P.862 的 C 程式包一層）。PyPI 上只有原始碼，要在這台機器上編譯（Xcode Command Line Tools 的 clang）。
#            編不起來不擋路：那一欄留空，其他照跑。
#
# 版本政策：第一次跑時取解出來的版本並鎖進 ../versions-tts.lock（PYSTOI_VERSION、PESQ_VERSION），之後都用鎖定的那一版；
# 要升級就刪掉那一行再跑。安裝時把環境裡現有的每個套件都當成 constraint，所以不會動 torch、transformers、speechbrain、numpy。
# 寫這支腳本時在 Linux 容器裡裝過的是 pystoi 0.4.1、pesq 0.0.4（numpy 2.4.4）；macOS arm64 上還沒有裝過。
set -euo pipefail
cd "$(dirname "$0")"
DEMOS=".."
ENV="${DEMOS}/.venv-tts"
PY="${ENV}/bin/python"
LOCK="${DEMOS}/versions-tts.lock"
command -v uv >/dev/null || { echo "需要 uv：brew install uv" >&2; exit 1; }
[[ -x "${PY}" ]] || { echo "還沒有 ${ENV}：先跑 ${DEMOS}/w08_d1_mean_vs_sample/setup.sh" >&2; exit 1; }
[[ -f "${LOCK}" ]] || { echo "找不到 ${LOCK}：先跑 ${DEMOS}/w08_d1_mean_vs_sample/setup.sh" >&2; exit 1; }
"${PY}" -c "import speechbrain" 2>/dev/null || { echo "環境裡沒有 speechbrain：先跑 ${DEMOS}/w09_d1_cloning_leak/setup.sh" >&2; exit 1; }

lock() { grep -E "^$1=" "${LOCK}" 2>/dev/null | cut -d= -f2 || true; }
spec() { local v; v="$(lock "$2")"; [[ -n "${v}" ]] && echo "$1==${v}" || echo "$1"; }
ver() { "${PY}" -c "from importlib.metadata import version; print(version('$1'))" 2>/dev/null || true; }

cons="$(mktemp)"
trap 'rm -f "${cons}"' EXIT
uv pip freeze --python "${PY}" | grep -v -i -E '^(pystoi|pesq)==' > "${cons}"
uv pip install --python "${PY}" "$(spec pystoi PYSTOI_VERSION)" -c "${cons}"
if ! uv pip install --python "${PY}" "$(spec pesq PESQ_VERSION)" -c "${cons}"; then
  echo
  echo "pesq 編不起來（上面是編譯器的訊息）。不擋路：PESQ 那一欄會是空的。要它的話：xcode-select --install 之後再跑一次 ./setup.sh" >&2
fi

STOI_V="$(ver pystoi)"
PESQ_V="$(ver pesq)"
{
  head -n 1 "${LOCK}"
  {
    tail -n +2 "${LOCK}" | grep -v -E '^(PYSTOI_VERSION|PESQ_VERSION)=' || true
    echo "PYSTOI_VERSION=${STOI_V}"
    [[ -n "${PESQ_V}" ]] && echo "PESQ_VERSION=${PESQ_V}"
  } | LC_ALL=C sort
} > "${LOCK}.tmp" && mv "${LOCK}.tmp" "${LOCK}"
uv pip freeze --python "${PY}" > "${DEMOS}/requirements-tts.lock.txt"
echo
echo "→ pystoi ${STOI_V}、pesq ${PESQ_V:-（沒裝起來）}；${LOCK}、${DEMOS}/requirements-tts.lock.txt"
"${PY}" demo.py selftest
echo
echo "下一步：./present.sh fetch（要網路）→ ./present.sh check → ./present.sh rehearse"

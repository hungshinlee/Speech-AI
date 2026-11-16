#!/usr/bin/env bash
# 此檔由 bin/publish_demos.py 從 private repo 的 <course>/demos/ 複製而來，請勿在公開 repo 直接編輯。
# W9 demo 3 多需要的一個套件：audioseal，裝進 W8 的 TTS 環境 speech_ai/demos/.venv-tts（W8 Demo 1 的 setup.sh 建的）。
#
#   ./setup.sh            裝 audioseal（不動環境裡其他套件的版本），更新 ../versions-tts.lock 與 ../requirements-tts.lock.txt，跑 selftest
#
# 為什麼要它：後半的 watermark 是 AudioSeal（generator 與 detector 的模型定義在這個套件裡）。前半的偵測器 AASIST 不是套件，
# 是 fetch clone 下來的一個 repo（../_media/w09_d3/aasist/），只需要 torch。
#
# 版本政策：第一次跑時取解出來的版本並鎖進 ../versions-tts.lock（AUDIOSEAL_VERSION），之後都用鎖定的那一版；要升級就刪掉那一行再跑。
# 安裝時把環境裡現有的每個套件都當成 constraint，所以 audioseal 帶進來的只會是環境裡還沒有的套件（omegaconf、einops 多半已經在），
# 不會動 torch、transformers、f5-tts、speechbrain。
set -euo pipefail
cd "$(dirname "$0")"
DEMOS=".."
ENV="${DEMOS}/.venv-tts"
LOCK="${DEMOS}/versions-tts.lock"
command -v uv >/dev/null || { echo "需要 uv：brew install uv" >&2; exit 1; }
[[ -x "${ENV}/bin/python" ]] || { echo "還沒有 ${ENV}：先跑 ${DEMOS}/w08_d1_mean_vs_sample/setup.sh" >&2; exit 1; }
[[ -f "${LOCK}" ]] || { echo "找不到 ${LOCK}：先跑 ${DEMOS}/w08_d1_mean_vs_sample/setup.sh" >&2; exit 1; }

v="$(grep -E '^AUDIOSEAL_VERSION=' "${LOCK}" | cut -d= -f2 || true)"
spec="audioseal"
[[ -n "${v}" ]] && spec="audioseal==${v}"

cons="$(mktemp)"
trap 'rm -f "${cons}"' EXIT
uv pip freeze --python "${ENV}/bin/python" | grep -v -i -E '^audioseal==' > "${cons}"
uv pip install --python "${ENV}/bin/python" "${spec}" -c "${cons}"

ver="$("${ENV}/bin/python" -c "from importlib.metadata import version; print(version('audioseal'))")"
{
  head -n 1 "${LOCK}"
  { tail -n +2 "${LOCK}" | grep -v -E '^AUDIOSEAL_VERSION=' || true; echo "AUDIOSEAL_VERSION=${ver}"; } | LC_ALL=C sort
} > "${LOCK}.tmp" && mv "${LOCK}.tmp" "${LOCK}"
uv pip freeze --python "${ENV}/bin/python" > "${DEMOS}/requirements-tts.lock.txt"
echo
echo "→ audioseal ${ver}；${LOCK}、${DEMOS}/requirements-tts.lock.txt"
"${ENV}/bin/python" demo.py selftest
echo
echo "下一步：./present.sh fetch（要網路）→ ./present.sh check → ./present.sh record（選配）→ ./present.sh rehearse"

#!/usr/bin/env bash
# 此檔由 bin/publish_demos.py 從 private repo 的 <course>/demos/ 複製而來，請勿在公開 repo 直接編輯。
# W9 demo 1 多需要的一個套件：speechbrain，裝進 W8 的 TTS 環境 speech_ai/demos/.venv-tts（W8 Demo 1 的 setup.sh 建的）。
#
#   ./setup.sh            裝 speechbrain（不動環境裡其他套件的版本），更新 ../versions-tts.lock 與 ../requirements-tts.lock.txt，跑 selftest
#
# 為什麼要它：(a) 那條路的條件是 x-vector，而 SpeechT5 的 TTS 訓練時用的是 SpeechBrain 的 spkrec-xvect-voxceleb 那個 encoder 的向量
# （Hugging Face 文件的做法）。上週用的是別人事先算好的 CMU ARCTIC 向量；這週要從 prompt 現算，所以要那個 encoder。
#
# 版本政策：第一次跑時取解出來的版本並鎖進 ../versions-tts.lock（SPEECHBRAIN_VERSION），之後都用鎖定的那一版；要升級就刪掉那一行再跑。
# 安裝時把環境裡現有的每個套件都當成 constraint，所以 speechbrain 帶進來的只會是新套件（hyperpyyaml、ruamel-yaml），不會動 torch、transformers、f5-tts。
set -euo pipefail
cd "$(dirname "$0")"
DEMOS=".."
ENV="${DEMOS}/.venv-tts"
LOCK="${DEMOS}/versions-tts.lock"
command -v uv >/dev/null || { echo "需要 uv：brew install uv" >&2; exit 1; }
[[ -x "${ENV}/bin/python" ]] || { echo "還沒有 ${ENV}：先跑 ${DEMOS}/w08_d1_mean_vs_sample/setup.sh" >&2; exit 1; }
[[ -f "${LOCK}" ]] || { echo "找不到 ${LOCK}：先跑 ${DEMOS}/w08_d1_mean_vs_sample/setup.sh" >&2; exit 1; }

v="$(grep -E '^SPEECHBRAIN_VERSION=' "${LOCK}" | cut -d= -f2 || true)"
spec="speechbrain"
[[ -n "${v}" ]] && spec="speechbrain==${v}"

cons="$(mktemp)"
trap 'rm -f "${cons}"' EXIT
uv pip freeze --python "${ENV}/bin/python" | grep -v -i -E '^(speechbrain|hyperpyyaml|ruamel[-.]yaml(-clib)?)==' > "${cons}"
uv pip install --python "${ENV}/bin/python" "${spec}" -c "${cons}"

ver="$("${ENV}/bin/python" -c "from importlib.metadata import version; print(version('speechbrain'))")"
{
  head -n 1 "${LOCK}"
  { tail -n +2 "${LOCK}" | grep -v -E '^SPEECHBRAIN_VERSION=' || true; echo "SPEECHBRAIN_VERSION=${ver}"; } | LC_ALL=C sort
} > "${LOCK}.tmp" && mv "${LOCK}.tmp" "${LOCK}"
uv pip freeze --python "${ENV}/bin/python" > "${DEMOS}/requirements-tts.lock.txt"
echo
echo "→ speechbrain ${ver}；${LOCK}、${DEMOS}/requirements-tts.lock.txt"
"${ENV}/bin/python" demo.py selftest
echo
echo "下一步：./present.sh fetch（要網路）→ ./present.sh check → ./present.sh record mood／accent → ./present.sh rehearse"

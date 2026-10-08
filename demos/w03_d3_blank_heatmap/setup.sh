#!/usr/bin/env bash
# 此檔由 bin/publish_demos.py 從 private repo 的 <course>/demos/ 複製而來，請勿在公開 repo 直接編輯。
# W3 demo 3 的模型套件，裝進 speech_ai/demos 共用的 .venv（先跑 ../setup.sh 建好基底）。
#
#   ./setup.sh            裝 torch／torchaudio／transformers／soundfile，版本寫進 ../versions.lock
#
# 版本政策（speech_ai/CLAUDE.md §6；根 CLAUDE.md「模型與套件版本只在 Mac 上實測過才寫死」）：
#   - torch 釘 2.14.0：§6 查證過的版本，「ctc_loss 在 MPS 上有了前向與反向」這句話是對這一版說的（2026-09-02 release blog）。
#     沒有這個釘，check 量到的 MPS 結果就不知道是對哪一版說的。
#   - torchaudio 不釘：§6 只知道「2.11、maintenance mode」，與 torch 2.14 配哪一版沒查到；讓 pip 依 torch 的釘解出相容版，
#     第一次跑的版本寫進 versions.lock，之後都用鎖定的那一版。forced_align 在不在由 ./present.sh check 回答。
#   - transformers、soundfile 不釘：寫這支腳本的環境連不到 PyPI，查不到現行版號，不憑印象釘；同樣第一次跑時鎖定。
#   - huggingface_hub 跟著 transformers 來。
# 要升級就刪掉 versions.lock 裡對應那一行再跑。
set -euo pipefail
cd "$(dirname "$0")"
DEMOS=".."
PY="$DEMOS/.venv/bin/python"
[[ -x "$PY" ]] || { echo "先跑 ${DEMOS}/setup.sh 建共用環境" >&2; exit 1; }
command -v uv >/dev/null || { echo "需要 uv：brew install uv" >&2; exit 1; }

TORCH_VERSION="2.14.0"

lock() { grep -E "^$1=" "$DEMOS/versions.lock" 2>/dev/null | cut -d= -f2 || true; }
spec() { local v; v="$(lock "$2")"; [[ -n "$v" ]] && echo "$1==$v" || echo "$1"; }

uv pip install --python "$PY" \
  "torch==${TORCH_VERSION}" \
  "$(spec torchaudio TORCHAUDIO_VERSION)" \
  "$(spec transformers TRANSFORMERS_VERSION)" \
  "$(spec soundfile SOUNDFILE_VERSION)"

ver() { "$PY" -c "from importlib.metadata import version; print(version('$1'))"; }
TORCHAUDIO_VERSION="$(ver torchaudio)"
TRANSFORMERS_VERSION="$(ver transformers)"
SOUNDFILE_VERSION="$(ver soundfile)"

{
  grep -E '^#' "$DEMOS/versions.lock" || true
  {
    grep -E '^[A-Z0-9_]+=' "$DEMOS/versions.lock" | grep -vE '^(TORCH_VERSION|TORCHAUDIO_VERSION|TRANSFORMERS_VERSION|SOUNDFILE_VERSION)=' || true
    echo "SOUNDFILE_VERSION=${SOUNDFILE_VERSION}"
    echo "TORCHAUDIO_VERSION=${TORCHAUDIO_VERSION}"
    echo "TORCH_VERSION=${TORCH_VERSION}"
    echo "TRANSFORMERS_VERSION=${TRANSFORMERS_VERSION}"
  } | LC_ALL=C sort
} > "$DEMOS/versions.lock.tmp" && mv "$DEMOS/versions.lock.tmp" "$DEMOS/versions.lock"
uv pip freeze --python "$PY" > "$DEMOS/requirements.lock.txt"
echo
echo "torch ${TORCH_VERSION} / torchaudio ${TORCHAUDIO_VERSION} / transformers ${TRANSFORMERS_VERSION} / soundfile ${SOUNDFILE_VERSION} → ${DEMOS}/versions.lock"
echo "下一步：./present.sh fetch（下載模型與那一句 LibriSpeech，要網路）→ ./present.sh check"

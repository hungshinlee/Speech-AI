#!/usr/bin/env bash
# 此檔由 bin/publish_demos.py 從 private repo 的 <course>/demos/ 複製而來，請勿在公開 repo 直接編輯。
# W4 demo 2 的套件，裝進 speech_ai/demos 共用的 .venv（先跑 ../setup.sh 建好基底）。
#
#   ./setup.sh            裝 sherpa-onnx 與 onnx，版本寫進 ../versions.lock
#
# 版本政策（根 CLAUDE.md「模型與套件版本只在 Mac 上實測過才寫死」）：
#   - sherpa-onnx 不釘：寫這支腳本的環境連不到 PyPI（2026-10-06），查不到現行版號；第一次跑時取最新版並鎖進 versions.lock，
#     之後都用鎖定的那一版。§6 查證過它明講支援 Apple Silicon（arm64 wheel）；Python API 的簽名 2026-10-06 從 GitHub 原始碼對過
#     （OnlineRecognizer.from_transducer、create_stream、accept_waveform、is_ready、decode_stream、tokens、timestamps）。
#   - onnx 不釘：只用來讀 encoder 的 metadata_props（decode_chunk_len、T）；沒裝也能跑，check 會改用 config 的預設值。
# 要升級就刪掉 versions.lock 裡對應那一行再跑。
set -euo pipefail
cd "$(dirname "$0")"
DEMOS=".."
PY="$DEMOS/.venv/bin/python"
[[ -x "$PY" ]] || { echo "先跑 ${DEMOS}/setup.sh 建共用環境" >&2; exit 1; }
command -v uv >/dev/null || { echo "需要 uv：brew install uv" >&2; exit 1; }

lock() { grep -E "^$1=" "$DEMOS/versions.lock" 2>/dev/null | cut -d= -f2 || true; }
spec() { local v; v="$(lock "$2")"; [[ -n "$v" ]] && echo "$1==$v" || echo "$1"; }

uv pip install --python "$PY" \
  "$(spec sherpa-onnx SHERPA_ONNX_VERSION)" \
  "$(spec onnx ONNX_VERSION)"

ver() { "$PY" -c "from importlib.metadata import version; print(version('$1'))"; }
SHERPA_ONNX_VERSION="$(ver sherpa-onnx)"
ONNX_VERSION="$(ver onnx)"

{
  grep -E '^#' "$DEMOS/versions.lock" || true
  {
    grep -E '^[A-Z0-9_]+=' "$DEMOS/versions.lock" | grep -vE '^(SHERPA_ONNX_VERSION|ONNX_VERSION)=' || true
    echo "ONNX_VERSION=${ONNX_VERSION}"
    echo "SHERPA_ONNX_VERSION=${SHERPA_ONNX_VERSION}"
  } | LC_ALL=C sort
} > "$DEMOS/versions.lock.tmp" && mv "$DEMOS/versions.lock.tmp" "$DEMOS/versions.lock"
uv pip freeze --python "$PY" > "$DEMOS/requirements.lock.txt"
echo
echo "sherpa-onnx ${SHERPA_ONNX_VERSION} / onnx ${ONNX_VERSION} → ${DEMOS}/versions.lock"
echo "下一步：./present.sh fetch（下載模型，要網路）→ ./present.sh check → ./present.sh rehearse"

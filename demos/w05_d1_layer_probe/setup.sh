#!/usr/bin/env bash
# 此檔由 bin/publish_demos.py 從 private repo 的 <course>/demos/ 複製而來，請勿在公開 repo 直接編輯。
# W5 demo 1 的額外套件，裝進 speech_ai/demos 共用的 .venv（先跑 ../setup.sh 建好基底，再跑 ../w03_d3_blank_heatmap/setup.sh 裝 torch／transformers／soundfile）。
#
#   ./setup.sh            裝 pyarrow（讀音素對齊的 parquet，只在 fetch 用到），版本寫進 ../versions.lock
#
# 版本政策（根 CLAUDE.md「模型與套件版本只在 Mac 上實測過才寫死」）：
#   - torch 2.14.0、transformers 5.17.0、soundfile 0.14.0 已由 W3 d3 的 setup.sh 鎖進 versions.lock，這裡不再裝、不再釘；缺的話先跑那支。
#   - pyarrow 不釘：寫這支腳本的環境連不到 PyPI，查不到現行版號；第一次跑時取最新版並寫進 versions.lock，之後都用鎖定的那一版。
# 要升級就刪掉 versions.lock 裡對應那一行再跑。
set -euo pipefail
cd "$(dirname "$0")"
DEMOS=".."
PY="$DEMOS/.venv/bin/python"
[[ -x "$PY" ]] || { echo "先跑 ${DEMOS}/setup.sh 建共用環境" >&2; exit 1; }
command -v uv >/dev/null || { echo "需要 uv：brew install uv" >&2; exit 1; }
"$PY" -c "import torch, transformers, soundfile" 2>/dev/null || { echo "缺 torch／transformers／soundfile：先跑 ${DEMOS}/w03_d3_blank_heatmap/setup.sh" >&2; exit 1; }

lock() { grep -E "^$1=" "$DEMOS/versions.lock" 2>/dev/null | cut -d= -f2 || true; }
spec() { local v; v="$(lock "$2")"; [[ -n "$v" ]] && echo "$1==$v" || echo "$1"; }

uv pip install --python "$PY" "$(spec pyarrow PYARROW_VERSION)"

PYARROW_VERSION="$("$PY" -c "from importlib.metadata import version; print(version('pyarrow'))")"
{
  grep -E '^#' "$DEMOS/versions.lock" || true
  {
    grep -E '^[A-Z0-9_]+=' "$DEMOS/versions.lock" | grep -vE '^PYARROW_VERSION=' || true
    echo "PYARROW_VERSION=${PYARROW_VERSION}"
  } | LC_ALL=C sort
} > "$DEMOS/versions.lock.tmp" && mv "$DEMOS/versions.lock.tmp" "$DEMOS/versions.lock"
uv pip freeze --python "$PY" > "$DEMOS/requirements.lock.txt"
echo
echo "pyarrow ${PYARROW_VERSION} → ${DEMOS}/versions.lock"
echo "下一步：./present.sh fetch（兩個模型、dev-clean 子集、音素對齊，要網路）→ ./present.sh check → ./present.sh extract → ./present.sh rehearse"

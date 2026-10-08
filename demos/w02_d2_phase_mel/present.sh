#!/usr/bin/env bash
# 此檔由 bin/publish_demos.py 從 private repo 的 <course>/demos/ 複製而來，請勿在公開 repo 直接編輯。
# W2 demo 2 課堂與課前的入口。
#
#   ./present.sh               上課用：現場重算一次，然後進入播放畫面
#   ./present.sh replay        現場失敗時：播課前彩排算好的版本（畫面標明非現場）
#   ./present.sh record        課前：錄一句話當素材（audio/sentence.wav）
#   ./present.sh rehearse      課前彩排：完整算一次並錄下退路，印出指標、產生兩張圖
#   ./present.sh fake          沒有素材也能演練流程（合成訊號，不是給學生聽的）
#
# 換素材：在子命令前加 --wav，例如  ./present.sh --wav audio/taigi.wav rehearse
# macOS 內建的是 bash 3.2：空陣列要寫成 ${opts[@]+"${opts[@]}"}，否則 set -u 會報錯。
set -euo pipefail
cd "$(dirname "$0")"
PY="../.venv/bin/python"   # 所有 demo 共用 speech_ai/demos/.venv
[[ -x "$PY" ]] || { echo "還沒有共用環境，先跑 ../setup.sh" >&2; exit 1; }

opts=()
while [[ $# -gt 0 && "$1" == --* ]]; do
  case "$1" in
    --wav) opts+=("$1" "$2"); shift 2 ;;
    *) echo "不認得的參數：$1" >&2; exit 2 ;;
  esac
done
cmd="${1:-show}"; shift || true
case "$cmd" in
  show|replay|record|rehearse) exec "$PY" demo.py ${opts[@]+"${opts[@]}"} "$cmd" "$@" ;;
  fake) exec "$PY" demo.py --fake show "$@" ;;
  *) echo "不認得的子命令：$cmd" >&2; sed -n '2,10p' "$0" >&2; exit 2 ;;
esac

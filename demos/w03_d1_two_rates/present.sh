#!/usr/bin/env bash
# 此檔由 bin/publish_demos.py 從 private repo 的 <course>/demos/ 複製而來，請勿在公開 repo 直接編輯。
# W3 demo 1 課堂與課前的入口。
#
#   ./present.sh               上課用：r 錄第 1 段（正常）、r 錄第 2 段（快一倍）、先問學生、再按 s 並排
#   ./present.sh rehearse      課前彩排：同上，但 s 會把兩段音、圖、數字存進 runs/rehearsal/（進版控 = 退路）
#   ./present.sh replay        現場收不到音時：載入 runs/rehearsal/ 的兩段，直接進並排畫面（畫面標明非現場）
#   ./present.sh replay DIR    載入別的資料夾（例如 runs/live/2026-…）
#   ./present.sh fake          沒有麥克風也能演練流程（合成的「句子」）
#   ./present.sh devices       列出音訊輸入裝置
#
# 畫面上的鍵：r 錄下一段，1／2 重錄某一段，s 並排，w／n 切窗長，p F0 點，z 頻率上限，q 離開。
set -euo pipefail
cd "$(dirname "$0")"
PY="../.venv/bin/python"   # 所有 demo 共用 speech_ai/demos/.venv
[[ -x "$PY" ]] || { echo "還沒有共用環境，先跑 ../setup.sh" >&2; exit 1; }

cmd="${1:-show}"; shift || true
case "$cmd" in
  show|rehearse|replay|fake|devices) exec "$PY" demo.py "$cmd" "$@" ;;
  *) echo "不認得的子命令：$cmd" >&2; sed -n '2,11p' "$0" >&2; exit 2 ;;
esac

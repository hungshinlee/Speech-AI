#!/usr/bin/env bash
# 此檔由 bin/publish_demos.py 從 private repo 的 <course>/demos/ 複製而來，請勿在公開 repo 直接編輯。
# W1 demo 2 課堂與課前的入口。
#
#   ./present.sh               上課用：先讓學生預測 → 跑 10 遍 → 印預算表、開時間軸圖
#   ./present.sh replay        現場失敗時：畫課前彩排的結果（畫面標明非現場）
#   ./present.sh record        課前：錄那句話（中間停 700 ms）→ audio/utterance.wav
#   ./present.sh endpoint      只看各門檻在句中／句尾的行為（Misconception 3 的實證）
#   ./present.sh check         課前體檢：套件、模型、暖機
#   ./present.sh rehearse      課前彩排：跑 10 遍，存 runs/rehearsal/utterance/
#   ./present.sh moshi         啟動 moshi-mlx 即時全雙工（戴耳機）
#   ./present.sh devices       列出音訊裝置
#   ./present.sh fake          沒有 Mac 也能演練整個流程（合成訊號＋fake 後端）
#
# 換素材：在子命令前加 --wav，例如  ./present.sh --wav audio/taigi.wav rehearse
# macOS 內建的是 bash 3.2：空陣列要寫成 ${opts[@]+"${opts[@]}"}，否則 set -u 會報錯。
set -euo pipefail
cd "$(dirname "$0")"
PY="../.venv/bin/python"
[[ -x "$PY" ]] || { echo "還沒有共用環境，先跑 ../setup.sh，再跑 ./setup.sh 裝模型套件" >&2; exit 1; }

opts=()
while [[ $# -gt 0 && "$1" == --* ]]; do
  case "$1" in
    --wav) opts+=("$1" "$2"); shift 2 ;;
    *) echo "不認得的參數：$1" >&2; exit 2 ;;
  esac
done
cmd="${1:-show}"; shift || true
case "$cmd" in
  show|replay|record|endpoint|check|rehearse|moshi|devices) exec "$PY" demo.py ${opts[@]+"${opts[@]}"} "$cmd" "$@" ;;
  fake) exec "$PY" demo.py --fake show "$@" ;;
  *) echo "不認得的子命令：$cmd" >&2; sed -n '2,14p' "$0" >&2; exit 2 ;;
esac

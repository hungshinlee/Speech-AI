#!/usr/bin/env bash
# 此檔由 bin/publish_demos.py 從 private repo 的 <course>/demos/ 複製而來，請勿在公開 repo 直接編輯。
# W2 demo 1 課堂與課前的入口。
#
#   ./present.sh               上課用：麥克風即時 spectrogram + pitch track + 母音圖
#   ./present.sh rehearse      課前彩排：同上，按 1／2／3 把畫面上這段音存成退路
#   ./present.sh replay vowels 現場收不到音時：載入退路（vowels／tones／question），畫面凍結但照樣能切窗長
#   ./present.sh fake          沒有麥克風也能演練流程（合成母音）
#   ./present.sh devices       列出音訊輸入裝置
#   ./present.sh praat vowels  用 Praat 開同一段退路音檔（精確讀數用）
#
# 畫面上的鍵：w／n 切窗長，space 凍結，p pitch，z 頻率上限，c 清軌跡，s 存檔，q 離開。
set -euo pipefail
cd "$(dirname "$0")"
PY="../.venv/bin/python"   # 所有 demo 共用 speech_ai/demos/.venv
[[ -x "$PY" ]] || { echo "還沒有共用環境，先跑 ../setup.sh" >&2; exit 1; }

cmd="${1:-show}"; shift || true
case "$cmd" in
  show|rehearse|replay|fake|devices) exec "$PY" demo.py "$cmd" "$@" ;;
  praat)
    wav="$PWD/runs/rehearsal/${1:-vowels}.wav"
    [[ -f "$wav" ]] || { echo "找不到 $wav，先 ./present.sh rehearse 並按 1／2／3" >&2; exit 1; }
    PRAAT="/Applications/Praat.app/Contents/MacOS/Praat"
    [[ -x "$PRAAT" ]] || { echo "找不到 Praat：brew install --cask praat" >&2; exit 1; }
    exec "$PRAAT" --send "$PWD/w02_d1.praat" "$wav" 75 400
    ;;
  *) echo "不認得的子命令：$cmd" >&2; sed -n '2,11p' "$0" >&2; exit 2 ;;
esac

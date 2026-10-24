#!/usr/bin/env bash
# 此檔由 bin/publish_demos.py 從 private repo 的 <course>/demos/ 複製而來，請勿在公開 repo 直接編輯。
# W5 demo 3 課堂與課前的入口。純 numpy、跑 CPU、終端機加一張 matplotlib 圖（沒有音訊）。
#
#   ./present.sh               上課用：單鍵選單。1 真值與天花板表，2 掃 N（逐個訓 critic、圖上逐點加），3 變異直方圖，
#                              b 純雙線性對照，r 換 seed，h 選單，q 離開（離開時存 runs/live/<時間>/，不進版控）
#   ./present.sh rehearse      課前彩排：不互動，照課堂順序 1 2 3 b 跑一遍，存 runs/rehearsal/（json、txt、兩張 png；進版控 = 退路）
#   ./present.sh replay        現場 Python 出狀況時：印出彩排的純文字輸出，圖用 runs/rehearsal/*.png
#   ./present.sh selftest      同一個 seed 重跑掃描，對 slides/assets/w05/w05-data.json → infonce_ceiling 逐一比對，再查不等式的性質
#   ./present.sh show --no-plot   只在終端機，不開圖
#
# 課堂上的講法與按鍵順序寫在投影片 p15 講稿的【操作】。
set -euo pipefail
cd "$(dirname "$0")"
PY="../.venv/bin/python"   # 所有 demo 共用 speech_ai/demos/.venv（numpy、matplotlib；common.py 要 scipy）
[[ -x "$PY" ]] || { echo "還沒有共用環境，先跑 ../setup.sh" >&2; exit 1; }

cmd="${1:-show}"; shift || true
case "$cmd" in
  show|rehearse|replay|selftest) exec "$PY" demo.py "$cmd" "$@" ;;
  *) echo "不認得的子命令：${cmd}" >&2; sed -n '2,10p' "$0" >&2; exit 2 ;;
esac

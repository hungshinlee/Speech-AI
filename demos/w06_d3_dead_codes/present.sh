#!/usr/bin/env bash
# 此檔由 bin/publish_demos.py 從 private repo 的 <course>/demos/ 複製而來，請勿在公開 repo 直接編輯。
# W6 demo 3 課堂與課前的入口。純 numpy、跑 CPU、終端機加一張 matplotlib 圖（沒有音訊、不碰模型）。
#
#   ./present.sh               上課用：單鍵選單。1 random+gradient，2 random+EMA，3 k-means init，4 EMA+reset，5 FSQ，
#                              i init 移進資料裡，t 投影片的表，r 換 seed，h 選單，q 離開（離開時存 runs/live/<時間>/，不進版控）
#   ./present.sh rehearse      課前彩排：不互動，照課堂順序 1 2 3 4 5 i t 跑一遍，存 runs/rehearsal/（json、txt、panels.png；進版控 = 退路）
#   ./present.sh replay        現場 Python 出狀況時：印出彩排的純文字輸出，圖用 runs/rehearsal/panels.png
#   ./present.sh selftest      同一個 seed 重跑五種設定，對 slides/assets/w06/w06-data.json → dead_codes 逐一比對，再查該有的性質
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

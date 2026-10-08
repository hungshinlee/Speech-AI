#!/usr/bin/env bash
# 此檔由 bin/publish_demos.py 從 private repo 的 <course>/demos/ 複製而來，請勿在公開 repo 直接編輯。
# W4 demo 1 課堂與課前的入口。純標準庫、跑 CPU、只在終端機裡（沒有視窗、沒有音訊）。
#
#   ./present.sh               上課用：單鍵選單。1／2／3／0 列 RNN-T 路徑（並排 CTC 個數），4 窮舉、5 forward 表逐格，r 換一組 y，
#                              6 greedy 有上限、u 拿掉上限，h 選單，q 離開（離開時存 runs/live/<時間>/，不進版控）
#   ./present.sh rehearse      課前彩排：不互動，照課堂順序全部跑一遍，存 runs/rehearsal/（進版控 = 退路）
#   ./present.sh replay        現場 Python 出狀況時：印出彩排的純文字輸出
#   ./present.sh selftest      對 slides/assets/w04/w04-data.json 逐條、逐位比對；隨機 y 窮舉 vs. forward vs. log 域；decode 有／沒上限
#
# 課堂上的講法與按鍵順序寫在投影片講稿的【操作】（p29 Paths、p33 Check the recursion、p31 Misconception 5）。
set -euo pipefail
cd "$(dirname "$0")"
PY="../.venv/bin/python"   # 所有 demo 共用 speech_ai/demos/.venv（common.py 要 numpy／scipy）
[[ -x "$PY" ]] || { echo "還沒有共用環境，先跑 ../setup.sh" >&2; exit 1; }

cmd="${1:-show}"; shift || true
case "$cmd" in
  show|rehearse|replay|selftest) exec "$PY" demo.py "$cmd" "$@" ;;
  *) echo "不認得的子命令：${cmd}" >&2; sed -n '2,9p' "$0" >&2; exit 2 ;;
esac

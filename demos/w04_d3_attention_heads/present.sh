#!/usr/bin/env bash
# 此檔由 bin/publish_demos.py 從 private repo 的 <course>/demos/ 複製而來，請勿在公開 repo 直接編輯。
# W4 demo 3 課堂與課前的入口。一律離線（HF_HUB_OFFLINE=1），避免教室網路出狀況時卡在連線。
#
#   ./present.sh               上課用：載入 Whisper small（eager）、轉錄、teacher-forced 拿 144 個 head 的 cross-attention，開視窗；
#                              鍵在視窗裡按：1 播音＋轉錄、2 這一層 12 個 head、3 下一層、0 全部 144 個、4 框出 OpenAI 的 alignment heads、
#                              5 DTW 與字的時間戳（p45），a 疊 W3 的 forced alignment（p46），h 選單，q 離開（存 runs/live/<時間>/，不進版控）
#   ./present.sh rehearse      課前彩排：不互動，全部跑一遍，存 runs/rehearsal/（進版控 = 現場的退路）
#   ./present.sh replay        現場模型出狀況時：從 runs/rehearsal/ 讀回，同一個視窗、同樣的鍵，不碰 torch
#   ./present.sh check         課前體檢：版本、模型在不在快取、eager 回傳權重／sdpa 不回傳、base85 對 HF 的 alignment_heads、
#                              自己的 DTW 對 transformers 的 return_token_timestamps、MPS 對 CPU → runs/check.json
#   ./present.sh fetch         課前一週：下載 openai/whisper-small（約 1 GB；鎖 commit → model.lock；要網路）。音訊重用 W3 Demo 3 的
#   ./present.sh fake          沒有模型也能演練流程（合成的 144 個 head，畫面標明 FAKE）
#   ./present.sh selftest      不需要模型：base85 解碼、DTW、中值濾波、分類判準、併字、字錯數、fake 整條流程
#   ./present.sh export        rehearsal.json 的摘要 → slides/assets/w04/w04-metrics.json（demo3）
#   ./present.sh recompute     改了判準之後：從 runs/rehearsal/ 已存的權重重算分類、DTW、圖與 rehearsal.json，不碰模型
#
# 模型套件（torch 2.14.0、transformers 5.17.0）已由 w03_d3_blank_heatmap/setup.sh 裝進共用 .venv；這個 demo 不另裝東西。
set -euo pipefail
cd "$(dirname "$0")"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
PY="../.venv/bin/python"   # 所有 demo 共用 speech_ai/demos/.venv
[[ -x "$PY" ]] || { echo "還沒有共用環境，先跑 ../setup.sh，再跑 ../w03_d3_blank_heatmap/setup.sh 裝 torch／transformers" >&2; exit 1; }

cmd="${1:-show}"; shift || true
case "$cmd" in
  show|rehearse|replay|check|fake|selftest|export|recompute) exec "$PY" demo.py "$cmd" "$@" ;;
  fetch) HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=0 exec "$PY" demo.py fetch "$@" ;;
  *) echo "不認得的子命令：${cmd}" >&2; sed -n '2,15p' "$0" >&2; exit 2 ;;
esac

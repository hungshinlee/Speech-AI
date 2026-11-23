#!/usr/bin/env bash
# 此檔由 bin/publish_demos.py 從 private repo 的 <course>/demos/ 複製而來，請勿在公開 repo 直接編輯。
# W10 demo 1 課堂與課前的入口。一律離線（HF_HUB_OFFLINE=1），避免教室網路出狀況時卡在連線。
#
#   ./present.sh               上課用：讀 audio/real_*.wav（沒有就只有模擬），算好全部版本，視窗裡單鍵操作：
#                              1 麥克風 y、2 y − x、3 對齊再減、4 NLMS 從零、5 插話＋固定步長、6 ＋Geigel、7 ＋軟步長、8 失真的天花板、
#                              9 VAD 與 Whisper（AEC 之前）、0（之後）、y／e／a 播、c 程式碼、t 表、m 真錄音／模擬互換、h 選單、q 離開
#                              （離開時存 runs/live/<時間>/，不進版控；--sim 只用模擬、--no-plot 鍵在終端機按）
#   ./present.sh record alone  錄真的回音，三段依序：alone（只有系統在說）、talk（你插話，畫面會提示）、loud（音量開到最大）
#   ./present.sh record talk     → audio/real_<名字>.wav。要在教室現場做就是上課前（或當場）跑這三行，再開 ./present.sh
#   ./present.sh record loud
#   ./present.sh record user   （選配）錄一句乾淨的插話給模擬用 → audio/user.wav（不進版控）
#   ./present.sh rehearse      課前彩排：不互動，真錄音與模擬各跑一遍（含 VAD 與 Whisper），存 runs/rehearsal/（進版控 = 現場的退路）
#   ./present.sh replay        現場出狀況時：從 runs/rehearsal/ 讀回，同樣的鍵，不錄音、不算、不碰 torch
#   ./present.sh check         課前體檢：版本、音訊裝置、播錄延遲三次、輸入路徑上有沒有系統自帶的處理、NLMS 的速度、VAD、Whisper → runs/check.json
#   ./present.sh fetch         課前一週：下載 Whisper small（W7 Demo 1 的同一個 commit；要網路）
#   ./present.sh make          重做模擬的三段（平常不必跑）
#   ./present.sh selftest      不需要模型與麥克風：NLMS 的恆等式、三種步長、整條流程
#   ./present.sh export        runs/rehearsal/rehearsal.json 的摘要 → slides/assets/w10/w10-metrics.json（demo1）
#
# 這個 demo 多兩個套件（silero-vad、pyroomacoustics），由本資料夾的 ./setup.sh 裝進共用 .venv。
set -euo pipefail
cd "$(dirname "$0")"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_DISABLE_PROGRESS_BARS=1
export TOKENIZERS_PARALLELISM=false
PY="../.venv/bin/python"   # 所有 demo 共用 speech_ai/demos/.venv
[[ -x "$PY" ]] || { echo "還沒有共用環境，先跑 ../setup.sh、../w03_d3_blank_heatmap/setup.sh（torch／transformers），再跑 ./setup.sh" >&2; exit 1; }

cmd="${1:-show}"; shift || true
case "$cmd" in
  show|rehearse|replay|record|check|make|selftest|export) exec "$PY" demo.py "$cmd" "$@" ;;
  fetch) HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=0 exec "$PY" demo.py fetch "$@" ;;
  *) echo "不認得的子命令：${cmd}" >&2; sed -n '2,21p' "$0" >&2; exit 2 ;;
esac

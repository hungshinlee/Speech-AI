#!/usr/bin/env bash
# 此檔由 bin/publish_demos.py 從 private repo 的 <course>/demos/ 複製而來，請勿在公開 repo 直接編輯。
# W7 demo 1 課堂與課前的入口。一律離線（HF_HUB_OFFLINE=1），避免教室網路出狀況時卡在連線。
#
#   ./present.sh               上課用：載入 Whisper small 與 wav2vec2-base-960h，終端機單鍵操作：
#                              1 靜音、2 白噪音×3、3 W3 那句+長靜音（兩個視窗）、4 自己錄的非語音、w CTC 對照、
#                              z encoder 歸零、Z 換平均向量、f 切 fallback、c 切 condition_on_prev、l 切語言、s W3 那句對照、h 選單、q 離開
#                              （離開時存 runs/live/<時間>/，不進版控）
#   ./present.sh rehearse      課前彩排：不互動，全部跑一遍，存 runs/rehearsal/（進版控 = 現場的退路）
#   ./present.sh replay        現場模型出狀況時：從 runs/rehearsal/ 讀回，同樣的鍵，不碰 torch
#   ./present.sh check         課前體檢：版本、兩個模型在不在快取、<|nospeech|> 的 id、encoder 歸零的 hook、MPS 對 CPU → runs/check.json
#   ./present.sh compare       §6 第 8 項：transformers／pywhispercpp／mlx-whisper 三方對比 → runs/compare.json（不是課堂上的鍵）
#   ./present.sh fetch         課前一週：下載兩個模型（commit 讀 W4 d3、W3 d3 的 model.lock；要網路）；--mlx 另抓 mlx-community/whisper-small-mlx
#   ./present.sh make          產生音訊 audio/（靜音、三個 SNR 的白噪音、W3 那句+長靜音；確定性，不進版控）
#   ./present.sh record        錄 10 秒自己的非語音 → audio/own.wav（不進版控、不上站）
#   ./present.sh fake          沒有模型也能演練流程（罐頭輸出，畫面標明 FAKE）
#   ./present.sh selftest      不需要模型：壓縮比、fallback 判定、SNR、CTC 摺疊、fake 整條流程
#   ./present.sh export        runs/rehearsal/rehearsal.json 的摘要 → slides/assets/w07/w07-metrics.json（demo1）
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
  show|rehearse|replay|check|compare|make|record|fake|selftest|export) exec "$PY" demo.py "$cmd" "$@" ;;
  fetch) HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=0 exec "$PY" demo.py fetch "$@" ;;
  *) echo "不認得的子命令：${cmd}" >&2; sed -n '2,18p' "$0" >&2; exit 2 ;;
esac

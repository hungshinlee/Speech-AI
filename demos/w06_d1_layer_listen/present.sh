#!/usr/bin/env bash
# 此檔由 bin/publish_demos.py 從 private repo 的 <course>/demos/ 複製而來，請勿在公開 repo 直接編輯。
# W6 demo 1 課堂與課前的入口。一律離線（HF_HUB_OFFLINE=1），避免教室網路出狀況時卡在連線。
#
#   ./present.sh               上課用：載三個 codec、算好兩段輸入的全部重建（幾秒到幾十秒），開三格的視窗；鍵在視窗裡按：
#                              o 原句、1 2 4 8 Mimi 前 n 層、e EnCodec 1.5→24 kbps、d DAC 1→8 層、m 換輸入、f F0 曲線、r 再播、t 指標表、h 選單、q 離開
#   ./present.sh rehearse      課前彩排：不互動，全部設定算一遍，重建 .wav、指標、圖存 runs/rehearsal/（進版控 = 現場的退路），寫 w06-metrics.json
#   ./present.sh replay        現場出狀況時：從 runs/rehearsal/ 讀回 .wav 與指標，同一個視窗、同樣的鍵，不碰 torch
#   ./present.sh record        課前：錄華語句子（6 秒；--from x.wav 匯入）→ ../_media/w06_d1/mandarin.wav（不進版控、不上站）
#   ./present.sh check         課前體檢：版本、三個 codec 載不載得起、config 的 f_r／V／N_q 對 encode 出來的矩陣、切列對層數參數、mps 對 cpu → runs/check.json
#   ./present.sh fetch         課前一週：下載三個 codec（鎖 commit → model.lock）、複製 W3 那句到 audio/（要網路）
#   ./present.sh fake          沒有模型也能演練流程（低通＋雜訊，畫面標明 FAKE）
#   ./present.sh selftest      bitrate 公式、F0 估計、指標恆等、裁靜音、fake 流程；有彩排紀錄時對 w06-metrics.json（不開視窗，約 20 秒）
#   ./present.sh show --no-plot   只在終端機，不開圖
#
# 課堂上的講法與按鍵順序寫在投影片 p30／p31 講稿的【操作】。
set -euo pipefail
cd "$(dirname "$0")"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
PY="../.venv/bin/python"   # 所有 demo 共用 speech_ai/demos/.venv
[[ -x "$PY" ]] || { echo "還沒有共用環境，先跑 ../setup.sh；torch／transformers 由 ../w03_d3_blank_heatmap/setup.sh 裝（沒有新套件）" >&2; exit 1; }

cmd="${1:-show}"; shift || true
case "$cmd" in
  show|rehearse|replay|record|check|fake|selftest|export) exec "$PY" demo.py "$cmd" "$@" ;;
  fetch)  HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=0 exec "$PY" demo.py fetch "$@" ;;
  *) echo "不認得的子命令：${cmd}" >&2; sed -n '2,13p' "$0" >&2; exit 2 ;;
esac

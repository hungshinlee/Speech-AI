#!/usr/bin/env bash
# 此檔由 bin/publish_demos.py 從 private repo 的 <course>/demos/ 複製而來，請勿在公開 repo 直接編輯。
# W3 demo 3 課堂與課前的入口。一律離線（HF_HUB_OFFLINE=1），避免教室網路出狀況時卡在連線。
#
#   ./present.sh               上課用：載入 wav2vec2、算 logits、開視窗；鍵在視窗裡按：1 播音、2 熱圖、3 四個數字（p37），
#                              4 正確文字 loss、5 接長 → inf、6 zero_infinity（p42），7 Viterbi 尖峰、a 疊圖，q 離開
#   ./present.sh rehearse      課前彩排：不互動，全部跑一遍，存 runs/rehearsal/（進版控 = 現場的退路）
#   ./present.sh replay        現場模型出狀況時：從 runs/rehearsal/ 讀回，同一個視窗、同樣的鍵，不碰 torch
#   ./present.sh check         課前體檢：版本、模型在不在快取、MPS 上的 ctc_loss 對 CPU、forced_align 還在不在 → runs/check.json
#   ./present.sh fetch         課前一週：下載模型（鎖 commit → model.lock）與那一句 LibriSpeech（要網路）
#   ./present.sh fake          沒有模型也能演練流程（合成的後驗，畫面標明 FAKE）
#
# 換句子：在子命令前加 --wav，例如  ./present.sh --wav audio/mine.wav rehearse（文字放 audio/mine.txt）
# macOS 內建的是 bash 3.2：空陣列要寫成 ${opts[@]+"${opts[@]}"}，否則 set -u 會報錯。
set -euo pipefail
cd "$(dirname "$0")"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
PY="../.venv/bin/python"   # 所有 demo 共用 speech_ai/demos/.venv
[[ -x "$PY" ]] || { echo "還沒有共用環境，先跑 ../setup.sh，再跑 ./setup.sh 裝 torch／transformers" >&2; exit 1; }

opts=()
while [[ $# -gt 0 && "$1" == --* ]]; do
  case "$1" in
    --wav) opts+=("$1" "$2"); shift 2 ;;
    *) echo "不認得的參數：$1" >&2; exit 2 ;;
  esac
done
cmd="${1:-show}"; shift || true
case "$cmd" in
  show|rehearse|replay|check|fake) exec "$PY" demo.py ${opts[@]+"${opts[@]}"} "$cmd" "$@" ;;
  fetch)  HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=0 exec "$PY" demo.py fetch "$@" ;;
  *) echo "不認得的子命令：${cmd}" >&2; sed -n '2,10p' "$0" >&2; exit 2 ;;
esac

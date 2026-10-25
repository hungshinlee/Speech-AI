#!/usr/bin/env bash
# 此檔由 bin/publish_demos.py 從 private repo 的 <course>/demos/ 複製而來，請勿在公開 repo 直接編輯。
# W5 demo 2 課堂與課前的入口。一律離線（HF_HUB_OFFLINE=1），避免教室網路出狀況時卡在連線。
#
#   ./present.sh               上課用：從快取讀 k-means 與整句表徵，開視窗；鍵在視窗裡按：1 三個純度、2 那句話（[ ] 移窗、f 整句、u 多兩列）、
#                              3 同一個字不同說話人、4 自錄、h 選單、q 離開
#   ./present.sh rehearse      課前彩排：不互動，全部算一遍，存 runs/rehearsal/（進版控 = 現場的退路），寫 w05-metrics.json → demo2
#   ./present.sh replay        現場出狀況時：從 runs/rehearsal/ 讀回所有數字與 unit 序列，同一個視窗、同樣的鍵，不碰 torch、不碰快取
#   ./present.sh cluster       課前：三組 k-means（MFCC 當場從 wav 算）→ ../_media/w05_d2/（show／rehearse 缺快取會自動跑）
#   ./present.sh extract       課前：那句話、念同一個字的其他說話人、自錄 wav 的整句表徵 → ../_media/w05_d2/seq_*.npz（要模型）
#   ./present.sh fetch         課前一週：對齊資料集的 words 欄 → data/words.json（要網路）
#   ./present.sh fake          沒有模型也能演練流程（合成的 unit，畫面標明 FAKE）
#   ./present.sh selftest      純度／k-means／DCT／LCS 的單元檢查 ＋ fake 流程 ＋ 彩排紀錄對 w05-metrics.json（不開視窗，約 20 秒）
#   ./present.sh show --no-plot   只在終端機，不開圖
#
# 前置：Demo 1 的 fetch 與 extract 要先跑完（子集、音素對齊、../_media/w05_d1/feats_hubert.npz）。沒有新套件要裝。
# 課堂上的講法與按鍵順序寫在投影片 p45 講稿的【操作】。
set -euo pipefail
cd "$(dirname "$0")"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
PY="../.venv/bin/python"   # 所有 demo 共用 speech_ai/demos/.venv
[[ -x "$PY" ]] || { echo "還沒有共用環境，先跑 ../setup.sh（torch／transformers 由 ../w03_d3_blank_heatmap/setup.sh 裝，pyarrow 由 ../w05_d1_layer_probe/setup.sh 裝）" >&2; exit 1; }

cmd="${1:-show}"; shift || true
case "$cmd" in
  show|rehearse|replay|cluster|extract|fake|selftest|export) exec "$PY" demo.py "$cmd" "$@" ;;
  fetch)  HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=0 HF_DATASETS_OFFLINE=0 exec "$PY" demo.py fetch "$@" ;;
  *) echo "不認得的子命令：${cmd}" >&2; sed -n '2,12p' "$0" >&2; exit 2 ;;
esac

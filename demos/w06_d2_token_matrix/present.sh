#!/usr/bin/env bash
# 此檔由 bin/publish_demos.py 從 private repo 的 <course>/demos/ 複製而來，請勿在公開 repo 直接編輯。
# W6 demo 2 課堂與課前的入口。一律離線（HF_HUB_OFFLINE=1），避免教室網路出狀況時卡在連線。
#
#   ./present.sh               上課用：載三個 codec（Demo 1 的 model.lock）、讀子集快取（沒有就先 extract，幾分鐘）、encode W3 那句，開視窗；鍵在視窗裡按：
#                              1 token 矩陣（[ ] 移窗）、2 使用率與熵、3 殘差曲線、4 bigram H2/H1、5 算帳、t 投影片的表、h 選單、q 離開
#   ./present.sh extract       課前：dev-clean 子集（W5 Demo 1 的 396 句，../_media/w05_d1/wav/）× 三個 codec → ../_media/w06_d2/codes_*.npz（一次；--force 重算）
#   ./present.sh rehearse      課前彩排：extract（若還沒有）→ 全部統計與算帳 → runs/rehearsal/（進版控 = 現場的退路）、slides/assets/w06/w06-metrics.json → demo2
#   ./present.sh replay        現場出狀況時：從 runs/rehearsal/ 讀回全部數字，同一個視窗、同樣的鍵，不碰 torch
#   ./present.sh check         課前體檢：對 W3 那句，三個 codec 的「各層 codeword 累加」對 quantizer 自己的 decode、從殘差重推 code、各路維度與層數；tokenizer → runs/check.json
#   ./present.sh fake          沒有模型也能演練流程（合成的 token 與曲線，畫面標明 FAKE）
#   ./present.sh selftest      熵／run／bigram 的函數、純 numpy RVQ 的殘差恆等、算帳對 w06-data.json、fake 流程；有彩排紀錄時對 w06-metrics.json（不開視窗，約 15 秒）
#   ./present.sh show --no-plot   只在終端機，不開圖
#
# 模型與 W3 那句都由 Demo 1 準備（../w06_d1_layer_listen：fetch／check）；子集的 wav 由 W5 Demo 1 準備（../w05_d1_layer_probe：fetch）。這裡沒有 fetch。
# 課堂上的講法與按鍵順序寫在投影片 p37／p46 講稿的【操作】。
set -euo pipefail
cd "$(dirname "$0")"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
PY="../.venv/bin/python"   # 所有 demo 共用 speech_ai/demos/.venv
[[ -x "$PY" ]] || { echo "還沒有共用環境，先跑 ../setup.sh；torch／transformers 由 ../w03_d3_blank_heatmap/setup.sh 裝（沒有新套件）" >&2; exit 1; }

cmd="${1:-show}"; shift || true
case "$cmd" in
  show|extract|rehearse|replay|check|fake|selftest|export) exec "$PY" demo.py "$cmd" "$@" ;;
  *) echo "不認得的子命令：${cmd}" >&2; sed -n '2,13p' "$0" >&2; exit 2 ;;
esac

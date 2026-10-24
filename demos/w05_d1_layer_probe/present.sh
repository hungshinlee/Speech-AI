#!/usr/bin/env bash
# 此檔由 bin/publish_demos.py 從 private repo 的 <course>/demos/ 複製而來，請勿在公開 repo 直接編輯。
# W5 demo 1 課堂與課前的入口。一律離線（HF_HUB_OFFLINE=1），避免教室網路出狀況時卡在連線。
#
#   ./present.sh               上課用：從快取讀表徵、開三格的視窗；鍵在視窗裡按：1 播一句＋13 層 shape、2 說話人探針、3 音素探針、
#                              4 CCA（l 切換 layer 0 ↔ log-mel）、w 換模型、a 兩個模型並排、t 摘要表（p37 的表格）、h 選單、q 離開
#   ./present.sh rehearse      課前彩排：不互動，兩個模型的三條曲線全部算一遍，存 runs/rehearsal/（進版控 = 現場的退路），寫 w05-metrics.json
#   ./present.sh replay        現場出狀況時：從 runs/rehearsal/ 讀回曲線，同一個視窗、同樣的鍵，不碰 torch
#   ./present.sh extract       課前：兩個模型對子集抽表徵 → ../_media/w05_d1/feats_<model>.npz（慢的那一步；show／rehearse 缺快取會自動跑）
#   ./present.sh check         課前體檢：版本、模型在不在快取、HuBERT 的 output_hidden_states 在 MPS 上跑不跑、mps 對 cpu → runs/check.json
#   ./present.sh fetch         課前一週：下載兩個模型（鎖 commit → model.lock）、dev-clean 子集（與 W3 Demo 3 共用 tar）、音素對齊（要網路）
#   ./present.sh fake          沒有模型也能演練流程（合成的表徵，畫面標明 FAKE）
#   ./present.sh selftest      探針與 CCA 的單元檢查 ＋ fake 流程 ＋ 彩排紀錄對 w05-metrics.json（不開視窗，約 2–4 分鐘）
#   ./present.sh show --no-plot   只在終端機，不開圖
#
# 課堂上的講法與按鍵順序寫在投影片 p37 講稿的【操作】。
set -euo pipefail
cd "$(dirname "$0")"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
PY="../.venv/bin/python"   # 所有 demo 共用 speech_ai/demos/.venv
[[ -x "$PY" ]] || { echo "還沒有共用環境，先跑 ../setup.sh，再跑 ./setup.sh 裝 pyarrow（torch／transformers 由 W3 d3 的 setup.sh 裝）" >&2; exit 1; }

cmd="${1:-show}"; shift || true
case "$cmd" in
  show|rehearse|replay|extract|check|fake|selftest|export) exec "$PY" demo.py "$cmd" "$@" ;;
  fetch)  HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=0 HF_DATASETS_OFFLINE=0 exec "$PY" demo.py fetch "$@" ;;
  *) echo "不認得的子命令：${cmd}" >&2; sed -n '2,13p' "$0" >&2; exit 2 ;;
esac

#!/usr/bin/env bash
# 此檔由 bin/publish_demos.py 從 private repo 的 <course>/demos/ 複製而來，請勿在公開 repo 直接編輯。
# W7 demo 2 課堂與課前的入口。一律離線（HF_HUB_OFFLINE=1），避免教室網路出狀況時卡在連線。
#
#   ./present.sh               上課用：載入 Whisper small、wav2vec2-base-960h、Qwen3-ASR，終端機單鍵操作：
#                              1 英文無 prompt、2 英文+名單、e 英文+台詞、3 華語無 prompt、4 華語+名單、5 華語+繁體句、6 華語+台詞、
#                              7 英文 CTC（greedy／beam δ=0／δ 三個值）、8 Qwen3-ASR 英文與華語（無／名單）、9 Qwen3-ASR 華語（繁體句／台詞）、
#                              p 全部 prompt、t 表、h 選單、q 離開（離開時存 runs/live/<時間>/，不進版控）
#   ./present.sh rehearse      課前彩排：不互動，全部跑一遍。英文 → runs/rehearsal/（進版控 = 現場的退路）；華語 → ../_media/w07_d2/private_rehearsal.*（不進版控）
#   ./present.sh replay        現場模型出狀況時：從兩份彩排紀錄讀回，同樣的鍵，不碰 torch
#   ./present.sh check         課前體檢：版本、三個模型在不在快取、Qwen3-ASR 在 MPS／CPU、prompt 槽、prefix beam 對 greedy → runs/check.json
#   ./present.sh pick          從 toml 的英文候選句挑一句：每句過 Whisper 無 prompt 與 CTC greedy，印哪些名字錯 → runs/pick.json
#   ./present.sh fetch         課前一週：下載模型（Whisper／wav2vec2 沿兄弟 demo 的 commit、Qwen3-ASR 取最新）、解英文候選句的 wav；要網路
#   ./present.sh record        錄一句自己的華語（含專有名詞）→ ../_media/w07_d2/mandarin.wav，接著輸入參考文字與名單
#   ./present.sh fake          沒有模型也能演練流程（罐頭輸出，畫面標明 FAKE）
#   ./present.sh selftest      不需要模型：trie／failure arc、prefix beam 對窮舉、名字命中、繁簡、洩漏、fake 整條流程
#   ./present.sh export        彩排紀錄的摘要 → slides/assets/w07/w07-metrics.json（demo2；華語只進數字）
#
# 模型套件（torch 2.14.0、transformers 5.17.0）已由 w03_d3_blank_heatmap/setup.sh 裝進共用 .venv；這個 demo 不另裝東西。
# Qwen3-ASR 的 transformers 原生版要 transformers ≥ 5.13.0（model card），5.17.0 夠。
set -euo pipefail
cd "$(dirname "$0")"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
PY="../.venv/bin/python"   # 所有 demo 共用 speech_ai/demos/.venv
[[ -x "$PY" ]] || { echo "還沒有共用環境，先跑 ../setup.sh，再跑 ../w03_d3_blank_heatmap/setup.sh 裝 torch／transformers" >&2; exit 1; }

cmd="${1:-show}"; shift || true
case "$cmd" in
  show|rehearse|replay|check|pick|record|fake|selftest|export) exec "$PY" demo.py "$cmd" "$@" ;;
  fetch) HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=0 exec "$PY" demo.py fetch "$@" ;;
  *) echo "不認得的子命令：${cmd}" >&2; sed -n '2,17p' "$0" >&2; exit 2 ;;
esac

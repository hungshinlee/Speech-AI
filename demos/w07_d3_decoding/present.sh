#!/usr/bin/env bash
# 此檔由 bin/publish_demos.py 從 private repo 的 <course>/demos/ 複製而來，請勿在公開 repo 直接編輯。
# W7 demo 3 課堂與課前的入口。一律離線（HF_HUB_OFFLINE=1），避免教室網路出狀況時卡在連線。
#
#   ./present.sh               上課用：載入 wav2vec2-base-960h、3-gram LM、MMS-1B-all、Whisper small、Qwen3-ASR，終端機單鍵操作：
#                              0 p32 的例子、1 greedy 與 beam λ=0、2 λ 掃描（視窗畫曲線）、3 一句看字（n 下一句）、m 華語同音字、
#                              e η 的調參、l LM 本身、t 表、h 選單、q 離開（離開時存 runs/live/<時間>/，不進版控）
#   ./present.sh rehearse      課前彩排：先在調參集上調 η，再全部跑一遍 → runs/rehearsal/（進版控 = 現場的退路；含 eta.json、lambda_sweep.png）；
#                              華語全文 → ../_media/w07_d3/private_rehearsal.*（不進版控）
#   ./present.sh replay        現場模型出狀況時：從彩排紀錄讀回，同樣的鍵，不碰 torch
#   ./present.sh check         課前體檢：版本、四個模型在不在快取、ARPA 讀得起來（第一次建 .npz 快取）、LM 的 backoff、beam λ=0 對 greedy、
#                              MMS-1B-all 在 MPS／CPU、一次有 LM 的 beam search 要多久 → runs/check.json
#   ./present.sh fetch         課前一週：下載 wav2vec2／Whisper／Qwen3-ASR（沿 Demo 2 的 commit）、MMS-1B-all（只抓 cmn 的 adapter）、3-gram ARPA；
#                              從 dev-clean.tar.gz 讀參考文字 → data/sentences.json；要網路
#   ./present.sh record        錄一句自己的華語 → ../_media/w07_d3/mandarin.wav，接著輸入參考文字
#   ./present.sh fake          沒有模型也能演練流程（合成 log 機率 + 小 LM，畫面標明 FAKE）
#   ./present.sh selftest      不需要模型：ARPA 與 backoff 對手算、LMBonus、p32 的窮舉對 json、WER／字元對齊、同音判定、fake 整條流程
#   ./present.sh export        彩排紀錄的摘要 → slides/assets/w07/w07-metrics.json（demo3）
#
# 模型套件（torch 2.14.0、transformers 5.17.0）已由 w03_d3_blank_heatmap/setup.sh 裝進共用 .venv；pypinyin 0.55.0 也已在 requirements.lock.txt。
# 這個 demo 不另裝東西；prefix beam search 用 ../w07_d2_biasing/ctc_prefix_beam.py、Whisper 與 Qwen3-ASR 用 Demo 2 的類別。
set -euo pipefail
cd "$(dirname "$0")"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
PY="../.venv/bin/python"   # 所有 demo 共用 speech_ai/demos/.venv
[[ -x "${PY}" ]] || { echo "還沒有共用環境，先跑 ../setup.sh，再跑 ../w03_d3_blank_heatmap/setup.sh 裝 torch／transformers" >&2; exit 1; }

cmd="${1:-show}"; shift || true
case "${cmd}" in
  show|rehearse|replay|check|record|fake|selftest|export) exec "${PY}" demo.py "${cmd}" "$@" ;;
  fetch) HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=0 exec "${PY}" demo.py fetch "$@" ;;
  *) echo "不認得的子命令：${cmd}" >&2; sed -n '2,18p' "$0" >&2; exit 2 ;;
esac

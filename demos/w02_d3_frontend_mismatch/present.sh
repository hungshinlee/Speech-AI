#!/usr/bin/env bash
# 此檔由 bin/publish_demos.py 從 private repo 的 <course>/demos/ 複製而來，請勿在公開 repo 直接編輯。
# W2 demo 3 課堂與課前的入口。一律離線（HF_HUB_OFFLINE=1），避免教室網路出狀況時卡在連線。
#
#   ./present.sh               上課用：載入 Whisper small（CPU），先用它自己的前端轉錄 40 句當基準，然後終端機單鍵操作：
#                              0 同配方自己算、1 HTK 尺度、2 峰值正規化、3 magnitude、4 floor、5 dither、6 pre-emphasis、7 CMVN、8 五個一起、
#                              a 全部依序、t 表、n 換特徵圖的句子、s 播那一句、h 選單、q 離開（離開時存 runs/live/<時間>/，不進版控）
#   ./present.sh rehearse      課前彩排：不互動，基準加全部 recipe 跑一遍，存 runs/rehearsal/（進版控 = 現場的退路；含 wer.png、features_<id>.png）
#   ./present.sh replay        現場模型出狀況時：從 runs/rehearsal/ 讀回，同樣的鍵、同樣的畫面，不碰 torch
#   ./present.sh check         課前體檢：版本、模型在不在快取、自己算的特徵對 WhisperFeatureExtractor、同配方轉錄要相同、一句多久、
#                              mel 第 64 個濾波器對 w02-data.json → runs/check.json
#   ./present.sh fetch         課前一週：下載 openai/whisper-small（commit 沿 W4 Demo 3 的 model.lock；本來就在快取的話只寫 model.lock）；
#                              從 ../_media/dev-clean.tar.gz 讀參考文字、每位說話人挑一句、解出 wav → ../_media/w02_d3/wav/、data/sentences.json
#   ./present.sh fake          沒有模型也能演練流程（合成的辨識器，畫面標明 FAKE）
#   ./present.sh selftest      不需要模型：自己的前端對 transformers 的 numpy 實作、兩種 mel 尺度對 w02-data.json、WER、FLAC 表頭、挑句、fake 流程
#   ./present.sh export        runs/rehearsal/rehearsal.json 的摘要 → slides/assets/w02/w02-metrics.json（demo3）
#
# 模型套件（torch 2.14.0、transformers 5.17.0）已由 w03_d3_blank_heatmap/setup.sh 裝進共用 .venv；soundfile 0.14.0 早在 requirements.lock.txt。
# 這個 demo 不另裝東西。前端全部在 ../dsp.py（兩種 mel 尺度、正規化開關）與這裡的 demo.py。
set -euo pipefail
cd "$(dirname "$0")"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
PY="../.venv/bin/python"   # 所有 demo 共用 speech_ai/demos/.venv
[[ -x "${PY}" ]] || { echo "還沒有共用環境，先跑 ../setup.sh，再跑 ../w03_d3_blank_heatmap/setup.sh 裝 torch／transformers" >&2; exit 1; }

cmd="${1:-show}"; shift || true
case "${cmd}" in
  show|rehearse|replay|check|fake|selftest|export) exec "${PY}" demo.py "${cmd}" "$@" ;;
  fetch) HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=0 exec "${PY}" demo.py fetch "$@" ;;
  *) echo "不認得的子命令：${cmd}" >&2; sed -n '2,16p' "$0" >&2; exit 2 ;;
esac

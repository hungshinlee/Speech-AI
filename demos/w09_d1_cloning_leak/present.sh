#!/usr/bin/env bash
# 此檔由 bin/publish_demos.py 從 private repo 的 <course>/demos/ 複製而來，請勿在公開 repo 直接編輯。
# W9 demo 1 課堂與課前的入口。一律離線（HF_HUB_OFFLINE=1），避免教室網路出狀況時卡在連線。
#
#   ./present.sh               上課用：載模型、把十二段（與有錄音時的加一格）全部算好（M5 Max 上一兩分鐘，課前先開），開一個視窗。鍵：
#                              p 四版 prompt  1 加噪  2 加音樂  3 電話頻寬  c 乾淨 ｜ s f k 只播一個系統  o 這一版的 prompt  g 真人錄音 ｜ t 數字 ｜ e 語氣  a 腔 ｜ r 再播  h 選單  q 離開
#   ./present.sh rehearse      課前彩排：不互動，全部算一遍，.wav、圖、數字存 runs/rehearsal/（進版控 = 現場的退路），寫 w09-metrics.json → demo1
#   ./present.sh replay        現場出狀況時：從 runs/rehearsal/ 讀回，同一個視窗、同樣的鍵，不碰 torch、不開 worker
#   ./present.sh check         課前體檢：SpeechBrain 的 x-vector 與 WavLMForXVector 在 cpu／mps 上跑不跑、SIM 的兩個參考點、Whisper、mlx-audio → runs/check.json
#   ./present.sh fetch         課前一週：下載 x-vector encoder、SIM 的 encoder、Whisper（鎖 commit → model.lock）；確認 W8 Demo 1／2 的模型都在（要網路）
#   ./present.sh record mood   錄「加一格」的 prompt（mood 或 accent）→ audio/own_<名字>.wav（不上站）
#   ./present.sh fake          沒有模型也能演練流程（畫面標明 FAKE）
#   ./present.sh selftest      四版 prompt、兩個自己定的量、WER、fake 流程；有彩排紀錄時對 w09-metrics.json（不開視窗）
#   ./present.sh export        只重寫 w09-metrics.json → demo1
#   ./present.sh show --no-plot   只在終端機，不開圖；--no-mlx 不開 worker（Qwen3-TTS 那一欄留空）
#
# 兩個環境：demo.py 在 ../.venv-tts（torch、f5-tts、speechbrain；W8 Demo 1 的 setup.sh 建、本資料夾的 setup.sh 加 speechbrain），
# Qwen3-TTS 由 W8 Demo 2 的 mlx_worker.py 在 ../.venv（mlx-audio）當子程序跑。replay／fake／selftest 不需要 torch，沒有 ../.venv-tts 時退到 ../.venv。
# 課堂上的講法與按鍵順序寫在投影片 p14–p17 講稿的【操作】。
set -euo pipefail
cd "$(dirname "$0")"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_DISABLE_PROGRESS_BARS=1
export TOKENIZERS_PARALLELISM=false
PY="../.venv-tts/bin/python"

cmd="${1:-show}"; shift || true
case "$cmd" in
  replay|fake|selftest|export|record)
    [[ -x "$PY" ]] || PY="../.venv/bin/python"
    [[ -x "$PY" ]] || { echo "沒有 ../.venv-tts 也沒有 ../.venv：先跑 ../setup.sh 與 ../w08_d1_mean_vs_sample/setup.sh" >&2; exit 1; }
    exec "$PY" demo.py "$cmd" "$@" ;;
  show|rehearse)
    [[ -x "$PY" ]] || { echo "還沒有 ../.venv-tts：先跑 ../w08_d1_mean_vs_sample/setup.sh，再跑 ./setup.sh" >&2; exit 1; }
    exec "$PY" demo.py "$cmd" "$@" ;;
  check)
    # 體檢要嚴格：torch 載入時就關掉 MPS 的 CPU fallback（殼層裡設了 1 也蓋掉），不支援的運算直接報錯
    [[ -x "$PY" ]] || { echo "還沒有 ../.venv-tts：先跑 ../w08_d1_mean_vs_sample/setup.sh，再跑 ./setup.sh" >&2; exit 1; }
    PYTORCH_ENABLE_MPS_FALLBACK=0 exec "$PY" demo.py check "$@" ;;
  fetch)
    [[ -x "$PY" ]] || { echo "還沒有 ../.venv-tts：先跑 ../w08_d1_mean_vs_sample/setup.sh，再跑 ./setup.sh" >&2; exit 1; }
    HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=0 exec "$PY" demo.py fetch "$@" ;;
  *) echo "不認得的子命令：${cmd}" >&2; sed -n '2,17p' "$0" >&2; exit 2 ;;
esac

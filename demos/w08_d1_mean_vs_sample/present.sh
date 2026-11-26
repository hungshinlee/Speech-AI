#!/usr/bin/env bash
# 此檔由 bin/publish_demos.py 從 private repo 的 <course>/demos/ 複製而來，請勿在公開 repo 直接編輯。
# W8 demo 1 課堂與課前的入口。一律離線（HF_HUB_OFFLINE=1），避免教室網路出狀況時卡在連線。
#
#   ./present.sh               上課用：載兩個模型、把全部版本算好（M5 Max 上幾十秒），開一個視窗；鍵在視窗裡按：
#                              1 SpeechT5、2 F5 seed 1、g 真人錄音、x 只看 2–6 kHz、3 F5 再抽兩次、z 三條 F0、s SpeechT5 換 seed、t 數字表、r 再播、h 選單、q 離開
#   ./present.sh rehearse      課前彩排：不互動，全部算一遍，.wav、mel、圖、數字存 runs/rehearsal/（進版控 = 現場的退路），寫 w08-metrics.json
#   ./present.sh replay        現場出狀況時：從 runs/rehearsal/ 讀回，同一個視窗、同樣的鍵，不碰 torch
#   ./present.sh check         課前體檢：版本、cpu／mps 的數值差與耗時；F5 在兩個裝置各抽一次 → runs/check/*.wav（用耳朵比，§6 第 7 項）、runs/check.json
#   ./present.sh fetch         課前一週：下載模型與 x-vector（鎖 commit → model.lock）、複製 target 錄音、解出 prompt（要網路）
#   ./present.sh fake          沒有模型也能演練流程（用真人錄音的 mel 冒充，畫面標明 FAKE）
#   ./present.sh selftest      頻率軸、細紋理指標、F5 的長度公式、fake 流程；有彩排紀錄時對 w08-metrics.json（不開視窗）
#   ./present.sh export        只重寫 w08-metrics.json → demo1
#   ./present.sh show --no-plot   只在終端機，不開圖
#
# 環境是 ../.venv-tts（./setup.sh 建），不是共用的 ../.venv——f5-tts 的相依跟共用環境的兩個釘衝突（README）。
# replay／fake／selftest 不需要 torch，沒有 ../.venv-tts 時退到 ../.venv。
# 課堂上的講法與按鍵順序寫在投影片 p12／p14 講稿的【操作】。
set -euo pipefail
cd "$(dirname "$0")"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_DISABLE_PROGRESS_BARS=1
export TOKENIZERS_PARALLELISM=false
PY="../.venv-tts/bin/python"

cmd="${1:-show}"; shift || true
case "$cmd" in
  replay|fake|selftest|export)
    [[ -x "$PY" ]] || PY="../.venv/bin/python"
    [[ -x "$PY" ]] || { echo "沒有 ../.venv-tts 也沒有 ../.venv：先跑 ./setup.sh" >&2; exit 1; }
    exec "$PY" demo.py "$cmd" "$@" ;;
  show|rehearse)
    [[ -x "$PY" ]] || { echo "還沒有 ../.venv-tts：先跑 ./setup.sh" >&2; exit 1; }
    exec "$PY" demo.py "$cmd" "$@" ;;
  check)
    # 體檢要嚴格：torch 載入時就關掉 MPS 的 CPU fallback（殼層裡設了 1 也蓋掉），不支援的運算直接報錯
    [[ -x "$PY" ]] || { echo "還沒有 ../.venv-tts：先跑 ./setup.sh" >&2; exit 1; }
    PYTORCH_ENABLE_MPS_FALLBACK=0 exec "$PY" demo.py check "$@" ;;
  fetch)
    [[ -x "$PY" ]] || { echo "還沒有 ../.venv-tts：先跑 ./setup.sh" >&2; exit 1; }
    HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=0 exec "$PY" demo.py fetch "$@" ;;
  *) echo "不認得的子命令：${cmd}" >&2; sed -n '2,17p' "$0" >&2; exit 2 ;;
esac

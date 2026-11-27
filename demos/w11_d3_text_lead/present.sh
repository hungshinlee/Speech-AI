#!/usr/bin/env bash
# 此檔由 bin/publish_demos.py 從 private repo 的 <course>/demos/ 複製而來，請勿在公開 repo 直接編輯。
# W11 demo 3 課堂與課前的入口。一律離線（HF_HUB_OFFLINE=1），避免教室網路出狀況時卡在連線。
#
#   ./present.sh               上課用：模型載起來（幾秒），按鍵現場生成；表格讀彩排的（runs/rehearsal/）。鍵：
#                              p 播問題   1 文字全部先寫 → 6 : 12   2 掃比例   t 表   3 兩個條件   4 cascade（$ 現場跑）
#                              5 算術題（按兩次：直接問、要求先推理）   6 後半的表   w 寫的字對聽到的字   l 再播一次   ? 選單   q 離開
#   ./present.sh rehearse      課前彩排：三題 × 五個設定、算術題兩次、cascade → runs/rehearsal/（進版控 = 現場的退路），寫 w11-metrics.json → demo3
#                              --preview 寫到 runs/preview/、不寫 metrics；--questions 1 只跑第一題；--no-cascade；--no-think
#   ./present.sh replay        現場出狀況時：只讀 runs/rehearsal/，token 照當時的時間戳重印、播當時的聲音，不載任何模型
#   ./present.sh check         課前體檢：排法、速度、分塊解碼、sequential 模式、Whisper、cascade 的三個後端 → runs/check.json
#   ./present.sh fetch         課前一週：下載 LFM2.5-Audio，合成四題的聲音（audio/ 裡已經有就不重做；要網路）
#   ./present.sh tts           只重新合成問題的聲音（--force 覆蓋）
#   ./present.sh record q1     用自己的聲音問（寫到 ../_media/w11_d3/，不進版控）
#   ./present.sh fake          沒有模型也能演練流程（畫面標明 FAKE，聲音是噪音）
#   ./present.sh selftest      trace.py 的每一條定義、設定檔、fake 整條流程；有彩排紀錄時對 w11-metrics.json
#   ./present.sh export        只重寫 w11-metrics.json → demo3
#
# 環境：共用的 ../.venv（mlx-audio 0.5.4 已經有 LFM2.5-Audio 的模組；torch、transformers、mlx-lm、pywhispercpp 也在裡面）。
# 這個 demo 不裝新套件、沒有 setup.sh。課堂上的講法與按鍵順序寫在投影片 p65、p73 講稿的【操作】。
set -euo pipefail
cd "$(dirname "$0")"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_DISABLE_PROGRESS_BARS=1
export TOKENIZERS_PARALLELISM=false
PY="../.venv/bin/python"
[[ -x "${PY}" ]] || { echo "還沒有 ../.venv：先跑 ../setup.sh" >&2; exit 1; }

cmd="${1:-show}"; shift || true
case "${cmd}" in
  show|replay|fake|selftest|export|rehearse|check|tts|record)
    exec "${PY}" demo.py "${cmd}" "$@" ;;
  fetch)
    # 只有這一步要網路
    HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=0 exec "${PY}" demo.py fetch "$@" ;;
  *) echo "不認得的子命令：${cmd}" >&2; sed -n '2,19p' "$0" >&2; exit 2 ;;
esac

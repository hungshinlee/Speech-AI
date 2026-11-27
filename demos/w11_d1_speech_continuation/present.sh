#!/usr/bin/env bash
# 此檔由 bin/publish_demos.py 從 private repo 的 <course>/demos/ 複製而來，請勿在公開 repo 直接編輯。
# W11 demo 1 課堂與課前的入口。一律離線（HF_HUB_OFFLINE=1），避免教室網路出狀況時卡在連線。
#
#   ./present.sh               上課用：讀回彩排的聲音與數字（runs/rehearsal/），不先載模型。鍵：
#                              p 開頭   1 2 3 4 一段續寫   c 只播續寫的部分   t 三欄的字   n 數字
#                              w Whisper 有多少把握   r 這位說話人實際上接著說的   m 開頭只過 Mimi 的 4 層
#                              g 現場再抽一段（這時才載模型）   ? 選單   q 離開
#   ./present.sh rehearse      課前彩排：同一個開頭抽四段、兩位裁判、文字 LLM → runs/rehearsal/（進版控 = 現場的退路），寫 w11-metrics.json → demo1
#                              --preview 寫到 runs/preview/、不寫 metrics；--draws 2 只抽兩段；--no-text 不跑文字 LLM；--device cpu
#   ./present.sh replay        現場出狀況時：只讀 runs/rehearsal/，同樣的鍵（除了 g），不載任何模型
#   ./present.sh check         課前體檢：token 的排法、cpu／mps 的差與速度、兩位裁判、文字 LLM 的兩種接法 → runs/check.json
#   ./present.sh fetch         課前一週：下載 Llama-Mimi（約 5 GB）與 Mimi，確認 Whisper、WavLM-SV、文字 LLM 在快取裡（要網路）
#   ./present.sh fake          沒有模型也能演練流程（畫面標明 FAKE，聲音是噪音）
#   ./present.sh selftest      token 的排法、解析、長度的算術、fake 整條流程；有彩排紀錄時對 w11-metrics.json
#   ./present.sh export        只重寫 w11-metrics.json → demo1
#
# 環境：共用的 ../.venv（torch、transformers、mlx-lm 都在裡面）。這個 demo 不裝新套件、沒有 setup.sh。
# 課堂上的講法與按鍵順序寫在投影片 p23 講稿的【操作】。
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
  show|replay|fake|selftest|export|rehearse)
    exec "${PY}" demo.py "${cmd}" "$@" ;;
  check)
    # 體檢要嚴格：關掉 MPS 的 CPU fallback（殼層裡設了 1 也蓋掉），不支援的運算直接報錯
    PYTORCH_ENABLE_MPS_FALLBACK=0 exec "${PY}" demo.py check "$@" ;;
  fetch)
    # 只有這一步要網路
    HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=0 exec "${PY}" demo.py fetch "$@" ;;
  *) echo "不認得的子命令：${cmd}" >&2; sed -n '2,19p' "$0" >&2; exit 2 ;;
esac

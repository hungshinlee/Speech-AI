#!/usr/bin/env bash
# 此檔由 bin/publish_demos.py 從 private repo 的 <course>/demos/ 複製而來，請勿在公開 repo 直接編輯。
# W11 demo 2 課堂與課前的入口。一律離線（HF_HUB_OFFLINE=1），避免教室網路出狀況時卡在連線。
#
#   ./present.sh               上課用：讀彩排的數字（runs/rehearsal/），另外把兩個模型載起來給現場的鍵。鍵：
#                              q 現場問一題（四種問法）  n 下一題  v 把回答念出來（Talker）
#                              a b c d 一列一列看  t 整張表  f 格式題  m c′（我自己念的）  i 三個落差  k 分三類
#                              2 換一個聲音  x 辨識後的字交給 θ  w 聽對了卻答錯  z 加噪音  s 靜音／白噪音  e 語氣  l 換語言  ? 選單  Q 離開
#   ./present.sh rehearse      課前彩排：兩種語言、全部條件 → runs/rehearsal/（進版控 = 現場的退路），寫 w11-metrics.json → demo2
#                              可以中斷再跑（接續）；--n 12 只跑 12 題的預覽 → runs/preview/；--langs zh；--only a,b,c1,d1；--probes 探針重跑
#   ./present.sh replay        現場出狀況時：只讀 runs/rehearsal/，同樣的鍵，不載任何模型
#   ./present.sh check         課前體檢：每一條路各走一次（θ 讀、聽、轉錄、出聲；θ0；Kokoro；Whisper）→ runs/check.json
#   ./present.sh fetch         課前一週：下載 θ（約 12.7 GB）、下載並轉出 θ0（下載約 15.8 GB）、確認 Kokoro 與 Whisper（要網路）
#   ./present.sh tts           把 50 題念成 wav（兩種語言、各兩個聲音）；rehearse 發現沒有時會自己做
#   ./present.sh record own    自己念 10 題（c′ 那一列）；record tone：同一句話錄高興與沮喪兩版（語氣的探針）。加 --lang en 錄英語版
#   ./present.sh fake          沒有模型也能演練流程（畫面標明 FAKE，數字沒有意義）
#   ./present.sh selftest      題目、計分規則、統計、fake 整條流程
#   ./present.sh export        只重寫 w11-metrics.json → demo2
#
# 環境：../.venv-sts（這個 demo 的 ./setup.sh 建的：新版 mlx-audio 與 mlx-lm）；
#       Kokoro 與 Whisper 沿用共用的 ../.venv，demo.py 用子程序叫它（aux_worker.py）。
# replay／fake／selftest／export 不需要模型，沒有 ../.venv-sts 時退到 ../.venv。
# 課堂上的講法與按鍵順序寫在投影片 p35、p46 講稿的【操作】。
set -euo pipefail
cd "$(dirname "$0")"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_DISABLE_PROGRESS_BARS=1
export TOKENIZERS_PARALLELISM=false
PY="../.venv-sts/bin/python"
need() { [[ -x "$PY" ]] || { echo "還沒有 ../.venv-sts：先跑 ./setup.sh" >&2; exit 1; }; }

cmd="${1:-show}"; shift || true
case "$cmd" in
  replay|fake|selftest|export)
    [[ -x "$PY" ]] || PY="../.venv/bin/python"
    [[ -x "$PY" ]] || { echo "沒有 ../.venv-sts 也沒有 ../.venv：先跑 ../setup.sh" >&2; exit 1; }
    exec "$PY" demo.py "$cmd" "$@" ;;
  show|rehearse|check|tts|record)
    need
    exec "$PY" demo.py "$cmd" "$@" ;;
  fetch)
    # 只有這一步要網路
    need
    HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=0 exec "$PY" demo.py fetch "$@" ;;
  *) echo "不認得的子命令：${cmd}" >&2; sed -n '2,21p' "$0" >&2; exit 2 ;;
esac

#!/usr/bin/env bash
# 此檔由 bin/publish_demos.py 從 private repo 的 <course>/demos/ 複製而來，請勿在公開 repo 直接編輯。
# W9 demo 3 課堂與課前的入口。一律離線（HF_HUB_OFFLINE=1），避免教室網路出狀況時卡在連線。
#
#   ./present.sh               上課用：載偵測器、watermark、codec 與 Vocos，啟動時全部算好，開一個視窗。鍵：
#                              前半 r 真人  空白鍵 現場對麥克風  1–6 生成器  c 真人過 codec  x 加映
#                              後半 w A／B  d 曲線  s 接合  a 重取樣  n 噪音  p MP3  e EnCodec（每按一次降一格）  m Mimi  v Vocos  b 長條圖
#                              t 摘要  y 再播  ? 選單  q 離開
#   ./present.sh rehearse      課前彩排：不互動，全部算一遍，.wav、圖、數字存 runs/rehearsal/（進版控 = 現場的退路），寫 w09-metrics.json → demo3
#   ./present.sh replay        現場出狀況時：從 runs/rehearsal/ 讀回，同一個視窗、同樣的鍵（除了空白鍵），不碰 torch
#   ./present.sh record        錄你自己的一句（→ ../_media/w09_d3/own.wav，不進版控）：彩排時當「現場對麥克風」那一格的預演
#   ./present.sh check         課前體檢：偵測器與 watermark 在 cpu／mps 上跑不跑、差多少；分數隨音量變多少；codec、Vocos、ffmpeg → runs/check.json
#   ./present.sh fetch         課前一週：clone AASIST（對 commit 與 sha256）、下載 AudioSeal 的權重（對 sha256）；確認前幾週的東西都在（要網路）
#   ./present.sh fake          沒有模型也能演練流程（畫面標明 FAKE）
#   ./present.sh selftest      取窗、調音量、接合、SNR、統計量、fake 流程；有彩排紀錄時對 w09-metrics.json（不開視窗）
#   ./present.sh export        只重寫 w09-metrics.json → demo3
#   ./present.sh show --no-plot   只在終端機，不開圖
#
# 環境：../.venv-tts（W8 Demo 1 的 setup.sh 建的；這個 demo 的 ./setup.sh 再裝一個 audioseal）。
# replay／fake／selftest／export 不需要 torch，沒有 ../.venv-tts 時退到 ../.venv。
# 課堂上的講法與按鍵順序寫在投影片 p85–p87 講稿的【操作】。
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
    [[ -x "$PY" ]] || { echo "沒有 ../.venv-tts 也沒有 ../.venv：先跑 ../setup.sh 與 ../w08_d1_mean_vs_sample/setup.sh" >&2; exit 1; }
    exec "$PY" demo.py "$cmd" "$@" ;;
  show|rehearse|record)
    [[ -x "$PY" ]] || { echo "還沒有 ../.venv-tts：先跑 ../w08_d1_mean_vs_sample/setup.sh，再跑 ./setup.sh" >&2; exit 1; }
    exec "$PY" demo.py "$cmd" "$@" ;;
  fetch)
    # 只有這一步要網路
    [[ -x "$PY" ]] || { echo "還沒有 ../.venv-tts：先跑 ../w08_d1_mean_vs_sample/setup.sh，再跑 ./setup.sh" >&2; exit 1; }
    HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=0 exec "$PY" demo.py fetch "$@" ;;
  check)
    # 體檢要嚴格：torch 載入時就關掉 MPS 的 CPU fallback（殼層裡設了 1 也蓋掉），不支援的運算直接報錯
    [[ -x "$PY" ]] || { echo "還沒有 ../.venv-tts：先跑 ../w08_d1_mean_vs_sample/setup.sh，再跑 ./setup.sh" >&2; exit 1; }
    PYTORCH_ENABLE_MPS_FALLBACK=0 exec "$PY" demo.py check "$@" ;;
  *) echo "不認得的子命令：${cmd}" >&2; sed -n '2,17p' "$0" >&2; exit 2 ;;
esac

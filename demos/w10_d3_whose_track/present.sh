#!/usr/bin/env bash
# 此檔由 bin/publish_demos.py 從 private repo 的 <course>/demos/ 複製而來，請勿在公開 repo 直接編輯。
# W10 demo 3 課堂與課前的入口。一律離線（HF_HUB_OFFLINE=1），避免教室網路出狀況時卡在連線。
#
#   ./present.sh               上課用：啟動時全部算好（分離、兩個 encoder、Whisper、diarization），開一個視窗。鍵：
#                              前半 1 完全重疊：混合 → 兩軌  p 播「對話」  2 逐塊的 SIM（ECAPA）  3 換 WavLM-SV  o 揭曉  n 每塊都取第一軌
#                                   4 挑軌、接起來、Whisper  l 播挑出來的  5 使用者不在的那一塊  a 門檻  c 換塊長  x 整段一次  g 換另一位
#                              後半 6 diarization 的時間軸  7 三個 DER  k 換成土法聚類  t 投影片的表  ? 選單  q 離開
#   ./present.sh rehearse      課前彩排：同樣算一遍 → runs/rehearsal/（進版控 = 現場的退路），寫 w10-metrics.json → demo3
#                              --no-diar 不跑 pyannote（還沒拿到 token 時）；--device cpu 全部放 CPU
#   ./present.sh replay        現場出狀況時：只讀 runs/rehearsal/，同一個視窗、同樣的鍵，不碰 torch
#   ./present.sh check         課前體檢：分離模型與兩個 encoder 在 cpu／mps 上跑不跑、要多久；
#                              pyannote 在 mps 與 cpu 上的時間戳一不一致；der.py 對 pyannote.metrics → runs/check.json
#   ./present.sh fetch         課前一週：解出要用的句子；下載 SepFormer、ECAPA、pyannote 的 pipeline（要網路；pyannote 要先同意條款並登入）
#   ./present.sh fake          沒有模型也能演練流程（畫面標明 FAKE，數字沒有意義）
#   ./present.sh selftest      接對話、切塊、SI-SDR、DER（對投影片 p83 的玩具）、fake 整條流程（不開視窗）
#   ./present.sh export        只重寫 w10-metrics.json → demo3
#   ./present.sh show --no-plot   只在終端機，不開圖
#
# 環境：../.venv-tts（W8 Demo 1 的 setup.sh 建的、W9 Demo 1 的 setup.sh 裝了 speechbrain）；
#       pyannote.audio 另住 ../.venv-diar（這個 demo 的 ./setup.sh 建的），demo.py 用子程序叫它。
# replay／fake／selftest／export 不需要 torch，沒有 ../.venv-tts 時退到 ../.venv。
# 課堂上的講法與按鍵順序寫在投影片 p74、p75、p84 講稿的【操作】。
set -euo pipefail
cd "$(dirname "$0")"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_DISABLE_PROGRESS_BARS=1
export TOKENIZERS_PARALLELISM=false
export PYANNOTE_METRICS_ENABLED=0
PY="../.venv-tts/bin/python"
need() { [[ -x "$PY" ]] || { echo "還沒有 ../.venv-tts：先跑 ../w08_d1_mean_vs_sample/setup.sh、../w09_d1_cloning_leak/setup.sh" >&2; exit 1; }; }

cmd="${1:-show}"; shift || true
case "$cmd" in
  replay|fake|selftest|export)
    [[ -x "$PY" ]] || PY="../.venv/bin/python"
    [[ -x "$PY" ]] || { echo "沒有 ../.venv-tts 也沒有 ../.venv：先跑 ../setup.sh" >&2; exit 1; }
    exec "$PY" demo.py "$cmd" "$@" ;;
  show|rehearse)
    need
    exec "$PY" demo.py "$cmd" "$@" ;;
  fetch)
    # 只有這一步要網路
    need
    HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=0 exec "$PY" demo.py fetch "$@" ;;
  check)
    # 體檢要嚴格：關掉 MPS 的 CPU fallback（殼層裡設了 1 也蓋掉），不支援的運算直接報錯
    need
    PYTORCH_ENABLE_MPS_FALLBACK=0 exec "$PY" demo.py check "$@" ;;
  *) echo "不認得的子命令：${cmd}" >&2; sed -n '2,20p' "$0" >&2; exit 2 ;;
esac

#!/usr/bin/env bash
# 此檔由 bin/publish_demos.py 從 private repo 的 <course>/demos/ 複製而來，請勿在公開 repo 直接編輯。
# W10 demo 2 課堂與課前的入口。一律離線（HF_HUB_OFFLINE=1），避免教室網路出狀況時卡在連線。
#
#   ./present.sh               上課用：讀回彩排的數字與聲音、載入模型（給鍵 l 現場重算），開一個視窗。鍵：
#                              聽 0 原句  1 帶噪 → 統計式 → MetricGAN+  u musical noise  x SepFormer  s 換 SNR  n 換句  b 換噪音
#                              看 2 CTC 的 WER  3 Whisper 的 WER  4 逐字稿並排  w 最糟的一句  5 SAR 對 WER  6 ω  7 偵測器
#                                 m 每位裁判  e 只增強不加噪  l 現場重算  t 投影片的表  ? 選單  q 離開
#   ./present.sh rehearse      課前彩排：全部算一遍 → runs/rehearsal/（進版控 = 現場的退路），寫 w10-metrics.json → demo2
#                              可以中斷，再跑一次會接下去。--noise white,babble 只算幾種噪音；--no-extra 不跑 SepFormer；--fresh 從頭算
#   ./present.sh rehearse --n 12   預覽：只用 12 句 → runs/preview/（不碰彩排紀錄）；看它用 ./present.sh replay --from runs/preview
#   ./present.sh replay        現場出狀況時：只讀 runs/rehearsal/，同一個視窗、同樣的鍵（除了 l），不碰 torch
#   ./present.sh check         課前體檢：模型在 cpu／mps 上跑不跑、要多久；彩排大約要幾分鐘 → runs/check.json
#   ./present.sh fetch         課前一週：下載 MetricGAN+ 與 SepFormer 的權重；確認前幾週的模型與 40 句都在（要網路）
#   ./present.sh fake          沒有模型也能演練流程（畫面標明 FAKE，數字沒有意義）
#   ./present.sh selftest      加噪音、兩次投影、observation adding、Wiener、fake 整條流程（不開視窗）
#   ./present.sh export        只重寫 w10-metrics.json → demo2
#   ./present.sh show --no-plot   只在終端機，不開圖
#
# 環境：../.venv-tts（W8 Demo 1 的 setup.sh 建的、W9 Demo 1 的 setup.sh 裝了 speechbrain；這個 demo 的 ./setup.sh 再裝 pystoi 與 pesq）。
# replay／fake／selftest／export 不需要 torch，沒有 ../.venv-tts 時退到 ../.venv。
# 課堂上的講法與按鍵順序寫在投影片 p49、p51、p57 講稿的【操作】。
set -euo pipefail
cd "$(dirname "$0")"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_DISABLE_PROGRESS_BARS=1
export TOKENIZERS_PARALLELISM=false
PY="../.venv-tts/bin/python"
need() { [[ -x "$PY" ]] || { echo "還沒有 ../.venv-tts：先跑 ../w08_d1_mean_vs_sample/setup.sh、../w09_d1_cloning_leak/setup.sh，再跑 ./setup.sh" >&2; exit 1; }; }

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

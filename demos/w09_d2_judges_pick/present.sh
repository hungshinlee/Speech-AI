#!/usr/bin/env bash
# 此檔由 bin/publish_demos.py 從 private repo 的 <course>/demos/ 複製而來，請勿在公開 repo 直接編輯。
# W9 demo 2 課堂與課前的入口。一律離線（HF_HUB_OFFLINE=1），避免教室網路出狀況時卡在連線。
#
#   ./present.sh               上課用：載 Qwen3-TTS 與三位裁判，讀回彩排的那一組，開一個視窗。鍵：
#                              g 表  l 長尾  a advantage ｜ w s m 三位裁判的第一名  h 我挑的  p 散點圖 ｜ z 換另一句  f F5 對照  n 現場再抽一組
#                              1–9 0 - = 播第 1–12 段  o prompt  t 摘要  r 再播  ? 選單  q 離開
#   ./present.sh rehearse      課前彩排：不互動，兩句各抽十二段（加 F5 的對照組），.wav、圖、數字存 runs/rehearsal/（進版控 = 現場的退路），寫 w09-metrics.json → demo2
#   ./present.sh listen en     彩排之後：十二段打亂順序、不顯示編號地播，你挑一段 → runs/rehearsal/picks.json（en 或 zh）
#   ./present.sh replay        現場出狀況時：從 runs/rehearsal/ 讀回，同一個視窗、同樣的鍵（除了 n），不碰 torch、不開 worker
#   ./present.sh check         課前體檢：MOS predictor 在 cpu／mps 上跑不跑、分數差多少；三位裁判對幾段已知的聲音；mlx-audio → runs/check.json
#   ./present.sh fetch         課前一週：下載 MOS predictor 的程式與權重（對 commit 與 sha256）；確認前面 demo 的模型都在（要網路）
#   ./present.sh fake          沒有模型也能演練流程（畫面標明 FAKE）；fake --real-judges 只有生成器是替身
#   ./present.sh selftest      對齊、音節、語速、F0 起伏、advantage、名次、fake 流程；有彩排紀錄時對 w09-metrics.json（不開視窗）
#   ./present.sh export        只重寫 w09-metrics.json → demo2（listen 之後跑一次，把你挑的寫進去）
#   ./present.sh show --no-plot   只在終端機，不開圖；--fresh 不讀彩排、現場從 seed 1 抽起；--no-mlx 不開 worker
#
# 兩個環境：demo.py 在 ../.venv-tts（torch、f5-tts；W8 Demo 1 的 setup.sh 建的），Qwen3-TTS 由 W8 Demo 2 的 mlx_worker.py 在 ../.venv（mlx-audio）當子程序跑。
# 這個 demo 不裝新套件。replay／listen／fake／selftest／export 不需要 torch，沒有 ../.venv-tts 時退到 ../.venv。
# 課堂上的講法與按鍵順序寫在投影片 p53–p55 講稿的【操作】。
set -euo pipefail
cd "$(dirname "$0")"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_DISABLE_PROGRESS_BARS=1
export TOKENIZERS_PARALLELISM=false
PY="../.venv-tts/bin/python"

cmd="${1:-show}"; shift || true
case "$cmd" in
  replay|listen|fake|selftest|export)
    [[ -x "$PY" ]] || PY="../.venv/bin/python"
    [[ -x "$PY" ]] || { echo "沒有 ../.venv-tts 也沒有 ../.venv：先跑 ../setup.sh 與 ../w08_d1_mean_vs_sample/setup.sh" >&2; exit 1; }
    exec "$PY" demo.py "$cmd" "$@" ;;
  show|rehearse|fetch)
    [[ -x "$PY" ]] || { echo "還沒有 ../.venv-tts：先跑 ../w08_d1_mean_vs_sample/setup.sh" >&2; exit 1; }
    exec "$PY" demo.py "$cmd" "$@" ;;
  check)
    # 體檢要嚴格：torch 載入時就關掉 MPS 的 CPU fallback（殼層裡設了 1 也蓋掉），不支援的運算直接報錯
    [[ -x "$PY" ]] || { echo "還沒有 ../.venv-tts：先跑 ../w08_d1_mean_vs_sample/setup.sh" >&2; exit 1; }
    PYTORCH_ENABLE_MPS_FALLBACK=0 exec "$PY" demo.py check "$@" ;;
  *) echo "不認得的子命令：${cmd}" >&2; sed -n '2,17p' "$0" >&2; exit 2 ;;
esac

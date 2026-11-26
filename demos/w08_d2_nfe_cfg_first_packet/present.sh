#!/usr/bin/env bash
# 此檔由 bin/publish_demos.py 從 private repo 的 <course>/demos/ 複製而來，請勿在公開 repo 直接編輯。
# W8 demo 2 課堂與課前的入口。一律離線（HF_HUB_OFFLINE=1），避免教室網路出狀況時卡在連線。
#
#   ./present.sh               上課用：載 F5（約十秒）、開 MLX worker（Qwen3-TTS、Kokoro 在第一次按 5／6 時載），開一個視窗；**按鍵時才生成**。鍵：
#                              1 NFE 掃描  e 另一句  2 CFG 掃描  3 sway 開／關  p EPSS 16 ｜ 4 F5 整句第一包  5 Qwen3-TTS 串流  6 Kokoro ｜ t 表  r 再播  h 選單  q 離開
#   ./present.sh rehearse      課前彩排：不互動，全部算一遍，.wav、mel、圖、時間存 runs/rehearsal/（進版控 = 現場的退路），寫 w08-metrics.json → demo2
#   ./present.sh replay        現場出狀況時：從 runs/rehearsal/ 讀回，同一個視窗、同樣的鍵，不碰 torch、不開 worker
#   ./present.sh check         課前體檢：版本、mlx-audio 有沒有 qwen3_tts；自寫的 sampler 對 f5-tts 的 sample() 逐格相同；一步兩次前向多久 → runs/check.json
#   ./present.sh fetch         課前一週：下載 Qwen3-TTS 與 Kokoro（鎖 commit → model.lock；F5 用 Demo 1 的）（要網路）
#   ./present.sh fake          沒有模型也能演練流程（畫面標明 FAKE）
#   ./present.sh selftest      字數、長度、時刻表、w ↔ α、fake 流程；有彩排紀錄時對 w08-metrics.json（不開視窗）
#   ./present.sh export        只重寫 w08-metrics.json → demo2
#   ./present.sh show --no-plot   只在終端機，不開圖；--no-mlx 不開 worker
#
# 兩個環境：demo.py 在 ../.venv-tts（torch、f5-tts；Demo 1 的 setup.sh 建），mlx_worker.py 由 demo.py 用 ../.venv/bin/python（mlx-audio 0.5.4）開。
# replay／fake／selftest 不需要 torch，沒有 ../.venv-tts 時退到 ../.venv。
# 課堂上的講法與按鍵順序寫在投影片 p62／p64、p84 講稿的【操作】。
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
  show|rehearse)
    [[ -x "$PY" ]] || { echo "還沒有 ../.venv-tts：先跑 ../w08_d1_mean_vs_sample/setup.sh" >&2; exit 1; }
    exec "$PY" demo.py "$cmd" "$@" ;;
  check)
    # 體檢要嚴格：torch 載入時就關掉 MPS 的 CPU fallback（殼層裡設了 1 也蓋掉），不支援的運算直接報錯
    [[ -x "$PY" ]] || { echo "還沒有 ../.venv-tts：先跑 ../w08_d1_mean_vs_sample/setup.sh" >&2; exit 1; }
    PYTORCH_ENABLE_MPS_FALLBACK=0 exec "$PY" demo.py check "$@" ;;
  fetch)
    [[ -x "$PY" ]] || PY="../.venv/bin/python"
    HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=0 exec "$PY" demo.py fetch "$@" ;;
  *) echo "不認得的子命令：${cmd}" >&2; sed -n '2,15p' "$0" >&2; exit 2 ;;
esac

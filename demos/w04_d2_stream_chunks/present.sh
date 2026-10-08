#!/usr/bin/env bash
# 此檔由 bin/publish_demos.py 從 private repo 的 <course>/demos/ 複製而來，請勿在公開 repo 直接編輯。
# W4 demo 2 課堂與課前的入口。一律離線（HF_HUB_OFFLINE=1；模型在 ~/.cache/sherpa-onnx，不碰網路）。
#
#   ./present.sh               上課用：載入 sherpa-onnx 的 streaming zipformer，開時間軸視窗；鍵在視窗裡按：
#                              1 餵 1280 ms、2 餵 640 ms、3 餵 320 ms（每種都以即時步調餵同一句、同時播音），a 摘要表＋第四格，
#                              p 播音，m 靜音，h 選單，q 離開（離開時存 runs/live/<時間>/，不進版控）
#   ./present.sh rehearse      課前彩排：不互動、不播音，三種餵法各跑一次，存 runs/rehearsal/（進版控 = 現場的退路）
#   ./present.sh replay        現場模型出狀況時：從 runs/rehearsal/ 讀回，同一個視窗、同樣的鍵，不碰 sherpa-onnx
#   ./present.sh check         課前體檢：版本、模型檔、encoder metadata（chunk／pad）、整句離線解一次對文字與時間戳 → runs/check.json
#   ./present.sh fetch         課前一週：下載模型 tar.bz2 到 ~/.cache/sherpa-onnx 並解壓（要網路）
#   ./present.sh fake          沒有模型也能演練流程（合成的發射時間，畫面標明 FAKE）
#   ./present.sh selftest      不需要模型：對齊、併字、比對、統計、D_alg、FAKE 的時鐘關係
#   ./present.sh export        rehearsal.json 的摘要 → slides/assets/w04/w04-metrics.json（demo2）
#   ./present.sh recompute     改了後處理（併字、比對、D_alg）之後：從已存的事件重算 runs/rehearsal/ 與 check.json，不碰模型、不改時鐘
#
# 子命令後面可加 --mute（show 不播音）。
set -euo pipefail
cd "$(dirname "$0")"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
PY="../.venv/bin/python"   # 所有 demo 共用 speech_ai/demos/.venv
[[ -x "$PY" ]] || { echo "還沒有共用環境，先跑 ../setup.sh，再跑 ./setup.sh 裝 sherpa-onnx" >&2; exit 1; }

cmd="${1:-show}"; shift || true
case "$cmd" in
  show|rehearse|replay|check|fake|selftest|export|recompute) exec "$PY" demo.py "$cmd" "$@" ;;
  fetch) HF_HUB_OFFLINE=0 exec "$PY" demo.py fetch "$@" ;;
  *) echo "不認得的子命令：${cmd}" >&2; sed -n '2,14p' "$0" >&2; exit 2 ;;
esac

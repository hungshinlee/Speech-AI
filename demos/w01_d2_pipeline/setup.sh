#!/usr/bin/env bash
# 此檔由 bin/publish_demos.py 從 private repo 的 <course>/demos/ 複製而來，請勿在公開 repo 直接編輯。
# W1 demo 2 的模型套件，裝進 speech_ai/demos 共用的 .venv（先跑 ../setup.sh 建好基底）。
#
#   ./setup.sh            裝 ASR / LLM / TTS / 全雙工四組套件，版本寫進 ../versions.lock
#
# 版本政策（speech_ai/CLAUDE.md §6，2026-09-16 查證）：
#   - pywhispercpp 1.5.1：有 arm64 wheel，課堂安裝最省事；Metal 走 whisper.cpp。
#   - mlx-audio 0.5.4：§6 查到的版本（2026-09-14）。
#   - moshi-mlx 0.3.0：PyPI 2025-08-04。**它釘 mlx<0.27，與 mlx-lm 衝突，所以裝在 uv tool 的獨立環境**（見下）。
#   - mlx-lm：**照 nlp_llm/demos 的做法釘 git commit**（PyPI 版落後 main 數月）；
#     沒有指定就用 nlp_llm/demos/versions.lock 裡那一個，確保兩門課的 LLM demo 跑同一份程式。
#   - mlx：與 nlp_llm 同版（0.32.2）。
# 這些版本**還沒在這台機器上一起裝過**——四組套件對 mlx 版本的要求可能互相牽制，第一次跑失敗就
# 把衝突的那一組拆到 requirements 裡註解掉、先跑其餘的，並把結果記進 README「彩排時要確認的事」。
set -euo pipefail
cd "$(dirname "$0")"
DEMOS=".."
PY="$DEMOS/.venv/bin/python"
[[ -x "$PY" ]] || { echo "先跑 $DEMOS/setup.sh 建共用環境" >&2; exit 1; }
[[ "$(uname -s)" == "Darwin" && "$(uname -m)" == "arm64" ]] || {
  echo "這台不是 Apple silicon 的 macOS。mlx 裝不起來；只能用 ./present.sh fake 演練流程。" >&2; exit 1; }
command -v uv >/dev/null || { echo "需要 uv：brew install uv" >&2; exit 1; }

PYWHISPERCPP_VERSION="1.5.1"
MLX_AUDIO_VERSION="0.5.4"
MOSHI_MLX_VERSION="0.3.0"
MLX_VERSION="0.32.2"
MLX_LM_REPO="https://github.com/ml-explore/mlx-lm"

MLX_LM_COMMIT="$(grep -E '^MLX_LM_COMMIT=' "$DEMOS/versions.lock" 2>/dev/null | cut -d= -f2 || true)"
if [[ -z "$MLX_LM_COMMIT" ]]; then
  MLX_LM_COMMIT="$(grep -E '^MLX_LM_COMMIT=' ../../../nlp_llm/demos/versions.lock 2>/dev/null | cut -d= -f2 || true)"
fi
if [[ -z "$MLX_LM_COMMIT" ]]; then
  MLX_LM_COMMIT="$(git ls-remote "$MLX_LM_REPO" refs/heads/main | cut -f1)"
  echo "mlx-lm：鎖定 main 目前的 HEAD = $MLX_LM_COMMIT"
fi

# moshi-mlx 0.3.0 釘 mlx>=0.26,<0.27，與 mlx-lm 需要的 0.32.2 不相容（2026-09-20 實測，uv 直接拒絕）。
# 它只給 `moshi` 子命令用，所以**不裝進共用 .venv**，改用 uv 的獨立工具環境：
#   uv tool install "moshi-mlx==0.3.0"     → 之後 demo.py 的 moshi 子命令用 `uvx --from moshi-mlx==0.3.0 python -m moshi_mlx.local`
# 這正是 §6 說的「發行方式的障礙不是專案的障礙」的又一例：同一台機器上兩個 mlx 版本並存，各跑各的。
uv pip install --python "$PY" \
  "pywhispercpp==${PYWHISPERCPP_VERSION}" \
  "mlx==${MLX_VERSION}" \
  "mlx-lm @ git+${MLX_LM_REPO}@${MLX_LM_COMMIT}" \
  "mlx-audio==${MLX_AUDIO_VERSION}" \
  "misaki[en,zh]"
# misaki：Kokoro 的文字前處理（G2P），mlx-audio 沒把它列成必要相依（2026-09-20 實測 ImportError）。
# [zh] 這個 extra 帶中文 G2P（jieba 等）；沒釘版本，第一次裝的版本會進 requirements.lock.txt。
echo
echo "moshi-mlx（獨立環境，與共用 .venv 的 mlx 版本互不干擾）："
uv tool install --force "moshi-mlx==${MOSHI_MLX_VERSION}" || echo "moshi-mlx 裝不起來；只影響 ./present.sh moshi"

{
  grep -E '^#' "$DEMOS/versions.lock" || true
  {
    grep -E '^[A-Z0-9_]+=' "$DEMOS/versions.lock" | grep -vE '^(PYWHISPERCPP_VERSION|MLX_AUDIO_VERSION|MOSHI_MLX_VERSION|MLX_VERSION|MLX_LM_COMMIT)=' || true
    echo "MLX_AUDIO_VERSION=${MLX_AUDIO_VERSION}"
    echo "MLX_LM_COMMIT=${MLX_LM_COMMIT}"
    echo "MLX_VERSION=${MLX_VERSION}"
    echo "MOSHI_MLX_VERSION=${MOSHI_MLX_VERSION}"
    echo "PYWHISPERCPP_VERSION=${PYWHISPERCPP_VERSION}"
  } | LC_ALL=C sort
} > "$DEMOS/versions.lock.tmp" && mv "$DEMOS/versions.lock.tmp" "$DEMOS/versions.lock"
uv pip freeze --python "$PY" > "$DEMOS/requirements.lock.txt"
echo
echo "下一步：./present.sh check（第一次會下載模型，要等）"

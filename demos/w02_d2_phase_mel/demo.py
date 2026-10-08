# -*- coding: utf-8 -*-
# 此檔由 bin/publish_demos.py 從 private repo 的 <course>/demos/ 複製而來，請勿在公開 repo 直接編輯。
"""W2 demo 2：把「從 spectrogram 重建波形」的失真拆成兩個因子來聽。

    magnitude ∈ {linear, mel-n 經 pseudo-inverse 還原} × 相位 ∈ {原始, Griffin-Lim(k 輪), 隨機}

  record     錄一句話當素材（或自己放一個 WAV）
  rehearse   課前：完整算一次，存進 runs/rehearsal/<素材>/（進版控，現場失敗時的退路）
  show       課堂：現場重算一次（存進 runs/live/），然後進入播放畫面
  replay     現場失敗時：直接播 runs/rehearsal/ 裡的版本（畫面標明非現場）

只依賴 numpy、scipy、matplotlib；播放用 macOS 內建的 afplay。
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tomllib
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import common as C  # noqa: E402
import dsp          # noqa: E402

CFG = tomllib.loads((HERE / "demo_config.toml").read_text(encoding="utf-8"))
FS = CFG["audio"]["fs"]
L = int(round(FS * CFG["stft"]["win_ms"] / 1000))
H = int(round(FS * CFG["stft"]["hop_ms"] / 1000))
N = CFG["stft"]["n_fft"]
ITERS = sorted(CFG["griffin_lim"]["iters"])
BINS = CFG["mel"]["bins"]


# ── 素材 ────────────────────────────────────────────────────────
def fake_sentence() -> np.ndarray:
    """合成的 /a/ /i/ /u/ 加一段噪音。只用來測流程，不是給學生聽的。"""
    rng = np.random.default_rng(1)
    parts = [dsp.synth_vowel(FS, 0.6, np.linspace(110, 160, 60), dsp.VOWELS[v]) for v in "aiu"]
    return np.concatenate(parts + [0.05 * rng.standard_normal(FS // 4)])


def load_source(args) -> tuple[str, np.ndarray]:
    if args.fake:
        return "FAKE", fake_sentence()
    path = Path(args.wav) if args.wav else HERE / CFG["audio"]["default"]
    if not path.exists():
        sys.exit(f"找不到素材 {path}。先跑 ./present.sh record，或用 --wav 指定一個 WAV。")
    x = C.load_wav(path, FS)
    return path.stem, x / (np.max(np.abs(x)) + 1e-12) * 0.5


# ── 計算 ────────────────────────────────────────────────────────
def compute(x: np.ndarray, out: Path, tag: str) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    X = dsp.stft(x, L, H, N)
    A, phase = np.abs(X), np.exp(1j * np.angle(X))
    n = len(x)
    mc = CFG["mel"]
    waves: dict[str, np.ndarray] = {"orig": x}
    gl_logs = {}

    def mel_mag(bins: int) -> np.ndarray:
        M = dsp.mel_filterbank(FS, N, bins, mc["scale"], mc["area_norm"])
        return np.sqrt(dsp.mel_inverse(M @ (A * A), M, mc["inverse"]))

    mags = {"lin": A, f"mel{BINS[0]}": mel_mag(BINS[0])}
    for name, mag in mags.items():
        C.note(f"  {name}: 原始相位、Griffin-Lim {ITERS[-1]} 輪……")
        waves[f"{name}_orig"] = dsp.istft(mag * phase, L, H, N, n)
        _, log, snaps = dsp.griffin_lim(mag, L, H, N, n, ITERS[-1],
                                        seed=CFG["griffin_lim"]["seed"], keep=tuple(ITERS))
        gl_logs[name] = log
        for k, w in snaps.items():
            waves[f"{name}_gl{k:03d}"] = w
    for b in BINS[1:]:
        waves[f"mel{b}_orig"] = dsp.istft(mel_mag(b) * phase, L, H, N, n)

    peak = max(float(np.max(np.abs(w))) for w in waves.values())   # 所有版本同一個增益
    for k, w in waves.items():
        C.save_wav(out / f"{k}.wav", w, FS, peak)

    metrics = {k: {"lsd_db": round(dsp.log_spectral_distance(x, w, L, H, N), 2),
                   "snr_db": round(dsp.snr_db(x, w), 1)} for k, w in waves.items() if k != "orig"}
    mono = {name: all(b["inconsistency"] <= a["inconsistency"] + 1e-12 for a, b in zip(log, log[1:]))
            for name, log in gl_logs.items()}
    meta = {"tag": tag, "time": C.now(), "seconds": round(n / FS, 2), "config": CFG,
            "versions": C.versions(), "metrics": metrics, "gl_monotone": mono,
            "gl_log": {name: [l for l in log if l["iter"] in ITERS] for name, log in gl_logs.items()}}
    (out / "metrics.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    figures(out, waves, gl_logs)
    return meta


def figures(out: Path, waves: dict, gl_logs: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4))
    for name, log in gl_logs.items():
        ax.semilogy([l["iter"] for l in log], [l["inconsistency"] for l in log], label=name)
    ax.set_xlabel("Griffin-Lim iteration")
    ax.set_ylabel(r"inconsistency  $\|C - P(C)\|^2 / \|C\|^2$")
    ax.set_title("Monotone non-increasing, but it does not reach zero")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "convergence.png", dpi=160)
    plt.close(fig)

    top = f"gl{ITERS[-1]:03d}"
    rows = [("lin", "linear magnitude"), (f"mel{BINS[0]}", f"mel-{BINS[0]} → linear")]
    cols = [("orig", "original phase"), (top, f"Griffin-Lim ×{ITERS[-1]}"), ("gl000", "random phase")]
    fig, axes = plt.subplots(2, 3, figsize=(13, 6), sharex=True, sharey=True)
    for i, (r, rl) in enumerate(rows):
        for j, (c, cl) in enumerate(cols):
            S = dsp.spectrogram_db(waves[f"{r}_{c}"], FS, CFG["stft"]["win_ms"], 5.0, N)
            axes[i, j].imshow(S, origin="lower", aspect="auto", cmap="magma",
                              vmin=S.max() - 70, vmax=S.max(), extent=[0, S.shape[1] * 0.005, 0, FS / 2000])
            axes[i, j].set_title(f"{rl} × {cl}", fontsize=10)
    for a in axes[:, 0]:
        a.set_ylabel("kHz")
    for a in axes[1]:
        a.set_xlabel("s")
    fig.tight_layout()
    fig.savefig(out / "spectrograms.png", dpi=140)
    plt.close(fig)


# ── 播放畫面 ────────────────────────────────────────────────────
def screen(run: Path, meta: dict, k: int, live: bool) -> None:
    m = meta["metrics"]
    mel = f"mel{BINS[0]}"

    def cell(key: str) -> str:
        return f"LSD {m[key]['lsd_db']:5.2f} dB"

    print("\033[2J\033[H", end="")
    C.banner(f"W2 Demo 2 — phase × magnitude   [{meta['tag']}，{meta['seconds']} s]",
             C.BLUE if live else C.AMBER)
    if not live:
        print(f"{C.AMBER}非現場：這是 {meta['time']} 彩排時算好的版本。{C.RESET}")
    gl = f"gl{k:03d}"
    print(f"\n{'':18}{'原始相位':^18}{f'Griffin-Lim ×{k}':^22}{'隨機相位':^18}")
    print(f"  {'linear':<14}  [1] {cell('lin_orig')}   [2] {cell('lin_' + gl)}     [3] {cell('lin_gl000')}")
    print(f"  {mel + ' → lin':<14}  [4] {cell(mel + '_orig')}   [5] {cell(mel + '_' + gl)}     [6] {cell(mel + '_gl000')}")
    print("\n  mel 掃描（原始相位，隔離 mel 的損失）：  " +
          "   ".join(f"[{key}] mel-{b}  {cell(f'mel{b}_orig')}" for key, b in zip("asdf", BINS)))
    log = {l["iter"]: l for l in meta["gl_log"]["lin"]}[k]
    print(f"\n  linear 的 Griffin-Lim 在第 {k} 輪：inconsistency = {log['inconsistency']:.4f}"
          f"（單調不增：{'是' if meta['gl_monotone']['lin'] else '否'}）")
    C.note("\n  [0] 原音   [ / ] 調 Griffin-Lim 輪數   [c] 收斂曲線   [g] spectrogram 對照   [q] 離開")
    C.note("  LSD = 與原音的 log-spectral distance，只看 magnitude；相位的損失它量不到，要用聽的。")


def player(run: Path, live: bool) -> None:
    meta = json.loads((run / "metrics.json").read_text(encoding="utf-8"))
    mel = f"mel{BINS[0]}"
    i = ITERS.index(CFG["griffin_lim"]["start"]) if CFG["griffin_lim"]["start"] in ITERS else len(ITERS) - 1
    while True:
        k = ITERS[i]
        gl = f"gl{k:03d}"
        screen(run, meta, k, live)
        files = {"0": "orig", "1": "lin_orig", "2": f"lin_{gl}", "3": "lin_gl000",
                 "4": f"{mel}_orig", "5": f"{mel}_{gl}", "6": f"{mel}_gl000",
                 **{key: f"mel{b}_orig" for key, b in zip("asdf", BINS)}}
        key = C.getkey()
        if key in ("q", "\x03", ""):
            return
        if key == "[":
            i = max(0, i - 1)
        elif key == "]":
            i = min(len(ITERS) - 1, i + 1)
        elif key in ("c", "g"):
            png = run / ("convergence.png" if key == "c" else "spectrograms.png")
            subprocess.run(["open", str(png)], check=False)
        elif key in files:
            print(f"  ▶ {files[key]}.wav")
            C.play(run / f"{files[key]}.wav")


# ── 入口 ────────────────────────────────────────────────────────
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--wav", help="素材 WAV（預設 demo_config.toml 的 audio.default）")
    ap.add_argument("--fake", action="store_true", help="用合成訊號測流程（不是給學生聽的）")
    ap.add_argument("cmd", choices=["record", "rehearse", "show", "replay"])
    args = ap.parse_args()

    if args.cmd == "record":
        path = Path(args.wav) if args.wav else HERE / CFG["audio"]["default"]
        x = C.record(CFG["audio"]["record_seconds"], FS)
        C.save_wav(path, x, FS, float(np.max(np.abs(x))) + 1e-12)
        print(f"存到 {path}")
        C.play(path)
        return

    tag, x = load_source(args)
    rehearsal = HERE / "runs" / "rehearsal" / tag
    if args.cmd == "replay":
        if not (rehearsal / "metrics.json").exists():
            sys.exit(f"沒有彩排紀錄 {rehearsal}。先跑 ./present.sh rehearse。")
        return player(rehearsal, live=False)

    out = rehearsal if args.cmd == "rehearse" else HERE / "runs" / "live" / f"{C.now().replace(':', '')}-{tag}"
    C.banner(f"計算中：{tag}（{len(x) / FS:.1f} s）→ {out.relative_to(HERE)}")
    meta = compute(x, out, tag)
    if args.cmd == "rehearse":
        for k, v in meta["metrics"].items():
            print(f"  {k:<14} LSD {v['lsd_db']:6.2f} dB   SNR {v['snr_db']:6.1f} dB")
        print(f"  Griffin-Lim 單調不增：{meta['gl_monotone']}")
        print(f"\n圖：{out / 'convergence.png'}\n    {out / 'spectrograms.png'}\n下一步：./present.sh replay 聽一遍。")
        return
    player(out, live=True)


if __name__ == "__main__":
    main()

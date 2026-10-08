# -*- coding: utf-8 -*-
# 此檔由 bin/publish_demos.py 從 private repo 的 <course>/demos/ 複製而來，請勿在公開 repo 直接編輯。
"""dsp.py 的自我檢查。只用合成訊號，不需要音訊裝置。setup.sh 最後會跑一次；換版本之後也跑一次。

    ../demos/.venv/bin/python selftest.py
"""
import numpy as np

import dsp

fs, L, H, N = 16000, 400, 160, 512
rng = np.random.default_rng(1)
x = np.concatenate([dsp.synth_vowel(fs, 0.5, np.linspace(120, 150, 50), dsp.VOWELS[v]) for v in "aiu"]
                   + [0.05 * rng.standard_normal(3000)])
fails = []


def check(name: str, ok: bool, detail: str) -> None:
    print(f"  {'OK  ' if ok else 'FAIL'} {name}: {detail}")
    if not ok:
        fails.append(name)


X = dsp.stft(x, L, H, N)
err = float(np.max(np.abs(x - dsp.istft(X, L, H, N, len(x)))))
check("STFT → ISTFT 完美重建（hop 不滿足 COLA 也成立）", err < 1e-10, f"最大誤差 {err:.1e}")

_, log, _ = dsp.griffin_lim(np.abs(X), L, H, N, len(x), 100)
inc = [l["inconsistency"] for l in log]
check("Griffin-Lim 的 inconsistency 單調不增", all(b <= a + 1e-12 for a, b in zip(inc, inc[1:])),
      f"{inc[0]:.3f} → {inc[-1]:.4f}")
check("……但 100 輪之後仍然不是 0（收到局部解）", inc[-1] > 1e-6, f"{inc[-1]:.4f}")

for scale in ("htk", "slaney"):
    M = dsp.mel_filterbank(fs, N, 80, scale)
    check(f"mel filterbank（{scale}）沒有空的 filter", bool((M.sum(1) > 0).all()), str(M.shape))
a = dsp.mel_to_hz(np.linspace(0, dsp.hz_to_mel(8000, "htk"), 82), "htk")[1:-1]
b = dsp.mel_to_hz(np.linspace(0, dsp.hz_to_mel(8000, "slaney"), 82), "slaney")[1:-1]
check("兩種 mel 尺度的中心頻率確實不同", float(np.max(np.abs(a - b))) > 50,
      f"80 個中心頻率最多差 {np.max(np.abs(a - b)):.0f} Hz")

for f in (90.0, 130.0, 220.0, 380.0):
    est, _ = dsp.f0_autocorr(dsp.synth_vowel(fs, 0.3, f, dsp.VOWELS["a"])[2000:2640], fs, 75, 400)
    check(f"F0 {f:.0f} Hz", abs(est - f) < 0.02 * f, f"估計 {est:.1f}")
est, _ = dsp.f0_autocorr(dsp.synth_vowel(fs, 0.3, 220.0, dsp.VOWELS["a"])[2000:2640], fs, 75, 150)
check("ceiling 設成 150 去追 220 Hz → octave error（這是預期的）", abs(est - 110) < 3, f"估計 {est:.1f}")

for v, (f1, f2, _) in dsp.VOWELS.items():
    e1, e2 = dsp.formants(dsp.synth_vowel(fs, 0.3, 130.0, dsp.VOWELS[v])[2000:2480], fs)
    check(f"/{v}/ 的 F1、F2", abs(e1 - f1) < 60 and abs(e2 - f2) < 60, f"估計 {e1:.0f}, {e2:.0f}；合成時給的是 {f1}, {f2}")

print("\n全部通過。" if not fails else f"\n{len(fails)} 項失敗：{fails}")
raise SystemExit(1 if fails else 0)

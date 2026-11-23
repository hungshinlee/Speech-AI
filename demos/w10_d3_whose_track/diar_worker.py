# -*- coding: utf-8 -*-
# 此檔由 bin/publish_demos.py 從 private repo 的 <course>/demos/ 複製而來，請勿在公開 repo 直接編輯。
"""W10 demo 3 的子程序：pyannote.audio 住在另一個環境（../.venv-diar），demo.py 用子程序叫它，結果寫成 json。

  run     一個 wav → pipeline 的輸出：[[開始, 結束, 標籤], …]
  score   同一份參考與系統輸出交給 pyannote.metrics 算三種計分的 DER（check 用來對 der.py）

為什麼另一個環境：pyannote.audio 4.1.1 要 matplotlib ≥ 3.11，而共用環境釘的是 3.10.9（投影片的圖是用它產的）。
pyannote.audio 4 預設會回傳使用統計（telemetry）；這裡在 import 之前就關掉。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import warnings
from pathlib import Path

os.environ["PYANNOTE_METRICS_ENABLED"] = "0"
warnings.filterwarnings("ignore")


def versions() -> dict:
    from importlib import metadata
    v = {}
    for pkg in ("pyannote.audio", "pyannote.metrics", "pyannote.core", "torch", "torchaudio"):
        try:
            v[pkg] = metadata.version(pkg)
        except metadata.PackageNotFoundError:
            v[pkg] = None
    return v


def run(a) -> dict:
    import numpy as np
    import torch
    from scipy.io import wavfile
    from pyannote.audio import Pipeline
    try:
        from pyannote.audio.telemetry import set_telemetry_metrics
        set_telemetry_metrics(False)
    except Exception:
        pass
    fs, x = wavfile.read(a.wav)
    x = x.astype(np.float32) / (32768.0 if x.dtype == np.int16 else 1.0)
    pipe = Pipeline.from_pretrained(a.model)
    dev = a.device
    if dev == "auto":
        dev = "mps" if torch.backends.mps.is_available() else "cpu"
    pipe.to(torch.device(dev))
    # 直接給波形，不經它的檔案解碼（那一條路要 torchcodec 與 ffmpeg）
    t0 = time.perf_counter()
    out = pipe({"waveform": torch.from_numpy(x)[None], "sample_rate": int(fs)})
    dt = time.perf_counter() - t0
    ann = getattr(out, "speaker_diarization", out)
    segs = [[round(float(t.start), 3), round(float(t.end), 3), str(s)] for t, _, s in ann.itertracks(yield_label=True)]
    res = {"segments": sorted(segs), "device": dev, "compute_s": dt, "versions": versions()}
    excl = getattr(out, "exclusive_speaker_diarization", None)
    if excl is not None:
        res["exclusive_segments"] = sorted([round(float(t.start), 3), round(float(t.end), 3), str(s)] for t, _, s in excl.itertracks(yield_label=True))
    return res


def score(a) -> dict:
    from pyannote.core import Annotation, Segment, Timeline
    from pyannote.metrics.diarization import DiarizationErrorRate

    def ann(path: str) -> Annotation:
        out = Annotation()
        for i, (s, e, lab) in enumerate(json.loads(Path(path).read_text())):
            out[Segment(float(s), float(e)), i] = str(lab)
        return out

    ref, hyp = ann(a.ref), ann(a.hyp)
    uem = Timeline([Segment(0.0, float(a.seconds))])
    scores = []
    # pyannote.metrics 的 collar 是「邊界前後合計」的寬度，所以前後各 c 秒要給 2c
    for collar, skip in ((a.collar, True), (a.collar, False), (0.0, False)):
        m = DiarizationErrorRate(collar=2 * collar, skip_overlap=skip)
        d = m(ref, hyp, uem=uem, detailed=True)
        scores.append({"collar": collar, "skip_overlap": skip, "der": float(d["diarization error rate"]), "fa": float(d["false alarm"]),
                       "miss": float(d["missed detection"]), "conf": float(d["confusion"]), "total": float(d["total"])})
    return {"scores": scores, "versions": versions()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run", "score"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--wav")
    ap.add_argument("--model")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--ref")
    ap.add_argument("--hyp")
    ap.add_argument("--seconds", type=float)
    ap.add_argument("--collar", type=float, default=0.25)
    a = ap.parse_args()
    try:
        res = run(a) if a.cmd == "run" else score(a)
    except Exception as e:
        res = {"error": f"{type(e).__name__}: {str(e)[:400]}"}
    Path(a.out).write_text(json.dumps(res), encoding="utf-8")
    if "error" in res:
        print(res["error"], file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

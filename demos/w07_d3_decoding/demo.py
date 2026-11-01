# -*- coding: utf-8 -*-
# 此檔由 bin/publish_demos.py 從 private repo 的 <course>/demos/ 複製而來，請勿在公開 repo 直接編輯。
"""W7 demo 3：解碼——先驗的位置與重量（一個旋鈕 λ 掃一遍；同一個 CTC 模型帶到華語數同音字）。

  show       課堂：載入 wav2vec2-base-960h、3-gram LM（與華語那半的 MMS-1B-all、Whisper small、Qwen3-ASR），終端機單鍵操作（下面的鍵）。
             離開時存 runs/live/<時間>/（不進版控）
  rehearse   課前：不互動，先在調參集上調 η，再照課堂順序全部跑一遍 → runs/rehearsal/（進版控 = 現場的退路；含 lambda_sweep.png、eta.json）；
             華語那半 → ../_media/w07_d3/private_rehearsal.*（不進版控、不上站——裡面有我自己錄音的文字）
  replay     現場模型出狀況時：從彩排紀錄讀回，同樣的鍵、同樣的畫面，不碰 torch
  check      課前體檢：版本、模型在不在快取、ARPA 讀得起來（第一次順便建 .npz 快取）、LM 的 backoff、beam λ=0 對 greedy、
             MMS-1B-all 在 MPS 還是 CPU、一次有 LM 的 beam search 要多久 → runs/check.json
  fetch      課前一週：下載 wav2vec2（commit 沿 W3 d3／W7 d2 的 model.lock）、MMS-1B-all（只抓 model.safetensors 與 cmn 的 adapter）、
             3-gram ARPA（openslr 11）；從 dev-clean.tar.gz 讀出測試集與調參集的參考文字 → data/sentences.json；要網路
  record     錄一句自己的華語 → ../_media/w07_d3/mandarin.wav，接著輸入參考文字 → mandarin.txt
  fake       沒有模型也能演練流程（合成的 log 機率 + 小 LM，畫面標明 FAKE，不是給學生看的）
  selftest   不需要模型：ARPA 解析與 backoff 對手算、LMBonus 的狀態、prefix beam 對 p32 的窮舉（w07-data.json）、WER／字元對齊、
             同音判定、句子挑選的確定性、fake 整條流程
  export     runs/rehearsal/ 的摘要 → slides/assets/w07/w07-metrics.json（鍵 demo3；華語只進數字，public_text 的句子才進文字）

畫面上的鍵（順序照投影片 p41 講稿的【操作】）：
  0  p32 的數字例子：九條路徑窮舉、greedy 的輸出、prefix beam 的輸出（對 w07-data.json）
  1  greedy（無 LM）與 beam λ=0 的 WER（測試集）
  2  λ 掃描：逐點印 WER、視窗上逐點畫曲線；最後印最佳 λ
  3  挑一句：λ=0／最佳 λ／過大 λ 三個版本的轉錄，錯的詞標色，分數拆成 ln P_ctc、λ·ln P_LM、η·詞數      n  換下一句
  m  華語：每個可用的輸入過 MMS-1B-all CTC（cmn adapter、greedy、無 LM；輸出轉繁體）、Whisper small、Qwen3-ASR；
     對參考文字做字元對齊，替換的字用 pypinyin 判同音
  e  調 η 的結果（調參集上每個 η 的 WER）      l  LM 本身：幾個 n-gram、<unk>、幾個例句的 ln P
  t  投影片 p41 的表      h  重印選單      q  離開

判定的定義（課堂上要念出來的）：
  WER        測試集全部參考詞數為分母的 corpus WER（Levenshtein，詞級；大寫、去標點；LibriSpeech 的參考本來就是這個格式）
  分數       ln P_ctc(prefix) + λ·Σ ln P_LM(w | 前兩詞)（含 </s>）+ η·詞數；LM 在詞界處乘，走在詞中間的字元不算；
             字典裡沒有的詞當 <unk> 再加 oov_ln_penalty + oov_ln_per_char×字元數；修剪時半個詞先當 OOV 估分（toml；沒有這兩件事 λ 一大 beam 會把空白吃掉、整句當一個 <unk>）
  同音       替換掉的那個字與參考的字，pypinyin 的讀音（含多音字）有交集：同音同調／同音異調；都沒有就是異音。繁簡轉成簡體再比，字體不算錯
"""
from __future__ import annotations

import argparse
import difflib
import importlib.util
import itertools
import json
import math
import random
import re
import sys
import time
import tomllib
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
D2_DIR = HERE.parent / "w07_d2_biasing"
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(D2_DIR))
sys.path.insert(0, str(HERE))
import common as C  # noqa: E402
from ctc_prefix_beam import greedy as ctc_greedy, prefix_beam_search  # noqa: E402
from arpa_lm import LN10, ArpaLM, LMBonus  # noqa: E402


def _load_d2():
    """Demo 2 的 demo.py 當模組載入：Whisper、Qwen、CTC 三個類別、繁簡對照表與 Out 都沿用（W6 d2 載 d1 的前例）。"""
    spec = importlib.util.spec_from_file_location("w07_d2_demo", D2_DIR / "demo.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


D2 = _load_d2()

CFG = tomllib.loads((HERE / "demo_config.toml").read_text(encoding="utf-8"))
FS = int(CFG["audio"]["fs"])
DIG = int(CFG["display"]["digits"])
MODEL_LOCK = HERE / "model.lock"
PRIVATE = (HERE / CFG["mandarin"]["dir"]).resolve()
METRICS = (HERE.parent.parent / "slides" / "assets" / "w07" / "w07-metrics.json").resolve()
DATA_JSON = (HERE.parent.parent / "slides" / "assets" / "w07" / "w07-data.json").resolve()
SENTENCES = HERE / "data" / "sentences.json"
SIBLING_LOCKS = {CFG["ctc"]["repo"]: [D2_DIR / "model.lock", HERE.parent / "w07_d1_hallucination" / "model.lock", HERE.parent / "w03_d3_blank_heatmap" / "model.lock"],
                 CFG["whisper"]["repo"]: [D2_DIR / "model.lock", HERE.parent / "w07_d1_hallucination" / "model.lock"],
                 CFG["qwen"]["repo"]: [D2_DIR / "model.lock"]}
PATTERNS = ["*.json", "*.txt", "*.safetensors", "*.jinja", "*.model", "*.tiktoken"]
MMS_PATTERNS = ["*.json", "*.txt", "model.safetensors", f"adapter.{CFG['mms']['adapter']}.safetensors"]


def _p(rel: str) -> Path:
    return (HERE / rel).resolve()


# ── 純函數（selftest 測這些）──────────────────────────────────────────────
def norm_en(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^A-Z0-9' ]+", " ", text.upper())).strip()


def align_words(ref: list[str], hyp: list[str]) -> dict:
    """詞級 Levenshtein 加回溯：{"errors", "S", "D", "I", "n_ref", "ops"}；ops 是 (op, ref_word|None, hyp_word|None)。"""
    R, H = len(ref), len(hyp)
    d = np.zeros((R + 1, H + 1), dtype=np.int32)
    d[:, 0] = np.arange(R + 1)
    d[0, :] = np.arange(H + 1)
    for i in range(1, R + 1):
        for j in range(1, H + 1):
            d[i, j] = min(d[i - 1, j] + 1, d[i, j - 1] + 1, d[i - 1, j - 1] + (ref[i - 1] != hyp[j - 1]))
    ops = []
    i, j = R, H
    while i > 0 or j > 0:
        if i > 0 and j > 0 and d[i, j] == d[i - 1, j - 1] + (ref[i - 1] != hyp[j - 1]):
            ops.append(("C" if ref[i - 1] == hyp[j - 1] else "S", ref[i - 1], hyp[j - 1]))
            i, j = i - 1, j - 1
        elif i > 0 and d[i, j] == d[i - 1, j] + 1:
            ops.append(("D", ref[i - 1], None))
            i -= 1
        else:
            ops.append(("I", None, hyp[j - 1]))
            j -= 1
    ops.reverse()
    S = sum(1 for o in ops if o[0] == "S")
    Dl = sum(1 for o in ops if o[0] == "D")
    I = sum(1 for o in ops if o[0] == "I")
    return {"errors": int(d[R, H]), "S": S, "D": Dl, "I": I, "n_ref": R, "ops": ops}


def wer_words(ref: str, hyp: str) -> dict:
    a = align_words(norm_en(ref).split(), norm_en(hyp).split())
    a["wer"] = a["errors"] / a["n_ref"] if a["n_ref"] else 0.0
    return a


def corpus_wer(refs: list[str], hyps: list[str]) -> dict:
    tot_e, tot_n, S, Dl, I = 0, 0, 0, 0, 0
    per = []
    for r, h in zip(refs, hyps):
        a = wer_words(r, h)
        tot_e += a["errors"]
        tot_n += a["n_ref"]
        S, Dl, I = S + a["S"], Dl + a["D"], I + a["I"]
        per.append(a["errors"])
    return {"errors": tot_e, "n_ref": tot_n, "wer": tot_e / tot_n if tot_n else 0.0, "S": S, "D": Dl, "I": I, "per_sentence": per}


def select_sentences(subset: dict, cfg: dict) -> dict:
    """每位說話人一句（長度在 min_s–max_s 之間、最接近 target_s）→ 依 seed 洗牌 → 前 n_test 是測試集、接著 n_tune 是調參集。確定性。"""
    by_spk: dict[str, list[dict]] = {}
    for u in subset["utts"]:
        by_spk.setdefault(str(u["spk"]), []).append(u)
    picked = []
    for spk in sorted(by_spk, key=lambda s: int(s) if s.isdigit() else s):
        us = by_spk[spk]
        ok = [u for u in us if float(cfg["min_s"]) <= float(u["dur_s"]) <= float(cfg["max_s"])] or us
        best = min(ok, key=lambda u: (abs(float(u["dur_s"]) - float(cfg["target_s"])), u["id"]))
        picked.append({"id": best["id"], "spk": spk, "dur_s": float(best["dur_s"])})
    rng = random.Random(int(cfg["seed"]))
    rng.shuffle(picked)
    n_test, n_tune = int(cfg["n_test"]), int(cfg["n_tune"])
    return {"test": picked[:n_test], "tune": picked[n_test:n_test + n_tune], "n_speakers": len(picked)}


_PUNCT_ZH = r"[\s，。、！？；：「」『』（）()《》〈〉—…,.!?;:\"'\-]"
# Demo 2 的對照表沒收的幾個字（訂票句會用到）。只加「繁 → 簡」那一向：週→周 只能單向——周也是繁體字（周圍、周杰倫），
# 簡 → 繁 時不敢把每個周都換成週，這正是「一簡對多繁」讓轉換自己出錯的例子（p63）。
_EXTRA_T2S = {"訂": "订", "週": "周", "張": "张", "們": "们", "實": "实", "驗": "验", "師": "师", "贏": "赢", "選": "选", "舉": "举", "萬": "万", "長": "长"}
_EXTRA_S2T = {"订": "訂", "张": "張", "实": "實", "验": "驗", "师": "師", "赢": "贏", "选": "選", "举": "舉", "万": "萬", "长": "長"}
T2S = {**D2.T2S, **_EXTRA_T2S}
S2T = {**D2.S2T, **_EXTRA_S2T}


def to_simplified(text: str) -> str:
    return "".join(T2S.get(ch, ch) for ch in text)


def norm_zh(text: str) -> str:
    """華語：去標點與空白、轉成簡體（字體不算錯；對照表沿 Demo 2 加幾個字，不是 OpenCC）。"""
    return to_simplified(re.sub(_PUNCT_ZH, "", text))


def to_traditional(text: str) -> tuple[str, int]:
    """簡 → 繁（Demo 2 的對照表加幾個字）；回傳 (轉好的字串, 還留著的簡體專用字數)——表裡沒有的字、一簡對多繁的字（周）就轉不了，
    那是這個「normalization 步驟」自己的錯。"""
    out = "".join(S2T.get(ch, ch) for ch in text)
    residual = sum(1 for ch in out if ch in S2T and ch not in T2S and ch not in D2.AMBIG)
    return out, residual


def align_chars(ref: str, hyp: str) -> dict:
    a = align_words(list(ref), list(hyp))
    a["cer"] = a["errors"] / a["n_ref"] if a["n_ref"] else 0.0
    return a


def _readings(ch: str) -> set[str]:
    try:
        from pypinyin import Style, pinyin
    except ImportError:
        return set()
    return {p for p in pinyin(ch, style=Style.TONE3, heteronym=True)[0] if p}


def homophone(ref_ch: str, hyp_ch: str) -> str:
    """"same_tone"（同音同調）／"same_base"（同音異調）／"different"（異音）／"unknown"（沒有 pypinyin 或非漢字）。"""
    r, h = _readings(ref_ch), _readings(hyp_ch)
    if not r or not h:
        return "unknown"
    if r & h:
        return "same_tone"
    rb = {re.sub(r"\d", "", p) for p in r}
    hb = {re.sub(r"\d", "", p) for p in h}
    return "same_base" if rb & hb else "different"


def classify_subs(ops: list[tuple]) -> dict:
    """字元對齊的 ops → 替換清單與同音計數。"""
    subs = []
    cnt = {"same_tone": 0, "same_base": 0, "different": 0, "unknown": 0}
    for op, r, h in ops:
        if op == "S":
            k = homophone(r, h)
            cnt[k] += 1
            subs.append({"ref": r, "hyp": h, "kind": k})
    cnt["homophone"] = cnt["same_tone"] + cnt["same_base"]
    return {"subs": subs, "counts": cnt}


def collapse(path: list[int], blank: int) -> list[int]:
    out, prev = [], None
    for i in path:
        if i != prev and i != blank:
            out.append(i)
        prev = i
    return out


def prefix_example(data: dict) -> dict:
    """p32 的 T=2 例子：窮舉九條路徑、按標籤合併、greedy，再用 prefix_beam_search 跑一次；全部對 w07-data.json。符號 id：∅=0、a=1、b=2。"""
    y1, y2 = data["y1"], data["y2"]
    sym = ["∅", "a", "b"]
    lp = np.log(np.array([[y1["∅"], y1["a"], y1["b"]], [y2["∅"], y2["a"], y2["b"]]]))
    paths = []
    by_label: dict[str, float] = {}
    for p1, p2 in itertools.product(range(3), repeat=2):
        prob = float(np.exp(lp[0, p1] + lp[1, p2]))
        lab = "".join(sym[i] for i in collapse([p1, p2], 0))                             # 摺疊重複、刪 blank
        paths.append({"path": sym[p1] + sym[p2], "prob": round(prob, 6), "label": lab})
        by_label[lab] = round(by_label.get(lab, 0.0) + prob, 6)
    best_single = max(paths, key=lambda d: d["prob"])
    greedy_label = "".join(sym[i] for i in ctc_greedy(lp, 0))
    beams = prefix_beam_search(lp, 0, beam=8, prune_logp=-50, n_best=8)
    beam_label = "".join(sym[i] for i in beams[0]["ids"])
    beam_probs = {"".join(sym[i] for i in b["ids"]): round(float(np.exp(b["logp"])), 6) for b in beams}
    return {"paths": paths, "by_label": by_label, "best_single_path": best_single, "greedy_label": greedy_label, "beam_label": beam_label, "beam_probs": beam_probs,
            "ok": greedy_label == data["greedy_label"] and beam_label == data["best_label"] and all(abs(by_label.get(k, 0) - v) < 1e-9 for k, v in data["by_label"].items())
            and all(abs(beam_probs.get(k, 0) - v) < 1e-9 for k, v in data["by_label"].items() if k in beam_probs)}


def word_colored(ref: str, hyp: str) -> str:
    """hyp 的詞對 ref 標色：替換／插入紅、刪除的參考詞用 [灰] 標在位置上。"""
    a = align_words(norm_en(ref).split(), norm_en(hyp).split())
    out = []
    for op, r, h in a["ops"]:
        if op == "C":
            out.append(h)
        elif op == "S":
            out.append(f"{C.RED}{h}{C.RESET}{C.GREY}[{r}]{C.RESET}")
        elif op == "I":
            out.append(f"{C.RED}+{h}{C.RESET}")
        else:
            out.append(f"{C.GREY}[-{r}]{C.RESET}")
    return " ".join(out)


def chars_colored(ref: str, hyp: str, subs_kind: dict | None = None) -> str:
    a = align_chars(ref, hyp)
    out = []
    for op, r, h in a["ops"]:
        if op == "C":
            out.append(h)
        elif op == "S":
            k = homophone(r, h)
            col = C.AMBER if k in ("same_tone", "same_base") else C.RED
            out.append(f"{col}{h}{C.RESET}{C.GREY}[{r}]{C.RESET}")
        elif op == "I":
            out.append(f"{C.RED}+{h}{C.RESET}")
        else:
            out.append(f"{C.GREY}[-{r}]{C.RESET}")
    return "".join(out)


# ── 模型 ─────────────────────────────────────────────────────────────────
def read_lock() -> dict[str, str]:
    lock = {}
    if MODEL_LOCK.exists():
        for line in MODEL_LOCK.read_text(encoding="utf-8").splitlines():
            parts = line.split()
            if len(parts) == 2:
                lock[parts[0]] = parts[1]
    for repo, sibs in SIBLING_LOCKS.items():
        if repo in lock:
            continue
        for sib in sibs:
            if sib.exists():
                for line in sib.read_text(encoding="utf-8").splitlines():
                    parts = line.split()
                    if len(parts) == 2 and parts[0] == repo:
                        lock[repo] = parts[1]
                    elif len(parts) == 1 and parts[0] and repo == CFG["ctc"]["repo"] and "w03" in str(sib):
                        lock[repo] = parts[0]
                if repo in lock:
                    break
    return lock


def model_dir(repo: str, patterns: list[str] | None = None, offline: bool = True) -> str:
    from huggingface_hub import snapshot_download
    return snapshot_download(repo, revision=read_lock().get(repo), local_files_only=offline, allow_patterns=patterns or PATTERNS)


def pick_device(name: str):
    import torch
    if name == "auto":
        name = "mps" if torch.backends.mps.is_available() else "cpu"
    return torch.device(name)


class MMS:
    """facebook/mms-1b-all（Wav2Vec2ForCTC）配一個語言 adapter：只拿 log softmax 與 greedy。"""
    name = "mms"

    def __init__(self, device: str | None = None):
        import torch
        from transformers import AutoProcessor, Wav2Vec2ForCTC
        D2.quiet_transformers()
        self.torch = torch
        m = CFG["mms"]
        path = model_dir(m["repo"], MMS_PATTERNS)
        self.proc = AutoProcessor.from_pretrained(path)
        self.proc.tokenizer.set_target_lang(m["adapter"])
        try:
            self.model = Wav2Vec2ForCTC.from_pretrained(path, target_lang=m["adapter"], ignore_mismatched_sizes=True)
            self.how = "from_pretrained(target_lang)"
        except Exception:
            self.model = Wav2Vec2ForCTC.from_pretrained(path, ignore_mismatched_sizes=True)
            self.model.load_adapter(m["adapter"])
            self.how = "load_adapter"
        self.model.eval()
        self.device = pick_device(device or m["device"])
        self.model.to(self.device)
        self.blank = int(self.model.config.pad_token_id)
        self.n_params = sum(p.numel() for p in self.model.parameters())
        self.vocab_size = int(self.model.lm_head.out_features)

    def transcribe(self, x: np.ndarray) -> dict:
        torch = self.torch
        inp = self.proc(x.astype(np.float32), sampling_rate=FS, return_tensors="pt")
        t0 = time.perf_counter()
        with torch.no_grad():
            logits = self.model(inp.input_values.to(self.device)).logits[0].float()
        dt = time.perf_counter() - t0
        lp = torch.log_softmax(logits, dim=-1).cpu().numpy()
        am = lp.argmax(-1)
        ids = ctc_greedy(lp, self.blank)
        text = self.proc.decode(ids) if ids else ""
        return {"text": str(text).replace(" ", ""), "T": int(lp.shape[0]), "blank_frac": float(np.mean(am == self.blank)), "seconds": dt}


# ── fake 後端 ───────────────────────────────────────────────────────────────
# (id, 真正的句子, 「聲學上聽到的」句子——同長度、幾個字聽錯)
FAKE_SENTENCES = [("f1", "THE CAT SAT ON THE MAT", "THE CAT SAT ON THE MAT"), ("f2", "HE WENT TO THE STORE", "HE WENT TO THE STOVE"),
                  ("f3", "SHE READ A BOOK", "SHE REED A BOOK"), ("f4", "WE SAW THE MAN", "WE SAW THE MAN"), ("f5", "THE DOG RAN HOME", "THE DOG RAN HOLE"),
                  ("f6", "I LIKE THE CAT", "I LIKE THE KAT")]
FAKE_ARPA = """\\data\\
ngram 1=19
ngram 2=18
ngram 3=10

\\1-grams:
-1.5\t<unk>\t-0.3
-99\t<s>\t-0.5
-1.2\t</s>
-0.6\tTHE\t-0.4
-1.3\tCAT\t-0.3
-1.3\tSAT\t-0.3
-1.1\tON\t-0.3
-1.4\tMAT\t-0.3
-1.2\tHE\t-0.3
-1.3\tWENT\t-0.3
-1.0\tTO\t-0.3
-1.4\tSTORE\t-0.3
-1.2\tSHE\t-0.3
-1.3\tREAD\t-0.3
-0.9\tA\t-0.3
-1.4\tBOOK\t-0.3
-1.4\tDOG\t-0.3
-1.4\tRAN\t-0.3
-1.4\tHOME\t-0.3

\\2-grams:
-0.5\t<s> THE\t-0.2
-0.3\tTHE CAT\t-0.2
-0.3\tCAT SAT\t-0.2
-0.2\tSAT ON\t-0.2
-0.2\tON THE\t-0.2
-0.4\tTHE MAT\t-0.2
-0.3\tMAT </s>
-0.7\t<s> HE\t-0.2
-0.3\tHE WENT\t-0.2
-0.2\tWENT TO\t-0.2
-0.3\tTO THE\t-0.2
-0.5\tTHE STORE\t-0.2
-0.3\tSTORE </s>
-0.8\t<s> SHE\t-0.2
-0.4\tSHE READ\t-0.2
-0.3\tREAD A\t-0.2
-0.4\tA BOOK\t-0.2
-0.3\tBOOK </s>

\\3-grams:
-0.1\t<s> THE CAT
-0.1\tTHE CAT SAT
-0.1\tCAT SAT ON
-0.1\tSAT ON THE
-0.1\tON THE MAT
-0.1\tTHE MAT </s>
-0.2\t<s> HE WENT
-0.1\tHE WENT TO
-0.1\tWENT TO THE
-0.2\tTO THE STORE

\\end\\
"""


class FakeCTC:
    """合成 log 機率：給一段「聲學上聽到的」文字（可以有錯），每個字元攤兩格、中間夾 blank；錯的那格把對的字也給一點機率，LM 一加就修回來。"""
    name = "fake"
    device = "fake"
    n_params = 0
    vocab = {"<pad>": 0, "<s>": 1, "</s>": 2, "<unk>": 3, "|": 4, **{ch: 5 + i for i, ch in enumerate("ETAONIHSRDLUMWCFGYPBVKJXQZ'")}}
    blank, delim = 0, 4

    def __init__(self):
        self.inv = {v: k for k, v in self.vocab.items()}

    def encode(self, text):
        return [self.vocab["|"] if ch == " " else self.vocab.get(ch, 3) for ch in text.upper()]

    def decode(self, ids):
        return "".join(self.inv.get(i, "") for i in ids).replace("|", " ").strip()

    def logprobs_for(self, heard: str, truth: str) -> np.ndarray:
        V = len(self.vocab)
        rows = []
        h_ids, t_ids = self.encode(heard), self.encode(truth)
        for k, c in enumerate(h_ids):
            p = np.full(V, 1e-5)
            p[c] = 0.9
            if k < len(t_ids) and t_ids[k] != c:          # 聽錯的那格：對的字也給三成
                p[c], p[t_ids[k]] = 0.55, 0.42
            p /= p.sum()
            rows.append(np.log(p))
            b = np.full(V, 1e-5)
            b[0] = 0.95
            b /= b.sum()
            rows.append(np.log(b))
        return np.array(rows)


class FakeMMS:
    name = "fake"
    device = "fake"

    def transcribe(self, x):
        return {"text": "我想定一张下周三去台南的票", "T": 200, "blank_frac": 0.6, "seconds": 0.0}


# ── 素材 ─────────────────────────────────────────────────────────────────
def load_sentences() -> dict:
    if not SENTENCES.exists():
        sys.exit(f"沒有 {SENTENCES.relative_to(HERE)}，先跑 ./present.sh fetch（會從 dev-clean.tar.gz 讀參考文字）")
    return json.loads(SENTENCES.read_text(encoding="utf-8"))


def wav_path(utt: str) -> Path:
    return _p(CFG["sentences"]["wav_dir"]) / f"{utt}.wav"


def load_lm(progress=None) -> ArpaLM:
    cache = _p(CFG["lm"]["cache"])
    if cache.exists():
        t0 = time.perf_counter()
        lm = ArpaLM.load(cache)
        lm.load_seconds = time.perf_counter() - t0
        lm.from_cache = True
        lm.oov_penalty = float(CFG["lm"].get("oov_ln_penalty", 0.0))
        lm.oov_per_char = float(CFG["lm"].get("oov_ln_per_char", 0.0))
        lm.source = CFG["lm"]["url"]                        # 紀錄裡寫來源網址，不寫這台機器的絕對路徑（rehearsal.json 會上站）
        return lm
    src = _p(CFG["lm"]["file"])
    if not src.exists():
        sys.exit(f"沒有 {src}，先跑 ./present.sh fetch")
    t0 = time.perf_counter()
    lm = ArpaLM.parse(src, progress=progress)
    lm.parse_seconds = time.perf_counter() - t0
    cache.parent.mkdir(parents=True, exist_ok=True)
    lm.save(cache)
    lm.load_seconds = lm.parse_seconds
    lm.from_cache = False
    lm.oov_penalty = float(CFG["lm"].get("oov_ln_penalty", 0.0))
    lm.oov_per_char = float(CFG["lm"].get("oov_ln_per_char", 0.0))
    lm.source = CFG["lm"]["url"]
    return lm


def mandarin_inputs() -> list[dict]:
    """toml 的 [[mandarin.inputs]] 裡檔案存在的那幾個；參考文字從 text 或 text_file 讀。"""
    out = []
    for m in CFG["mandarin"]["inputs"]:
        p = _p(m["path"])
        if not p.exists():
            continue
        text = m.get("text", "")
        if not text and m.get("text_file"):
            tf = _p(m["text_file"])
            text = tf.read_text(encoding="utf-8").strip() if tf.exists() else ""
        out.append({"key": m["key"], "label": m["label"], "path": p, "text": text, "public_text": bool(m.get("public_text", False))})
    return out


# ── 解碼 ─────────────────────────────────────────────────────────────────
def lm_bonus(lm: ArpaLM | None, lam: float, eta: float, ctc) -> LMBonus:
    id2ch = {v: k for k, v in ctc.vocab.items() if len(k) == 1}
    return LMBonus(lm, lam, eta, id2ch, ctc.delim, partial_as_oov=bool(CFG["lm"].get("score_partial_as_oov", True)))


def decode_with_lm(lp: np.ndarray, lm: ArpaLM | None, lam: float, eta: float, ctc, beam: int, prune: float) -> dict:
    b = lm_bonus(lm, lam, eta, ctc)
    best = prefix_beam_search(lp, ctc.blank, beam=beam, bonus=b, prune_logp=prune, n_best=1)[0]
    st = b.start()
    for c in best["ids"]:
        st = b.step(st, c)
    parts = b.parts(st)
    return {"text": ctc.decode(best["ids"]), "logp_ctc": best["logp"], "bonus": best["bonus"], "score": best["score"], **parts}


# ── Demo ────────────────────────────────────────────────────────────────────
class Demo:
    def __init__(self, out, sents: dict, ctc, lm: ArpaLM | None, mms, whisper, qwen, stored: dict | None = None, stored_private: dict | None = None, tag: str = "",
                 logprobs: dict[str, np.ndarray] | None = None, eta: float | None = None):
        self.out, self.ctc, self.lm, self.mms, self.wh, self.qw = out, ctc, lm, mms, whisper, qwen
        self.sents = sents
        self.res: dict = dict(stored or {})
        self.res_priv: dict = dict(stored_private or {})
        self.replay = stored is not None
        self.tag = tag
        self.lp = logprobs or {}
        self.dec = CFG["decode"]
        self.eta = eta if eta is not None else (float(self.dec["eta"]) if self.dec["eta"] != "auto" else 1.0)
        self.eta_source = "toml" if self.dec["eta"] != "auto" else ("eta.json" if eta is not None else "預設 1.0（還沒調）")
        self.pick_i = 0
        self.plot = None

    def _get(self, key: str, compute, private: bool = False):
        store = self.res_priv if private else self.res
        if key in store:
            return store[key]
        if self.replay:
            self.out(f"  {C.AMBER}彩排紀錄裡沒有 {key}{C.RESET}")
            return None
        store[key] = compute()
        return store[key]

    # 語音 → log 機率（一次算完，之後每個 λ 都重用）
    def ensure_logprobs(self, which: str = "test") -> None:
        if self.replay:
            return
        todo = [s for s in self.sents[which] if s["id"] not in self.lp]
        if not todo:
            return
        t0 = time.perf_counter()
        for s in todo:
            x = C.load_wav(wav_path(s["id"]), FS)
            self.lp[s["id"]] = self.ctc.logprobs(x)
        C.note(f"  {which}：{len(todo)} 句的 log 機率算好（{self.ctc.device}，{time.perf_counter() - t0:.1f} s）")

    def _decode_set(self, which: str, lam: float, eta: float) -> dict:
        refs = [s["text"] for s in self.sents[which]]
        hyps, parts = [], []
        t0 = time.perf_counter()
        for s in self.sents[which]:
            r = decode_with_lm(self.lp[s["id"]], self.lm, lam, eta, self.ctc, int(self.dec["beam"]), float(self.dec["prune_logp"]))
            hyps.append(r["text"])
            parts.append({k: r[k] for k in ("logp_ctc", "lm_ln", "n_words", "n_oov")})
        w = corpus_wer(refs, hyps)
        return {"lam": lam, "eta": eta, "hyps": hyps, "parts": parts, "seconds": time.perf_counter() - t0, **{k: w[k] for k in ("errors", "n_ref", "wer", "S", "D", "I", "per_sentence")}}

    # 0：p32 的例子
    def panel_example(self) -> None:
        o = self.out
        data = json.loads(DATA_JSON.read_text(encoding="utf-8"))["prefix_example"] if DATA_JSON.exists() else None
        if data is None:
            o("  沒有 w07-data.json"); return
        o.banner("p32 的數字例子：T = 2，兩格的機率 → 九條路徑 → 按標籤合併 → greedy 對 prefix beam")
        o(f"  y¹ = {data['y1']}    y² = {data['y2']}")
        r = self._get("example", lambda: prefix_example(data))
        if not r:
            return
        o("  九條路徑（path → 標籤，機率）：")
        for p in r["paths"]:
            lab = p["label"] or '""'
            o(f"    {p['path']:<4s} → {lab:<4s} {p['prob']:.4f}")
        o("  按標籤合併：" + "、".join(f"{k or chr(34) * 2} {v:.4f}" for k, v in sorted(r["by_label"].items(), key=lambda kv: -kv[1])))
        o(f"  單條最高的路徑 {r['best_single_path']['path']}（{r['best_single_path']['prob']:.2f}）→ greedy 給 {C.RED}{r['greedy_label']!r}{C.RESET}")
        o(f"  prefix beam search（合併之後再搜）給 {C.RED}{r['beam_label']!r}{C.RESET}：" + "、".join(f"{k or chr(34) * 2} {v:.4f}" for k, v in r["beam_probs"].items()))
        o(f"  與 w07-data.json 逐位相同：{'是' if r['ok'] else C.RED + '否' + C.RESET}")

    # 1：greedy 與 beam λ=0
    def panel_baseline(self) -> None:
        o = self.out
        o.banner(f"測試集 {len(self.sents['test'])} 句（dev-clean，每位說話人一句）：greedy（無 LM）與 prefix beam λ = 0")
        self.ensure_logprobs("test")

        def compute():
            refs = [s["text"] for s in self.sents["test"]]
            g = [self.ctc.decode(ctc_greedy(self.lp[s["id"]], self.ctc.blank)) for s in self.sents["test"]]
            wg = corpus_wer(refs, g)
            b0 = self._decode_set("test", 0.0, 0.0)
            T = sum(int(self.lp[s["id"]].shape[0]) for s in self.sents["test"])
            dur = sum(float(s["dur_s"]) for s in self.sents["test"])
            return {"greedy": {"hyps": g, **{k: wg[k] for k in ("errors", "n_ref", "wer", "S", "D", "I", "per_sentence")}}, "beam0": b0, "T_total": T, "dur_total": dur}
        r = self._get("baseline", compute)
        if not r:
            return
        g, b = r["greedy"], r["beam0"]
        o(f"  共 {r['dur_total']:.0f} s 語音、{r['T_total']} frame、{g['n_ref']} 個參考詞")
        o(f"  greedy            WER {C.RED}{100 * g['wer']:.2f}%{C.RESET}（{g['errors']} 錯：S {g['S']}／D {g['D']}／I {g['I']}）")
        o(f"  prefix beam λ=0   WER {C.RED}{100 * b['wer']:.2f}%{C.RESET}（{b['errors']} 錯：S {b['S']}／D {b['D']}／I {b['I']}；beam {self.dec['beam']}，{b['seconds']:.1f} s）")
        same = sum(1 for a, c in zip(g["hyps"], b["hyps"]) if norm_en(a) == norm_en(c))
        o(f"  兩者輸出相同的句子 {same}/{len(g['hyps'])}——沒有 LM 時，合併 prefix 只在機率很接近的地方改變答案")

    # 2：λ 掃描
    def panel_sweep(self) -> None:
        o = self.out
        lams = [float(v) for v in self.dec["lambdas"]]
        o.banner(f"λ 掃描：score = ln P_ctc + λ·ln P_LM + η·詞數；beam {self.dec['beam']}、η = {self.eta:g}（{self.eta_source}）固定，λ ∈ {lams}")
        self.ensure_logprobs("test")
        base = self.res.get("baseline")
        g_wer = base["greedy"]["wer"] if base else None
        stored = self.res.get("sweep")
        if stored is None and self.replay:
            o(f"  {C.AMBER}彩排紀錄裡沒有 sweep{C.RESET}"); return
        rows = stored["rows"] if stored else []
        self._plot_start(lams, g_wer)
        if stored:
            for r in rows:
                self._print_sweep_row(r)
                self._plot_update([x["lam"] for x in rows[:rows.index(r) + 1]], [x["wer"] for x in rows[:rows.index(r) + 1]])
        else:
            t0 = time.perf_counter()
            for lam in lams:
                r = self._decode_set("test", lam, self.eta)
                rows.append(r)
                self._print_sweep_row(r)
                self._plot_update([x["lam"] for x in rows], [x["wer"] for x in rows])
            best = min(rows, key=lambda r: (r["wer"], r["lam"]))
            self.res["sweep"] = {"lambdas": lams, "eta": self.eta, "beam": int(self.dec["beam"]), "rows": rows, "best_lam": best["lam"], "best_wer": best["wer"],
                                 "seconds": time.perf_counter() - t0, "n_test": len(self.sents["test"])}
            stored = self.res["sweep"]
        best = min(rows, key=lambda r: (r["wer"], r["lam"]))
        left = rows[0]["wer"]
        right = rows[-1]["wer"]
        o(f"\n  最佳 λ = {C.BOLD}{best['lam']:g}{C.RESET}，WER {C.RED}{100 * best['wer']:.2f}%{C.RESET}；λ = 0 是 {100 * left:.2f}%，λ = {rows[-1]['lam']:g} 是 {100 * right:.2f}%"
          + (f"；greedy {100 * g_wer:.2f}%" if g_wer is not None else "") + f"；整個掃描 {stored['seconds']:.0f} s")
        shape = "U 形（左邊下降、右邊回升）" if right > best["wer"] and left > best["wer"] else ("單調下降（右邊還沒回升——λ 要掃更大）" if right <= best["wer"] else "左邊就是最低——LM 沒有幫上忙")
        o(f"  形狀：{shape}")
        self._plot_finish(best)

    def _print_sweep_row(self, r: dict) -> None:
        self.out(f"  λ = {r['lam']:<4g} WER {100 * r['wer']:6.2f}%   S {r['S']:3d}  D {r['D']:3d}  I {r['I']:3d}   ({r['seconds']:.1f} s)")

    def _plot_start(self, lams, g_wer):
        try:
            import matplotlib
            if self.tag in ("rehearsal", "selftest", "fake"):
                matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            self.plt = plt
            if self.tag not in ("rehearsal", "selftest", "fake"):
                plt.ion()
            self.fig, self.ax = plt.subplots(figsize=(7.5, 4.5))
            self.ax.set_xlabel("λ (weight of the external 3-gram LM)")
            self.ax.set_ylabel("WER % on the dev-clean subset")
            self.ax.set_title(f"wav2vec2-base-960h + LibriSpeech 3-gram (pruned 1e-7), beam {self.dec['beam']}, η = {self.eta:g}")
            if g_wer is not None:
                self.ax.axhline(100 * g_wer, color="grey", ls="--", lw=1, label="greedy, no LM")
            self.line, = self.ax.plot([], [], "o-", color="#1f4e79", label="prefix beam + λ·LM")
            self.ax.set_xlim(-0.1, max(lams) * 1.05)
            self.ax.legend(loc="best")
            self.ax.grid(alpha=0.3)
            if self.tag not in ("rehearsal", "selftest", "fake"):
                self.fig.show()
                plt.pause(0.01)
        except Exception as e:
            self.plt = None
            C.note(f"  （沒有視窗：{type(e).__name__}: {e}；只印數字）")

    def _plot_update(self, xs, ys):
        if not getattr(self, "plt", None):
            return
        self.line.set_data(xs, [100 * y for y in ys])
        self.ax.relim()
        self.ax.autoscale_view(scalex=False)
        if self.tag not in ("rehearsal", "selftest", "fake"):
            self.fig.canvas.draw_idle()
            self.plt.pause(0.01)

    def _plot_finish(self, best):
        if not getattr(self, "plt", None):
            return
        self.ax.plot([best["lam"]], [100 * best["wer"]], "o", ms=12, mfc="none", mec="#c0392b", mew=2)
        self.ax.annotate(f"best λ = {best['lam']:g}\n{100 * best['wer']:.2f}%", (best["lam"], 100 * best["wer"]), textcoords="offset points", xytext=(12, 10), fontsize=9)
        if self.tag in ("rehearsal", "selftest", "fake"):
            d = HERE / "runs" / ("rehearsal" if self.tag == "rehearsal" else "_selftest")
            d.mkdir(parents=True, exist_ok=True)
            self.fig.savefig(d / "lambda_sweep.png", dpi=150, bbox_inches="tight")
            self.plt.close(self.fig)
        else:
            self.fig.canvas.draw_idle()
            self.plt.pause(0.01)

    # 3：挑一句看字
    def _ranked_sentences(self) -> list[int]:
        sw = self.res.get("sweep")
        if not sw:
            return list(range(len(self.sents["test"])))
        rows = {r["lam"]: r for r in sw["rows"]}
        r0, rb = rows[0.0] if 0.0 in rows else sw["rows"][0], rows[sw["best_lam"]]
        rl = rows.get(float(self.dec["large_lambda"]), sw["rows"][-1])
        fix = [r0["per_sentence"][i] - rb["per_sentence"][i] for i in range(len(self.sents["test"]))]        # λ=0 → 最佳：修好幾個
        broke = [rl["per_sentence"][i] - rb["per_sentence"][i] for i in range(len(self.sents["test"]))]      # 最佳 → 過大：又壞幾個
        return sorted(range(len(fix)), key=lambda i: (-fix[i], -broke[i], i))

    def panel_sentence(self, advance: bool = False) -> None:
        o = self.out
        sw = self.res.get("sweep")
        if not sw:
            o("  先按 2 跑 λ 掃描"); return
        order = self._ranked_sentences()
        if advance:
            self.pick_i = (self.pick_i + 1) % len(order)
        i = order[self.pick_i]
        s = self.sents["test"][i]
        rows = {r["lam"]: r for r in sw["rows"]}
        lams = [0.0, sw["best_lam"], float(self.dec["large_lambda"]) if float(self.dec["large_lambda"]) in rows else sw["rows"][-1]["lam"]]
        o.banner(f"一句看字：{s['id']}（說話人 {s['spk']}，{s['dur_s']:.1f} s）——λ = 0／最佳 {sw['best_lam']:g}／過大 {lams[2]:g}；第 {self.pick_i + 1}/{len(order)} 句（n 換下一句）")
        o(f"  參考：{s['text']}")
        base = self.res.get("baseline")
        if base:
            o(f"  greedy：{word_colored(s['text'], base['greedy']['hyps'][i])}")
        for lam, label in zip(lams, ("λ = 0", f"最佳 λ = {sw['best_lam']:g}", f"過大 λ = {lams[2]:g}")):
            r = rows[lam]
            hyp, pt = r["hyps"][i], r["parts"][i]
            err = r["per_sentence"][i]
            o(f"\n  {C.BOLD}{label:<14s}{C.RESET} {word_colored(s['text'], hyp)}")
            o(f"    錯 {err}；ln P_ctc {pt['logp_ctc']:.{DIG}f}；ln P_LM {pt['lm_ln']:.{DIG}f} → λ·ln P_LM {lam * pt['lm_ln']:.{DIG}f}；{pt['n_words']} 詞 → η·詞數 {self.eta * pt['n_words']:.{DIG}f}"
              + (f"；{pt['n_oov']} 個詞不在 LM 字典裡（當 <unk>）" if pt.get("n_oov") else ""))
        if self.lm is not None and not self.replay:
            self._lm_view(s["text"], rows[lams[2]]["hyps"][i], rows[0.0]["hyps"][i])
        self.res.setdefault("sentence_picks", {})[s["id"]] = {"index": i, "ref": s["text"], "hyps": {str(l): rows[l]["hyps"][i] for l in lams}, "errors": {str(l): rows[l]["per_sentence"][i] for l in lams},
                                                             "greedy": base["greedy"]["hyps"][i] if base else None}

    def _lm_view(self, ref: str, hyp_large: str, hyp0: str) -> None:
        """過大 λ 的版本裡被換掉的詞：LM 給參考詞與換上去的詞各多少（三元、含 backoff）。"""
        o = self.out
        a = align_words(norm_en(ref).split(), norm_en(hyp_large).split())
        hw = norm_en(hyp_large).split()
        changes = [(k, r, h) for k, (op, r, h) in enumerate(a["ops"]) if op in ("S", "I")]
        if not changes:
            return
        o(f"  {C.GREY}LM 的看法（ln P，三元含 backoff；過大 λ 的版本對參考）：{C.RESET}")
        pos = 0
        for op, r, h in a["ops"]:
            if op in ("S", "I", "C"):
                hist = tuple(hw[max(0, pos - 2):pos])
                if op in ("S", "I"):
                    lp_h = self.lm.logp(h, ("<s>",) + hist if pos < 2 else hist)
                    lp_r = self.lm.logp(r, ("<s>",) + hist if pos < 2 else hist) if r else None
                    o(f"    …{' '.join(hist)} → {C.RED}{h}{C.RESET} {lp_h:.2f}" + (f"   參考 {r} {lp_r:.2f}" if r else "   （插入）") + (f"  （不在字典裡：<unk> {self.lm.oov_cost(h):+.1f}）" if not self.lm.in_vocab(h) else ""))
                pos += 1

    # m：華語
    def panel_mandarin(self) -> None:
        o = self.out
        inputs = mandarin_inputs() if not self.replay else [dict(v, path=None) for v in self.res_priv.get("mandarin_inputs", [])]
        if not inputs:
            o("  沒有可用的華語輸入：W1 d2 的 utterance.wav、W7 d2 的 mandarin.wav 都不在，先 ./present.sh record"); return
        o.private = True
        try:
            o.banner("華語同音字：同一句話 → MMS-1B-all CTC（cmn adapter、greedy、無 LM、轉繁體）／Whisper small／Qwen3-ASR；替換的字：同音=琥珀、異音=紅")
            self.res_priv["mandarin_inputs"] = [{"key": m["key"], "label": m["label"], "text": m["text"], "public_text": m["public_text"]} for m in inputs]
            for m in inputs:
                o(f"\n  {C.BOLD}{m['label']}{C.RESET}（{m['key']}）")
                o(f"  參考：{m['text'] or '（沒有參考文字：只印輸出）'}")
                x = C.load_wav(m["path"], FS) if m["path"] else None
                for model, label in (("mms", "MMS-1B-all CTC"), ("whisper", "Whisper small"), ("qwen", "Qwen3-ASR")):
                    r = self._get(f"zh:{m['key']}:{model}", lambda mm=model, mx=x, mi=m: self._mandarin_run(mm, mx, mi), private=True)
                    if r is None:
                        continue
                    if r.get("skipped"):
                        o(f"  {label:<16s} {C.GREY}{r['skipped']}{C.RESET}"); continue
                    shown = r["text_trad"] if model == "mms" else r["text"]
                    o(f"  {label:<16s} {C.RED}{shown!r}{C.RESET}   ({r['seconds']:.2f} s" + (f"；T = {r['T']}，blank {100 * r['blank_frac']:.0f}%" if model == "mms" else "") + ")")
                    if model == "mms":
                        o(f"  {'':<16s} 原始輸出（簡體）{r['text']!r}" + (f"；轉繁體後還有 {r['trad_residual']} 個字對照表裡沒有" if r["trad_residual"] else ""))
                    if m["text"]:
                        c = r["align"]["counts"]
                        o(f"  {'':<16s} {chars_colored(norm_zh(m['text']), norm_zh(shown))}")
                        o(f"  {'':<16s} CER {100 * r['align']['cer']:.1f}%（S {r['align']['S']}／D {r['align']['D']}／I {r['align']['I']}）；替換 {r['align']['S']} 個字裡同音 {c['homophone']}"
                          f"（同調 {c['same_tone']}、異調 {c['same_base']}）、異音 {c['different']}" + (f"、無法判定 {c['unknown']}" if c["unknown"] else ""))
            o("\n  （字元級 CTC 沒有輸出端的先驗，同音字只能靠聲音分；Whisper 與 Qwen3-ASR 的 decoder 讀得到前文——看兩邊替換掉的字是不是同音）")
        finally:
            o.private = False

    def _mandarin_run(self, model: str, x: np.ndarray, m: dict) -> dict:
        if model == "mms":
            if self.mms is None:
                return {"skipped": f"MMS-1B-all 載不起來（{getattr(self, 'mms_error', None)}）"}
            r = self.mms.transcribe(x)
            r["text_trad"], r["trad_residual"] = to_traditional(r["text"])
            shown = r["text_trad"]
        elif model == "whisper":
            if self.wh is None:
                return {"skipped": "Whisper 沒載"}
            r = self.wh.transcribe(x, CFG["whisper"]["language_zh"], None)
            shown = r["text"]
        else:
            if self.qw is None:
                return {"skipped": f"Qwen3-ASR 載不起來（{getattr(self, 'qw_error', None)}）——只放 MMS 與 Whisper 兩格，理由課堂上說"}
            r = self.qw.transcribe(m["path"], x, CFG["qwen"]["language_zh"], None)
            shown = r["text"]
        r["script"] = D2.script_judge(shown)
        if m["text"]:
            a = align_chars(norm_zh(m["text"]), norm_zh(shown))
            cl = classify_subs(a["ops"])
            r["align"] = {"cer": a["cer"], "errors": a["errors"], "S": a["S"], "D": a["D"], "I": a["I"], "n_ref": a["n_ref"], "subs": cl["subs"], "counts": cl["counts"]}
        return r

    # e：η
    def panel_eta(self) -> None:
        o = self.out
        r = self.res.get("eta_tuning")
        o.banner(f"η（每個詞的插入獎勵）：調參集 {len(self.sents['tune'])} 句、λ = {self.dec['tune_lambda']} 固定")
        if not r:
            o(f"  還沒調（rehearse 會調）；目前 η = {self.eta:g}（{self.eta_source}）"); return
        for row in r["rows"]:
            mark = " ←" if row["eta"] == r["best_eta"] else ""
            o(f"  η = {row['eta']:<4g} WER {100 * row['wer']:6.2f}%   S {row['S']:3d}  D {row['D']:3d}  I {row['I']:3d}{mark}")
        o(f"  取 η = {r['best_eta']:g}；它只在 rehearse 時調一次，課堂上固定——一次只變一個量")

    def tune_eta(self) -> float:
        self.ensure_logprobs("tune")
        rows = []
        for eta in [float(v) for v in self.dec["eta_grid"]]:
            r = self._decode_set("tune", float(self.dec["tune_lambda"]), eta)
            rows.append({k: r[k] for k in ("eta", "wer", "errors", "S", "D", "I", "seconds")})
        best = min(rows, key=lambda r: (r["wer"], r["eta"]))
        self.res["eta_tuning"] = {"tune_lambda": float(self.dec["tune_lambda"]), "rows": rows, "best_eta": best["eta"], "n_tune": len(self.sents["tune"])}
        self.eta, self.eta_source = best["eta"], "rehearse 調的"
        return best["eta"]

    # l：LM 本身
    def panel_lm(self) -> None:
        o = self.out
        o.banner("外部 LM：LibriSpeech 官方 3-gram（pruned 1e-7，openslr 11），自己讀 ARPA、自己 backoff")
        info = self._get("lm_info", self._lm_info)
        if not info:
            return
        o(f"  1-gram {info['n_unigrams']:,}、2-gram {info['n_bigrams']:,}、3-gram {info['n_trigrams']:,}；<unk> {'有' if info['unk'] else '沒有（用地板值）'}，OOV 另加 {info.get('oov_penalty', 0.0):g} + {info.get('oov_per_char', 0.0):g}×字元數（ln）；載入 {info['load_seconds']:.1f} s（{'快取' if info['from_cache'] else '第一次讀 ARPA'}）")
        for ex in info["examples"]:
            o(f"  ln P = {ex['lnp']:7.2f}   {ex['text']}" + (f"   （{ex['note']}）" if ex.get("note") else ""))
        o(f"  測試集參考句的 perplexity {info['ppl_test']:.1f}（每詞）")

    def _lm_info(self) -> dict:
        lm = self.lm
        exs = [("THE CAT SAT ON THE MAT", ""), ("MAT THE ON SAT CAT THE", "同樣的詞倒過來"), ("THINK OF LADY GLENARVAN THINK OF MARY GRANT", "Demo 2 那句"),
               ("THINK OF LADY GLENARVIN THINK OF MARY GRANT", "Demo 2 的錯字")]
        out = []
        for text, note in exs:
            lp, per = lm.sentence_logp(text.split())
            out.append({"text": text, "lnp": lp, "per_word": per, "note": note, "oov": [w for w in text.split() if not lm.in_vocab(w)]})
        return {"n_unigrams": lm.n_unigrams, "n_bigrams": lm.n_bigrams, "n_trigrams": lm.n_trigrams, "unk": lm.unk, "oov_penalty": lm.oov_penalty, "oov_per_char": lm.oov_per_char, "load_seconds": getattr(lm, "load_seconds", 0.0),
                "from_cache": getattr(lm, "from_cache", True), "examples": out, "ppl_test": lm.perplexity([s["text"].split() for s in self.sents["test"]])}

    # t：投影片的表
    def panel_table(self) -> None:
        o = self.out
        o.banner("投影片 p41 的表")
        b, sw, ex = self.res.get("baseline"), self.res.get("sweep"), self.res.get("example")
        if b:
            o(f"  greedy, no LM                 {100 * b['greedy']['wer']:.2f}%")
            o(f"  prefix beam, λ = 0            {100 * b['beam0']['wer']:.2f}%")
        if sw:
            rows = {r["lam"]: r for r in sw["rows"]}
            large = float(self.dec["large_lambda"])
            o(f"  prefix beam, best λ = {sw['best_lam']:g}      {100 * sw['best_wer']:.2f}%   (η = {sw['eta']:g}, beam {sw['beam']}, {sw['n_test']} sentences)")
            if large in rows:
                o(f"  prefix beam, large λ = {large:g}     {100 * rows[large]['wer']:.2f}%   (S {rows[large]['S']}, D {rows[large]['D']}, I {rows[large]['I']})")
            last = sw["rows"][-1]
            if last["lam"] != large:
                o(f"  prefix beam, λ = {last['lam']:g}           {100 * last['wer']:.2f}%   (S {last['S']}, D {last['D']}, I {last['I']})")
        if ex:
            o(f"  p32's example in code         greedy → {ex['greedy_label']!r}; prefix beam → {ex['beam_label']!r}；{'對 json 逐位相同' if ex['ok'] else '與 json 不同！'}")
        zh = {k: v for k, v in self.res_priv.items() if k.startswith("zh:")}
        if zh:
            o("  華語（數字；文字只有 public_text 的句子上投影片）：")
            for k, v in zh.items():
                if v.get("align"):
                    c = v["align"]["counts"]
                    o(f"    {k:<24s} CER {100 * v['align']['cer']:.1f}%  替換 {v['align']['S']}：同音 {c['homophone']}／異音 {c['different']}")

    def menu(self) -> None:
        o = self.out
        o("\n  0 p32 的例子   1 greedy 與 beam λ=0   2 λ 掃描   3 一句看字（n 下一句）   m 華語同音字")
        o("  e η 的調參   l LM 本身   t 表   h 選單   q 離開")

    def dispatch(self, k: str) -> bool:
        if k == "0": self.panel_example()
        elif k == "1": self.panel_baseline()
        elif k == "2": self.panel_sweep()
        elif k == "3": self.panel_sentence()
        elif k == "n": self.panel_sentence(advance=True)
        elif k == "m": self.panel_mandarin()
        elif k == "e": self.panel_eta()
        elif k == "l": self.panel_lm()
        elif k == "t": self.panel_table()
        elif k == "h": self.menu()
        elif k == "q": return False
        return True

    def run_all(self) -> None:
        for k in "l012":
            self.dispatch(k)
        self.panel_sentence()
        for _ in range(2):
            self.panel_sentence(advance=True)
        self.pick_i = 0
        self.dispatch("m")
        self.dispatch("e")
        self.dispatch("t")

    def versions(self) -> dict:
        v = C.versions()
        from importlib import metadata
        for pkg in ("torch", "transformers", "huggingface_hub", "pypinyin"):
            try:
                v[pkg] = metadata.version(pkg)
            except Exception:
                v[pkg] = None
        v["models"] = read_lock()
        return v

    def save(self, folder: Path) -> None:
        folder.mkdir(parents=True, exist_ok=True)
        dev = {"ctc": str(getattr(self.ctc, "device", "")), "mms": str(getattr(self.mms, "device", "")) if self.mms else None,
               "whisper": str(getattr(self.wh, "device", "")) if self.wh else None, "qwen": str(getattr(self.qw, "device", "")) if self.qw else None}
        pub_zh = {}
        for k, v in self.res_priv.items():
            if k.startswith("zh:") and v and not v.get("skipped"):
                key = k.split(":")[1]
                m = next((mi for mi in self.res_priv.get("mandarin_inputs", []) if mi["key"] == key), {})
                entry = {"seconds": v.get("seconds"), "script": v.get("script", {}).get("verdict") if v.get("script") else None}
                if v.get("align"):
                    entry["cer"] = v["align"]["cer"]
                    entry.update({x: v["align"][x] for x in ("S", "D", "I", "n_ref")})
                    entry["homophone_counts"] = v["align"]["counts"]
                if m.get("public_text"):
                    entry["text"] = v["text_trad"] if "text_trad" in v else v["text"]
                    entry["ref"] = m.get("text")
                    if v.get("align"):
                        entry["subs"] = v["align"]["subs"]
                pub_zh[k] = entry
        log = {"time": C.now(), "tag": self.tag, "backend": self.ctc.name, "device": dev, "versions": self.versions(), "config": CFG, "eta": self.eta, "eta_source": self.eta_source,
               "sentences": self.sents, "lm": {"source": getattr(self.lm, "source", None), "n_unigrams": getattr(self.lm, "n_unigrams", None), "n_bigrams": getattr(self.lm, "n_bigrams", None),
                                               "n_trigrams": getattr(self.lm, "n_trigrams", None), "unk": getattr(self.lm, "unk", None), "oov_penalty": getattr(self.lm, "oov_penalty", None), "oov_per_char": getattr(self.lm, "oov_per_char", None)} if self.lm is not None else None,
               "mandarin": {"inputs": [{"key": m["key"], "label": m["label"], "public_text": m["public_text"]} for m in self.res_priv.get("mandarin_inputs", [])], "results": pub_zh},
               "mms_error": getattr(self, "mms_error", None), "qwen_error": getattr(self, "qw_error", None), "results": self.res}
        (folder / "rehearsal.json").write_text(json.dumps(log, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
        (folder / "rehearsal.txt").write_text(self.out.text(), encoding="utf-8")
        if "eta_tuning" in self.res and self.tag == "rehearsal":
            (folder / "eta.json").write_text(json.dumps({"eta": self.eta, "time": log["time"], "tune_lambda": self.res["eta_tuning"]["tune_lambda"], "n_tune": len(self.sents["tune"])}, indent=1) + "\n", encoding="utf-8")
        C.note(f"  英文與華語的數字存在 {folder}/（rehearsal.json、rehearsal.txt" + ("、eta.json、lambda_sweep.png" if self.tag == "rehearsal" else "") + "）")
        if self.res_priv:
            PRIVATE.mkdir(parents=True, exist_ok=True)
            name = "private_rehearsal" if self.tag == "rehearsal" else f"private_{folder.name}"
            plog = {"time": log["time"], "tag": self.tag, "results": self.res_priv}
            (PRIVATE / f"{name}.json").write_text(json.dumps(plog, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
            (PRIVATE / f"{name}.txt").write_text(self.out.text(private=True), encoding="utf-8")
            C.note(f"  華語的全文存在 {PRIVATE}/{name}.json、.txt（不進版控、不上站）")


# ── 建構 ─────────────────────────────────────────────────────────────────────
def read_eta() -> float | None:
    if CFG["decode"]["eta"] != "auto":
        return float(CFG["decode"]["eta"])
    p = HERE / "runs" / "rehearsal" / "eta.json"
    if p.exists():
        return float(json.loads(p.read_text(encoding="utf-8"))["eta"])
    return None


def build(cmd: str, out) -> Demo:
    if cmd == "replay":
        src = HERE / "runs" / "rehearsal" / "rehearsal.json"
        if not src.exists():
            sys.exit("沒有 runs/rehearsal/rehearsal.json，先跑 ./present.sh rehearse")
        log = json.loads(src.read_text(encoding="utf-8"))
        psrc = PRIVATE / "private_rehearsal.json"
        plog = json.loads(psrc.read_text(encoding="utf-8")) if psrc.exists() else {}
        out.banner(f"REPLAY：{log['time']} 的彩排（backend {log['backend']}）" + ("" if plog else "；沒有華語那半的全文"), C.AMBER)
        res = dict(log["results"])
        for lam_rows in (res.get("sweep", {}).get("rows", []),):
            for r in lam_rows:
                r["lam"] = float(r["lam"])
        if "sweep" in res:
            res["sweep"]["best_lam"] = float(res["sweep"]["best_lam"])
        d = Demo(out, log["sentences"], FakeCTC(), None, None, None, None, stored=res, stored_private=plog.get("results", {}), tag="replay", eta=float(log["eta"]))
        d.eta_source = log.get("eta_source", "彩排")
        return d
    if cmd == "fake":
        out.banner("FAKE：合成的 log 機率與六句話的小 LM，只為了走流程，不是模型的輸出", C.RED)
        return _fake_demo(out, tag="fake")
    sents = load_sentences()
    t0 = time.perf_counter()
    ctc = D2.CTC(CFG["ctc"]["device"])
    lm = load_lm(progress=lambda n: C.note(f"    …讀了 {n:,} 條 n-gram"))
    C.note(f"  wav2vec2-base-960h {ctc.n_params / 1e6:.0f} M（{ctc.device}）；LM {lm.n_unigrams:,}／{lm.n_bigrams:,}／{lm.n_trigrams:,}（{'快取' if lm.from_cache else '第一次讀 ARPA，已存快取'}，{lm.load_seconds:.1f} s）")
    mms, mms_err = None, None
    if mandarin_inputs():
        try:
            mms = MMS()
        except Exception as e:
            mms_err = f"{type(e).__name__}: {e}"
            if pick_device(CFG["mms"]["device"]).type == "mps":
                try:
                    C.note(f"  MMS-1B-all 在 MPS 上載不起來（{mms_err[:80]}），退 CPU")
                    mms, mms_err = MMS("cpu"), None
                except Exception as e2:
                    mms_err = f"{type(e2).__name__}: {e2}"
        wh = D2.Whisper()
        qw, qw_err = None, None
        try:
            qw = D2.Qwen()
        except Exception as e:
            qw_err = f"{type(e).__name__}: {e}"
            try:
                C.note(f"  Qwen3-ASR 在 MPS 上載不起來（{qw_err[:80]}），退 CPU")
                qw, qw_err = D2.Qwen("cpu"), None
            except Exception as e2:
                qw_err = f"{type(e2).__name__}: {e2}"
        C.note("  " + (f"MMS-1B-all {mms.n_params / 1e9:.2f} B（{mms.device}，{mms.how}，lm_head {mms.vocab_size}）" if mms else f"MMS 載入失敗：{mms_err}") + f"、Whisper small（{wh.device}）、"
               + (f"Qwen3-ASR {qw.n_params / 1e9:.1f} B（{qw.device}）" if qw else f"Qwen3-ASR 載入失敗：{qw_err}"))
    else:
        wh, qw, qw_err = None, None, None
        C.note("  （沒有華語輸入：m 那格會跳過；W1 d2 的 utterance.wav 或 ./present.sh record）")
    C.note(f"  載入共 {time.perf_counter() - t0:.1f} s")
    d = Demo(out, sents, ctc, lm, mms, wh, qw, tag="rehearsal" if cmd == "rehearse" else "", eta=read_eta())
    d.mms_error, d.qw_error = mms_err, qw_err
    return d


def _fake_demo(out, tag: str) -> Demo:
    fc = FakeCTC()
    lm = ArpaLM.parse(_write_fake_arpa())
    lm.oov_penalty, lm.oov_per_char = -1.0, -0.5
    sents = {"test": [{"id": k, "spk": "0", "dur_s": 2.0, "text": t} for k, t, _ in FAKE_SENTENCES[:4]], "tune": [{"id": k, "spk": "0", "dur_s": 2.0, "text": t} for k, t, _ in FAKE_SENTENCES[4:]]}
    lps = {k: fc.logprobs_for(h, t) for k, t, h in FAKE_SENTENCES}
    d = Demo(out, sents, fc, lm, FakeMMS(), D2.FakeWhisper(), D2.FakeQwen(), tag=tag, logprobs=lps, eta=0.5)
    d.eta_source = "fake"
    d._fake_inputs = True
    return d


def _write_fake_arpa() -> Path:
    p = HERE / "runs" / "_selftest" / "fake.arpa"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(FAKE_ARPA, encoding="utf-8")
    return p


# fake 模式下華語輸入也用罐頭（沒有錄音檔也能走流程）
_real_mandarin_inputs = mandarin_inputs


def _fake_mandarin_inputs() -> list[dict]:
    return [{"key": "fake", "label": "FAKE 訂票句", "path": None, "text": "我想訂一張下週三去台南的票", "public_text": True}]


# ── check ───────────────────────────────────────────────────────────────────
def check() -> None:
    C.banner("課前體檢", C.AMBER)
    res: dict = {"time": C.now()}
    d = Demo(D2.Out(), {"test": [], "tune": []}, FakeCTC(), None, None, None, None)
    v = d.versions()
    for k in ("python", "platform", "numpy", "scipy", "matplotlib", "torch", "transformers", "huggingface_hub", "pypinyin", "models"):
        print(f"  {k:<16s} {v.get(k)}")
    res["versions"] = v
    if not v.get("pypinyin"):
        print(f"  {C.AMBER}沒有 pypinyin：同音判定會是「無法判定」。uv pip install --python ../.venv/bin/python pypinyin{C.RESET}")
    try:
        import torch
    except ImportError:
        print("  沒有 torch：先在 w03_d3_blank_heatmap 跑 ./setup.sh"); return
    mps = bool(torch.backends.mps.is_available())
    res["mps_available"] = mps
    print(f"  {'mps':<16s} available={mps}")
    for repo, pats in ((CFG["ctc"]["repo"], PATTERNS), (CFG["mms"]["repo"], MMS_PATTERNS), (CFG["whisper"]["repo"], PATTERNS), (CFG["qwen"]["repo"], PATTERNS)):
        try:
            p = model_dir(repo, pats)
            extra = ""
            if repo == CFG["mms"]["repo"]:
                ad = "adapter." + CFG["mms"]["adapter"] + ".safetensors"
                extra = "；" + ad + ("在" if (Path(p) / ad).exists() else C.RED + "不在" + C.RESET)
            print(f"  {'model':<16s} {repo} → {p}{extra}")
        except Exception as e:
            print(f"  {'model':<16s} {repo} 不在快取：{type(e).__name__}\n  先跑 ./present.sh fetch"); return
    # 句子與 wav
    sents = load_sentences()
    missing = [s["id"] for s in sents["test"] + sents["tune"] if not wav_path(s["id"]).exists()]
    print(f"  句子            測試 {len(sents['test'])}、調參 {len(sents['tune'])}（{sents.get('n_speakers')} 位說話人）；缺 wav {missing or '沒有'}")
    res["sentences"] = {"n_test": len(sents["test"]), "n_tune": len(sents["tune"]), "missing_wav": missing}
    # LM
    lm = load_lm(progress=lambda n: print(f"    …讀了 {n:,} 條 n-gram"))
    print(f"  LM              {lm.n_unigrams:,}／{lm.n_bigrams:,}／{lm.n_trigrams:,}；<unk> {lm.unk!r}；<s> {lm.bos_id is not None}；</s> {lm.eos_id is not None}；"
          f"{'快取' if lm.from_cache else '第一次讀 ARPA（已存 .npz）'} {lm.load_seconds:.1f} s")
    ex = {"THE|<s>": lm.logp("THE", ("<s>",)), "ZEBRA|<s>": lm.logp("ZEBRA", ("<s>",)), "OF THE": lm.logp("THE", ("OF",)), "OF ZEBRA": lm.logp("ZEBRA", ("OF",)),
          "oov XQZV": lm.logp("XQZV", ("THE",)), "ppl test refs": lm.perplexity([s["text"].split() for s in sents["test"]])}
    print("  LM 例子（ln P）  " + "；".join(f"{k} {val:.2f}" for k, val in ex.items()))
    oov = sorted({w for s in sents["test"] + sents["tune"] for w in s["text"].split() if not lm.in_vocab(w)})
    print(f"  參考句不在字典裡的詞 {len(oov)}：{oov[:12]}{'…' if len(oov) > 12 else ''}；OOV 罰 {lm.oov_penalty:g} + {lm.oov_per_char:g}×字元數（ln）")
    res["lm"] = {"n": [lm.n_unigrams, lm.n_bigrams, lm.n_trigrams], "unk": lm.unk, "load_seconds": lm.load_seconds, "from_cache": lm.from_cache, "examples": ex, "oov_in_refs": oov}
    # CTC：一句 greedy 對 beam λ=0、一次有 LM 的 beam search 要多久
    ctc = D2.CTC(CFG["ctc"]["device"])
    s0 = sents["test"][0]
    x = C.load_wav(wav_path(s0["id"]), FS)
    t0 = time.perf_counter(); lp = ctc.logprobs(x); t_lp = time.perf_counter() - t0
    g = ctc.decode(ctc_greedy(lp, ctc.blank))
    t0 = time.perf_counter(); b0 = decode_with_lm(lp, None, 0.0, 0.0, ctc, int(CFG["decode"]["beam"]), float(CFG["decode"]["prune_logp"])); t_b0 = time.perf_counter() - t0
    t0 = time.perf_counter(); b1 = decode_with_lm(lp, lm, 0.5, 1.0, ctc, int(CFG["decode"]["beam"]), float(CFG["decode"]["prune_logp"])); t_b1 = time.perf_counter() - t0
    print(f"  ctc {s0['id']}（{len(x) / FS:.1f} s，T = {lp.shape[0]}，log 機率 {1000 * t_lp:.0f} ms，{ctc.device}）\n    參考        {s0['text']}\n    greedy      {g}\n"
          f"    beam λ=0    {b0['text']}  ({1000 * t_b0:.0f} ms)\n    beam λ=.5   {b1['text']}  ln P_ctc {b1['logp_ctc']:.2f}、ln P_LM {b1['lm_ln']:.2f}、{b1['n_words']} 詞  ({1000 * t_b1:.0f} ms)")
    n_est = len(sents["test"]) * len(CFG["decode"]["lambdas"]) * t_b1
    print(f"  估整個 λ 掃描 {len(sents['test'])} 句 × {len(CFG['decode']['lambdas'])} 個 λ ≈ {n_est:.0f} s")
    res["ctc"] = {"device": str(ctc.device), "utt": s0["id"], "T": int(lp.shape[0]), "greedy": g, "beam0": b0["text"], "beam_lam05": b1["text"], "ms_logprobs": 1000 * t_lp, "ms_beam0": 1000 * t_b0,
                  "ms_beam_lm": 1000 * t_b1, "sweep_estimate_s": n_est}
    # 華語：MMS 在 MPS／CPU
    inputs = mandarin_inputs()
    print(f"  華語輸入        {[m['key'] for m in inputs] or '沒有（W1 d2 的 utterance.wav、W7 d2 的 mandarin.wav 都不在；./present.sh record）'}")
    res["mandarin_inputs"] = [m["key"] for m in inputs]
    res["mms"] = {}
    if inputs:
        m0 = inputs[0]
        x0 = C.load_wav(m0["path"], FS)
        for dev in (["mps", "cpu"] if mps else ["cpu"]):
            try:
                t0 = time.perf_counter(); mm = MMS(dev); tl = time.perf_counter() - t0
                r = mm.transcribe(x0)
                r2 = mm.transcribe(x0)
                trad, resid = to_traditional(r["text"])
                entry = {"load_s": tl, "how": mm.how, "lm_head": mm.vocab_size, "n_params": mm.n_params, "T": r["T"], "blank_frac": r["blank_frac"], "seconds_first": r["seconds"], "seconds_second": r2["seconds"],
                         "n_chars": len(r["text"]), "trad_residual": resid}
                if m0["public_text"]:
                    entry["text"], entry["text_trad"] = r["text"], trad
                print(f"  mms-1b-all {dev:<4s} 載入 {tl:.1f} s（{mm.how}；lm_head {mm.vocab_size}）；{m0['key']} → {trad!r}（{len(r['text'])} 字，T = {r['T']}，blank {100 * r['blank_frac']:.0f}%，{r['seconds']:.2f}／{r2['seconds']:.2f} s）")
                if m0["text"]:
                    a = align_chars(norm_zh(m0["text"]), norm_zh(trad))
                    cl = classify_subs(a["ops"])
                    print(f"    對參考：CER {100 * a['cer']:.1f}%，替換 {a['S']}（同音 {cl['counts']['homophone']}、異音 {cl['counts']['different']}、無法判定 {cl['counts']['unknown']}）")
                    entry["cer"], entry["counts"] = a["cer"], cl["counts"]
                res["mms"][dev] = entry
                del mm
                break
            except Exception as e:
                res["mms"][dev] = {"error": f"{type(e).__name__}: {e}"}
                print(f"  mms-1b-all {dev:<4s} 失敗：{type(e).__name__}: {str(e)[:200]}")
    (HERE / "runs").mkdir(exist_ok=True)
    (HERE / "runs" / "check.json").write_text(json.dumps(res, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    C.note("  結果存在 runs/check.json（進版控；華語只有 public_text 那句的文字）")


# ── fetch／record／export ─────────────────────────────────────────────────
def fetch() -> None:
    import urllib.request
    from huggingface_hub import HfApi, snapshot_download
    C.banner("下載模型、LM，讀參考文字", C.AMBER)
    lock = read_lock()
    lines = []
    for repo, pats in ((CFG["ctc"]["repo"], PATTERNS), (CFG["mms"]["repo"], MMS_PATTERNS), (CFG["whisper"]["repo"], PATTERNS), (CFG["qwen"]["repo"], PATTERNS)):
        sha = lock.get(repo) or HfApi().model_info(repo).sha
        path = snapshot_download(repo, revision=sha, allow_patterns=pats)
        lines.append(f"{repo} {sha}")
        print(f"  {repo} @ {sha[:12]} → {path}")
    MODEL_LOCK.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("  commit 寫進 model.lock（wav2vec2、Whisper、Qwen3-ASR 與 Demo 2 同一個；MMS-1B-all 是這次抓到的，只抓 model.safetensors 與 cmn 的 adapter）")
    # LM
    dst = _p(CFG["lm"]["file"])
    if dst.exists():
        print(f"  LM 已在：{dst}")
    else:
        dst.parent.mkdir(parents=True, exist_ok=True)
        print(f"  下載 {CFG['lm']['url']} …")
        urllib.request.urlretrieve(CFG["lm"]["url"], dst)
        print(f"  → {dst}（{dst.stat().st_size / 1e6:.1f} MB）")
    cache = _p(CFG["lm"]["cache"])
    if cache.exists():
        cache.unlink()
        print("  舊的 .npz 快取刪掉，check 會重建")
    # 句子
    subset = json.loads(_p(CFG["sentences"]["subset"]).read_text(encoding="utf-8"))
    sel = select_sentences(subset, CFG["sentences"])
    want = {s["id"] for s in sel["test"] + sel["tune"]}
    texts = read_transcripts(_p(CFG["sentences"]["dev_clean_tar"]), want)
    for s in sel["test"] + sel["tune"]:
        s["text"] = texts.get(s["id"], "")
    missing_text = [s["id"] for s in sel["test"] + sel["tune"] if not s["text"]]
    missing_wav = [s["id"] for s in sel["test"] + sel["tune"] if not wav_path(s["id"]).exists()]
    if missing_wav:
        done = D2.extract_from_tar(missing_wav)
        print(f"  從 dev-clean.tar.gz 解出 wav {done}")
    SENTENCES.parent.mkdir(exist_ok=True)
    SENTENCES.write_text(json.dumps({"source": {"audio": subset["source"]["audio"], "subset": CFG["sentences"]["subset"], "license": "CC BY 4.0", "time": C.now(), "selection": CFG["sentences"]},
                                     "n_speakers": sel["n_speakers"], "test": sel["test"], "tune": sel["tune"]}, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"  測試集 {len(sel['test'])} 句、調參集 {len(sel['tune'])} 句 → {SENTENCES.relative_to(HERE)}" + (f"；{C.RED}沒讀到參考文字 {missing_text}{C.RESET}" if missing_text else ""))
    print("\n下一步：./present.sh check → ./present.sh rehearse")


def read_transcripts(tar_path: Path, want: set[str]) -> dict[str, str]:
    """從 dev-clean.tar.gz 的 *.trans.txt 讀指定句子的參考文字。"""
    import tarfile
    out = {}
    if not tar_path.exists():
        print(f"  {C.RED}沒有 {tar_path}（W3 d3 的 fetch 會下載）{C.RESET}")
        return out
    chapters = {"-".join(u.split("-")[:2]) for u in want}
    with tarfile.open(tar_path, "r:gz") as tf:
        for m in tf:
            if not m.name.endswith(".trans.txt"):
                continue
            ch = Path(m.name).name.replace(".trans.txt", "")
            if ch not in chapters:
                continue
            for line in tf.extractfile(m).read().decode("utf-8").splitlines():
                parts = line.strip().split(" ", 1)
                if len(parts) == 2 and parts[0] in want:
                    out[parts[0]] = parts[1]
            if len(out) == len(want):
                break
    return out


def record() -> None:
    m = next(mi for mi in CFG["mandarin"]["inputs"] if mi["key"] == "own")
    PRIVATE.mkdir(parents=True, exist_ok=True)
    C.banner(f"錄 {m['record_seconds']:g} 秒華語，一句自然的話（同音字多一點更好）", C.AMBER)
    x = C.record(float(m["record_seconds"]), FS)
    C.save_wav(_p(m["path"]), x, FS)
    print(f"  → {_p(m['path'])}（不進版控、不上站）。想聽：afplay {_p(m['path'])}")
    ref = input("  參考文字（繁體，照講的打；Enter 跳過）：").strip()
    if ref:
        _p(m["text_file"]).write_text(ref + "\n", encoding="utf-8")
    print("  之後要改就直接編輯 mandarin.txt。")


def export() -> None:
    src = HERE / "runs" / "rehearsal" / "rehearsal.json"
    if not src.exists():
        sys.exit("沒有 runs/rehearsal/rehearsal.json，先跑 ./present.sh rehearse")
    log = json.loads(src.read_text(encoding="utf-8"))
    if log.get("backend") != "wav2vec2":
        sys.exit(f"rehearsal.json 的 backend 是 {log.get('backend')}，不是真模型的紀錄，不匯出")
    R = log["results"]
    sw, base, ex = R.get("sweep"), R.get("baseline"), R.get("example")
    data = json.loads(METRICS.read_text(encoding="utf-8")) if METRICS.exists() else {}
    picks = R.get("sentence_picks", {})
    data["demo3"] = {
        "source": "demos/w07_d3_decoding/runs/rehearsal/rehearsal.json (Mandarin: counts, plus text only for the W1 ticket sentence)",
        "time": log["time"], "device": log["device"], "models": log["versions"].get("models"), "transformers": log["versions"].get("transformers"), "torch": log["versions"].get("torch"),
        "test_set": {"n_sentences": len(log["sentences"]["test"]), "n_ref_words": base["greedy"]["n_ref"] if base else None, "dur_s": base.get("dur_total") if base else None, "utts": [s["id"] for s in log["sentences"]["test"]]},
        "lm": log.get("lm"), "eta": log.get("eta"), "eta_tuning": R.get("eta_tuning"), "beam": sw["beam"] if sw else None,
        "greedy_wer": base["greedy"]["wer"] if base else None, "beam0_wer": base["beam0"]["wer"] if base else None,
        "sweep": {"lambdas": sw["lambdas"], "wer": [r["wer"] for r in sw["rows"]], "S": [r["S"] for r in sw["rows"]], "D": [r["D"] for r in sw["rows"]], "I": [r["I"] for r in sw["rows"]],
                  "best_lam": sw["best_lam"], "best_wer": sw["best_wer"], "large_lam": float(CFG["decode"]["large_lambda"]), "seconds": sw["seconds"]} if sw else None,
        "example_p32": {"greedy": ex["greedy_label"], "beam": ex["beam_label"], "matches_json": ex["ok"]} if ex else None,
        "sentence_picks": picks,
        "mandarin": log.get("mandarin"), "mms_error": log.get("mms_error"), "qwen_error": log.get("qwen_error"),
    }
    METRICS.parent.mkdir(parents=True, exist_ok=True)
    METRICS.write_text(json.dumps(data, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    d3 = data["demo3"]
    print(f"→ {METRICS}（demo3）：greedy {100 * d3['greedy_wer']:.2f}% → beam λ=0 {100 * d3['beam0_wer']:.2f}% → 最佳 λ {d3['sweep']['best_lam']:g} {100 * d3['sweep']['best_wer']:.2f}%"
          f" → λ {d3['sweep']['lambdas'][-1]:g} {100 * d3['sweep']['wer'][-1]:.2f}%；η {d3['eta']}")


# ── selftest ───────────────────────────────────────────────────────────────
def selftest() -> None:
    global mandarin_inputs
    C.banner("selftest（不需要模型）", C.AMBER)
    fails = []

    def chk(name: str, cond: bool, extra: str = "") -> None:
        print(f"  {'ok ' if cond else 'FAIL'} {name}{('  ' + extra) if extra else ''}")
        if not cond:
            fails.append(name)
    # ARPA
    lm = ArpaLM.parse(_write_fake_arpa())
    chk("ARPA：1/2/3-gram 的數目", (lm.n_unigrams, lm.n_bigrams, lm.n_trigrams) == (19, 18, 10), f"{lm.n_unigrams}/{lm.n_bigrams}/{lm.n_trigrams}")
    chk("ARPA：<unk>、<s>、</s>", lm.unk == "<unk>" and lm.bos_id is not None and lm.eos_id is not None)
    chk("三元在表裡：P(CAT | <s> THE) = 10^-0.1", abs(lm.logp("CAT", ("<s>", "THE")) - (-0.1 * LN10)) < 1e-6)
    # backoff：P(DOG | <s> THE)：沒有三元 → bo(<s> THE) = -0.2 + P(DOG | THE)；沒有二元 → bo(THE) = -0.4 + P(DOG) = -1.4 → -2.0
    chk("backoff 兩層：P(DOG | <s> THE) = 10^(-0.2-0.4-1.4)", abs(lm.logp("DOG", ("<s>", "THE")) - (-2.0 * LN10)) < 1e-6, f"{lm.logp('DOG', ('<s>', 'THE')) / LN10:.3f}")
    # backoff 一層：P(MAT | ON THE) 有三元 -0.1；P(STORE | ON THE)：沒有三元 → bo(ON THE) -0.2 + P(STORE | THE) = -0.5 → -0.7
    chk("backoff 一層：P(STORE | ON THE) = 10^(-0.2-0.5)", abs(lm.logp("STORE", ("ON", "THE")) - (-0.7 * LN10)) < 1e-6)
    # 歷史不在表裡：P(THE | DOG RAN)：沒有三元、沒有 (DOG RAN) 二元 → bo 0 + P(THE | RAN)：沒有二元 → bo(RAN) -0.3 + P(THE) -0.6 → -0.9
    chk("歷史二元不存在：bo = 0", abs(lm.logp("THE", ("DOG", "RAN")) - (-0.9 * LN10)) < 1e-6)
    chk("OOV → <unk>：P(XYZ | <s>) = bo(<s>) -0.5 + P(<unk>) -1.5", abs(lm.logp("XYZ", ("<s>",)) - (-2.0 * LN10)) < 1e-6)
    lm.oov_penalty, lm.oov_per_char = -2.0, -0.5
    chk("OOV 罰：固定 -2 加每字元 -0.5（XYZ 三個字元 → -3.5）", abs(lm.logp("XYZ", ("<s>",)) - (-2.0 * LN10 - 3.5)) < 1e-6)
    chk("OOV 罰不影響字典裡的詞", abs(lm.logp("CAT", ("<s>", "THE")) - (-0.1 * LN10)) < 1e-6 and lm.oov_cost("CAT") == 0.0)
    lm.oov_penalty, lm.oov_per_char = 0.0, 0.0
    tot, per = lm.sentence_logp("THE CAT SAT ON THE MAT".split())
    hand = (-0.5 + -0.1 + -0.1 + -0.1 + -0.1 + -0.1 + -0.1) * LN10     # <s> THE 二元(-0.5)，之後六個三元各 -0.1（含 THE MAT </s>）
    chk("整句 ln P（含 </s>）對手算", abs(tot - hand) < 1e-6 and len(per) == 7, f"{tot / LN10:.2f} 對 {hand / LN10:.2f}")
    tmp = HERE / "runs" / "_selftest" / "lm.npz"
    lm.save(tmp)
    lm2 = ArpaLM.load(tmp)
    chk("save/load 後查表相同", abs(lm2.logp("DOG", ("<s>", "THE")) - lm.logp("DOG", ("<s>", "THE"))) < 1e-9 and lm2.unk == "<unk>")
    # LMBonus
    fc = FakeCTC()
    b = lm_bonus(lm, 1.0, 0.5, fc)
    st = b.start()
    for c in fc.encode("THE CAT"):
        st = b.step(st, c)
    chk("LMBonus：走完 THE、CAT 還沒接詞界 → 只算了 THE", st[4] == 1 and abs(st[3] - lm.logp("THE", ("<s>",))) < 1e-9 and "".join(st[2]) == "CAT")
    chk("LMBonus：score = λ·ln P + η·詞數（罰都是 0 時半個詞不估分）", abs(b.score(st) - (lm.logp("THE", ("<s>",)) + 0.5)) < 1e-9)
    lm.oov_penalty, lm.oov_per_char = -2.0, -0.5
    chk("LMBonus：修剪時半個詞 CAT 先當 OOV 估 -2 - 1.5", abs(b.score(st) - (lm.logp("THE", ("<s>",)) + 0.5 - 3.5)) < 1e-9)
    b_off = LMBonus(lm, 1.0, 0.5, {v: k for k, v in fc.vocab.items() if len(k) == 1}, fc.delim, partial_as_oov=False)
    chk("partial_as_oov=False 時不估", abs(b_off.score(st) - (lm.logp("THE", ("<s>",)) + 0.5)) < 1e-9)
    lm.oov_penalty, lm.oov_per_char = 0.0, 0.0
    fin = b.finish(st)
    want = lm.logp("THE", ("<s>",)) + lm.logp("CAT", ("<s>", "THE")) + lm.logp("</s>", ("THE", "CAT")) + 1.0
    chk("LMBonus：finish 補上最後一個詞與 </s>", abs(fin - want) < 1e-9, f"{fin:.4f} 對 {want:.4f}")
    st2 = b.step(st, fc.delim)
    st2 = b.step(st2, fc.delim)
    chk("連續兩個詞界不會多算一個空詞", st2[4] == 2 and st2[2] == ())
    b0 = lm_bonus(None, 0.0, 0.0, fc)
    chk("沒有 LM、λ=η=0 時 bonus 全是 0", b0.finish(b0.step(b0.step(b0.start(), fc.vocab["A"]), fc.delim)) == 0.0)
    # p32 的例子
    if DATA_JSON.exists():
        data = json.loads(DATA_JSON.read_text(encoding="utf-8"))["prefix_example"]
        ex = prefix_example(data)
        chk("p32：九條路徑、合併、greedy b、prefix beam a——對 w07-data.json 逐位相同", ex["ok"], f"greedy {ex['greedy_label']!r}、beam {ex['beam_label']!r}、a {ex['by_label'].get('a')}、b {ex['by_label'].get('b')}")
        chk("p32：prefix beam 的 P(a) = 0.35、P(b) = 0.3225", abs(ex["beam_probs"]["a"] - 0.35) < 1e-9 and abs(ex["beam_probs"]["b"] - 0.3225) < 1e-9)
    else:
        chk("w07-data.json 在", False, str(DATA_JSON))
    # WER／對齊
    a = wer_words("A CAT SAT", "THE CAT SAT DOWN")
    chk("WER：p62 的例子 S 1／I 1／N 3 → 2/3", (a["S"], a["D"], a["I"], a["n_ref"]) == (1, 0, 1, 3) and abs(a["wer"] - 2 / 3) < 1e-12)
    cw = corpus_wer(["A B C", "D E"], ["A X C", "D E F"])
    chk("corpus WER：(1+1)/5", cw["errors"] == 2 and cw["n_ref"] == 5 and cw["per_sentence"] == [1, 1])
    chk("norm_en：大寫、去標點、保留撇號", norm_en("Don't stop, now.") == "DON'T STOP NOW")
    # 華語
    chk("norm_zh：去標點、轉簡體", norm_zh("我想訂一張，下週三去台南的票。") == "我想订一张下周三去台南的票")
    trad, resid = to_traditional("我想定一张下周三去台南的票")
    chk("to_traditional：张→張；周不動（一簡對多繁）", "張" in trad and "周" in trad and "週" not in trad, f"{trad}（殘留 {resid}）")
    ca = align_chars("我想订一张下周三去台南的票", "我想定一张下周三去台南的票")
    chk("字元對齊：訂→定 一個替換", (ca["S"], ca["D"], ca["I"]) == (1, 0, 0))
    cl = classify_subs(ca["ops"])
    try:
        import pypinyin  # noqa: F401
        chk("同音判定：訂／定 同音同調（ding4）", cl["counts"]["same_tone"] == 1, str(cl["subs"]))
        chk("同音判定：伯／柏 同音異調（bo2／bai3｜bo2）", homophone("伯", "柏") in ("same_tone", "same_base"), homophone("伯", "柏"))
        chk("同音判定：張／票 異音", homophone("张", "票") == "different")
    except ImportError:
        chk("沒有 pypinyin → unknown（容器裡可接受）", cl["counts"]["unknown"] == 1)
    # 句子挑選
    subset_p = _p(CFG["sentences"]["subset"])
    if subset_p.exists():
        subset = json.loads(subset_p.read_text(encoding="utf-8"))
        s1, s2 = select_sentences(subset, CFG["sentences"]), select_sentences(subset, CFG["sentences"])
        ids_t, ids_u = {s["id"] for s in s1["test"]}, {s["id"] for s in s1["tune"]}
        chk("句子挑選：確定性、測試集與調參集不重疊、每位說話人一句", s1 == s2 and not (ids_t & ids_u) and len({s["spk"] for s in s1["test"] + s1["tune"]}) == len(ids_t | ids_u),
            f"{len(s1['test'])} + {len(s1['tune'])}，{s1['n_speakers']} 位說話人")
        chk("句子長度都在範圍內", all(CFG["sentences"]["min_s"] <= s["dur_s"] <= CFG["sentences"]["max_s"] for s in s1["test"] + s1["tune"]))
    else:
        print(f"  （跳過句子挑選：沒有 {subset_p}）")
    # fake 整條流程
    mandarin_inputs = _fake_mandarin_inputs
    out = D2.Out()
    try:
        d = _fake_demo(out, tag="selftest")
        d.tune_eta()
        d.run_all()
        for k in "012n3melth":
            d.dispatch(k)
        sw = d.res["sweep"]
        chk("fake：λ 掃描跑完、有最佳 λ", len(sw["rows"]) == len(CFG["decode"]["lambdas"]) and "best_lam" in sw, f"best λ {sw['best_lam']}，WER {[round(r['wer'], 3) for r in sw['rows']]}")
        chk("fake：λ=0 有錯、加 LM 之後修好（STOVE→STORE、REED→READ）", sw["rows"][0]["errors"] > 0 and sw["best_wer"] < sw["rows"][0]["wer"])
        chk("fake：greedy 與 beam λ=0 都有結果", d.res["baseline"]["greedy"]["n_ref"] > 0)
        chk("fake：華語三格在私密水槽", all(k.startswith("zh:fake:") for k in d.res_priv if k.startswith("zh:")) and len([k for k in d.res_priv if k.startswith("zh:")]) == 3)
        chk("fake：公開 txt 裡沒有華語參考句", "訂一張" not in out.text() and "訂一張" in out.text(private=True))
        tmp = HERE / "runs" / "_selftest"
        d.save(tmp)
        j = json.loads((tmp / "rehearsal.json").read_text(encoding="utf-8"))
        chk("rehearsal.json 可讀；public_text 的華語有文字", j["backend"] == "fake" and j["mandarin"]["results"]["zh:fake:mms"].get("text"))
        chk("lambda_sweep.png 存了", (tmp / "lambda_sweep.png").exists())
        rp = Demo(D2.Out(), d.sents, FakeCTC(), None, None, None, None, stored=d.res, stored_private=d.res_priv, tag="replay", eta=d.eta)
        chk("replay 從紀錄重建", rp.dispatch("1") and rp.dispatch("2") and rp.dispatch("3") and rp.dispatch("m") and rp.dispatch("t"))
        import shutil
        shutil.rmtree(tmp)
        for f in (PRIVATE / "private__selftest.json", PRIVATE / "private__selftest.txt"):
            if f.exists():
                f.unlink()
    except Exception as e:
        import traceback
        traceback.print_exc()
        chk("fake 整條流程", False, f"{type(e).__name__}: {e}")
    finally:
        mandarin_inputs = _real_mandarin_inputs
    print(f"\n  {'全部通過' if not fails else '失敗：' + ', '.join(fails)}")
    if fails:
        sys.exit(1)


# ── main ───────────────────────────────────────────────────────────────────
def main() -> None:
    global mandarin_inputs
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["show", "rehearse", "replay", "check", "fetch", "record", "fake", "selftest", "export"])
    args = ap.parse_args()
    if args.cmd == "check": check(); return
    if args.cmd == "fetch": fetch(); return
    if args.cmd == "record": record(); return
    if args.cmd == "selftest": selftest(); return
    if args.cmd == "export": export(); return

    out = D2.Out()
    if args.cmd == "fake":
        mandarin_inputs = _fake_mandarin_inputs
    d = build(args.cmd, out)
    if args.cmd == "rehearse":
        if CFG["decode"]["eta"] == "auto":
            C.note("  先在調參集上調 η …")
            d.tune_eta()
            C.note(f"  η = {d.eta:g}")
        d.run_all()
        d.save(HERE / "runs" / "rehearsal")
        print("\n下一步：./present.sh 自己按一遍 0 1 2 3 n m t；把 rehearsal.json 的數字填進 p41 的表、.gen 區塊、【操作】【講述】；./present.sh export 寫進 w07-metrics.json。")
        return
    out.banner("W7 Demo 3：解碼——先驗的重量 λ 掃一遍；同一個 CTC 帶到華語" + (f"   [{d.tag}]" if d.tag else ""))
    d.menu()
    while True:
        k = C.getkey()
        if not d.dispatch(k):
            break
    if args.cmd == "show":
        d.save(HERE / "runs" / "live" / C.now().replace(":", "-"))


if __name__ == "__main__":
    main()

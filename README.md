# Speech-AI — Speech Processing and Human-Machine Interaction

**語音處理與人機互動** · 中文版說明：[README.zh-TW.md](README.zh-TW.md)

A graduate course (Master's / PhD): 14 weeks × 3 hours, lectures throughout, in Mandarin; slides and the course site are in English. Assessment is a midterm report (40%) and a final paper (60%), both written in the INTERSPEECH long-paper format and done in teams of 2–4.

**Course site:** <https://hungshinlee.github.io/Speech-AI/>

**Lecturer:** [Hung-Shin Lee (李鴻欣)](https://web.ntnu.edu.tw/~hslee/)

This repository holds the *public* side of the course — the website source, the slides, the classroom demo code, the supplementary guides and the report template. The teaching materials it is generated from (the full course outline, speaker notes, exam questions and rubrics) live in a private repository and are not here; see [What is deliberately not here](#what-is-deliberately-not-here).

## The course in one sentence

The course is built backwards from a single question: **how do you build a spoken dialogue system that can listen while it speaks, be interrupted, and know when it is its turn to talk?** Starting from that end goal — a half- and full-duplex spoken dialogue AI system — it works out what has to be known about signal processing, representation learning, generative models and interaction modeling, and teaches exactly that, aligned with the state of the field in 2025–2026. The depth is graduate: derivation skeletons, the literature behind each idea, and open problems.

Three threads run through all fourteen weeks and are asked of every layer: **Representation** (what does this layer operate on — waveform, spectrogram, continuous SSL feature, discrete token — and what decides the frame rate), **Latency** (how much of the latency budget does this module spend, and is it algorithmic latency from lookahead or compute latency), and **Supervision** (where does this capability come from — labeled data, self-supervision, synthetic data, or human preference).

## Course map

| Part I — Foundations | Part II — Modules | Part III — Dialogue Systems |
|---|---|---|
| W1 A systems view of spoken dialogue and the latency budget | W7 Speech recognition: three paradigms, streaming, and the LLM turn | W11 Audio-native language models: building half-duplex spoken dialogue |
| W2 Speech signals, auditory front-ends, and the physics of representation | W8 Speech synthesis I: generative paradigms and streaming TTS | W12 Full-duplex I: linguistic foundations and modeling of turn-taking |
| W3 The alignment problem: from HMM to CTC | W9 Speech synthesis II: controllability, post-training alignment, evaluation, and misuse | W13 Full-duplex II: architectures, multi-stream modeling, and data |
| W4 Sequence models and streaming architectures: Transformer, Conformer, RNN-T | W10 The physical layer of full-duplex: VAD, AEC, enhancement, separation, speaker | W14 Evaluation, alignment, deployment, and open problems |
| W5 Self-supervised representation learning: where speech foundation models come from | | |
| W6 Neural audio codecs and discretization: turning speech into a language | | |

**W6 is the hinge of the course.** Everything in Part III rests on three things decided there — frame rate, token budget, and the separation of semantic from acoustic information — so that is the week not to miss. W3, W6 and W8 are the three mathematical peaks.

## What is published so far

Weeks are opened on the site as they are taught. As of 2026-10-19, **W1–W6** are open: week pages, slides, and the demo code for every week — so all of Part I is published. The remaining weeks appear on the course map as greyed-out placeholders; their pages are generated but not rendered, so there are no dead links.

| | W1 | W2 | W3 | W4 | W5 | W6 | W7–W14 |
|---|---|---|---|---|---|---|---|
| Week page (positioning, learning objectives, references, demos) | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | generated, not yet published |
| Slides (`slides/wNN.qmd`, reveal.js) | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | — |
| Demo code (`demos/wNN_dK_*/`) | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | — |

Beyond the week pages:

- **Syllabus** carries the course setup, the three-part structure, and the **latency budget** — the shared coordinate system every week is measured against.
- **Slides** indexes the decks; each deck has a briefing slide before every live demo and the measured result as a fragment after it. Where a rehearsal contradicted the lecture's prediction (W6 demo 1: Mimi's first codebook alone keeps neither the words nor the voice), the slide says so rather than keeping the prediction.
- **Resources** lists the core textbooks, one paper per week, the toolchain the demos run on, and what cannot be shown on the demo hardware.
- **Supplements** backs the two assessed deliverables with five items: **26 problem cards** for an INTERSPEECH 2027 paper (Taiwan's languages, spoken dialogue, codecs and SSL, evaluation, robustness; resources checked against their official pages on 2026-10-16; the method is left to you), the **midterm report** guide (from a problem card to a proof of concept), the **final paper** guide (from MVP to a conference-ready paper), **writing and narrative logic** for speech papers, and the **LaTeX report template** used by both reports (INTERSPEECH 2026 long-paper kit with guidance boxes; download the ZIP and import it into Prism). The four guides are in Chinese; the template is in English.

## For students

- **You do not need to run anything.** The course is lectures only. Every demo is run live by the lecturer, and every number on the slides comes from a rehearsal record checked into `demos/*/runs/rehearsal/`. The code is published so you can read it, re-run it, and change it.
- **Each week page has a Demos section** — one row per demo with *Why*, *What you see* and *Result*, linking to the demo's folder and its student README. The slide footer names the demo (`Demo K`) so the three line up.
- **The slides have no speaker notes.** Pressing `S` in a deck opens an empty notes panel. That is intentional, not a bug: the Chinese lecture script is part of the teaching materials and is not published.
- **Demos were tested on one machine** — a MacBook Pro M5 Max (64 GB unified memory, no CUDA), Python 3.12. `demos/README.md` says which demos are portable (standard library, or CPU-only numpy/scipy/PyTorch) and which need Apple silicon (`mlx`), and pins every package and model revision in `demos/versions.lock` and the per-demo `model.lock`. The lecturer's own recordings (behind W2 demo 2, W3 demo 1 and W6 demo 1) are not published; those READMEs say how to record your own input.
- **References are filtered before they reach the site.** Entries the course outline marks as unverified are dropped from the site rather than shown with a caveat, and the references added in the W2–W6 revisions were checked title and first author against the arXiv abstract page, the ACL Anthology, the ISCA archive, PMLR, or Crossref. Where a source is a blog post, model card, or specification rather than a peer-reviewed paper, the sentence says so.
- The site has a **presentation mode** (press `z`, or use the navbar button) that collapses the side panels for projection.

## Repository layout

Most of this repository is *generated*. Files marked ⚙︎ are overwritten on the next publish and must not be edited by hand; the first line of each such file says where its original lives.

| Path | What it is |
|---|---|
| `index.qmd`, `syllabus.qmd`, `resources.qmd` | Hand-written site pages (English) |
| `docs/site-en.md` | Source of the English prose blocks pulled into Syllabus and Resources (latency budget, textbooks, one-paper-per-week, toolchain) |
| `docs/course-map-en.md` | The course map in English, kept for reference; the one on the home page is generated |
| `weeks/w01.qmd` … `w14.qmd` | ⚙︎ Week pages, filtered from the private course outline by `scripts/build_weeks.py` |
| `_includes/*.md` | ⚙︎ Generated fragments: course map, weekly schedule, reading table, latency budget, textbooks, toolchain |
| `slides.qmd`, `supplements.qmd` | ⚙︎ Slide index and Supplements index |
| `slides/wNN.qmd`, `slides/theme.scss`, `slides/assets/` | ⚙︎ Slides with speaker notes **stripped**, published from the private originals; `assets/wNN/` holds the figures and the `wNN-data.json` / `wNN-metrics.json` every number on the slides is read from |
| `slides/scripts/` | ⚙︎ Analysis scripts referenced on the W1 slides (energy VAD and overlap / response / stop latency on a two-channel recording, two-track timeline figure) |
| `demos/` | ⚙︎ Classroom demo code, rehearsal records, and student-facing READMEs, published by whitelist; `demos/README.md` is the entry point, `common.py` / `dsp.py` / `selftest.py` are shared, `setup.sh` builds the shared `.venv` from `versions.lock` |
| `supplements/*.md` | ⚙︎ Supplementary guides (Chinese), published from the private originals with their private blocks removed |
| `supplements/Speech_AI_Report_Template.md`, `supplements/speech-ai-report-template.zip` | ⚙︎ The report template's one-page guide and the LaTeX project as a ZIP, both built from the private `report-template/` |
| `supplements/*.qmd` | ⚙︎ Quarto wrappers around the guides, generated by `build_weeks.py` |
| `scripts/build_weeks.py`, `scripts/visibility.py` | The filter: which sections of the outline are public (deny by default, whitelist), how citation markers are handled, which weeks are published, how the Demos tables are laid out |
| `styles.scss`, `_present-mode.html`, `_quarto.yml` | Site theme, presentation mode, Quarto configuration |
| `.github/workflows/publish.yml` | CI: a leak guard plus `quarto render`, deployed to GitHub Pages |
| `CLAUDE.md` | Maintainer notes: build procedure, rendering pitfalls, decisions |

## What is deliberately not here

The course outline (`course-outline.md`) is the single source of truth for the course and lives only in the private repository. Week pages on this site are produced from it by a **deny-by-default filter**: only three sections per week are public — *Positioning*, *Learning objectives*, and *References* — plus a student-facing *Demos* section that links to the published code. Lesson timing, the misconception list, demo scripts for the lecturer, and the private design notes stay private. The same goes for the Chinese speaker notes in the slides, the lecturer's demo READMEs (rehearsal notes and fallbacks), the lecturer's own recordings, the private blocks of the supplementary guides, and all assessment material (questions and rubrics).

If `docs/course-outline.md` ever appears in this repository, that is a regression, not a feature.

## How the site is built

The normal entry point is `bin/publish.py` in the private repository. One run does, in order:

1. re-extracts the speaker notes from the slide originals (so the lecturer's printable notes never drift from the decks);
2. publishes the slides **with the notes stripped**, together with `theme.scss` and `assets/`;
3. copies through four whitelist channels, each listed in the private `sites.json`: the demo folders (`public_demos`, with the lecturer's recordings excluded and absolute paths in rehearsal records redacted), the supplementary guides (`public_supplements`, private blocks removed), the report template (`public_report_template`, packed into the ZIP plus its one-page guide), and handout PDFs when a course has them;
4. runs the filtered build of the week pages and the `_includes/` fragments from `course-outline.md`;
5. checks that nothing teacher-only leaked across `weeks/`, `_includes/`, `slides/`, `demos/` and `supplements/` — a single surviving `::: {.notes}` block aborts the publish.

It does **not** commit or push; that is done by hand so the diff is always seen.

```bash
cd ~/Course-Hub && python3 bin/publish.py speech_ai
```

To rebuild only the week pages from a checkout of the outline (this path does not update slides, demos or supplements):

```bash
export COURSE_OUTLINE=~/Course-Hub/speech_ai/course-outline.md
python3 scripts/build_weeks.py   # after every outline change
quarto preview                   # local preview (brew install --cask quarto)
quarto render                    # produces _site/
```

Pushing to `main` triggers GitHub Actions, which **does not** rebuild the week pages (CI has no access to the outline). It only runs a leak check and `quarto render`. The cost of that design is that the site drifts if the local build step is forgotten.

Opening a new week requires two coordinated edits: add the week to `PUBLISHED_WEEKS` in `scripts/build_weeks.py` (controls the links on the home page, course map and resources) and add its page to the `render` list and sidebar in `_quarto.yml` (controls whether the page exists). The slide index links a deck — and its week page — as soon as `slides/wNN.qmd` exists, so a deck published before its week is open leaves a dead link on the Slides page until the week opens; publish the two together.

Supplementary guides are written in the private repository and listed in `sites.json`; each one carries three comment lines under its H1 (`en`, `order`, `summary`) that drive the Supplements index. The English title of each week is written in the outline on the line after the week heading (`<!-- en: ... -->`), and the build fails if one is missing.

## Language

The site skeleton and the body of every published week page are in English. The English for each week is written in the outline alongside the Chinese and extracted at build time; a week without an English block falls back to Chinese rather than breaking, and the build prints the list of weeks still to be translated. Two things stay Chinese on purpose: each week page's `subtitle` (the Chinese week title, kept for cross-reference) and the four `supplements/` guides, which match the language of instruction. The report template is in English, like the papers it is for.

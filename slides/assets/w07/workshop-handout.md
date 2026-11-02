# W7 · Error-analysis workshop — one transcript, as the systems wrote it

*Ten segments, copied verbatim from the rehearsal outputs of this week's three demos (2026-11-02). Nothing has been edited. The reference for each segment is on the back.*

**Your task.** (1) Align each segment with its reference and mark every error as S, D or I. (2) Name the kind of each error — the seven-row table on the slide is the guide; one segment does not fit any of the seven. (3) Trace each error to its cause: **paradigm** (the model's structure only allows this), **decoding** (a search or a knob), or **data** (the prior it learned — or the scorer's rules).

| # | output |
|---|---|
| 1 | Mr. Quilter is the apostle of the middle classes, and we are glad to welcome his gospel. |
| 2 | I am the one to go after Walt. If any one has top, I'll go down, Mister Thomas. |
| 3 | None might exert or confess or in any way meal with the religious experience of another. |
| 4 | Colonel Leonidas Talbot regarded the white flag with feelings which triumph and sadness were mingled strangely. |
| 5 | Think of Glenarvan Glenarvan. Think of Mary Grant. |
| 6 | 我想定一張下周山去臺南的票 |
| 7 | 我想订一张下周三去台南的票。 |
| 8 | I kind of took that picture, because the burnwasty was really fine |
| 9 | I'm sorry, I'm sorry, I'm sorry, I'm sorry, I'm sorry, I'm sorry, I'm sorry, I'm sorry, I'm sorry, I'm sorry, … (continues to the token limit) |
| 10 | Thanks for watching, and don't forget to subscribe to the channel. |

Answers are discussed in class; W14 returns to this page with semantic metrics and calibration.

<div style="page-break-after: always"></div>

# Reference — what was actually in the audio

| # | reference |
|---|---|
| 1 | MISTER QUILTER IS THE APOSTLE OF THE MIDDLE CLASSES AND WE ARE GLAD TO WELCOME HIS GOSPEL |
| 2 | I AM THE ONE TO GO AFTER WALT IF ANYONE HAS TO I'LL GO DOWN MISTER THOMAS |
| 3 | NONE MIGHT EXHORT OR CONFESS OR IN ANY WAY MEDDLE WITH THE RELIGIOUS EXPERIENCE OF ANOTHER |
| 4 | COLONEL LEONIDAS TALBOT REGARDED THE WHITE FLAG WITH FEELINGS IN WHICH TRIUMPH AND SADNESS WERE MINGLED STRANGELY |
| 5 | THINK OF LADY GLENARVAN THINK OF MARY GRANT |
| 6 | 我想訂一張下週三去台南的票（a Taiwanese speaker; reference in Traditional characters） |
| 7 | 我想訂一張下週三去台南的票（the same recording as 6, a different system） |
| 8 | *no speech* — 10 s of white noise |
| 9 | *no speech* — the acoustic evidence was removed before decoding |
| 10 | *no speech* — 8 s of near-silence; the system had been given a line of text as context |

Segments 1–5 are LibriSpeech sentences (references as distributed, upper case, no punctuation); 6–7 are the booking sentence from W1's Demo 2.

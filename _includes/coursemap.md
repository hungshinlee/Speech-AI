<!-- 此檔由 scripts/build_weeks.py 從課程大綱過濾產生，請勿直接編輯。大綱正本在 private repo（$COURSE_OUTLINE），改完請重跑腳本。 -->

## Course Map

<div class="coursemap">
<p class="cm-goal"><strong>Endpoint — a full-duplex spoken dialogue system:</strong> listen while speaking, survive barge-in, and answer within roughly 300 ms. The 14 weeks are built backwards from it; W1 sets out the systems view and the latency budget that the rest has to fit.</p>
<div class="cm-grid">
<div class="cm-col">
<div class="cm-part">Part I — Foundations</div>
<a class="cm-wk" href="weeks/w01.html"><span class="cm-n">W1</span><span class="cm-t">Systems view &amp; latency</span></a>
<div class="cm-wk cm-soon"><span class="cm-n">W2</span><span class="cm-t">Signals &amp; front-end</span></div>
<div class="cm-wk cm-soon"><span class="cm-n">W3</span><span class="cm-t">Alignment: HMM → CTC</span></div>
<div class="cm-wk cm-soon"><span class="cm-n">W4</span><span class="cm-t">Sequence models &amp; streaming</span></div>
<div class="cm-wk cm-soon"><span class="cm-n">W5</span><span class="cm-t">SSL representations</span></div>
<div class="cm-wk cm-hinge cm-soon"><span class="cm-n">W6</span><span class="cm-t">Codecs &amp; discretization</span></div>
</div>
<div class="cm-col">
<div class="cm-part">Part II — Modules</div>
<div class="cm-wk cm-soon"><span class="cm-n">W7</span><span class="cm-t">ASR</span></div>
<div class="cm-wk cm-soon"><span class="cm-n">W8</span><span class="cm-t">TTS I: generative</span></div>
<div class="cm-wk cm-soon"><span class="cm-n">W9</span><span class="cm-t">TTS II: control &amp; eval</span></div>
<div class="cm-wk cm-soon"><span class="cm-n">W10</span><span class="cm-t">VAD / AEC / sep / spk</span></div>
</div>
<div class="cm-col">
<div class="cm-part">Part III — Dialogue Systems</div>
<div class="cm-wk cm-soon"><span class="cm-n">W11</span><span class="cm-t">Audio-native LM</span></div>
<div class="cm-wk cm-soon"><span class="cm-n">W12</span><span class="cm-t">Turn-taking</span></div>
<div class="cm-wk cm-soon"><span class="cm-n">W13</span><span class="cm-t">Full-duplex arch. &amp; data</span></div>
<div class="cm-wk cm-soon"><span class="cm-n">W14</span><span class="cm-t">Evaluation &amp; deployment</span></div>
</div>
</div>
<p class="cm-note"><strong>W6 is the hinge.</strong> Everything in Part III rests on what is decided there — the <em>frame rate</em>, the <em>token budget</em>, and the <em>split between semantic and acoustic information</em> — so that is the week not to miss.</p>
</div>

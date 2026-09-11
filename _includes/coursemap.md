<!-- 此檔由 scripts/build_weeks.py 自動產生，請勿直接編輯；請改 docs/course-outline.md 後重跑腳本。 -->

## Course Map

<div class="coursemap">
<p class="cm-goal"><strong>Endpoint — a full-duplex spoken dialogue system:</strong> listen while speaking, survive barge-in, and answer within roughly 300 ms. The 14 weeks are built backwards from it; W1 sets out the systems view and the latency budget that the rest has to fit.</p>
<div class="cm-grid">
<div class="cm-col">
<div class="cm-part">Part I — Foundations</div>
<a class="cm-wk" href="weeks/w01.html"><span class="cm-n">W1</span><span class="cm-t">Systems view &amp; latency</span></a>
<a class="cm-wk" href="weeks/w02.html"><span class="cm-n">W2</span><span class="cm-t">Signals &amp; front-end</span></a>
<a class="cm-wk" href="weeks/w03.html"><span class="cm-n">W3</span><span class="cm-t">Alignment: HMM → CTC</span></a>
<a class="cm-wk" href="weeks/w04.html"><span class="cm-n">W4</span><span class="cm-t">Sequence models &amp; streaming</span></a>
<a class="cm-wk" href="weeks/w05.html"><span class="cm-n">W5</span><span class="cm-t">SSL representations</span></a>
<a class="cm-wk cm-hinge" href="weeks/w06.html"><span class="cm-n">W6</span><span class="cm-t">Codecs &amp; discretization</span></a>
</div>
<div class="cm-col">
<div class="cm-part">Part II — Modules</div>
<a class="cm-wk" href="weeks/w07.html"><span class="cm-n">W7</span><span class="cm-t">ASR</span></a>
<a class="cm-wk" href="weeks/w08.html"><span class="cm-n">W8</span><span class="cm-t">TTS I: generative</span></a>
<a class="cm-wk" href="weeks/w09.html"><span class="cm-n">W9</span><span class="cm-t">TTS II: control &amp; eval</span></a>
<a class="cm-wk" href="weeks/w10.html"><span class="cm-n">W10</span><span class="cm-t">VAD / AEC / sep / spk</span></a>
</div>
<div class="cm-col">
<div class="cm-part">Part III — Dialogue Systems</div>
<a class="cm-wk" href="weeks/w11.html"><span class="cm-n">W11</span><span class="cm-t">Audio-native LM</span></a>
<a class="cm-wk" href="weeks/w12.html"><span class="cm-n">W12</span><span class="cm-t">Turn-taking</span></a>
<a class="cm-wk" href="weeks/w13.html"><span class="cm-n">W13</span><span class="cm-t">Full-duplex arch. &amp; data</span></a>
<a class="cm-wk" href="weeks/w14.html"><span class="cm-n">W14</span><span class="cm-t">Evaluation &amp; deployment</span></a>
</div>
</div>
<p class="cm-note"><strong>W6 is the hinge.</strong> Everything in Part III rests on what is decided there — the <em>frame rate</em>, the <em>token budget</em>, and the <em>split between semantic and acoustic information</em> — so that is the week not to miss.</p>
</div>

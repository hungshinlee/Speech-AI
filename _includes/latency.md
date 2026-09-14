<!-- 此檔由 scripts/build_weeks.py 從課程大綱過濾產生，請勿直接編輯。大綱正本在 private repo（$COURSE_OUTLINE），改完請重跑腳本。 -->

## The Latency Budget: The Course's Shared Coordinate System

Hand this diagram out in week 1, then fill in each module's contribution as the course goes.

Human turn transitions are a distribution, not a single mean. Across ten languages on five continents, informal conversation follows one norm — *minimal gap, minimal overlap* — with transitions clustering near zero and cross-language means differing by at most about ±250 ms. Responses that do not answer the question, or that run against its bias, are delayed by up to a second, and listeners read that delay as meaning (Stivers et al., *PNAS* 106(26): 10587–10592, 2009). The engineering target for "natural" comes from that distribution, not from the single figure that usually circulates second-hand.

Each cell holds the time that elapses **after** the user stops speaking — not the module's total processing time. A streaming ASR has already consumed most of the utterance by then; counting that work again inflates the whole budget by an order of magnitude.

The template is cascade-shaped. A native full-duplex model has no endpoint-detection cell and no ASR tail: it emits on every frame. The budget stops fitting in W13, and that is the definition of full duplex rather than a flaw in the table.

```
user stops speaking
      ──▶ [endpoint detection] ──▶ [ASR tail]
      ──▶ [LLM prefill + first token]
      ──▶ [TTS first packet] ──▶ [playout buffer]
      ──▶ user hears
```

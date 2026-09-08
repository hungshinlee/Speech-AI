<!-- 此檔由 scripts/build_weeks.py 自動產生，請勿直接編輯；請改 docs/course-outline.md 後重跑腳本。 -->

## The Latency Budget: The Course's Shared Coordinate System

Hand this diagram out in week 1, then fill in each module's contribution as the course goes. The median response time at a turn transition in human conversation is on the order of 200 ms (`[主題]` search *turn-taking gap distribution, Levinson & Torreira, universals in turn-taking timing*); that is where the engineering target for "natural" comes from.

```
user stops speaking
      ──▶ [endpoint detection] ──▶ [ASR tail]
      ──▶ [LLM prefill + first token]
      ──▶ [TTS first packet] ──▶ [playout buffer]
      ──▶ user hears
```

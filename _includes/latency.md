<!-- 此檔由 scripts/build_weeks.py 自動產生，請勿直接編輯；請改 docs/course-outline.md 後重跑腳本。 -->

## 延遲預算：全課的共同座標系

第一週就把這張表發下去，之後每週回填該模組的貢獻。人類對話中 turn 轉換的中位反應時間約在 200 ms 量級（`[主題]` 檢索 *turn-taking gap distribution, Levinson & Torreira, universals in turn-taking timing*），這是「自然感」的工程目標來源。

```
使用者停止說話 ──▶ [endpoint 偵測] ──▶ [ASR 尾段] ──▶ [LLM prefill + first token]
                                                              ──▶ [TTS first packet] ──▶ [播放緩衝] ──▶ 使用者聽到
```

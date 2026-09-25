## llm baseline — `Qwen/Qwen3-4B-Instruct-2507` on `data_heldout` (fitted on `data` calib)

| decision | n | T | acc | macro-F1 | NLL raw → cal | Brier raw → cal | ECE raw → cal | ms/decision |
|---|---|---|---|---|---|---|---|---|
| grounded | 1000 | 12.47 | 0.787 | 0.787 | 3.630 → 0.466 | 0.410 → 0.301 | 0.207 → 0.073 | 63.1 |
| relevance | 4655 | 13.54 | 0.503 | 0.397 | 10.168 → 1.155 | 0.968 → 0.688 | 0.478 → 0.222 | 151.6 |
| sufficient | 1040 | 17.36 | 0.653 | 0.645 | 6.512 → 0.678 | 0.677 → 0.475 | 0.336 → 0.115 | 285.8 |

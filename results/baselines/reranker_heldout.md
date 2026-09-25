## reranker baseline — `BAAI/bge-reranker-v2-m3` on `data_heldout` (mapping fitted on `data` calib)

| decision | n | T | acc | macro-F1 | NLL raw → cal | Brier raw → cal | ECE raw → cal | ms/decision |
|---|---|---|---|---|---|---|---|---|
| grounded | 1000 | 1.00 | 0.737 | 0.737 | 0.545 → 0.545 | 0.361 → 0.361 | 0.025 → 0.025 | 5.0 |
| relevance | 4655 | 1.00 | 0.478 | 0.317 | 1.450 → 1.450 | 0.721 → 0.721 | 0.322 → 0.322 | 13.4 |
| sufficient | 1040 | 1.00 | 0.597 | 0.555 | 0.653 → 0.653 | 0.464 → 0.464 | 0.109 → 0.109 | 17.9 |

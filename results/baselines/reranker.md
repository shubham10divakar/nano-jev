## reranker baseline — `BAAI/bge-reranker-v2-m3`

| decision | n | T | acc | macro-F1 | NLL raw → cal | Brier raw → cal | ECE raw → cal | ms/decision |
|---|---|---|---|---|---|---|---|---|
| grounded | 500 | 1.00 | 0.798 | 0.798 | 0.449 → 0.449 | 0.288 → 0.288 | 0.028 → 0.028 | 3.5 |
| relevance | 1984 | 1.00 | 0.694 | 0.618 | 0.593 → 0.593 | 0.366 → 0.366 | 0.026 → 0.026 | 7.4 |
| sufficient | 1492 | 1.00 | 0.692 | 0.692 | 0.590 → 0.590 | 0.403 → 0.403 | 0.022 → 0.022 | 15.4 |

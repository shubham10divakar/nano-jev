# Cascade: `runs/nano-jev-v1.0` → `Qwen/Qwen3-4B-Instruct-2507`

## grounded on `data` (threshold chosen on calib: 0.8)

| accept Nano-Jev if confidence ≥ | escalated to LLM | accuracy | ms/decision |
|---|---|---|---|
| always (Nano-Jev only) | 0.0% | 0.844 | 1.0 |
| 0.60 | 7.0% | 0.862 | 4.5 |
| 0.70 | 13.6% | 0.862 | 7.8 |
| 0.80 ← chosen | 25.6% | 0.868 | 13.8 |
| 0.85 | 33.6% | 0.856 | 17.9 |
| 0.90 | 45.6% | 0.858 | 23.9 |
| 0.95 | 72.2% | 0.852 | 37.3 |
| 0.99 | 100.0% | 0.844 | 51.3 |
| never (LLM only) | 100.0% | 0.844 | 51.3 |

## grounded on `data_heldout` (threshold chosen on calib: 0.8)

| accept Nano-Jev if confidence ≥ | escalated to LLM | accuracy | ms/decision |
|---|---|---|---|
| always (Nano-Jev only) | 0.0% | 0.675 | 1.3 |
| 0.60 | 9.4% | 0.697 | 7.2 |
| 0.70 | 19.4% | 0.724 | 13.5 |
| 0.80 ← chosen | 30.1% | 0.745 | 20.3 |
| 0.85 | 39.0% | 0.754 | 26.0 |
| 0.90 | 53.0% | 0.788 | 34.8 |
| 0.95 | 72.4% | 0.794 | 47.1 |
| 0.99 | 100.0% | 0.787 | 64.6 |
| never (LLM only) | 100.0% | 0.787 | 64.6 |


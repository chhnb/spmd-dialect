# Cost Model Report

Train rows: 255
Validation rows: 333

| Model | MAPE | Exact Strategy Accuracy | Top-1 within 5% | Normalized Regret |
|---|---:|---:|---:|---:|
| selector_baseline | 53.92% | 100.00% | 50.00% | 17.96% |
| regressor_enhanced | 24.19% | 61.11% | 33.33% | 89.23% |

## Interpretation

- `selector_baseline` is the current best held-out strategy picker.
- `regressor_enhanced` cuts time-prediction MAPE substantially, but its strategy ranking still needs work.
- Next step is to inject L2-L5 profiling features from the representative NCU runs.

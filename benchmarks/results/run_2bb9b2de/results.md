# Model Comparison Report

## Summary

- Datasets evaluated: 6

### Model Wins

| Model | Wins |
|-------|------|
| AutoARIMA_multi | 2 |
| AutoTheta_multi | 2 |
| AutoETS_multi | 2 |

## Per-Dataset Results

### m4_yearly

Best model: **AutoARIMA_multi**

| Model | MAE |
|-------|------|
| AutoARIMA_multi | 662.1101 |
| AutoETS_multi | 675.3016 |
| AutoTheta_multi | 696.2509 |
| Naive | 725.7706 |
| ADIDA_multi | 941.1945 |
| IMAPA_multi | 941.1945 |
| CrostonClassic_multi | 1336.5052 |
| HistoricAverage_multi | 1375.2539 |
| SeasonalNaive_12 | 1646.2866 |

### m4_quarterly

Best model: **AutoTheta_multi**

| Model | MAE |
|-------|------|
| AutoTheta_multi | 611.5235 |
| Naive | 613.1002 |
| AutoETS_multi | 636.8785 |
| ADIDA_multi | 660.3799 |
| IMAPA_multi | 660.3799 |
| AutoARIMA_multi | 675.8952 |
| CrostonClassic_multi | 786.6835 |
| SeasonalNaive_12 | 859.0440 |
| HistoricAverage_multi | 903.0085 |

### m4_monthly

Best model: **AutoTheta_multi**

| Model | MAE |
|-------|------|
| AutoTheta_multi | 298.9783 |
| AutoETS_multi | 305.2616 |
| AutoARIMA_multi | 316.0925 |
| ADIDA_multi | 327.2521 |
| IMAPA_multi | 327.2521 |
| Naive | 337.5796 |
| CrostonClassic_multi | 382.2534 |
| SeasonalNaive_12 | 389.2697 |
| HistoricAverage_multi | 719.2594 |

### m4_weekly

Best model: **AutoETS_multi**

| Model | MAE |
|-------|------|
| AutoETS_multi | 268.9698 |
| AutoTheta_multi | 284.7382 |
| Naive | 288.6616 |
| AutoARIMA_multi | 309.5234 |
| ADIDA_multi | 325.2467 |
| IMAPA_multi | 325.2467 |
| CrostonClassic_multi | 400.6015 |
| SeasonalNaive_12 | 456.2076 |
| HistoricAverage_multi | 498.4009 |

### m4_daily

Best model: **AutoETS_multi**

| Model | MAE |
|-------|------|
| AutoETS_multi | 168.5725 |
| AutoTheta_multi | 168.9836 |
| Naive | 172.5825 |
| AutoARIMA_multi | 178.5926 |
| ADIDA_multi | 196.7238 |
| IMAPA_multi | 196.7238 |
| CrostonClassic_multi | 223.2128 |
| SeasonalNaive_12 | 230.7855 |
| HistoricAverage_multi | 371.7761 |

### m4_hourly

Best model: **AutoARIMA_multi**

| Model | MAE |
|-------|------|
| AutoARIMA_multi | 1574.2190 |
| HistoricAverage_multi | 1718.5166 |
| AutoTheta_multi | 1725.7703 |
| AutoETS_multi | 1732.0458 |
| Naive | 1737.9074 |
| CrostonClassic_multi | 1773.3189 |
| ADIDA_multi | 1959.4899 |
| IMAPA_multi | 1959.4899 |
| SeasonalNaive_12 | 2064.9482 |

# Model Comparison Report

## Summary

- Datasets evaluated: 6

### Model Wins

| Model | Wins |
|-------|------|
| AutoETS_multi | 3 |
| AutoTheta_multi | 1 |
| Naive | 1 |
| AutoARIMA_multi | 1 |

## Per-Dataset Results

### m4_yearly

Best model: **AutoTheta_multi**

| Model | MAE |
|-------|------|
| AutoTheta_multi | 625.6398 |
| AutoETS_multi | 628.8089 |
| AutoARIMA_multi | 703.3100 |
| Naive | 704.5344 |
| ADIDA_multi | 896.3301 |
| IMAPA_multi | 896.3301 |
| HistoricAverage_multi | 1256.0736 |
| CrostonClassic_multi | 1277.5437 |
| SeasonalNaive_12 | 1548.3706 |

### m4_quarterly

Best model: **AutoETS_multi**

| Model | MAE |
|-------|------|
| AutoETS_multi | 426.2086 |
| AutoTheta_multi | 434.9598 |
| AutoARIMA_multi | 444.3283 |
| Naive | 455.6248 |
| ADIDA_multi | 482.6108 |
| IMAPA_multi | 482.6108 |
| CrostonClassic_multi | 629.7582 |
| SeasonalNaive_12 | 692.3769 |
| HistoricAverage_multi | 700.1872 |

### m4_monthly

Best model: **AutoETS_multi**

| Model | MAE |
|-------|------|
| AutoETS_multi | 479.6843 |
| AutoARIMA_multi | 486.8295 |
| AutoTheta_multi | 492.6513 |
| ADIDA_multi | 516.1245 |
| IMAPA_multi | 516.1245 |
| Naive | 537.4511 |
| CrostonClassic_multi | 573.6945 |
| SeasonalNaive_12 | 587.8750 |
| HistoricAverage_multi | 868.5690 |

### m4_weekly

Best model: **AutoETS_multi**

| Model | MAE |
|-------|------|
| AutoETS_multi | 247.2046 |
| Naive | 249.0600 |
| AutoTheta_multi | 251.8259 |
| AutoARIMA_multi | 266.5749 |
| ADIDA_multi | 276.3355 |
| IMAPA_multi | 276.3355 |
| CrostonClassic_multi | 318.0720 |
| SeasonalNaive_12 | 366.0698 |
| HistoricAverage_multi | 408.2354 |

### m4_daily

Best model: **Naive**

| Model | MAE |
|-------|------|
| Naive | 109.5718 |
| AutoETS_multi | 110.7810 |
| AutoTheta_multi | 111.5107 |
| AutoARIMA_multi | 114.2458 |
| ADIDA_multi | 120.3991 |
| IMAPA_multi | 120.3991 |
| CrostonClassic_multi | 147.9806 |
| SeasonalNaive_12 | 156.2632 |
| HistoricAverage_multi | 287.2378 |

### m4_hourly

Best model: **AutoARIMA_multi**

| Model | MAE |
|-------|------|
| AutoARIMA_multi | 840.1836 |
| HistoricAverage_multi | 1186.1758 |
| AutoTheta_multi | 1214.2896 |
| AutoETS_multi | 1219.8057 |
| Naive | 1220.7415 |
| CrostonClassic_multi | 1245.4757 |
| ADIDA_multi | 1365.2138 |
| IMAPA_multi | 1365.2138 |
| SeasonalNaive_12 | 1451.4866 |

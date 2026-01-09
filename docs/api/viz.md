# Visualization Module

```{eval-rst}
.. currentmodule:: temporalcv.viz
```

The `temporalcv.viz` module provides Tufte-inspired visualizations for cross-validation, gate results, and prediction intervals. All visualizations follow Edward Tufte's principles for data graphics.

## Tufte Principles Applied

1. **Maximize data-ink ratio** — Remove chartjunk, use minimal spines
2. **Direct labeling** — Label data directly, minimize legends
3. **Muted colors** — Avoid saturated colors that distract
4. **Small multiples** — Enable comparison through consistent formatting

## When to Use

```{mermaid}
flowchart TD
    Start[Need Visualization?] --> Q1{What are you visualizing?}
    Q1 -->|CV folds| CVFoldsDisplay
    Q1 -->|Gate results| GateDisplay[GateResultDisplay or GateComparisonDisplay]
    Q1 -->|Prediction intervals| PredictionIntervalDisplay
    Q1 -->|Model metrics| MetricComparisonDisplay

    CVFoldsDisplay --> Usage1[Show train/test/gap structure]
    GateDisplay --> Usage2[Show HALT/WARN/PASS status]
    PredictionIntervalDisplay --> Usage3[Show coverage and width]
    MetricComparisonDisplay --> Usage4[Compare MAE, RMSE across models]
```

## API Patterns

The module provides two API styles:

### sklearn-style Display Classes

Best for complex visualizations and method chaining:

```python
from temporalcv.viz import CVFoldsDisplay

display = CVFoldsDisplay.from_cv(cv, X, y)
display.plot(tufte=True, title="Walk-Forward CV")
fig = display.figure_  # Access matplotlib figure
```

### statsmodels-style Functions

Best for quick one-liners:

```python
from temporalcv.viz import plot_cv_folds

ax = plot_cv_folds(cv, X, title="Walk-Forward CV")
plt.show()
```

---

## Display Classes

### CVFoldsDisplay

Visualize cross-validation fold structure with train/test/gap regions.

```{eval-rst}
.. autoclass:: temporalcv.viz.CVFoldsDisplay
   :members:
   :special-members: __init__
```

**Example:**

```python
from temporalcv import WalkForwardCV
from temporalcv.viz import CVFoldsDisplay

cv = WalkForwardCV(n_splits=5, test_size=20, extra_gap=5)
display = CVFoldsDisplay.from_cv(cv, X, y)
display.plot(title="Walk-Forward with Gap")
```

### GateResultDisplay

Visualize a single validation gate result (HALT/WARN/PASS).

```{eval-rst}
.. autoclass:: temporalcv.viz.GateResultDisplay
   :members:
   :special-members: __init__
```

**Example:**

```python
from temporalcv.gates import gate_signal_verification
from temporalcv.viz import GateResultDisplay

result = gate_signal_verification(model, X, y, n_shuffles=100)
display = GateResultDisplay.from_gate(result)
display.plot()
```

### GateComparisonDisplay

Compare multiple gate results side by side.

```{eval-rst}
.. autoclass:: temporalcv.viz.GateComparisonDisplay
   :members:
   :special-members: __init__
```

**Example:**

```python
from temporalcv.gates import run_gates
from temporalcv.viz import GateComparisonDisplay

report = run_gates([gate1, gate2, gate3])
display = GateComparisonDisplay.from_report(report)
display.plot(orientation="horizontal")
```

### PredictionIntervalDisplay

Visualize prediction intervals with coverage highlighting.

```{eval-rst}
.. autoclass:: temporalcv.viz.PredictionIntervalDisplay
   :members:
   :special-members: __init__
   :exclude-members: coverage_
```

**Example:**

```python
from temporalcv.conformal import SplitConformalPredictor
from temporalcv.viz import PredictionIntervalDisplay

conformal = SplitConformalPredictor(alpha=0.10)
conformal.calibrate(cal_preds, cal_actuals)
intervals = conformal.predict_interval(test_preds)

display = PredictionIntervalDisplay.from_conformal(intervals, test_actuals)
display.plot(show_coverage=True)
display.plot_width()  # Show interval width variation
```

### MetricComparisonDisplay

Compare metrics across multiple models.

```{eval-rst}
.. autoclass:: temporalcv.viz.MetricComparisonDisplay
   :members:
   :special-members: __init__
```

**Example:**

```python
from temporalcv.viz import MetricComparisonDisplay

results = {
    "Baseline": {"MAE": 0.20, "RMSE": 0.28},
    "Model A": {"MAE": 0.15, "RMSE": 0.22},
    "Model B": {"MAE": 0.12, "RMSE": 0.19},
}

display = MetricComparisonDisplay.from_dict(results, baseline="Baseline")
display.plot(show_best=True)  # Highlights best model per metric
display.plot_relative()  # Shows % improvement vs baseline
```

---

## Functions

### plot_cv_folds

```{eval-rst}
.. autofunction:: temporalcv.viz.plot_cv_folds
```

### plot_gate_result

```{eval-rst}
.. autofunction:: temporalcv.viz.plot_gate_result
```

### plot_gate_comparison

```{eval-rst}
.. autofunction:: temporalcv.viz.plot_gate_comparison
```

### plot_prediction_intervals

```{eval-rst}
.. autofunction:: temporalcv.viz.plot_prediction_intervals
```

### plot_interval_width

```{eval-rst}
.. autofunction:: temporalcv.viz.plot_interval_width
```

### plot_metric_comparison

```{eval-rst}
.. autofunction:: temporalcv.viz.plot_metric_comparison
```

---

## Styling Primitives

For custom visualizations, use the low-level styling functions:

### apply_tufte_style

Apply Tufte's principles to any matplotlib axes:

```python
from temporalcv.viz import apply_tufte_style

fig, ax = plt.subplots()
ax.plot(x, y)
apply_tufte_style(ax)  # Removes top/right spines, subtle colors
```

```{eval-rst}
.. autofunction:: temporalcv.viz.apply_tufte_style
```

### direct_label

Label data points directly (eliminates legends):

```python
from temporalcv.viz import direct_label

ax.plot(x, y)
direct_label(ax, x[-1], y[-1], "Series A", offset=(5, 0))
```

```{eval-rst}
.. autofunction:: temporalcv.viz.direct_label
```

### create_tufte_figure

Create a figure with Tufte styling pre-applied:

```python
from temporalcv.viz import create_tufte_figure

fig, axes = create_tufte_figure(nrows=2, ncols=2)
# All axes already have Tufte styling
```

```{eval-rst}
.. autofunction:: temporalcv.viz.create_tufte_figure
```

### Color Palettes

```python
from temporalcv.viz import TUFTE_PALETTE, COLORS

# TUFTE_PALETTE: primary, secondary, accent, success, warning, etc.
# COLORS: train, test, gap, pass, warn, halt, prediction, actual
```

---

## Common Patterns

### Subplot Layout

```python
from temporalcv.viz import (
    CVFoldsDisplay,
    GateComparisonDisplay,
    create_tufte_figure,
)

fig, (ax1, ax2) = create_tufte_figure(nrows=1, ncols=2, figsize=(12, 4))

CVFoldsDisplay.from_cv(cv, X).plot(ax=ax1, title="CV Structure")
GateComparisonDisplay.from_report(report).plot(ax=ax2, title="Gates")

plt.tight_layout()
```

### Saving Figures

```python
display = CVFoldsDisplay.from_cv(cv, X)
display.plot()

# High-quality PNG for reports
display.figure_.savefig("cv_folds.png", dpi=300, bbox_inches="tight")

# Vector format for publication
display.figure_.savefig("cv_folds.svg", format="svg", bbox_inches="tight")
```

### Disabling Tufte Style

If you prefer matplotlib defaults:

```python
display.plot(tufte=False)  # Use default matplotlib styling
```

---

## See Also

- [Example 00: Quickstart](../tutorials/examples_index.md) — Basic visualization usage
- [Example 05: Conformal Prediction](../tutorials/examples_index.md) — Interval visualization
- [Example 16: Failure Case - Rolling Stats](../tutorials/examples_index.md) — Gate visualization

## References

- Tufte, E. R. (1983). *The Visual Display of Quantitative Information*.
- Tufte, E. R. (2001). *Envisioning Information*.
- [scikit-learn Visualization API](https://scikit-learn.org/stable/visualizations.html)

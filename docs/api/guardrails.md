# Guardrails

Unified guardrails suite for temporal validation patterns.

## Overview

The guardrails module provides a convenience layer over validation gates, aggregating
multiple checks into a single validation pass.

## Core Classes

### `GuardrailResult`

Result from guardrail validation:

```python
@dataclass
class GuardrailResult:
    passed: bool              # All guardrails passed
    errors: List[str]         # List of error messages
    warnings: List[str]       # List of warning messages
    details: Dict[str, Any]   # Additional diagnostic info
```

## Functions

### `run_all_guardrails`

Run comprehensive validation in one call:

```python
from temporalcv.guardrails import run_all_guardrails

result = run_all_guardrails(
    model_metric=0.15,
    baseline_metric=0.20,
    n_samples=100,
)

if not result.passed:
    print(f"Guardrails failed: {result.errors}")
```

## Best Practices

1. **Run guardrails before deployment** - Catch common issues early
2. **Check both errors and warnings** - Warnings indicate edge cases
3. **Use gates for fine-grained control** - Guardrails are for convenience

## See Also

- [`gates`](gates.md): Individual validation gates
- [`run_gates`](gates.md#run_gates): Gate aggregation function

---
orphan: true
---

# [Component Name] Model Card

**Version**: 1.0.0
**Module**: `temporalcv.[module_name]`
**Type**: [Validation gate | Cross-validator | Statistical test | Metric | Predictor]
**License**: MIT
**Knowledge Tier**: [T1] Academic claims; [T2] Empirical findings; [T3] Heuristics/assumptions

---

## Component Details

[High-level description of what this component does - 2-3 sentences]

**Key Insight**: [The core principle or unique contribution - 1 sentence]

---

## Intended Use

### Primary Use Cases

- [Use case 1 - with context of when/why]
- [Use case 2]
- [Use case 3]
- [Use case 4 for publication-quality work, if applicable]

### Out-of-Scope Uses

- **[Anti-use case 1]**: [Why this is the wrong tool]
- **[Anti-use case 2]**: [Alternative approach]
- **[Anti-use case 3]**: [What to use instead]

### Target Users

- [User type 1]
- [User type 2]
- **Prerequisites**: [Required background knowledge]

---

## Parameters

| Parameter | Type | Default | Description | Tier |
|-----------|------|---------|-------------|------|
| `param1` | type | value | What it controls | [T1/T2/T3] |
| `param2` | type | value | What it controls | [T1/T2/T3] |
| `param3` | type | value | What it controls | [T1/T2/T3] |

### Parameter Selection Guide

[If applicable - guidance on how to choose parameter values]

| Parameter | Recommended Value | When to Adjust |
|-----------|------------------|----------------|
| `param1` | default | [Conditions] |
| `param2` | default | [Conditions] |

---

## Assumptions

| Assumption | Required For | Violation Consequence | Validation Method |
|------------|--------------|----------------------|-------------------|
| [Assumption 1] | [What breaks] | [What happens] | [How to check] |
| [Assumption 2] | [What breaks] | [What happens] | [How to check] |
| [Assumption 3] | [What breaks] | [What happens] | [How to check] |

---

## Performance Characteristics

### Time Complexity

- **O([complexity])** - [Dominant factor]
- [Additional complexity notes if relevant]

### Space Complexity

- **O([complexity])** - [Memory requirements]
- [Additional memory notes if relevant]

### Typical Runtimes

| Dataset Size | Configuration | Runtime |
|--------------|--------------|---------|
| [Small] | [Config] | [Time] |
| [Medium] | [Config] | [Time] |
| [Large] | [Config] | [Time] |

---

## Examples

### Basic Usage

```python
from temporalcv.[module] import [ComponentName]

# Minimal example
[code example]
```

### Advanced Usage

```python
# More complex example with explanations
[code example with comments]
```

### Common Pitfall

```python
# ❌ BAD - [Why this is wrong]
[anti-pattern code]

# ✅ GOOD - [Correct approach]
[correct code]
```

---

## Limitations

### Known Constraints

1. **[Limitation 1]**: [Description and impact]
   - **Workaround**: [If available]

2. **[Limitation 2]**: [Description and impact]
   - **Workaround**: [If available]

3. **[Limitation 3]**: [Description and impact]
   - **Workaround**: [If available]

### Common Misconfigurations

| Misconfiguration | Symptom | Fix |
|------------------|---------|-----|
| [Wrong setting] | [What happens] | [Correct setting] |
| [Wrong setting] | [What happens] | [Correct setting] |

---

## Statistical Properties

[If applicable - for statistical tests, metrics, or methods with distributional properties]

### Under Null Hypothesis

- [What the distribution/behavior looks like]
- [Expected values, p-value interpretation, etc.]

### Under Alternative Hypothesis

- [What changes]
- [Power considerations]

---

## References

### Academic Sources [T1]

1. **\[Primary Paper\]**: \[Author\] (\[Year\]). "\[Title\]." *\[Journal\]* \[Volume\](\[Issue\]): \[Pages\].
   - **DOI**: [doi]
   - **Key Contribution**: [What this paper establishes]

2. **[Secondary Paper]**: [Author] ([Year]). "[Title]." *[Conference/Journal]*.
   - **Key Contribution**: [What this adds]

### Empirical Sources [T2]

1. **[Source]**: [Description of empirical finding]
   - **Context**: [Where this was observed]

### Implementation Notes [T3]

1. **[Design Decision]**: [Why this choice was made]
   - **Alternatives Considered**: [Other options and why they were rejected]

---

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0.0 | YYYY-MM-DD | Initial release |

---

## See Also

- `[Related Component 1](link.md)` - \[How it relates\]
- `[Related Component 2](link.md)` - \[How it relates\]
- `[API Documentation](../api/module.md)` - \[Full API reference\]

---

**Contributing**: To improve this model card, please submit a PR or open an issue at [https://github.com/brandonmbehring-dev/temporalcv](https://github.com/brandonmbehring-dev/temporalcv).

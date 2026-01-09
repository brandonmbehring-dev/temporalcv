# README Audit Report: temporalcv

**Date**: 2026-01-09
**Auditor**: Claude (Opus 4.5)
**Scope**: README.md, public-facing documentation, visual assets

---

## Executive Summary

The temporalcv README has **structural strengths** but suffers from **tonal inconsistencies** and **underutilized visual assets** that diminish professional perception.

**Overall Grade: B-** (Functional but not compelling)

**Key Insight**: The most professional READMEs (scikit-learn, Prophet) are *understated*. They state what the library does, show one example, and get out of the way.

---

## Implementation Decisions

| Decision | Choice |
|----------|--------|
| Tone | Capability-positive (Prophet style) |
| Visual assets | Use existing PNGs + create simple text/SVG logo |
| Comparison table | Broader ecosystem (sklearn + sktime + darts) |
| Diagrams | Hybrid: Mermaid for ASCII art, keep PNGs for complex visuals |
| Logo | Simple text-styled SVG |
| Failure examples | Keep KFold Trap iconic, table + links for others |
| Quick Start | Minimal (gates only), CV details in Core Features |

---

## Benchmark Repositories Analyzed

| Repository | Stars | Key Strength |
|------------|-------|--------------|
| **scikit-learn** | 62k | Authoritative minimalism, community trust |
| **Darts** | 8.5k | Visual output examples, model comparison tables |
| **sktime** | 9.4k | Badge grid, progressive disclosure, dual examples |
| **Prophet** | 19.9k | Peer-reviewed paper prominence, problem clarity |
| **StatsForecast** | 4k | Quantified speed claims, sklearn-like API showcase |
| **statsmodels** | 11k | Comprehensive feature taxonomy, multi-channel install |

---

## Current State Assessment

### Strengths

1. Visual assets exist: `kfold_trap.png`, `temporal_cv_gap.png`, `gates_halt.png`
2. Core value proposition clear: Leakage detection + temporal CV
3. Badges present: CI, PyPI, Docs, Coverage, Colab
4. Quick start included: Installation + minimal example

### Weaknesses

| Issue | Severity | Evidence |
|-------|----------|----------|
| Narrative tone | High | "Standard CV methods are fundamentally broken" — hyperbolic |
| Missing logo | Medium | No branded visual identity at top |
| No feature tables | Medium | Features listed as prose, not scannable matrix |
| ASCII diagrams | Medium | Dated appearance, poor mobile rendering |

---

## Gap Analysis vs. Best Practices

### Visual Hierarchy

| Element | scikit-learn | Darts | sktime | temporalcv |
|---------|--------------|-------|--------|------------|
| Logo | ✓ | ✓ | ✓ | ✗ |
| Tagline | ✓ | ✓ | ✓ | ✓ |
| Badge row | ✓ | ✓ | ✓ | ✓ |
| Feature table | ✗ | ✓ | ✓ | ✗ |

### Tone Comparison

**Current (adversarial)**:
```
"Standard cross-validation methods are fundamentally broken"
```

**Best Practice (capability-positive)**:
```
"Prophet works best with time series that have strong seasonal effects"
```

---

## Proposed New Structure

```
1. Logo + Tagline
2. Badge row
3. Quick Start (install + 5-line gates example)
4. Visual: KFold Trap diagram
5. Feature comparison table
6. Core capabilities (WalkForwardCV, metrics)
7. Failure patterns table (links to examples)
8. Documentation links
9. Citation
10. License
```

---

## Consolidated Insights from Prior Audits

| Source | Key Insight |
|--------|-------------|
| gemini-audit-report.md | ASCII art diagrams look dated, render poorly on mobile |
| gemini-audit-report.md | Phrases like "Sound familiar?" unprofessional |
| gemini-audit-report.md | "Buried Quickstart" - users scroll past theory |
| docs/audits/ | README cognitive overload - architecture before code |

---

## Action Items

1. **Create text-styled SVG logo**
2. **Rewrite opening** with capability-positive tone
3. **Add feature comparison table** (vs sklearn, sktime, darts)
4. **Replace ASCII pipeline** with Mermaid diagram
5. **Keep KFold Trap** as iconic visual, condense other failures
6. **Verify all images render** on GitHub

---

## Conclusion

Less narrative, more tables. Less "what's wrong with others," more "what we do well."

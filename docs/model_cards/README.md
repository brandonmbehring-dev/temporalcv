# Model Cards

Model cards for temporalcv validation components, following the [Mitchell et al. (2019)](https://arxiv.org/abs/1810.03993) framework adapted for ML infrastructure.

---

## Overview

Model cards provide transparent documentation of component behavior, assumptions, and limitations. Each card includes:

- **Component Details**: Version, module, license, knowledge tier
- **Intended Use**: Primary use cases and out-of-scope applications
- **Parameters**: Complete parameter reference with knowledge tier tags
- **Assumptions**: What must hold for correct behavior
- **Limitations**: Known constraints and common misconfigurations
- **References**: Academic sources tagged by confidence tier

---

## Available Model Cards

| Component | Type | Card |
|-----------|------|------|
| **WalkForwardCV** | Cross-validator | [walk_forward_cv.md](walk_forward_cv.md) |
| **gate_signal_verification** | Validation gate | [gate_signal_verification.md](gate_signal_verification.md) |

---

## Knowledge Tier System

All claims are tagged with confidence levels:

| Tier | Meaning | Example |
|------|---------|---------|
| **[T1]** | Academically validated | DM test (Diebold & Mariano 1995) |
| **[T2]** | Empirical finding | Block permutation default |
| **[T3]** | Assumption/heuristic | α=0.05 significance level |

---

## Using Model Cards

### Before Deployment

1. Read the **Intended Use** section to verify your use case is supported
2. Check **Assumptions** against your data and pipeline
3. Review **Limitations** for known constraints

### During Development

1. Use **Parameters** table as a quick reference
2. Check **Common Misconfigurations** to avoid pitfalls
3. Follow **Examples** for correct usage patterns

### For Documentation

1. Cite **References** in publications
2. Use **Knowledge Tier** tags for claim confidence
3. Note **Limitations** in methodology sections

---

## Contributing

To add a new model card:

1. Copy the template structure from existing cards
2. Fill in all sections with accurate information
3. Tag all claims with appropriate knowledge tiers
4. Add academic references for [T1] claims
5. Document empirical sources for [T2] claims
6. Justify [T3] heuristics explicitly

---

## Reference

Mitchell, M., Wu, S., Zaldivar, A., Barnes, P., Vasserman, L., Hutchinson, B., Spitzer, E., Raji, I.D., & Gebru, T. (2019). Model Cards for Model Reporting. *Proceedings of the Conference on Fairness, Accountability, and Transparency*, 220-229.

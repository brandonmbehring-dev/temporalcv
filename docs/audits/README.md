# Audit Reports

This directory contains external audits of the temporalcv library.

---

## Canonical Status

| Report | Date | Auditor | Status | Verdict |
|--------|------|---------|--------|---------|
| `gemini-audit-report.md` | 2026-01-05 | Gemini CLI | **Superseded** | Critical concerns raised |
| `../../../gemini_audit_report.md` | 2026-01-08 | Gemini | **Current** | Release Candidate Ready |

**Active Audit**: The January 8, 2026 audit is the canonical version.

---

## Audit History

### January 5, 2026 — Initial Audit

The first Gemini audit identified a potential methodological concern with `gate_signal_verification`:

> **Concern**: "If your model beats a shuffled target, features contain information about the target. This information could be leakage (bad) OR valid predictive signal (good)."

### Resolution

This concern was addressed by clarifying the gate's semantics in the docstring:

```python
# Current interpretation (gates.py:304-313):
# - HALT: Model has signal → investigate source (leakage vs legitimate)
# - PASS: Model has no signal → concerning (learned nothing)
#
# "HALT is expected and confirms the gate is working correctly."
```

The gate's purpose is **signal verification**, not leakage detection. A HALT result means "you have signal—now verify it's legitimate." A PASS result is actually the concerning outcome (model learned nothing).

### January 8, 2026 — Follow-Up Audit

After the semantic clarification, the follow-up audit concluded:

> **Verdict**: "Release Candidate Ready. The code quality, type safety, and documentation are superior to many existing PyPI packages."

---

## Other Audits

| Report | Date | Focus |
|--------|------|-------|
| `codex-audit-2025-12-23.md` | 2025-12-23 | Initial Codex audit |
| `codex-audit-2025-12-23-critique.md` | 2025-12-23 | Self-critique |
| `codex-audit-2026-01-05.md` | 2026-01-05 | Pre-release Codex audit |
| `codex-audit-claude-sonnet-2026-01-05.md` | 2026-01-05 | Claude Sonnet review |

---

## How to Audit

If you're conducting your own audit:

1. **Run the test suite**: `pytest tests/ -v --cov=temporalcv`
2. **Check golden references**: `pytest tests/test_golden_reference.py -v`
3. **Run Monte Carlo tests**: `pytest tests/monte_carlo/ -v --run-slow`
4. **Review validation evidence**: See `docs/validation_evidence.md`
5. **Check for `[T3]` tags**: These indicate heuristic (non-paper-validated) components

---

## Reporting Issues

If you find issues during an audit:

1. Check if already documented in [Known Caveats](../validation_evidence.md#8-known-caveats)
2. Open a GitHub issue with [Audit] prefix
3. Use the security advisory process for sensitive issues (see SECURITY.md)

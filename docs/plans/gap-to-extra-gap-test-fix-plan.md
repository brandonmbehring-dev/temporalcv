# Implementation Plan: Fix 83 Test Failures from gap→extra_gap Rename

**Status**: Planning Complete
**Created**: 2026-01-05
**Estimated Scope**: 83 test failures across 11 test files + 3 source file bugs

---

## Executive Summary

The `gap` → `extra_gap` parameter rename was applied to source code and documentation but **not to test files**. This causes 83 test failures (5.5% of suite).

### Root Causes Identified

1. **Source Code Bugs** (3 issues in `src/temporalcv/cv.py`):
   - Line 1509: References `self.verbose` before assignment (should be `verbose`)
   - Line 1717: Uses `gap=self.gap` (should be `extra_gap=self.extra_gap`)
   - Line 1928: Uses `gap={self.gap}` in `__repr__` (should be `extra_gap={self.extra_gap}`)

2. **Test File Parameter Changes** (121 occurrences across 11 files):
   - Constructor calls: `WalkForwardCV(gap=N)` → `WalkForwardCV(extra_gap=N)`
   - Function calls: `gate_temporal_boundary(gap=N)` → `gate_temporal_boundary(extra_gap=N)`
   - `walk_forward_evaluate(gap=N)` → `walk_forward_evaluate(extra_gap=N)`

3. **Test Attribute Assertions** (12 occurrences):
   - `assert cv.gap == N` → `assert cv.extra_gap == N`
   - `assert nested_cv.gap == N` → `assert nested_cv.extra_gap == N`
   - But keep: `assert info.gap == N` (SplitInfo.gap is a **computed property**, not renamed)

4. **Special Cases**:
   - `PurgedWalkForward(gap=N)` in cv_financial.py uses separate `gap` param (NOT renamed)
   - SplitInfo/SplitResult `.gap` property (computed field, unchanged)
   - Docstring examples need update in cv.py (3 locations)

---

## Semantic Changes to Understand

### Old Semantics (gap parameter)
```python
WalkForwardCV(horizon=5, gap=5)  # gap >= horizon required
# total_separation = gap (user specified both)
```

### New Semantics (extra_gap parameter)
```python
WalkForwardCV(horizon=5, extra_gap=0)  # minimum safe
WalkForwardCV(horizon=5, extra_gap=5)  # extra safety margin
# total_separation = horizon + extra_gap
```

**Key Insight**: The new semantics make `horizon` the primary parameter and `extra_gap` is the **additional** separation beyond the minimum requirement.

---

## Implementation Strategy

### Phase 1: Fix Source Code Bugs (MUST DO FIRST)
**Priority**: CRITICAL - Tests cannot pass until these are fixed

**Files**: `src/temporalcv/cv.py` (3 changes)

1. **Line 1509** - Variable reference bug:
   ```python
   # BEFORE:
   if self.verbose >= 1:  # ERROR: self.verbose doesn't exist yet

   # AFTER:
   if verbose >= 1:  # Use parameter, not attribute
   ```

2. **Line 1717** - NestedWalkForwardCV.fit() outer CV initialization:
   ```python
   # BEFORE:
   outer_cv = WalkForwardCV(
       n_splits=self.n_outer_splits,
       horizon=self.horizon,
       gap=self.gap,  # ERROR: AttributeError
       window_type=self.window_type,
       window_size=self.window_size,
       test_size=1,
   )

   # AFTER:
   outer_cv = WalkForwardCV(
       n_splits=self.n_outer_splits,
       horizon=self.horizon,
       extra_gap=self.extra_gap,  # FIXED
       window_type=self.window_type,
       window_size=self.window_size,
       test_size=1,
   )
   ```

3. **Line 1928** - NestedWalkForwardCV.__repr__():
   ```python
   # BEFORE:
   return (
       f"NestedWalkForwardCV(search={search_type!r}, "
       f"n_outer={self.n_outer_splits}, n_inner={self.n_inner_splits}, "
       f"horizon={self.horizon}, gap={self.gap})"  # ERROR
   )

   # AFTER:
   return (
       f"NestedWalkForwardCV(search={search_type!r}, "
       f"n_outer={self.n_outer_splits}, n_inner={self.n_inner_splits}, "
       f"horizon={self.horizon}, extra_gap={self.extra_gap})"  # FIXED
   )
   ```

4. **Docstring examples** (3 locations in cv.py):
   - Line 22: `cv = WalkForwardCV(n_splits=5, gap=2, ...)` → `extra_gap=2`
   - Line 540: `cv = WalkForwardCV(n_splits=5, gap=2)` → `extra_gap=2`
   - Line 815: `gap={info.gap}` - KEEP (computed property)

**Verification**: Run `pytest tests/test_cv.py::TestNestedWalkForwardCV::test_basic_nested_cv -v`

---

### Phase 2: Fix Test Files (11 files, 121 parameter changes, 12 attribute changes)

**Approach**: File-by-file systematic replacement with semantic verification.

#### File 1: `tests/test_cv.py` (34 parameter + 12 attribute changes)

**Parameter changes (34 total)**:
```text
# Pattern 1: WalkForwardCV constructor (24 occurrences)
WalkForwardCV(n_splits=N, gap=X)              → WalkForwardCV(n_splits=N, extra_gap=X)
WalkForwardCV(n_splits=N, gap=X, horizon=H)   → WalkForwardCV(n_splits=N, extra_gap=X, horizon=H)

# Pattern 2: walk_forward_evaluate (2 occurrences)
walk_forward_evaluate(..., gap=X)             → walk_forward_evaluate(..., extra_gap=X)

# Pattern 3: NestedWalkForwardCV (2 occurrences)
NestedWalkForwardCV(..., gap=X)               → NestedWalkForwardCV(..., extra_gap=X)
```

**Attribute changes (12 total)**:
```text
# Pattern A: Direct attribute access (10 occurrences)
assert cv.gap == N                            → assert cv.extra_gap == N
assert nested_cv.gap == 4                     → assert nested_cv.extra_gap == 4

# Pattern B: Computed property (UNCHANGED - 2 occurrences)
assert info.gap == 2                          → KEEP (info is SplitInfo, .gap is computed)
assert split.gap >= 2                         → KEEP (split is SplitResult, .gap is computed)
```

**Specific Line Changes**:
- L172: `gap=2` → `extra_gap=2`
- L183: `gap=2` → `extra_gap=2`
- L187: `"gap=2"` in repr assertion → `"extra_gap=2"`
- L163: `assert cv.gap == 0` → `assert cv.extra_gap == 0`
- L178: `assert cv.gap == 2` → `assert cv.extra_gap == 2`
- L255-295: 8 constructor calls with `gap=`
- L413-630: 10 constructor calls + 6 attribute assertions
- L862, L878, L902, L919: walk_forward_evaluate calls
- L1059: `assert nested_cv.gap == 4` → `assert nested_cv.extra_gap == 4`
- L1070: `gap=2` in constructor → `extra_gap=2`

**Lines with .gap to KEEP** (computed properties):
- L107: `assert info.gap == 2` - KEEP (SplitInfo.gap property)
- L262: `assert info.gap >= 2` - KEEP
- L691: `assert sr.gap == 2` - KEEP (SplitResult.gap property)
- L924: `assert split.gap >= 2` - KEEP

**Horizon+Gap Semantics Tests** (require semantic update):
- L599-614: Tests for horizon validation
  - OLD: `WalkForwardCV(horizon=3, gap=3)` meant gap >= horizon ✓
  - NEW: `WalkForwardCV(horizon=3, extra_gap=0)` means minimum safe separation
  - Action: Change `gap=X` → `extra_gap=X` but keep test logic same (tests now verify `horizon + extra_gap` separation)

#### File 2: `tests/test_gates.py` (4 changes)

**All changes**: `gate_temporal_boundary()` calls

```text
# Lines 646, 657, 669, 680
gate_temporal_boundary(train_end_idx=X, test_start_idx=Y, horizon=H, gap=G)
→
gate_temporal_boundary(train_end_idx=X, test_start_idx=Y, horizon=H, extra_gap=G)
```

**Semantic check**: These tests verify `actual_gap >= horizon + extra_gap`, which matches new semantics.

#### File 3: `tests/test_edge_cases.py` (4 changes)

**Pattern**: WalkForwardCV edge case tests

```text
# Lines 227, 285, 295, 305
WalkForwardCV(..., gap=X)  →  WalkForwardCV(..., extra_gap=X)
```

**Semantic considerations**:
- L227: `gap=5` → `extra_gap=5` (large separation test)
- L285: `gap=2, horizon=1` → `extra_gap=2, horizon=1` (total_sep = 1+2 = 3)
- L295: `gap=0` → `extra_gap=0` (minimum separation)
- L305: `gap=0` → `extra_gap=0`

#### File 4: `tests/test_cv_financial.py` (2 changes)

**CRITICAL**: `PurgedWalkForward` has separate `gap` parameter (NOT `extra_gap`)!

```python
# Line 290-291: PurgedWalkForward uses BOTH purge_gap AND gap
cv_no_gap = PurgedWalkForward(n_splits=3, test_size=20, gap=0, purge_gap=0)
cv_with_gap = PurgedWalkForward(n_splits=3, test_size=20, gap=10, purge_gap=0)
```

**Check**: Does `PurgedWalkForward` inherit from WalkForwardCV?

```bash
grep -A5 "class PurgedWalkForward" src/temporalcv/cv_financial.py
```

If YES → `gap` → `extra_gap`
If NO → KEEP `gap` as-is

**Action**: Investigate inheritance first, then update.

#### File 5: `tests/test_integration.py` (3 changes)

**Pattern**: Integration tests using WalkForwardCV

```text
WalkForwardCV(..., gap=X)  →  WalkForwardCV(..., extra_gap=X)
```

#### File 6: `tests/test_cross_fit.py` (changes TBD)

**Pattern**: CrossFitCV tests - verify if CrossFitCV uses `gap` or `extra_gap`

#### File 7-11: Remaining test files

- `tests/anti_patterns/test_boundary_violations.py`
- `tests/property/test_cv_invariants.py`
- `tests/benchmarks/test_gate_benchmarks.py`
- `tests/benchmarks/test_cv_benchmarks.py`
- `tests/integration/test_full_workflow.py`

**Strategy**: Grep for exact patterns, replace systematically.

---

### Phase 3: Verification Strategy

**Goal**: Ensure all 83 failures → 0 failures without breaking 1,644 passing tests.

#### Step 1: Unit Test Verification (per file)

After each file fix:
```bash
pytest tests/test_cv.py -v  # After fixing test_cv.py
pytest tests/test_gates.py -v  # After fixing test_gates.py
# etc.
```

**Success Criteria**: File-specific failures → 0, no new failures.

#### Step 2: Incremental Regression Check

After every 3 files:
```bash
pytest tests/ -x  # Stop on first failure
```

**Success Criteria**: No regressions in previously passing tests.

#### Step 3: Full Suite Validation

After all fixes:
```bash
pytest tests/ -v --tb=short
```

**Success Criteria**:
- Total collected: 1,741 tests
- Passed: 1,741 (100%)
- Failed: 0

#### Step 4: Gap Semantic Validation

**Purpose**: Ensure new semantics `total_separation = horizon + extra_gap` is enforced.

**Test Cases to Verify**:
1. `WalkForwardCV(horizon=3, extra_gap=0)` → min 3-step separation
2. `WalkForwardCV(horizon=3, extra_gap=2)` → min 5-step separation
3. Edge case: `WalkForwardCV(horizon=1, extra_gap=0)` → 1-step separation
4. NestedWalkForwardCV: Inner and outer CV both respect formula

**Verification Script**:
```python
cv = WalkForwardCV(n_splits=3, horizon=3, extra_gap=2, test_size=1)
X = np.arange(200).reshape(-1, 1)
for train, test in cv.split(X):
    actual_gap = test[0] - train[-1] - 1
    assert actual_gap >= (3 + 2), f"Gap {actual_gap} < required {3+2}"
```

#### Step 5: Edge Cases to Test

1. **Computed properties unchanged**:
   ```python
   info = SplitInfo(train_start=0, train_end=99, test_start=102, test_end=105)
   assert info.gap == 2  # (102 - 99 - 1) - should still work
   ```

2. **Repr strings updated**:
   ```python
   cv = WalkForwardCV(extra_gap=5)
   assert "extra_gap=5" in repr(cv)
   assert "gap=" not in repr(cv)  # Old param name gone
   ```

3. **Backward compatibility**: No old `gap` param accepted
   ```python
   with pytest.raises(TypeError, match="unexpected keyword argument 'gap'"):
       WalkForwardCV(gap=5)
   ```

---

## Detailed Change Breakdown

### Source Code Changes (3 files)

| File | Lines | Type | Change |
|------|-------|------|--------|
| src/temporalcv/cv.py | 22, 540 | docstring | `gap=2` → `extra_gap=2` |
| src/temporalcv/cv.py | 1509 | bugfix | `self.verbose` → `verbose` |
| src/temporalcv/cv.py | 1717 | bugfix | `gap=self.gap` → `extra_gap=self.extra_gap` |
| src/temporalcv/cv.py | 1928 | bugfix | `gap={self.gap}` → `extra_gap={self.extra_gap}` |

### Test File Changes Summary

| File | Constructor Calls | Attribute Asserts | Function Calls | Total |
|------|-------------------|-------------------|----------------|-------|
| tests/test_cv.py | 24 | 10 | 2 | 36 |
| tests/test_gates.py | 0 | 0 | 4 | 4 |
| tests/test_edge_cases.py | 4 | 0 | 0 | 4 |
| tests/test_cv_financial.py | 2 | 0 | 0 | 2 |
| tests/test_integration.py | ~3 | ~0 | ~0 | ~3 |
| tests/test_cross_fit.py | TBD | TBD | TBD | TBD |
| tests/anti_patterns/test_boundary_violations.py | TBD | TBD | TBD | TBD |
| tests/property/test_cv_invariants.py | TBD | TBD | TBD | TBD |
| tests/benchmarks/* | TBD | TBD | TBD | TBD |
| **TOTAL** | ~50 | ~10 | ~6 | **~66** |

**Note**: "TBD" files require detailed grep to count exact occurrences.

---

## Risk Assessment

### High Risk Areas

1. **Semantic Confusion**:
   - Risk: Tests might encode old `gap >= horizon` logic
   - Mitigation: Review each horizon+gap test for semantic correctness

2. **PurgedWalkForward edge case**:
   - Risk: cv_financial.py may use different `gap` semantics
   - Mitigation: Check class inheritance and documentation

3. **Computed vs. Stored Attributes**:
   - Risk: Accidentally change `.gap` property on SplitInfo/SplitResult
   - Mitigation: Grep for `\.gap` and verify it's a computed property before changing

### Medium Risk Areas

1. **Docstring examples**: May have been updated inconsistently
2. **Error messages**: May still reference old `gap` parameter name
3. **Benchmark tests**: May have hardcoded expectations

### Low Risk Areas

1. Property tests: Should adapt automatically if core logic correct
2. Integration tests: High-level, less parameter-specific

---

## Execution Checklist

### Pre-Flight Checks
- [ ] Backup current test results: `pytest tests/ -v > before_fix.txt`
- [ ] Verify current failure count: 83 failures expected
- [ ] Create git branch: `git checkout -b fix/gap-to-extra-gap-test-suite`

### Phase 1: Source Code Fixes
- [ ] Fix cv.py line 1509 (self.verbose → verbose)
- [ ] Fix cv.py line 1717 (gap=self.gap → extra_gap=self.extra_gap)
- [ ] Fix cv.py line 1928 (__repr__ gap → extra_gap)
- [ ] Fix cv.py docstrings (lines 22, 540)
- [ ] Run: `pytest tests/test_cv.py::TestNestedWalkForwardCV::test_basic_nested_cv -v`
- [ ] Verify: Test passes (no AttributeError)

### Phase 2A: Core Test Files
- [ ] Fix tests/test_cv.py (36 changes)
  - [ ] 24 constructor calls
  - [ ] 10 attribute assertions
  - [ ] 2 function calls
  - [ ] Verify: `pytest tests/test_cv.py -v` (all pass)
- [ ] Fix tests/test_gates.py (4 changes)
  - [ ] Verify: `pytest tests/test_gates.py -v` (all pass)
- [ ] Fix tests/test_edge_cases.py (4 changes)
  - [ ] Verify: `pytest tests/test_edge_cases.py -v` (all pass)

### Phase 2B: Special Case Files
- [ ] Investigate PurgedWalkForward inheritance
- [ ] Fix tests/test_cv_financial.py (2 changes, conditional)
  - [ ] Verify: `pytest tests/test_cv_financial.py -v`
- [ ] Fix tests/test_integration.py (~3 changes)
  - [ ] Verify: `pytest tests/test_integration.py -v`

### Phase 2C: Remaining Test Files
- [ ] Fix tests/test_cross_fit.py
- [ ] Fix tests/anti_patterns/test_boundary_violations.py
- [ ] Fix tests/property/test_cv_invariants.py
- [ ] Fix tests/benchmarks/test_gate_benchmarks.py
- [ ] Fix tests/benchmarks/test_cv_benchmarks.py
- [ ] Fix tests/integration/test_full_workflow.py

### Phase 3: Full Validation
- [ ] Run full test suite: `pytest tests/ -v`
- [ ] Verify: 1,741 passed, 0 failed
- [ ] Run semantic validation script (gap formula check)
- [ ] Run edge case tests (computed properties, repr, backward compat)
- [ ] Compare before/after: `diff before_fix.txt after_fix.txt`

### Post-Flight
- [ ] Review git diff: `git diff`
- [ ] Verify no unintended changes (e.g., .gap property removals)
- [ ] Update CHANGELOG.md entry (if exists)
- [ ] Commit: `git commit -m "fix: Complete gap→extra_gap test suite migration"`

---

## Critical Files for Implementation

These 5 files are the most critical to understand and modify correctly:

1. **src/temporalcv/cv.py** (lines 1459-1930)
   - Contains NestedWalkForwardCV class with 3 bugs blocking all tests
   - Has docstring examples that need updating
   - Most complex source code changes

2. **tests/test_cv.py** (entire file, 1200+ lines)
   - 36 changes needed (30% of total test failures)
   - Contains both parameter AND attribute changes
   - Has horizon+gap semantic tests requiring careful review
   - Pattern to follow for other test files

3. **tests/test_gates.py** (lines 646-680)
   - Tests gate_temporal_boundary() with gap parameter
   - Simpler pattern: only function parameter changes
   - Good reference for semantic correctness of gap formula

4. **src/temporalcv/cv_financial.py** (check inheritance)
   - Need to verify if PurgedWalkForward gap param is separate or inherited
   - Determines whether tests/test_cv_financial.py needs changes

5. **tests/test_edge_cases.py** (lines 227-305)
   - Edge cases for gap=0, large gap, etc.
   - Validates boundary conditions of new formula
   - Critical for ensuring no regressions in edge cases

---

## Success Criteria

1. **All tests pass**: 1,741/1,741 (100%)
2. **No regressions**: Previously passing tests still pass
3. **Semantic correctness**: `total_separation = horizon + extra_gap` enforced
4. **No backwards compatibility**: Old `gap` parameter rejected with clear error
5. **Consistent naming**: All references to `gap` updated (except computed properties)
6. **Documentation aligned**: Docstrings and examples use `extra_gap`

---

## Notes

- **Computed properties are UNCHANGED**: `SplitInfo.gap`, `SplitResult.gap` remain as-is
- **PurgedWalkForward**: Requires investigation - may have separate `gap` parameter
- **Semantic shift**: Old `gap` was total separation; new `extra_gap` is additional margin
- **Test philosophy**: Fix parameters first, verify semantics second

---

## Estimated Timeline

- Phase 1 (Source bugs): 30 minutes
- Phase 2A (Core tests): 1 hour
- Phase 2B (Special cases): 30 minutes
- Phase 2C (Remaining tests): 1 hour
- Phase 3 (Validation): 30 minutes
- **Total**: 3.5 hours

**Complexity**: Medium (systematic but requires semantic understanding)
**Risk Level**: Low-Medium (well-scoped, automated verification available)

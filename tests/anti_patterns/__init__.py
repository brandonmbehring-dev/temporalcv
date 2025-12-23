"""
Anti-pattern tests: Verify the library catches common ML leakage bugs.

These tests verify that temporalcv correctly identifies and halts on:
- Lag leakage (future information in features)
- Boundary violations (train/test overlap)
- Other temporal leakage patterns

Category mapping from lever_of_archimedes/patterns/data_leakage_prevention.md:
- Bug #1 (Shuffling): test_lag_leakage.py
- Bug #2 (Boundary violations): test_boundary_violations.py
- Bug #3-10: Covered by gate framework tests
"""

"""
    Stationarity

Unit Root and Stationarity Testing Module.

Provides implementations of classic stationarity tests:
- ADF (Augmented Dickey-Fuller): H0 = unit root (non-stationary)
- KPSS: H0 = stationary (opposite null hypothesis from ADF)

Joint interpretation logic for robust stationarity inference.

# Knowledge Tiers
- [T1] ADF test (Dickey & Fuller 1979)
- [T1] KPSS test (Kwiatkowski et al. 1992)
- [T2] Joint ADF+KPSS interpretation for robust inference

# Example
```julia
using TemporalValidation.Stationarity

# Single test
result = adf_test(series)
result.is_stationary

# Joint analysis (recommended)
joint = check_stationarity(series)
joint.conclusion  # STATIONARY, NON_STATIONARY, etc.
joint.recommended_action

# Auto-differencing
diff_series, d = difference_until_stationary(series)
```

# References
- Dickey & Fuller (1979). "Distribution of the Estimators for Autoregressive
  Time Series with a Unit Root." JASA 74(366), 427-431.
- Kwiatkowski et al. (1992). "Testing the null hypothesis of stationarity
  against the alternative of a unit root." J. Econometrics 54, 159-178.
"""
module Stationarity

using Statistics
using LinearAlgebra

# =============================================================================
# Types
# =============================================================================

include("types.jl")

# =============================================================================
# Tests
# =============================================================================

include("adf.jl")
include("kpss.jl")
include("joint.jl")

# =============================================================================
# Exports
# =============================================================================

# Enums and types
export StationarityConclusion, STATIONARY, NON_STATIONARY
export DIFFERENCE_STATIONARY, INSUFFICIENT_EVIDENCE
export StationarityTestResult, JointStationarityResult

# Test functions
export adf_test, kpss_test
export check_stationarity
export difference_until_stationary, integration_order

end # module Stationarity

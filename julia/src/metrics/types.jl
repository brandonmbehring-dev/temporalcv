# =============================================================================
# Metrics Type Definitions
# =============================================================================

"""
Type definitions for the Metrics module.

Provides result structs for various forecast evaluation metrics:
- MoveConditionalResult: MC-SS and move-conditional MAE
- BrierScoreResult: Brier score with Murphy decomposition
- VolatilityStratifiedResult: Metrics stratified by volatility regime

Knowledge Tiers
---------------
[T1] All struct fields correspond to standard statistical quantities
[T2] MC-SS formulation from myga-forecasting-v2 Phase 11
[T2] Reliability threshold of 10 samples per direction (rule of thumb)
"""

using Statistics

# =============================================================================
# Move Direction Enum
# =============================================================================

"""
    MoveDirection

Direction of value change for 3-class classification.

- `UP`: Value exceeds positive threshold
- `DOWN`: Value below negative threshold
- `FLAT`: Value within threshold range

Used for move-conditional evaluation where persistence (predict no change)
is the baseline. The 3-class framework gives persistence a fair baseline
(it correctly predicts FLAT periods).

Knowledge Tier: [T2] 3-class framework from myga-forecasting-v2.
"""
@enum MoveDirection begin
    UP
    DOWN
    FLAT
end

# =============================================================================
# Move Conditional Result
# =============================================================================

"""
    MoveConditionalResult

Move-conditional evaluation results.

Evaluates performance conditional on actual movement direction:
- UP: actual > threshold
- DOWN: actual < -threshold
- FLAT: |actual| <= threshold

# Fields
- `mae_up::Float64`: MAE for upward moves
- `mae_down::Float64`: MAE for downward moves
- `mae_flat::Float64`: MAE for flat periods
- `n_up::Int`: Count of upward moves
- `n_down::Int`: Count of downward moves
- `n_flat::Int`: Count of flat periods
- `skill_score::Float64`: MC-SS = 1 - (model_mae_moves / persistence_mae_moves)
- `move_threshold::Float64`: Threshold used for classification

# Knowledge Tiers
- [T1] Skill score formula: SS = 1 - (model_error / baseline_error) (Murphy 1988)
- [T2] MC-SS = skill score computed on moves only (myga-forecasting-v2)
- [T2] 70th percentile threshold for "significant" moves
"""
struct MoveConditionalResult
    mae_up::Float64
    mae_down::Float64
    mae_flat::Float64
    n_up::Int
    n_down::Int
    n_flat::Int
    skill_score::Float64
    move_threshold::Float64
end

"""Total sample count."""
n_total(r::MoveConditionalResult) = r.n_up + r.n_down + r.n_flat

"""Count of significant moves (UP + DOWN)."""
n_moves(r::MoveConditionalResult) = r.n_up + r.n_down

"""
    is_reliable(r::MoveConditionalResult)

Check if results are statistically reliable.

Requires at least 10 samples per move direction.

Knowledge Tier: [T3] 10 samples per direction is rule of thumb.
"""
is_reliable(r::MoveConditionalResult) = r.n_up >= 10 && r.n_down >= 10

"""Fraction of samples that are moves (not FLAT)."""
function move_fraction(r::MoveConditionalResult)
    total = n_total(r)
    total == 0 ? 0.0 : n_moves(r) / total
end

function Base.show(io::IO, r::MoveConditionalResult)
    ss = isnan(r.skill_score) ? "NaN" : round(r.skill_score, digits=3)
    print(io, "MoveConditionalResult(skill_score=$ss, n_moves=$(n_moves(r)), n_flat=$(r.n_flat))")
end

# =============================================================================
# Brier Score Result
# =============================================================================

"""
    BrierScoreResult

Brier score with Murphy decomposition.

The Brier score measures probabilistic forecast accuracy and can be
decomposed into reliability, resolution, and uncertainty components.

# Fields
- `brier_score::Float64`: Overall Brier score (lower is better, 0 = perfect)
- `reliability::Float64`: Calibration error (lower is better)
- `resolution::Float64`: Ability to separate outcomes (higher is better)
- `uncertainty::Float64`: Inherent outcome uncertainty (data property)
- `n_samples::Int`: Number of samples
- `n_classes::Int`: Number of classes (2 or 3)

# Murphy Decomposition
Brier = Reliability - Resolution + Uncertainty

A well-calibrated forecast has low reliability (close to 0).
A skillful forecast has high resolution (ability to predict extreme outcomes).

# Knowledge Tier
[T1] Murphy (1973). "A New Vector Partition of the Probability Score."
     Journal of Applied Meteorology, 12, 595-600.
"""
struct BrierScoreResult
    brier_score::Float64
    reliability::Float64
    resolution::Float64
    uncertainty::Float64
    n_samples::Int
    n_classes::Int
end

"""Check if Murphy decomposition sums correctly (debugging)."""
function decomposition_valid(r::BrierScoreResult; tol::Float64=1e-6)
    expected = r.reliability - r.resolution + r.uncertainty
    abs(r.brier_score - expected) < tol
end

function Base.show(io::IO, r::BrierScoreResult)
    bs = round(r.brier_score, digits=4)
    print(io, "BrierScoreResult(brier=$bs, n=$(r.n_samples), classes=$(r.n_classes))")
end

# =============================================================================
# Volatility Stratified Result
# =============================================================================

"""
    VolatilityStratifiedResult

Metrics stratified by volatility regime.

Useful for understanding model performance across different market conditions.

# Fields
- `mae_low::Float64`: MAE in low volatility regime
- `mae_medium::Float64`: MAE in medium volatility regime
- `mae_high::Float64`: MAE in high volatility regime
- `n_low::Int`: Count in low volatility
- `n_medium::Int`: Count in medium volatility
- `n_high::Int`: Count in high volatility
- `vol_thresholds::Tuple{Float64, Float64}`: (low, high) thresholds

# Knowledge Tier
[T3] Volatility stratification is a practical technique, thresholds are
     typically data-driven (33rd/67th percentiles).
"""
struct VolatilityStratifiedResult
    mae_low::Float64
    mae_medium::Float64
    mae_high::Float64
    n_low::Int
    n_medium::Int
    n_high::Int
    vol_thresholds::Tuple{Float64, Float64}
end

"""Total sample count."""
n_total(r::VolatilityStratifiedResult) = r.n_low + r.n_medium + r.n_high

function Base.show(io::IO, r::VolatilityStratifiedResult)
    print(io, "VolatilityStratifiedResult(n_low=$(r.n_low), n_medium=$(r.n_medium), n_high=$(r.n_high))")
end

# =============================================================================
# Interval Score Result
# =============================================================================

"""
    IntervalScoreResult

Result from interval score computation.

# Fields
- `score::Float64`: Total interval score (lower is better)
- `width_penalty::Float64`: Penalty for interval width
- `coverage_penalty::Float64`: Penalty for miscoverage
- `coverage::Float64`: Empirical coverage rate
- `mean_width::Float64`: Mean interval width
- `alpha::Float64`: Nominal miscoverage rate
- `n_samples::Int`: Number of samples

# Knowledge Tier
[T1] Gneiting & Raftery (2007). Strictly proper scoring rules.
"""
struct IntervalScoreResult
    score::Float64
    width_penalty::Float64
    coverage_penalty::Float64
    coverage::Float64
    mean_width::Float64
    alpha::Float64
    n_samples::Int
end

function Base.show(io::IO, r::IntervalScoreResult)
    s = round(r.score, digits=4)
    cov = round(r.coverage, digits=3)
    print(io, "IntervalScoreResult(score=$s, coverage=$cov, n=$(r.n_samples))")
end

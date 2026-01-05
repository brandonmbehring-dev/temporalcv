# =============================================================================
# CVFinancial Module - Purged Cross-Validation for Financial ML
# =============================================================================
#
# Implements cross-validation strategies for financial time series that account
# for label overlap (e.g., forward returns that share days).
#
# Based on: De Prado (2018), Advances in Financial Machine Learning, Ch. 7
#
# Key concepts [T1]:
# - Purging: Remove training samples within purge_gap of any test sample
# - Embargo: Additional % of samples removed after test set
# - Label overlap: When labels use future data (5-day forward returns share 4 days)

module CVFinancial

using Random
using Statistics
using Combinatorics

# =============================================================================
# Types
# =============================================================================

"""
Represents a single purged cross-validation split.

# Fields
- `train_indices::Vector{Int}`: Indices for training set
- `test_indices::Vector{Int}`: Indices for test set
- `n_purged::Int`: Number of samples purged from training
- `n_embargoed::Int`: Number of samples embargoed from training
"""
struct PurgedSplit
    train_indices::Vector{Int}
    test_indices::Vector{Int}
    n_purged::Int
    n_embargoed::Int
end

"""
Configuration for purged K-fold cross-validation.

# Fields
- `n_splits::Int`: Number of folds
- `purge_gap::Int`: Number of samples to purge around test set
- `embargo_pct::Float64`: Fraction of samples to embargo after test set
"""
struct PurgedKFold
    n_splits::Int
    purge_gap::Int
    embargo_pct::Float64

    function PurgedKFold(;
        n_splits::Int = 5,
        purge_gap::Int = 0,
        embargo_pct::Float64 = 0.0
    )
        n_splits >= 2 || error("n_splits must be >= 2, got $n_splits")
        purge_gap >= 0 || error("purge_gap must be >= 0, got $purge_gap")
        0.0 <= embargo_pct < 1.0 || error("embargo_pct must be in [0, 1), got $embargo_pct")
        new(n_splits, purge_gap, embargo_pct)
    end
end

"""
Configuration for combinatorial purged cross-validation (CPCV).

All (n choose k) combinations of test folds.

# Fields
- `n_splits::Int`: Number of folds
- `n_test_splits::Int`: Number of folds to use as test set
- `purge_gap::Int`: Number of samples to purge around test set
- `embargo_pct::Float64`: Fraction of samples to embargo after test set
"""
struct CombinatorialPurgedCV
    n_splits::Int
    n_test_splits::Int
    purge_gap::Int
    embargo_pct::Float64

    function CombinatorialPurgedCV(;
        n_splits::Int = 5,
        n_test_splits::Int = 2,
        purge_gap::Int = 0,
        embargo_pct::Float64 = 0.0
    )
        n_splits >= 2 || error("n_splits must be >= 2, got $n_splits")
        n_test_splits >= 1 || error("n_test_splits must be >= 1, got $n_test_splits")
        n_test_splits < n_splits || error("n_test_splits must be < n_splits")
        purge_gap >= 0 || error("purge_gap must be >= 0, got $purge_gap")
        0.0 <= embargo_pct < 1.0 || error("embargo_pct must be in [0, 1), got $embargo_pct")
        new(n_splits, n_test_splits, purge_gap, embargo_pct)
    end
end

"""
Configuration for purged walk-forward cross-validation.

# Fields
- `n_splits::Int`: Number of walk-forward splits
- `train_size::Int`: Fixed training window size (0 = expanding)
- `purge_gap::Int`: Number of samples to purge between train and test
- `embargo_pct::Float64`: Fraction of samples to embargo after test set
"""
struct PurgedWalkForward
    n_splits::Int
    train_size::Int
    purge_gap::Int
    embargo_pct::Float64

    function PurgedWalkForward(;
        n_splits::Int = 5,
        train_size::Int = 0,
        purge_gap::Int = 0,
        embargo_pct::Float64 = 0.0
    )
        n_splits >= 1 || error("n_splits must be >= 1, got $n_splits")
        train_size >= 0 || error("train_size must be >= 0, got $train_size")
        purge_gap >= 0 || error("purge_gap must be >= 0, got $purge_gap")
        0.0 <= embargo_pct < 1.0 || error("embargo_pct must be in [0, 1), got $embargo_pct")
        new(n_splits, train_size, purge_gap, embargo_pct)
    end
end

# =============================================================================
# Core Functions
# =============================================================================

include("purged.jl")

# =============================================================================
# Exports
# =============================================================================

export PurgedSplit
export PurgedKFold, CombinatorialPurgedCV, PurgedWalkForward

export compute_label_overlap, estimate_purge_gap
export split, get_n_splits
export apply_purge_and_embargo

end # module CVFinancial

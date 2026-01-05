# =============================================================================
# Compare Module - Model Comparison Framework
# =============================================================================
#
# Provides unified interface for comparing forecasting models across datasets.
# Integrates with StatisticalTests for DM test significance.

module Compare

using Statistics

# Import dm_test from parent module (will be available via TemporalValidation)
# We'll use a late binding approach

# =============================================================================
# Submodule Files
# =============================================================================

include("types.jl")
include("adapters.jl")
include("runner.jl")

# =============================================================================
# Exports
# =============================================================================

# Types
export ModelResult, ComparisonResult, ComparisonReport
export ForecastAdapter, NaiveAdapter, SeasonalNaiveAdapter

# Functions from types
export get_metric, get_ranking, to_dict, to_markdown
export compute_comparison_metrics

# Functions from runner
export run_comparison, run_benchmark_suite, compare_to_baseline

# Adapters
export model_name, package_name, fit_predict, get_params

end # module Compare

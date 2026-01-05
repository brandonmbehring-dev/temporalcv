# =============================================================================
# Benchmarks Module - Dataset Infrastructure for Time Series Benchmarking
# =============================================================================
#
# Provides benchmark dataset loaders and generators for model comparison.
# Designed with idiomatic Julia patterns: multiple dispatch, keyword args.
#
# Key exports:
# - TimeSeriesDataset, DatasetMetadata: Core types
# - create_synthetic_dataset: AR(1) generator
# - create_electricity_like_dataset: Seasonal pattern generator
# - to_benchmark_tuple: Convert to Compare.run_benchmark_suite format
#
# Example
# -------
# ```julia
# using TemporalValidation
#
# # Create synthetic datasets
# datasets = [create_synthetic_dataset(seed=i) for i in 1:3]
#
# # Convert for benchmark runner
# tuples = [to_benchmark_tuple(ds) for ds in datasets]
#
# # Run comparisons
# report = run_benchmark_suite(tuples; adapters=[NaiveAdapter()])
# ```

module Benchmarks

using Dates
using Random
using Statistics

# =============================================================================
# Submodule Files
# =============================================================================

include("types.jl")
include("synthetic.jl")

# =============================================================================
# Exports
# =============================================================================

# Types (from types.jl)
export DatasetNotFoundError
export DatasetMetadata
export TimeSeriesDataset

# Type utilities (from types.jl)
export to_dict
export n_obs, has_exogenous
export get_train_test_split
export validate_dataset

# Generators (from synthetic.jl)
export create_ar1_series
export create_synthetic_dataset
export create_electricity_like_dataset

# Bundled datasets (from synthetic.jl)
export create_bundled_test_datasets

# Benchmark integration (from synthetic.jl)
export to_benchmark_tuple
export to_benchmark_tuples

end # module Benchmarks

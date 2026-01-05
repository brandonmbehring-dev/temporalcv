"""
Time-series-aware bagging and bootstrap methods.

Provides bootstrap resampling strategies that preserve temporal dependence
structure, unlike i.i.d. bootstrap methods.

# Exports
- `BootstrapStrategy`: Abstract type for bootstrap strategies
- `MovingBlockBootstrap`: Kunsch (1989) block bootstrap
- `StationaryBootstrap`: Politis & Romano (1994) geometric blocks
- `TimeSeriesBagger`: Generic bagging wrapper

# References
- Kunsch (1989) "The Jackknife and the Bootstrap for General Stationary Observations"
- Politis & Romano (1994) "The Stationary Bootstrap" JASA 89(428)
"""
module Bagging

using Statistics
using Random

# Include type definitions first
include("types.jl")

# Include bootstrap strategies
include("block_bootstrap.jl")
include("stationary_bootstrap.jl")

# Include bagger
include("bagger.jl")

# Exports
export BootstrapStrategy
export bootstrap_sample

export MovingBlockBootstrap
export StationaryBootstrap

export TimeSeriesBagger
export fit!, predict, predict_with_uncertainty, predict_interval

end # module Bagging

"""
Conformal prediction for time series.

Provides conformal prediction methods with finite-sample coverage guarantees,
adapted for time series data with potential distribution shift.

# Exports
- `PredictionInterval`: Container for prediction intervals
- `SplitConformalPredictor`: Static split conformal for i.i.d. data
- `AdaptiveConformalPredictor`: Online adaptation for distribution shift

# References
- Vovk, Gammerman, Shafer (2005) "Algorithmic Learning in a Random World"
- Gibbs & Candès (2021) "Adaptive Conformal Inference Under Distribution Shift"
"""
module Conformal

using Statistics
using Random

# Include type definitions first
include("types.jl")

# Include predictor implementations
include("split.jl")
include("adaptive.jl")

# Exports
export PredictionInterval
export width, mean_width, coverage

export SplitConformalPredictor
export calibrate!, predict_interval

export AdaptiveConformalPredictor
export initialize!, update!

end # module Conformal

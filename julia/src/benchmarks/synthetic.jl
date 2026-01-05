# =============================================================================
# Synthetic Dataset Generation
# =============================================================================
#
# Provides AR(1) and other synthetic dataset generators for testing and
# benchmarking. Uses idiomatic Julia with multiple dispatch and keyword args.

using Random

# =============================================================================
# AR(1) Generation
# =============================================================================

"""
    create_ar1_series(n_obs; ar_coef=0.9, noise_std=0.1, seed=nothing) -> Vector{Float64}

Generate a single AR(1) time series.

# Mathematical Definition
```
y[t] = φ * y[t-1] + ε[t]
where ε[t] ~ N(0, σ²)
```

# Arguments
- `n_obs::Int`: Number of observations
- `ar_coef::Float64=0.9`: AR(1) coefficient φ (persistence)
- `noise_std::Float64=0.1`: Innovation standard deviation σ
- `seed::Union{Int, Nothing}=nothing`: Random seed for reproducibility

# Returns
`Vector{Float64}` of length `n_obs`

# Example
```julia
series = create_ar1_series(100; ar_coef=0.95, seed=42)
```

# Note
For stationary process, require |φ| < 1. High persistence (φ > 0.9) is typical
for financial time series.
"""
function create_ar1_series(
    n_obs::Int;
    ar_coef::Float64 = 0.9,
    noise_std::Float64 = 0.1,
    seed::Union{Int, Nothing} = nothing
)::Vector{Float64}
    n_obs >= 1 || error("n_obs must be >= 1, got $n_obs")
    abs(ar_coef) < 1.0 || @warn "ar_coef >= 1.0 produces non-stationary series"
    noise_std > 0 || error("noise_std must be > 0, got $noise_std")

    rng = isnothing(seed) ? Random.default_rng() : Random.MersenneTwister(seed)

    values = Vector{Float64}(undef, n_obs)
    values[1] = randn(rng) * noise_std

    for t in 2:n_obs
        values[t] = ar_coef * values[t-1] + randn(rng) * noise_std
    end

    return values
end


"""
    create_ar1_series(n_series, n_obs; kwargs...) -> Matrix{Float64}

Generate multiple independent AR(1) series.

# Arguments
- `n_series::Int`: Number of series to generate
- `n_obs::Int`: Observations per series
- `ar_coef::Float64=0.9`: AR(1) coefficient
- `noise_std::Float64=0.1`: Innovation standard deviation
- `seed::Union{Int, Nothing}=nothing`: Random seed

# Returns
`Matrix{Float64}` of shape (n_series, n_obs)

# Example
```julia
series = create_ar1_series(5, 100; seed=42)  # 5 series, 100 obs each
```
"""
function create_ar1_series(
    n_series::Int,
    n_obs::Int;
    ar_coef::Float64 = 0.9,
    noise_std::Float64 = 0.1,
    seed::Union{Int, Nothing} = nothing
)::Matrix{Float64}
    n_series >= 1 || error("n_series must be >= 1, got $n_series")
    n_obs >= 1 || error("n_obs must be >= 1, got $n_obs")

    rng = isnothing(seed) ? Random.default_rng() : Random.MersenneTwister(seed)

    values = Matrix{Float64}(undef, n_series, n_obs)

    for s in 1:n_series
        values[s, 1] = randn(rng) * noise_std
        for t in 2:n_obs
            values[s, t] = ar_coef * values[s, t-1] + randn(rng) * noise_std
        end
    end

    return values
end


# =============================================================================
# Synthetic Dataset Factory
# =============================================================================

"""
    create_synthetic_dataset(; n_obs=200, n_series=1, frequency="W",
                              horizon=2, train_fraction=0.8, ar_coef=0.9,
                              noise_std=0.1, seed=42) -> TimeSeriesDataset

Create synthetic AR(1) dataset for testing and benchmarking.

# Arguments
- `n_obs::Int=200`: Observations per series
- `n_series::Int=1`: Number of time series
- `frequency::String="W"`: Temporal frequency ("D", "W", "M", "H", "Y")
- `horizon::Int=2`: Forecast horizon
- `train_fraction::Float64=0.8`: Fraction allocated to training
- `ar_coef::Float64=0.9`: AR(1) persistence coefficient
- `noise_std::Float64=0.1`: Innovation standard deviation
- `seed::Int=42`: Random seed for reproducibility

# Returns
`TimeSeriesDataset` with:
- AR(1) values
- Metadata including train/test split
- Characteristics dict with generation parameters

# Example
```julia
# Single series
dataset = create_synthetic_dataset(n_obs=100, ar_coef=0.95)
train, test = get_train_test_split(dataset)

# Multiple series
dataset = create_synthetic_dataset(n_obs=100, n_series=5, seed=123)
```

# Note
The default parameters (ar_coef=0.9, noise_std=0.1) produce high-persistence
series typical of financial data. Adjust for your domain:
- Macroeconomic data: ar_coef ≈ 0.95-0.99
- Daily returns: ar_coef ≈ 0.0-0.1 (nearly white noise)
- Unemployment rate: ar_coef ≈ 0.98 (very sticky)
"""
function create_synthetic_dataset(;
    n_obs::Int = 200,
    n_series::Int = 1,
    frequency::String = "W",
    horizon::Int = 2,
    train_fraction::Float64 = 0.8,
    ar_coef::Float64 = 0.9,
    noise_std::Float64 = 0.1,
    seed::Int = 42
)::TimeSeriesDataset
    # Validate
    n_obs >= 1 || error("n_obs must be >= 1, got $n_obs")
    n_series >= 1 || error("n_series must be >= 1, got $n_series")
    0.0 < train_fraction < 1.0 || error("train_fraction must be in (0, 1), got $train_fraction")
    horizon >= 1 || error("horizon must be >= 1, got $horizon")

    # Generate values
    values = if n_series == 1
        create_ar1_series(n_obs; ar_coef=ar_coef, noise_std=noise_std, seed=seed)
    else
        create_ar1_series(n_series, n_obs; ar_coef=ar_coef, noise_std=noise_std, seed=seed)
    end

    # Compute split
    train_end_idx = floor(Int, n_obs * train_fraction)

    # Validate split leaves room for horizon
    train_end_idx > horizon || error(
        "train_end_idx ($train_end_idx) must be > horizon ($horizon) " *
        "to allow sufficient training data"
    )
    train_end_idx <= n_obs - horizon || error(
        "train_end_idx ($train_end_idx) leaves only $(n_obs - train_end_idx) test observations, " *
        "but need at least $horizon for horizon-step forecasting"
    )

    # Build metadata
    total_observations = n_series > 1 ? n_series * n_obs : n_obs

    metadata = DatasetMetadata(
        name = "synthetic_ar1",
        frequency = frequency,
        horizon = horizon,
        n_series = n_series,
        total_observations = total_observations,
        train_end_idx = train_end_idx,
        characteristics = Dict{String, Any}(
            "ar_coef" => ar_coef,
            "noise_std" => noise_std,
            "synthetic" => true,
            "seed" => seed
        ),
        license = "synthetic",
        source_url = ""
    )

    return TimeSeriesDataset(values=values, metadata=metadata)
end


# =============================================================================
# Benchmark Suite Integration
# =============================================================================

"""
    to_benchmark_tuple(dataset::TimeSeriesDataset) -> Tuple{String, Vector{Float64}, Vector{Float64}}

Convert TimeSeriesDataset to format expected by `run_benchmark_suite`.

# Returns
`(name, train, test)` tuple compatible with Compare.run_benchmark_suite

# Example
```julia
dataset = create_synthetic_dataset(seed=42)
name, train, test = to_benchmark_tuple(dataset)
report = run_benchmark_suite([to_benchmark_tuple(ds) for ds in datasets]; adapters=[...])
```

# Note
For multi-series datasets, this flattens to the first series. Use
`to_benchmark_tuples` (plural) for multi-series handling.
"""
function to_benchmark_tuple(dataset::TimeSeriesDataset)::Tuple{String, Vector{Float64}, Vector{Float64}}
    train, test = get_train_test_split(dataset)

    # Handle multi-series by taking first series
    train_vec = train isa Vector ? train : vec(train[1, :])
    test_vec = test isa Vector ? test : vec(test[1, :])

    return (dataset.metadata.name, train_vec, test_vec)
end


"""
    to_benchmark_tuples(dataset::TimeSeriesDataset) -> Vector{Tuple{String, Vector{Float64}, Vector{Float64}}}

Convert multi-series TimeSeriesDataset to vector of benchmark tuples.

Each series becomes a separate tuple named "name_1", "name_2", etc.

# Example
```julia
dataset = create_synthetic_dataset(n_series=5, seed=42)
tuples = to_benchmark_tuples(dataset)  # 5 tuples
report = run_benchmark_suite(tuples; adapters=[...])
```
"""
function to_benchmark_tuples(dataset::TimeSeriesDataset)::Vector{Tuple{String, Vector{Float64}, Vector{Float64}}}
    train, test = get_train_test_split(dataset)
    name = dataset.metadata.name
    n_series = dataset.metadata.n_series

    if train isa Vector
        # Single series
        return [(name, train, test)]
    else
        # Multi-series: each row is a series
        tuples = Tuple{String, Vector{Float64}, Vector{Float64}}[]
        for s in 1:n_series
            push!(tuples, ("$(name)_$s", vec(train[s, :]), vec(test[s, :])))
        end
        return tuples
    end
end


# =============================================================================
# Bundled Test Datasets
# =============================================================================

"""
    create_bundled_test_datasets(; seeds=[42, 123, 456]) -> Vector{TimeSeriesDataset}

Create standard bundled datasets for offline CI testing.

Returns 3 datasets with different random seeds but identical parameters:
- n_obs=150, ar_coef=0.9, noise_std=0.1, horizon=2

# Example
```julia
datasets = create_bundled_test_datasets()
tuples = [to_benchmark_tuple(ds) for ds in datasets]
report = run_benchmark_suite(tuples; adapters=[...])
```
"""
function create_bundled_test_datasets(;
    seeds::Vector{Int} = [42, 123, 456]
)::Vector{TimeSeriesDataset}
    return [
        create_synthetic_dataset(
            n_obs = 150,
            ar_coef = 0.9,
            noise_std = 0.1,
            seed = seed,
            horizon = 2
        )
        for seed in seeds
    ]
end


# =============================================================================
# Electricity-like Dataset
# =============================================================================

"""
    create_electricity_like_dataset(; n_obs=336, n_series=1, horizon=24,
                                     train_fraction=0.8, seed=42) -> TimeSeriesDataset

Create synthetic dataset mimicking electricity demand patterns.

Generates hourly-like data with:
- Daily seasonality (period 24)
- Trend component
- AR(1) residual noise

# Arguments
- `n_obs::Int=336`: Observations (default: 14 days × 24 hours)
- `n_series::Int=1`: Number of series
- `horizon::Int=24`: Forecast horizon (default: 1 day ahead)
- `train_fraction::Float64=0.8`: Training fraction
- `seed::Int=42`: Random seed

# Example
```julia
dataset = create_electricity_like_dataset(n_obs=720, horizon=48)  # 30 days
train, test = get_train_test_split(dataset)
```

# Note
This is a simplified synthetic approximation. For real electricity benchmarks,
use external datasets like GluonTS Electricity or Monash.
"""
function create_electricity_like_dataset(;
    n_obs::Int = 336,
    n_series::Int = 1,
    horizon::Int = 24,
    train_fraction::Float64 = 0.8,
    seed::Int = 42
)::TimeSeriesDataset
    rng = Random.MersenneTwister(seed)

    if n_series == 1
        values = _generate_electricity_series(n_obs, rng)
    else
        values = Matrix{Float64}(undef, n_series, n_obs)
        for s in 1:n_series
            values[s, :] = _generate_electricity_series(n_obs, rng)
        end
    end

    train_end_idx = floor(Int, n_obs * train_fraction)
    total_observations = n_series > 1 ? n_series * n_obs : n_obs

    metadata = DatasetMetadata(
        name = "synthetic_electricity",
        frequency = "H",
        horizon = horizon,
        n_series = n_series,
        total_observations = total_observations,
        train_end_idx = train_end_idx,
        characteristics = Dict{String, Any}(
            "synthetic" => true,
            "seasonality" => 24,
            "seed" => seed
        ),
        license = "synthetic",
        source_url = ""
    )

    return TimeSeriesDataset(values=values, metadata=metadata)
end


"""Generate single electricity-like series."""
function _generate_electricity_series(n_obs::Int, rng::AbstractRNG)::Vector{Float64}
    values = Vector{Float64}(undef, n_obs)

    # Parameters
    base_level = 100.0
    daily_amplitude = 30.0
    trend_per_obs = 0.01
    ar_coef = 0.3
    noise_std = 5.0

    prev_residual = 0.0

    for t in 1:n_obs
        # Daily seasonality (hour of day pattern)
        hour = mod(t - 1, 24)
        # Peak in afternoon, trough at night
        seasonal = daily_amplitude * sin(2π * (hour - 6) / 24)

        # Slight upward trend
        trend = trend_per_obs * t

        # AR(1) residual
        innovation = randn(rng) * noise_std
        residual = ar_coef * prev_residual + innovation
        prev_residual = residual

        values[t] = base_level + seasonal + trend + residual
    end

    return values
end

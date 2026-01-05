# =============================================================================
# Benchmark Types - Dataset Infrastructure
# =============================================================================
#
# Core types for benchmark dataset handling.
# Provides DatasetMetadata, TimeSeriesDataset, and DatasetNotFoundError.

using Dates


# =============================================================================
# DatasetNotFoundError
# =============================================================================

"""
    DatasetNotFoundError <: Exception

Exception raised when a dataset cannot be found or accessed.

Used for datasets that require manual download (M5) or API keys (FRED).

# Fields
- `dataset_name::String`: Name of the dataset
- `download_url::String`: URL where dataset can be obtained
- `instructions::String`: Step-by-step instructions for obtaining the dataset
"""
struct DatasetNotFoundError <: Exception
    dataset_name::String
    download_url::String
    instructions::String
end

function Base.showerror(io::IO, e::DatasetNotFoundError)
    print(io, "DatasetNotFoundError: Dataset '$(e.dataset_name)' not found.\n")
    print(io, "Download URL: $(e.download_url)\n")
    print(io, "\nInstructions:\n$(e.instructions)")
end


# =============================================================================
# DatasetMetadata
# =============================================================================

"""
    DatasetMetadata

Metadata describing a benchmark dataset.

# Required Fields
- `name::String`: Dataset identifier
- `frequency::String`: Temporal frequency ("D", "W", "M", "H", "Y")
- `horizon::Int`: Forecast horizon (>= 1)
- `n_series::Int`: Number of time series (>= 1)
- `total_observations::Int`: Total observation count (>= 1)

# Optional Fields
- `train_end_idx::Union{Int, Nothing}`: Standard train/test split index
- `characteristics::Dict{String, Any}`: Domain-specific metadata
- `license::String`: Licensing information
- `source_url::String`: Data source URL
- `official_split::Bool`: Whether split follows competition protocol
- `truncated::Bool`: Whether series were shortened
- `original_series_lengths::Union{Vector{Int}, Nothing}`: Pre-truncation lengths
- `split_source::String`: Source of split definition

# Example
```julia
meta = DatasetMetadata(
    name = "treasury_rates",
    frequency = "W",
    horizon = 4,
    n_series = 1,
    total_observations = 1000,
    train_end_idx = 800,
    characteristics = Dict{String, Any}("high_persistence" => true)
)
```
"""
struct DatasetMetadata
    name::String
    frequency::String
    horizon::Int
    n_series::Int
    total_observations::Int
    train_end_idx::Union{Int, Nothing}
    characteristics::Dict{String, Any}
    license::String
    source_url::String
    official_split::Bool
    truncated::Bool
    original_series_lengths::Union{Vector{Int}, Nothing}
    split_source::String

    function DatasetMetadata(;
        name::String,
        frequency::String,
        horizon::Int,
        n_series::Int,
        total_observations::Int,
        train_end_idx::Union{Int, Nothing} = nothing,
        characteristics::Dict{String, Any} = Dict{String, Any}(),
        license::String = "",
        source_url::String = "",
        official_split::Bool = false,
        truncated::Bool = false,
        original_series_lengths::Union{Vector{Int}, Nothing} = nothing,
        split_source::String = ""
    )
        # Validation
        isempty(name) && error("name cannot be empty")
        frequency in ["D", "W", "M", "H", "Y", "Q"] ||
            error("frequency must be one of D, W, M, H, Y, Q; got '$frequency'")
        horizon >= 1 || error("horizon must be >= 1, got $horizon")
        n_series >= 1 || error("n_series must be >= 1, got $n_series")
        total_observations >= 1 || error("total_observations must be >= 1, got $total_observations")

        if !isnothing(train_end_idx)
            train_end_idx >= 1 || error("train_end_idx must be >= 1")
            train_end_idx <= total_observations ||
                error("train_end_idx ($train_end_idx) exceeds total_observations ($total_observations)")
        end

        new(name, frequency, horizon, n_series, total_observations,
            train_end_idx, characteristics, license, source_url,
            official_split, truncated, original_series_lengths, split_source)
    end
end


"""
    to_dict(meta::DatasetMetadata) -> Dict{String, Any}

Convert metadata to dictionary for serialization.
"""
function to_dict(meta::DatasetMetadata)::Dict{String, Any}
    return Dict{String, Any}(
        "name" => meta.name,
        "frequency" => meta.frequency,
        "horizon" => meta.horizon,
        "n_series" => meta.n_series,
        "total_observations" => meta.total_observations,
        "train_end_idx" => meta.train_end_idx,
        "characteristics" => meta.characteristics,
        "license" => meta.license,
        "source_url" => meta.source_url,
        "official_split" => meta.official_split,
        "truncated" => meta.truncated,
        "original_series_lengths" => meta.original_series_lengths,
        "split_source" => meta.split_source
    )
end


# =============================================================================
# TimeSeriesDataset
# =============================================================================

"""
    TimeSeriesDataset

Container for time series data with metadata.

# Fields
- `values::Union{Vector{Float64}, Matrix{Float64}}`: Time series values
  - Vector for single series: (n_obs,)
  - Matrix for multi-series: (n_series, n_obs)
- `metadata::DatasetMetadata`: Dataset metadata
- `timestamps::Union{Vector{Date}, Nothing}`: Optional temporal index
- `exogenous::Union{Matrix{Float64}, Nothing}`: Optional exogenous features

# Example
```julia
# Single series
ds = TimeSeriesDataset(
    values = randn(100),
    metadata = DatasetMetadata(name="test", frequency="D", horizon=1, n_series=1, total_observations=100)
)

# Multi-series
ds = TimeSeriesDataset(
    values = randn(5, 100),  # 5 series, 100 observations each
    metadata = DatasetMetadata(name="test", frequency="D", horizon=1, n_series=5, total_observations=500)
)
```
"""
struct TimeSeriesDataset
    values::Union{Vector{Float64}, Matrix{Float64}}
    metadata::DatasetMetadata
    timestamps::Union{Vector{Date}, Nothing}
    exogenous::Union{Matrix{Float64}, Nothing}

    function TimeSeriesDataset(;
        values::Union{Vector{Float64}, Matrix{Float64}},
        metadata::DatasetMetadata,
        timestamps::Union{Vector{Date}, Nothing} = nothing,
        exogenous::Union{Matrix{Float64}, Nothing} = nothing
    )
        # Validate values match metadata
        if values isa Vector
            length(values) == metadata.total_observations ||
                error("values length ($(length(values))) != total_observations ($(metadata.total_observations))")
            metadata.n_series == 1 ||
                error("Vector values requires n_series=1, got $(metadata.n_series)")
        else
            size(values, 1) == metadata.n_series ||
                error("values rows ($(size(values,1))) != n_series ($(metadata.n_series))")
            size(values, 1) * size(values, 2) == metadata.total_observations ||
                error("values size ($(size(values,1) * size(values,2))) != total_observations ($(metadata.total_observations))")
        end

        # Validate timestamps if provided
        if !isnothing(timestamps)
            n = values isa Vector ? length(values) : size(values, 2)
            length(timestamps) == n ||
                error("timestamps length ($(length(timestamps))) != n_obs ($n)")
        end

        # Validate exogenous if provided
        if !isnothing(exogenous)
            n = values isa Vector ? length(values) : size(values, 2)
            size(exogenous, 1) == n ||
                error("exogenous rows ($(size(exogenous,1))) != n_obs ($n)")
        end

        new(values, metadata, timestamps, exogenous)
    end
end


"""
    n_obs(ds::TimeSeriesDataset) -> Int

Return number of observations per series.
"""
function n_obs(ds::TimeSeriesDataset)::Int
    return ds.values isa Vector ? length(ds.values) : size(ds.values, 2)
end


"""
    has_exogenous(ds::TimeSeriesDataset) -> Bool

Check if dataset has exogenous features.
"""
function has_exogenous(ds::TimeSeriesDataset)::Bool
    return !isnothing(ds.exogenous)
end


"""
    get_train_test_split(ds::TimeSeriesDataset) -> Tuple

Split dataset into train and test based on metadata.train_end_idx.

# Returns
For single series: `(train::Vector{Float64}, test::Vector{Float64})`
For multi-series: `(train::Matrix{Float64}, test::Matrix{Float64})`

# Throws
Error if train_end_idx is not set in metadata.
"""
function get_train_test_split(ds::TimeSeriesDataset)
    idx = ds.metadata.train_end_idx
    isnothing(idx) && error("train_end_idx not set in metadata. Cannot split dataset.")

    if ds.values isa Vector
        train = ds.values[1:idx]
        test = ds.values[(idx+1):end]
        return (train, test)
    else
        train = ds.values[:, 1:idx]
        test = ds.values[:, (idx+1):end]
        return (train, test)
    end
end


# =============================================================================
# Validation
# =============================================================================

"""
    validate_dataset(ds::TimeSeriesDataset) -> Bool

Validate dataset for protocol compliance.

Checks:
- Values not empty
- n_series >= 1
- horizon >= 1

# Returns
`true` if valid.

# Throws
`ArgumentError` if validation fails.
"""
function validate_dataset(ds::TimeSeriesDataset)::Bool
    # Check values not empty
    n = ds.values isa Vector ? length(ds.values) : size(ds.values, 2)
    n > 0 || throw(ArgumentError("Dataset values cannot be empty"))

    # Check n_series
    ds.metadata.n_series >= 1 ||
        throw(ArgumentError("n_series must be >= 1, got $(ds.metadata.n_series)"))

    # Check horizon
    ds.metadata.horizon >= 1 ||
        throw(ArgumentError("horizon must be >= 1, got $(ds.metadata.horizon)"))

    return true
end

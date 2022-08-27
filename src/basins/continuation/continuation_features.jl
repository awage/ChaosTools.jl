export FeaturizingSeedingContinuation
import ProgressMeter

# The recurrences based method is rather flexible because it works
# in two independent steps: it first finds attractors and then matches them.
struct FeaturizingSeedingContinuation{A, M, S, E} <: BasinsFractionContinuation
    mapper::A
    metric::M
    threshold::Float64
    seeds_from_attractor::S
    info_extraction::E
end

"""
    FeaturizingSeedingContinuation(mapper::AttractorsViaFeaturizing; kwargs...)
A method for [`basins_fractions_continuation`](@ref).
It uses seeding of previous attractors to find new ones, which is the main performance
bottleneck. Will write more once we have the paper going.

## Keyword Arguments
- `metric, threshold`: Given to [`match_attractor_ids!`](@ref) which is the function
  used to match attractors between each parameter slice.
- `info_extraction = identity`: A function that takes as an input an attractor (`Dataset`)
  and outputs whatever information should be stored. It is used to return the
  `attractors_info` in [`basins_fractions_continuation`](@ref).
- `seeds_from_attractor`: A function that takes as an input an attractor and returns
  an iterator of initial conditions to be seeded from the attractor for the next
  parameter slice. By default, we sample some points from existing attractors according
  to how many points the attractors themselves contain. A maximum of `10` seeds is done
  per attractor.
"""
function FeaturizingSeedingContinuation(
        mapper::AttractorsViaFeaturizing; metric = Euclidean(),
        threshold = Inf, seeds_from_attractor = _default_seeding_process,
        info_extraction = identity
    )
    return FeaturizingSeedingContinuation(
        mapper, metric, threshold, seeds_from_attractor, info_extraction
    )
end

# function _default_seeding_process(attractor::AbstractDataset)
#     max_possible_seeds = 10
#     seeds = round(Int, log(10, length(attractor)))
#     seeds = clamp(seeds, 1, max_possible_seeds)
#     return (rand(attractor.data) for _ in 1:seeds)
# end

function basins_fractions_continuation(
        continuation::FeaturizingSeedingContinuation, prange, pidx, ics::Function;
        samples_per_parameter = 100, show_progress = false,
    )
    # show_progress && @info "Starting basins fraction continuation."
    # show_progress && @info "p = $(prange[1])"
    if show_progress
        progress=ProgressMeter.Progress(length(prange); desc="Continuating basins fractions:")
    end

    (; mapper, metric, threshold) = continuation
    # first parameter is run in isolation, as it has no prior to seed from
    # TODO: Make this use ProgressMeter.jl
    N = samples_per_parameter * length(prange)
    feature_array = Vector{Vector{Float64}}(undef, N)
    i = 1
    for p in prange
        # show_progress && @show fs
        set_parameter!(mapper.ds, pidx, p)
        for k in 1:samples_per_parameter 
            feature_array[i] = extract_features(mapper, ics())
            i += 1
        end
        show_progress && next!(progress)
    end
    # Cluster over the all range of parameters
    cluster_labels,  = cluster_features(feature_array, mapper.cluster_config)
    fs = basins_fractions(cluster_labels[1:samples_per_parameter]) # Vanilla fractions method with Array input
    fractions_curves = [fs]
    for k in 1:length(prange)-1
        range = 1:samples_per_parameter .+ k*samples_per_parameter
        fs = basins_fractions(cluster_labels[range]) # Vanilla fractions method with Array input
        push!(fractions_curves,fs)
    end
    return fractions_curves
end


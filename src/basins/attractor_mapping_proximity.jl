"""
    AttractorsViaProximity(ds::DynamicalSystem, attractors::Dict; kwargs...)
Map initial conditions to attractors based on whether the trajectory reaches `ε`-distance
close to any of the user-provided `attractors`. They have to be in a form of a dictionary
mapping attractor labels to `Dataset`s containing the attractors.

The state of the system gets stepped, and at each step the minimum distance to all
attractors is computed. If any of these distances is `< ε`, then the label of the nearest
attractor is returned.

Because in this method all possible attractors are already known to the user,
the method can also be called _supervised_.

## Keywords
* `Ttr = 100`: Transient time to first evolve the system for before checking for proximity.
* `Δt = 1`: Integration step time (only valid for continuous systems).
* `horizon_limit = 1e3`: If the maximum distance of the trajectory from any of the given
  attractors exceeds this limit, it is assumed
  that the trajectory diverged (gets labelled as `-1`).
* `mx_chk_lost = 1000`: If the integrator has been stepped this many times without
  coming `ε`-near to any attractor,  it is assumed
  that the trajectory diverged (gets labelled as `-1`).
* `diffeq = NamedTuple()`: Keywords propagated to DifferentialEquations.jl
  (only valid for continuous systems).
"""
struct AttractorsViaProximity{I, AK, D, T, N, K} <: AttractorMapper
    integ::I
    attractors::Dict{AK, Dataset{D, T}}
    ε::Float64
    Δt::N
    Ttr::N
    mx_chk_lost::Int
    horizon_limit::Float64
    search_trees::K
    dist::Vector{Float64}
    idx::Vector{Int}
    maxdist::Float64
end
function AttractorsViaProximity(ds::DynamicalSystem, attractors::Dict; 
        ε=1e-3, Δt=1, Ttr=100, mx_chk_lost=1000, horizon_limit=1e3, diffeq = NamedTuple(),
        warnε = false
    )
    @assert dimension(ds) == dimension(first(attractors)[2])
    search_trees = Dict(k => KDTree(att.data, Euclidean()) for (k, att) in attractors)

    # TODO: After creating StroboscopicMap, the API will be simpler here...?

    # TODO: For Poincare/Stroboscopic maps, `integrator` should return the objects
    # themselves
    integ = integrator(ds; diffeq)

    mapper = AttractorsViaProximity(
        integ, attractors,
        ε, Δt, eltype(Δt)(Ttr), mx_chk_lost, horizon_limit,
        search_trees, [Inf], [0], 0.0,
    )
    
    # Minimum distance between attractors
    if warnε && length(attractors) > 1
        minε = Inf
        for (k, A) in attractors
            for (m, tree) in search_trees
                k == m && continue
                for p in A # iterate over all points of attractor
                    Neighborhood.NearestNeighbors.knn_point!(
                        tree, p, false, mapper.dist, mapper.idx, Neighborhood.alwaysfalse
                    )
                    mapper.dist[1] < minε && (minε = mapper.dist[1])
                end
            end
        end
        println("Distance given is: $(ε)")
        println("Minimum distance between attractros is: $(minε)")
    end

    return mapper
end

function (mapper::AttractorsViaProximity)(u0)
    reinit!(mapper.integ, u0)
    maxdist = 0.0
    mapper.Ttr > 0 && step!(mapper.integ, mapper.Ttr)
    lost_count = 0
    while lost_count < mapper.mx_chk_lost
        step!(mapper.integ, mapper.Δt)
        lost_count += 1
        u = get_state(mapper.integ)
        for (k, tree) in mapper.search_trees # this is a `Dict`
            Neighborhood.NearestNeighbors.knn_point!(
                tree, u, false, mapper.dist, mapper.idx, Neighborhood.alwaysfalse
            )
            if mapper.dist[1] < mapper.ε
                return k
            elseif maxdist < mapper.dist[1]
                maxdist = mapper.dist[1]
                maxdist > mapper.horizon_limit && return -1
            end
        end
    end
    return -1
end

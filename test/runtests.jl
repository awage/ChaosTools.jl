using ChaosTools
using Test
using ChaosTools.DynamicalSystemsBase, ChaosTools.DelayEmbeddings
using StatsBase
standardize = DelayEmbeddings.standardize
test_value = (val, vmin, vmax) -> @test vmin ≤ val ≤ vmax

# TODO: Make all tests use this `testfile` approach.
defaultname(file) = uppercasefirst(replace(splitext(basename(file))[1], '_' => ' '))
testfile(file, testname=defaultname(file)) = @testset "$testname" begin; include(file); end

@testset "ChaosTools" begin

testfile("basins/clustering_tests.jl", "Clustering tests (analytic)")
testfile("basins/attractor_mapping_tests.jl", "Attractor mappers")
testfile("basins/matching_attractors_tests.jl", "Matching attractors")
testfile("basins/basins_continuation_tests.jl", "Fractions continuation")
include("basins/uncertainty_tests.jl")
include("basins/tipping_points_tests.jl")
include("basins/proximity_deduce_ε_tests.jl")

include("orbitdiagrams/orbitdiagram_tests.jl")
include("orbitdiagrams/poincare_tests.jl")

include("chaosdetection/lyapunov_exponents.jl")
#include("chaosdetection/gali_tests.jl") # TODO: make those faster
include("chaosdetection/partially_predictable_tests.jl")
include("chaosdetection/01test.jl")
#include("chaosdetection/expansionentropy_tests.jl")  # TODO: make those faster

include("period_return/periodicity_tests.jl")
include("period_return/period_tests.jl")
include("period_return/yin_tests.jl")

testfile("rareevents/return_time_tests.jl", "Return times")

include("dimensions/entropydim.jl")
include("dimensions/correlationdim.jl")
testfile("dimensions/higuchi.jl")
include("nlts_tests.jl")
include("dyca_tests.jl")

end

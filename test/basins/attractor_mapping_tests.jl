using ChaosTools
using DynamicalSystemsBase
using Test
using LinearAlgebra
using OrdinaryDiffEq
using Random

@testset "AttractorMappers" begin

@testset "Henon map" begin
    u1 = [0.0, 0.0] # converges to attractor
    u2 = [0, 2.0] # diverges to inf
    ds = Systems.henon(zeros(2); a = 1.4, b = 0.3)
    xg = yg = range(-2.0, 2.0; length=100)
    grid = (xg, yg)
    sampler, = statespace_sampler(Random.MersenneTwister(1234); 
    min_bounds = minimum.(grid), max_bounds = maximum.(grid))
    ics = Dataset([sampler() for i in 1:1000])

    @testset "Recurrences method" begin
        mapper = AttractorsViaRecurrences(ds, grid)
        @test 1  == mapper(u1)
        @test -1 == mapper(u2)

        fs = basin_fractions(mapper, sampler)
        @test 0.1 < fs[1] < 0.9
        @test 0.1 < fs[-1] < 0.9
        @test sum(values(fs)) == 1

        # Deterministic test, should be tested with accuracy
        fs, labels = basin_fractions(mapper, ics)
        @test eltype(labels) == Int
        @test fs[1] == 0.451
        @test fs[-1] == 0.549
        @test sum(values(fs)) == 1
    end

    @tesetset "Proximity method" begin
        A = trajectory(ds, 1000; Ttr = 100)

    end

end
end

using ADMPS
using Test

@testset "ADMPS.jl" begin
    @testset "cuda_patch.jl" begin
        println("cuda_patch tests running...")
        include("cuda_patch.jl")
    end

    @testset "hamiltonianmodels" begin
        println("hamiltonianmodels tests running...")
        include("hamiltonianmodels.jl")
    end

    @testset "exampletensors" begin
        println("exampletensors tests running...")
        include("exampletensors.jl")
    end

    @testset "environment" begin
        println("environment tests running...")
        include("environment.jl")
    end

    @testset "autodiff" begin
        println("autodiff tests running...")
        include("autodiff.jl")
    end

    @testset "variationalmps" begin
        println("variationalmps tests running...")
        include("variationalmps.jl")
    end
end

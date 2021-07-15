using ADMPS
using Test

@testset "ADMPS.jl" begin
    @testset "environment" begin
        println("environment tests running...")
        include("environment.jl")
    end
end

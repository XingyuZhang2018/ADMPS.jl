using OMEinsum
using ADMPS: Grassmann
using TeneT: leftorth
using LinearAlgebra: I
using Random
using Test

@testset "retract!" begin
    Random.seed!(100)
    χ = 5
    d = 2
    x = rand(ComplexF64, χ,d,χ)

    M = Grassmann()
    retract!(M,x)
    @test ein"abc,abd->cd"(x,conj(x)) ≈ I(χ)
end

@testset "project!" begin
    Random.seed!(100)
    χ = 5
    d = 2
    g = rand(ComplexF64, χ,d,χ)
    x = rand(ComplexF64, χ,d,χ)
    M = Grassmann()
    retract!(M,x)
    project_tangent!(M, g, x)
    @test ein"abc,abd->cd"(g, conj(x)) ≈ zeros(χ,χ) atol = 1e-8


    g = rand(ComplexF64, χ,d,χ)
    x = rand(ComplexF64, χ,d,χ)
    M = Grassmann()
    retract!(M,x)
    g = project_tangent(M, g, x)
    @test ein"abc,abd->cd"(g, conj(x)) ≈ zeros(χ,χ) atol = 1e-8
end

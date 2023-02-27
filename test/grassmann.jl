using OMEinsum
using Optim: project_tangent!, retract!
using ADMPS: Grassmann
using TeneT: leftorth
using LinearAlgebra: I

@testset "retract!" begin
    Random.seed!(100)
    χ = 5
    d = 2
    x = rand(ComplexF64, χ,d,χ)

    M = Grassmann()
    Optim.retract!(M,x)
    @test ein"abc,abd->cd"(x,conj(x)) ≈ I(χ)
end

@testset "project!" begin
    Random.seed!(100)
    χ = 5
    d = 2
    g = rand(ComplexF64, χ,d,χ)
    x = rand(ComplexF64, χ,d,χ)

    M = Grassmann()
    Optim.retract!(M,x)
    project_tangent!(M, g, x)
    @test ein"abc,abd->cd"(g, conj(x)) ≈ zeros(χ,χ) atol = 1e-8
end

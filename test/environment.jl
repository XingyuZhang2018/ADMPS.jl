using ADMPS
using ADMPS:leftenv,rightenv,norm_FL,norm_FR
using CUDA
using KrylovKit
using LinearAlgebra
using Random
using Test
using OMEinsum
CUDA.allowscalar(false)

@testset "eigsolve with $atype{$dtype}" for atype in [Array, CuArray], dtype in [Float64]
    Random.seed!(100)
    D,d = 10,2
    A = atype(rand(D,d,D))
    工 = ein"asc,bsd -> abcd"(A,conj(A))
    λLs, Ls, info = eigsolve(L -> ein"ab,abcd -> cd"(L,工), atype(rand(D,D)), 1, :LM)
    λL, L = λLs[1], Ls[1]
    @test imag(λL) ≈ 0
    @test Array(ein"ab,ab -> "(L,L))[] ≈ 1 
    @test λL * L ≈ ein"ab,abcd -> cd"(L,工)

    λRs, Rs, info = eigsolve(R -> ein"abcd,cd -> ab"(工,R), atype(rand(D,D)), 1, :LM)
    λR, R = λRs[1], Rs[1]
    @test imag(λR) ≈ 0
    @test Array(ein"ab,ab -> "(R,R))[] ≈ 1 
    @test λR * R ≈ ein"abcd,cd -> ab"(工,R)
    @test λL ≈ λR
end

@testset "leftenv and rightenv with $atype{$dtype}" for atype in [Array, CuArray], dtype in [Float64]
    Random.seed!(100)
    d = 2
    D = 10

    β = rand(dtype)
    Au = atype(rand(dtype,D,d,D))
    Ad = atype(rand(dtype,D,d,D))
    M = atype(model_tensor(Ising(β)))
    
    λL,FL = leftenv(Au, Ad, M)
    @test λL * FL ≈ ein"((adf,abc),dgeb),fgh -> ceh"(FL,Au,M,Ad)

    λR,FR = rightenv(Au, Ad, M)
    @test λR * FR ≈ ein"((abc,ceh),dgeb),fgh -> adf"(Au,FR,M,Ad)
end

@testset "normalization leftenv and rightenv with $atype{$dtype}" for atype in [Array, CuArray], dtype in [Float64]
    Random.seed!(100)
    d = 2
    D = 10

    Au = atype(rand(dtype,D,d,D))
    Ad = atype(rand(dtype,D,d,D))

    λL,FL = norm_FL(Au, Ad)
    @test λL * FL ≈ ein"(ad,acb), dce -> be"(FL,Au,Ad)
    λR,FR = norm_FR(Au, Ad)
    @test λR * FR ≈ ein"(be,acb), dce -> ad"(FR,Au,Ad)
end
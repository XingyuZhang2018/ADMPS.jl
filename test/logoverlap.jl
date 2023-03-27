using ADMPS, Random, LinearAlgebra, Zygote, OMEinsum, ChainRulesCore, Test
atype = Array

using ADMPS:factory_logoverlap, factory_compress_fidelity
@testset "logoverlap factory with $atype" for atype in [Array]
    Random.seed!(101)
    D,χ = 2,5

    M = rand(2,2,2,2)
    Au = Matrix(qr!(rand(ComplexF64,χ*D,χ)).Q)
    Ad = Matrix(qr!(rand(ComplexF64,χ*D,χ)).Q)

    logoverlap = factory_logoverlap(χ,D,atype)
    compress_fidelity = factory_compress_fidelity(χ,D,atype)
    @test logoverlap(Au,Ad,M) != 0.0 # Just test interface
    @test compress_fidelity(Au,Ad,M) != 0.0 # Just test interface
end
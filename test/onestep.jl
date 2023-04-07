using ADMPS, Random, LinearAlgebra, Test
atype = Array
dtype = ComplexF64

using ADMPS: factory_onestep
@testset "gradient with $atype" for atype in [Array]
    Random.seed!(101)
    D,χ = 2,16

    M = rand(2,2,2,2)
    Au = Matrix(qr!(rand(ComplexF64,χ*D,χ)).Q)
    Ad = Matrix(qr!(rand(ComplexF64,χ*D,χ)).Q)

    onestep = factory_onestep(χ,D,atype)
    Ad = onestep(M, Au, Ad)
    @warn "The compress fidelity should near zero"
end
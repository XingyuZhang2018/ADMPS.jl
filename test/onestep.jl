using ADMPS, Random, LinearAlgebra, Test
atype = Array
dtype = ComplexF64
using ADMPS: onestep


@testset "gradient with $atype{$dtype}" for atype in [Array], dtype in [ComplexF64]
    Random.seed!(101)
    D,χ = 2,5

    M = rand(2,2,2,2)
    Au = Matrix(qr!(rand(dtype,χ*D,χ)).Q)
    Ad = Matrix(qr!(rand(dtype,χ*D,χ)).Q)

    Ad = onestep(M, Au, Ad)
    @warn "The compress fidelity should near zero"
end
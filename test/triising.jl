# EXACT: 1.3813564717043518
using ADMPS, Random, LinearAlgebra, Test
atype = Array
dtype = ComplexF64

@testset "Solve Dimer Covering with $atype{$dtype}" for atype in [Array], dtype in [ComplexF64]
    Random.seed!(101)
    D,χ = 2,32

    M = zeros(ComplexF64,(2,2,2,2))
    M[2,1,1,1]=1.0
    M[1,2,1,1]=1.0
    M[2,2,1,1]=1.0
    M[1,2,2,2]=1.0
    M[2,1,2,2]=1.0
    M[1,1,2,2]=1.0

    Au = random_mps(χ,D)
    Ad = random_mps(χ,D)

    Au, Ad = optimizemps(Au, Ad, M)
end
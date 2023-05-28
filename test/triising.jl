# EXACT: 1.3813564717043518 0.3230659669
using ADMPS, Random, LinearAlgebra, Test, HDF5
atype = Array
dtype = ComplexF64

@testset "Solve AFM-Triangular Ising model with $atype" for atype in [Array]
    Random.seed!(103)
    D,χ = 2,64

    M = zeros(ComplexF64,(2,2,2,2))
    M[2,1,1,1]=1.0
    M[1,2,1,1]=1.0
    M[2,2,1,1]=1.0
    M[1,2,2,2]=1.0
    M[2,1,2,2]=1.0
    M[1,1,2,2]=1.0

    # M = Complex.(rand(D,D,D,D))

    Au = random_mps(χ,D)
    Ad = random_mps(χ,D)

    Au, Ad = optimizemps(Au, Ad, M, verbosity=-1,poweriter=2000)


    # h5write("A.h5","Au",Au,"Ad",Ad)
end
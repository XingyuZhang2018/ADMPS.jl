# EXACT: 1.3385151519348453
using ADMPS, Random, LinearAlgebra, Test, OptimKit, CUDA
atype = CuArray
dtype = ComplexF64

@testset "Test naive Ax0 solver" for atype in [Array]
    Random.seed!(102)
    D,χ = 2,64 # Too large chi will cause "f<0 as krylov solver fail for extremly small n1"

    M = zeros(ComplexF64,(2,2,2,2))
    M[1,1,1,2] = 1.0
    M[1,1,2,1] = 1.0
    M[1,2,1,1] = 1.0
    M[2,1,1,1] = 1.0

    A = random_mps(χ,D)
    
    Mnew = zeros(ComplexF64,(3,2,3,2))
    Mnew[1:2,:,1:2,:] .= M
    Mnew[3,:,3,:] .= -1.3385151519348453 * Matrix(I,2,2)
    MM = reshape(permutedims(Mnew,(1,4,3,2)),(:,2))



end

@testset "Test solve Ax0 for dimer covering" for atype in [CuArray]
    Random.seed!(102)
    D,χ = 2,64 # Too large chi will cause "f<0 as krylov solver fail for extremly small n1"

    M = zeros(ComplexF64,(2,2,2,2))
    M[1,1,1,2] = 1.0
    M[1,1,2,1] = 1.0
    M[1,2,1,1] = 1.0
    M[2,1,1,1] = 1.0

    A = random_mps(χ,D)

    A = solveAx0(atype(M), atype(A), 1.3385151519348453, atype;alg=ConjugateGradient(;maxiter=2000, gradtol=1e-10, verbosity=2))
end

# EXACT: 2.533737289 0.9296954
using ADMPS, Random, LinearAlgebra, Test, OMEinsum, CUDA
atype = Array
dtype = ComplexF64

function tensor(β)
    a = reshape(ComplexF64[1 0 0 0; 0 0 0 0; 0 0 0 0; 0 0 0 1], 2,2,2,2)
    cβ, sβ = sqrt(cosh(β)), sqrt(sinh(β))
    q = 1/sqrt(2) * [cβ+sβ cβ-sβ; cβ-sβ cβ+sβ]
    M = ein"abcd,ai,bj,ck,dl -> ijkl"(a,q,q,q,q)
    return M
end

# @testset "Solve Dimer Covering with $atype (Keep Up-Dn symmetry)" for atype in [Array]
    # Random.seed!(104)
    # D,χ = 2,16
    # βc = log(1+sqrt(2))/2
    # β = 0.97*βc

    # Au = random_mps(χ,D)

    # Au, Ad = optimizemps(Au, deepcopy(Au), tensor(β)) # result: log(Z)= 0.9296615817523873
# end


@testset "Solve Ising model with $atype (abandon Up-Dn symmetry)" for atype in [CuArray]
    Random.seed!(103)
    D,χ = 2,64
    βc = log(1+sqrt(2))/2
    β = 1.0 * βc

    Au = random_mps(χ,D;atype=atype)
    Ad = random_mps(χ,D;atype=atype)
    # Ad = Array(qr(Au+0.1*Ad).Q)

    # seed 104 40 steps :
    # log(Z)= 0.9296718151859314
    # AuAd overlap = 0.9999837498761848

    # seed 105 80 steps : coverge to AuAd overlap = 0.9791574944853408 logZ=0.929990697945239
    Au, Ad = optimizemps(Au, Ad, atype(tensor(β)),verbosity=-1,poweriter=2000)
end
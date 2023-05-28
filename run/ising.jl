# EXACT: 2.533737289 0.92969540209
using ADMPS, Random, LinearAlgebra, OMEinsum, CUDA
atype = Array
dtype = ComplexF64

function tensor(β)
    a = reshape(ComplexF64[1 0 0 0; 0 0 0 0; 0 0 0 0; 0 0 0 1], 2,2,2,2)
    cβ, sβ = sqrt(cosh(β)), sqrt(sinh(β))
    q = 1/sqrt(2) * [cβ+sβ cβ-sβ; cβ-sβ cβ+sβ]
    M = ein"abcd,ai,bj,ck,dl -> ijkl"(a,q,q,q,q)
    return M
end

if ARGS[2] == "gpu"
    atype = CuArray
else
    atype = Array
end

Random.seed!(105)
D,χ = 2,parse(Int,ARGS[1])
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
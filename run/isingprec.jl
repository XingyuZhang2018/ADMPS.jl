# EXACT: 1.3813564717043518 0.3230659669
using ADMPS, Random, LinearAlgebra, OMEinsum, CUDA
atype = Array
dtype = ComplexF64

if ARGS[2] == "gpu"
    atype = CuArray
else
    atype = Array
end

Random.seed!(109)
D,χ = 2,parse(Int,ARGS[1])

Au = random_mps(χ,D;atype=atype)
Ad = random_mps(χ,D;atype=atype)


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

b = 0.75
V  = exp( b.* [0 1;1 0])
VI = exp(-b.* [0 1;1 0])
M = tensor(β)
VMV = ein"ijkl,pj,ql->ipkq"(M,V,VI)

Au, Ad = optimizemps(Au, Ad, atype(VMV),verbosity=-1,poweriter=2000)
# Au, Ad = optimizemps(Au, Ad, atype(VMV),verbosity=-1,poweriter=2000,savefile="/data/yangqi/ADMPS/isingprec/chi$(χ).h5")
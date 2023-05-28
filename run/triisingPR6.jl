# EXACT: 1.3813564717043518 0.3230659669
using ADMPS, Random, LinearAlgebra, OMEinsum, CUDA
atype = Array
dtype = ComplexF64

if ARGS[2] == "gpu"
    atype = CuArray
else
    atype = Array
end

R = 6
r = 1/10^R

Random.seed!(109)
D,χ = 2,parse(Int,ARGS[1])

Au = random_mps(χ,D;atype=atype)
Ad = random_mps(χ,D;atype=atype)

M = zeros(ComplexF64,(2,2,2,2))
M[2,1,1,2]=1.0
M[1,1,1,1]=1.0
M[2,1,1,1]=1.0
M[1,2,2,1]=1.0
M[2,2,2,2]=1.0
M[1,2,2,2]=1.0
M = M.+rand(2,2,2,2).*r

run(`mkdir -p /data/yangqi/ADMPS/triisingPR$(R)`) 
Au, Ad = optimizemps(Au, Ad, atype(M),verbosity=-1,poweriter=2000,savefile="/data/yangqi/ADMPS/triisingPR$(R)/chi$(χ).h5")
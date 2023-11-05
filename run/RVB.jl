# EXACT: 1.3813564717043518 0.3230659669
using ADMPS, Random, LinearAlgebra, OMEinsum, CUDA
atype = Array
dtype = ComplexF64

r = 1E-3

Random.seed!(109)
D,χ = 7,8

Au = random_mps(χ,D;atype=atype)
Ad = random_mps(χ,D;atype=atype)

A = [0 1.0 0;-1 0 0; 0 0 1]
P = zeros(2,3,3,3,3)
P[1,1,3,3,3] = 1.0
P[2,2,3,3,3] = 1.0

P[1,3,1,3,3] = 1.0
P[2,3,2,3,3] = 1.0

P[1,3,3,1,3] = 1.0
P[2,3,3,2,3] = 1.0

P[1,3,3,3,1] = 1.0
P[2,3,3,3,2] = 1.0

PA = ein"abcde,id,je->abcij"(P,A,A)
# PA = ein"abcde,pb,qc,di,ej->apqij"(P,sqrt(A),sqrt(A),sqrt(A),sqrt(A))
M = reshape(ein"abcij,apqmn->bpcqimjn"(PA,PA),(9,9,9,9,1,1))

l = [1,3,5,6,7,8,9]
M = M[l,l,l,l]

run(`mkdir -p /data/yangqi/ADMPS/tmp`)
Au, Ad = optimizemps(Au, Ad, atype(M),verbosity=-1,poweriter=2000,savefile="/data/yangqi/ADMPS/tmp/chi$(χ).h5")
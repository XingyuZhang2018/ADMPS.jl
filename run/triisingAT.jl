# EXACT: 1.3813564717043518 0.3230659669
using ADMPS, Random, LinearAlgebra, OMEinsum, CUDA
atype = Array
dtype = ComplexF64

if ARGS[2] == "gpu"
    atype = CuArray
else
    atype = Array
end

Random.seed!(105)
D,χ = 4,parse(Int,ARGS[1])

Au = random_mps(χ,D;atype=atype)
Ad = random_mps(χ,D;atype=atype)

M = zeros(ComplexF64,(2,2,2,2))
M[2,1,1,1]=1.0
M[1,1,1,2]=1.0
M[2,1,1,2]=1.0
M[1,2,2,2]=1.0
M[2,2,2,1]=1.0
M[1,2,2,1]=1.0

function to4(A)
    D = size(A,1)
    A4 = ein"abcd,cefg,hijb,jkle->ahikfldg"(A,A,A,A)
    reshape(A4,(D^2,D^2,D^2,D^2))
end

M = to4(M + 0.001*permutedims(M,(1,4,3,2)))

run(`mkdir -p /data/yangqi/ADMPS/triisingAT`)
Au, Ad = optimizemps(Au, Ad, atype(M),verbosity=-1,poweriter=2000,savefile="/data/yangqi/ADMPS/triisingAT/chi$(χ).h5")
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
D,χ = 4,parse(Int,ARGS[1])

Au = random_mps(χ,D;atype=atype)
Ad = random_mps(χ,D;atype=atype)

Ma = zeros(ComplexF64,(2,2,2,2))
Ma[2,2,1,1]=1.0
Ma[1,1,2,2]=1.0
Ma[1,2,1,2]=1.0
Ma[2,2,1,2]=1.0
Ma[2,1,2,1]=1.0
Ma[1,1,2,1]=1.0

Mb = zeros(ComplexF64,(2,2,2,2))
Mb[2,1,1,2]=1.0
Mb[1,2,2,1]=1.0
Mb[1,1,1,1]=1.0
Mb[2,1,1,1]=1.0
Mb[2,2,2,2]=1.0
Mb[1,2,2,2]=1.0

function to4(A1,A2)
    D = size(A1,1)
    A4 = ein"abcd,cefg,hijb,jkle->ahikfldg"(A1,A1,A2,A2)
    reshape(A4,(D^2,D^2,D^2,D^2))
end

run(`mkdir -p /data/yangqi/ADMPS/triisingto4`) 
Au, Ad = optimizemps(Au, Ad, atype(to4(Ma,Mb)),verbosity=-1,poweriter=2000,savefile="/data/yangqi/ADMPS/triisingto4/chi$(χ).h5")
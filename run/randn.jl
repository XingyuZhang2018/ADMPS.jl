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
D,χ = 2,parse(Int,ARGS[1])

M = ComplexF64.(randn(Float64,(2,2,2,2)))
# M = rand(ComplexF64,(2,2,2,2))

Au = random_mps(χ,D;atype=atype)
Ad = random_mps(χ,D;atype=atype)


Au, Ad = optimizemps(Au, Ad, atype(M),verbosity=-1,poweriter=2000,savefile="/data/yangqi/ADMPS/randn105/chi$(χ).h5")
# EXACT: 1.3813564717043518 0.3230659669
using ADMPS, Random, LinearAlgebra, OMEinsum, CUDA, HDF5
atype = Array
dtype = ComplexF64

Random.seed!(109)
D,χ = 2,12
loadχ = 8

Aui = h5read("/data/yangqi/ADMPS/triising/chi8.h5", "Au50")
Adi = h5read("/data/yangqi/ADMPS/triising/chi8.h5", "Ad50")
include("./src/leftorth.jl")

Aui = reshape(Aui,(loadχ,D,loadχ))
Adi = reshape(Adi,(loadχ,D,loadχ))

Au = zeros(ComplexF64,(χ,D,χ))
Ad = zeros(ComplexF64,(χ,D,χ))

Au[1:loadχ,1:D,1:loadχ] .= Aui
Ad[1:loadχ,1:D,1:loadχ] .= Adi

Au = Au .+ rand(size(Au)).*1e-7
Ad = Ad .+ rand(size(Ad)).*1e-7

Au = leftorth(Au)[1]
Ad = leftorth(Ad)[1]

Au = reshape(Au,(χ*D,χ))
Ad = reshape(Ad,(χ*D,χ))


M = zeros(ComplexF64,(2,2,2,2))
M[2,1,1,1]=1.0
M[1,2,1,1]=1.0
M[2,2,1,1]=1.0
M[1,2,2,2]=1.0
M[2,1,2,2]=1.0
M[1,1,2,2]=1.0

Au, Ad = optimizemps(Au, Ad, atype(M),verbosity=-1,poweriter=400,savefile="/data/yangqi/ADMPS/loadchi/chi12.h5")
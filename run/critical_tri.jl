using HDF5
using ADMPS
using KrylovKit
using ADMPS: norm_FL, factory_logoverlap

# EXACT: 1.3813564717043518 0.3230659669
using ADMPS, Random, LinearAlgebra, OMEinsum, CUDA
atype = Array
dtype = ComplexF64


Random.seed!(109)
D,χ = 2,12
file = "/data/yangqi/ADMPS/triisingPR9/chi$(χ).h5"
logovlp = factory_logoverlap(χ,D)
M = zeros(ComplexF64,(2,2,2,2))
M[2,1,1,1]=1.0
M[1,1,1,2]=1.0
M[2,1,1,2]=1.0
M[1,2,2,2]=1.0
M[2,2,2,1]=1.0
M[1,2,2,1]=1.0

for i in 50:55
    Au24 = h5read(file, "Au$(i)")
    Ad24 = h5read(file, "Ad$(i)")
    Au25 = h5read(file, "Au$(i+1)")
    Ad25 = h5read(file, "Ad$(i+1)")

    tt(x)=reshape(x,(χ,D,χ))

    @show norm_FL(tt(Au24),tt(Au25))[1]
    @show norm_FL(tt(Ad24),tt(Ad25))[1]
    @show log(abs(norm_FL(tt(Au24),tt(Ad24))[1]))
    @show log(abs(norm_FL(tt(Au25),tt(Ad25))[1]))

    @show -logovlp(Au24,Ad24,M)[1]
    @show -logovlp(Au25,Ad25,M)[1]

    @show -logovlp(Au24,Ad24,M)[1]-log(abs(norm_FL(tt(Au24),tt(Ad24))[1]))
    @show -logovlp(Au25,Ad25,M)[1]-log(abs(norm_FL(tt(Au25),tt(Ad25))[1]))

    function LMmap(Au,Ad,M,λ=0.0)
        function f(FL)
            FL = ein"((adf,abc),dgeb),fgh -> ceh"(FL,conj(Au),M,Ad) + λ * FL
            return FL
        end
    end

    λ24,FL24 = eigsolve(LMmap(tt(Au24),tt(Ad24),M), Array(rand(eltype(Au24), χ,2,χ)), 6, :LM; ishermitian = false)
    λ25,FL25 = eigsolve(LMmap(tt(Au25),tt(Ad25),M), Array(rand(eltype(Au24), χ,2,χ)), 6, :LM; ishermitian = false)
    covm = zeros(ComplexF64,(6,6))
    for i in 1:6
        for j in 1:6
            covm[i,j] = abs(ein"abc,abc -> "(conj(FL24[i]),FL25[j])[])
        end
    end

    λn24,FLn24 = eigsolve(FL -> ein"(ad,acb),dce -> be"(FL,conj(tt(Au24)),tt(Ad24))
    , Array(rand(eltype(Au24), χ,χ)), 6, :LM; ishermitian = false)
    λn25,FLn25 = eigsolve(FL -> ein"(ad,acb),dce -> be"(FL,conj(tt(Au25)),tt(Ad25))
    , Array(rand(eltype(Au24), χ,χ)), 6, :LM; ishermitian = false)
    covmn = zeros(ComplexF64,(6,6))
    for i in 1:6
        for j in 1:6
            covmn[i,j] = abs(ein"ab,ab-> "(conj(FLn24[i]),FLn25[j])[])
        end
    end
    @show covmn
    # Au, Ad = optimizemps(Au, Ad, atype(M),verbosity=-1,poweriter=2000,savefile="/data/yangqi/ADMPS/triising/sf25chi$(χ).h5")
end
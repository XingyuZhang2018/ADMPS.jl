using HDF5
using ADMPS
using OMEinsum,KrylovKit
using ADMPS: norm_FL
using LinearAlgebra,Random

D,χ=2,12
file = "/data/yangqi/ADMPS/triisingPR6/chi$(χ).h5"
tt(x)=reshape(x,(χ,D,χ))
M = zeros(ComplexF64,(2,2,2,2))
M[2,1,1,1]=1.0
M[1,1,1,2]=1.0
M[2,1,1,2]=1.0
M[1,2,2,2]=1.0
M[2,2,2,1]=1.0
M[1,2,2,1]=1.0

# Random.seed!(109)
# M = ComplexF64.(randn(Float64,(2,2,2,2)))


# include("./src/leftorth.jl")

for i in 50:120
    OAu = ein"dbeg,agc->adbce"(M,tt(h5read(file, "Au$(i)"))) |> x->reshape(x, (χ*D,D,D*χ))
    ALu, Cu = leftorth(OAu)
    OAd = ein"dgeb,agc->adbce"(M,tt(h5read(file, "Ad$(i)"))) |> x->reshape(x, (χ*D,D,D*χ))
    ALd, Cd = leftorth(OAd)
    @show opnorm(Cu)*opnorm(Cu^(-1))
    @show opnorm(Cd)*opnorm(Cd^(-1))
    # @show norm(Cu[1:χ,χ+1:2χ]),norm(Cd[1:χ,χ+1:2χ])
end
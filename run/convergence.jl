using HDF5
using ADMPS
using OMEinsum,KrylovKit
using ADMPS: norm_FL
using LinearAlgebra

D,χ=2,16
file = "/data/yangqi/ADMPS/randn109/chi$(χ).h5"
tt(x)=reshape(x,(χ,D,χ))


for i in 1:200
    Aui = h5read(file, "Au$(i)")
    Adi = h5read(file, "Ad$(i)")
    Auip = h5read(file, "Au$(i+1)")

    # print("iter $(i):\n")
    # print("Convergence",abs(norm_FL(tt(Aui),tt(Auip))[1]),'\n')

    λn,FLn = eigsolve(FL -> ein"(ad,acb),dce -> be"(FL,conj(tt(Adi)),tt(Adi))
    , Array(rand(eltype(Aui), χ,χ)), 6, :LM; ishermitian = false,krylovdim=200)
    # print("Cor-length:")
    print(1/log(abs(λn[3]/λn[4])),'\n')


    # non-normality=====
    # dm = reshape(ein"ijk,ajb->iakb"(reshape(conj(Aui),χ,D,χ),reshape(Aui,χ,D,χ)),(χ^2,χ^2))
    
    # w,V = eigen(dm)
    # # w,V = eigen(randn(χ^2,χ^2))
    
    # U,S,V = svd(V)
    # print(S[1]/S[end],'\n')
    # =====
end
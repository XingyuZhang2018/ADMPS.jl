using HDF5
using ADMPS
using OMEinsum,KrylovKit
using ADMPS: norm_FL
using LinearAlgebra

D,χ=2,16
file = "/data/yangqi/ADMPS/triising/chi$(χ).h5"
tt(x)=reshape(x,(χ,D,χ))

function measure(A)
    l = Matrix{Float64}(LinearAlgebra.I,χ,χ)
    T0 = reshape(ein"abc,dfe,bf->adce"(conj(tt(A)),tt(A),[1.0 0;0 0]),(χ^2,χ^2))
    T1 = reshape(ein"abc,dfe,bf->adce"(conj(tt(A)),tt(A),[0.0 0;0 1.0]),(χ^2,χ^2))
    λ,r = eigsolve(FR->ein"abc,dbe,ce->ad"(conj(tt(A)),tt(A),FR),Array(rand(eltype(A),χ,χ)),2,:LM; ishermitian = false,krylovdim=200)
    
    return 1/log(abs(λ[1]/λ[2]))
end

for i in 1:300
    Aui = h5read(file, "Au$(i)")
    Adi = h5read(file, "Ad$(i)")

    print(i,'\t',measure(Aui),'\t',measure(Adi),'\n')
end
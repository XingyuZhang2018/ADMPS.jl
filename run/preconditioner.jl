using LinearAlgebra, KrylovKit, OMEinsum

M = zeros(ComplexF64,(2,2,2,2))
M[2,1,1,1]=1.0
M[1,2,1,1]=1.0
M[2,2,1,1]=1.0
M[1,2,2,2]=1.0
M[2,1,2,2]=1.0
M[1,1,2,2]=1.0


function onorm2(X)
    return maximum(abs.(eigen(X).values))
end

function nonnormality(X)
    X = ein"ijkj->ik"(X)
    w,V = eigen(X)
    return onorm2(V)*onorm2(V^(-1))
end


for i = 1:100
U = randn((2,2)) |> UpperTriangular
UMU = ein"ijkl,ai,kb->ajbl"(M,U,U^(-1))
M2 = reshape(ein"ijkl,albp->iajkbp"(M,M),(4,2,4,2))
UM2 = reshape(ein"ijkl,albp->iajkbp"(UMU,UMU),(4,2,4,2))

    @show nonnormality(M2)
    @show nonnormality(UM2)
    if nonnormality(UM2) < 0.6*nonnormality(M2)
        print("Found")
        print(UMU)
        break
    end
end
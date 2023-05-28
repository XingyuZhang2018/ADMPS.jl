βc = log(1+sqrt(2))/2
β = 1.0*βc

using Random
Random.seed!(107)
D,χ = 2,16
M = zeros(ComplexF64,(2,2,2,2))
M[2,1,1,1]=1.0
M[1,2,1,1]=1.0
M[2,2,1,1]=1.0
M[1,2,2,2]=1.0
M[2,1,2,2]=1.0
M[1,1,2,2]=1.0

using LinearAlgebra,OMEinsum,KrylovKit
A = qr(rand(ComplexF64,(χ*D,χ))).Q |> Array |> x->reshape(x,(χ,D,χ))
Al,Ar,Au,Ad = deepcopy(A),deepcopy(A),deepcopy(A),deepcopy(A)
for i in 1:10
    w1,Ad = eigsolve(Ad->ein"ijkl,aib,ckd,bjc->ald"(M,Al,Ar,Ad), Ad, 1)
    Ad = Ad[1]/norm(Ad[1])
    w2,Au = eigsolve(Au->ein"ijkl,aib,ckd,ald->bjc"(M,Al,Ar,Au), Au, 1)
    Au = Au[1]/norm(Au[1])

    w3,Ar = eigsolve(Ar->ein"ijkl,aib,ald,bjc->dkc"(M,Al,Ar,Ad), Ar, 1)
    Ar = Ar[1]/norm(Ar[1])
    w4,Al = eigsolve(Al->ein"ijkl,dkc,ald,bjc->aib"(M,Al,Ar,Au), Al, 1)
    Al = Al[1]/norm(Al[1])
    w = [w1[1] w2[1] w3[1] w4[1]]
    @show map(abs,w)
end

Au = qr(reshape(Au,(χ*D,χ))).Q |> Array
Au, Ad = optimizemps(Au, deepcopy(Au), tensor(β)) # result: log(Z)= 0.9296615817523873
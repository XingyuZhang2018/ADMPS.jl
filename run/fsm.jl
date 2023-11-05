using LinearAlgebra
using OMEinsum
using Plots

M = zeros(ComplexF64,(2,2,2,2))
# M[2,1,1,2]=0.05
# M[1,2,2,1]=0.05
M[2,1,1,2]=1.0
M[1,2,2,1]=1.0
M[1,1,1,1]=1.0
M[2,1,1,1]=1.0
M[2,2,2,2]=1.0
M[1,2,2,2]=1.0

L,R = ([1.0 1.0],[1.0 1.0])
Iter = 6

function spec(Iter,Lin,Rin)
    L,R = (Lin,Rin) |> y-> map(x->reshape(x,(1,2,1)),y)
    D = 1
    for i in 1:Iter
        D = D*2
        L = reshape(ein"ijk,jplq->iqlkp"(L,M),D,2,D)
    end
    O = reshape(ein"ijk,pjl->ipkl"(L,R),(D,D))
    w,v = eigen(O)
    print(opnorm(v)*opnorm(inv(v)))
    return w./2.0.^Iter
end


function specM(Iter)
    D = 2
    result = deepcopy(M)
    for i in 1:Iter-1
        D = D*2
        result = reshape(ein"aijk,jplq->aiplkq"(result,M),2,D,2,D)
    end
    O = ein"ijik->jk"(result)
    w,v = eigen(O)
    print(opnorm(v)*opnorm(inv(v)))
    return w./2.0.^Iter
end

function plotri(w;label=nothing)
    mw = sort(abs.(w))[end]
    scatter!(real(w)./mw,imag(w)./mw,label=label)
end

plot(sin.(collect(0:0.01:2*pi)),cos.(collect(0:0.01:2*pi)),aspect_ratio=:equal,legend=true,label=nothing,size=(800,800))
plotri(spec(5,[1.0,1.0],[1.0,1.0]);label="L=5,[1.0,1.0];[1.0,1.0]")
plotri(spec(7,[1.0,1.0],[1.0,1.0]);label="L=7,[1.0,1.0];[1.0,1.0]")
plotri(spec(10,[1.0,1.0],[1.0,1.0]);label="L=10,[1.0,1.0];[1.0,1.0]")
savefig("./nonhermitian/spec01.png")

plot(sin.(collect(0:0.01:2*pi)),cos.(collect(0:0.01:2*pi)),aspect_ratio=:equal,legend=true,label=nothing,size=(800,800))
plotri(spec(5,[1.0,0.0],[1.0,0.0]);label="L=5,[1.0,0.0];[1.0,0.0]")
plotri(spec(7,[1.0,0.0],[1.0,0.0]);label="L=7,[1.0,0.0];[1.0,0.0]")
plotri(spec(10,[1.0,0.0],[1.0,0.0]);label="L=10,[1.0,0.0];[1.0,0.0]")
savefig("./nonhermitian/spec02.png")

plot(sin.(collect(0:0.01:2*pi)),cos.(collect(0:0.01:2*pi)),aspect_ratio=:equal,legend=true,label=nothing,size=(800,800))
plotri(specM(5);label="L=5,PBC")
plotri(specM(7);label="L=7,PBC")
plotri(specM(10);label="L=10,PBC")
savefig("./nonhermitian/specPBC.png")
using ADMPS
using LinearAlgebra
using CUDA
using KrylovKit
using ITensors
using Statistics
using OMEinsum
using LogExpFunctions
using ChainRulesCore

function getH(N)
    # Heisenberg model or XY
    sites = siteinds("S=1/2",N)
    os = OpSum()
    for j=1:N-1
    # os += "Sz",j,"Sz",j+1 # comment this line for XY model
        os += 1/2,"S+",j,"S-",j+1
        os += 1/2,"S-",j,"S+",j+1
    end
    H = MPO(os,sites)
    aH = [array(H[i]) for i in 1:N]

    aH[1]   = reshape(aH[1],(1,size(aH[1])...))
    aH[end] = reshape(aH[end],(size(aH[end],1),1,size(aH[end])[2:3]...))

    for i in 1:N
        aH[i] = permutedims(aH[i],(1,3,2,4))
    end
    return aH
end

function prepare_HL(H,LHL)
    HL = H[1]
    HbondDim = size(H[1],3)
    for i in 2:LHL
        HL = reshape(ein"ldru,rgfa->ldgfua"(HL,H[i]),1,2^i,HbondDim,2^i)
    end
    return permutedims(HL[1,:,:,:],(3,2,1))
end

function prepare_HR(H,LHL)
    HR = H[end]
    HbondDim = size(H[1],3)
    for i in 2:LHL
        HR = reshape(ein"ldru,rgfa->ldgfua"(H[i],HR),HbondDim,2^i,1,2^i)
    end
    return permutedims(HR[:,:,1,:],(3,1,2))
end

function getAR(AL)
    λ, ρ = norm_FR(AL,conj(AL),Matrix{eltype(AL)}(I,D,D))
    
    # can use cholesky to obtain C
    # A more stable substitution for cholesky decomposition
    F = svd(real(ρ./tr(ρ)))
    C = Diagonal(sqrt.(F.S)) * F.Vt
    _, C = ADMPS.qrpos(C)

    AR = ein"ij,jkl,lm->ikm"(inv(adjoint(C)),AL,(adjoint(C))) 
    return AR
end

D = 16
beta = 1.0

function effective_hamiltonian_01(FL,FR,Hm,AL,AR)
    h_eff_0 = reshape(ein"ijk,ajb->iakb"(FL,FR),(D^2,D^2))
    FL1 = ein"((adf,abc),dgeb),fgh -> ceh"(FL,AL,Hm,conj(AL))
    FR1 = ein"((ceh,abc),dgeb),fgh -> adf"(FR,AR,Hm,conj(AR))
    h_eff_1 = reshape(ein"ijk,ajb->iakb"(FL1,FR1),(D^2,D^2))
    return h_eff_0, h_eff_1
end

function free_energy_persite(h_eff_0, h_eff_1, beta)
    w0 = eigvals(Hermitian(h_eff_0))
    w1 = eigvals(Hermitian(h_eff_1))
    f0 = -1.0/beta*logsumexp(-beta * w0)
    f1 = -1.0/beta*logsumexp(-beta * w1)
    f_per_site = (f1 - f0)/2
    return f_per_site
end

AL = Array(qr(randn(Float64,D*2,D)).Q)
LHL = Int(log2(D))
H = getH(10)
Hm = eltype(AL).(H[5])
W = size(Hm,1)

function leftenv(Au, Ad, Hm, FL)
    function FLmap(FL)
        FL = ein"((adf,abc),dgeb),fgh -> ceh"(FL,Au,Hm,conj(Ad))
        FL[:,1,:] = FL[:,1,:] - Matrix{eltype(FL)}(I,D,D)*mean(diag(FL[:,1,:]))
        return FL
    end
    
    λs, FLs, info = eigsolve(FLmap, FL, 1, :LM; ishermitian = false)
    return λs[1], FLs[1]
end

function ChainRulesCore.rrule(::typeof(leftenv), Au::AbstractArray{T}, Ad::AbstractArray{T}, M::AbstractArray{T}, FL::AbstractArray{T}; kwargs...) where {T}
    λl, FL = leftenv(Au, Ad, M, FL)
    # @show λl
    function back((dλ, dFL))
        dFL -= Array(ein"abc,abc ->"(conj(FL), dFL))[] * FL
        ξl, info = linsolve(FR -> ein"((abc,ceh),dgeb),fgh -> adf"(Au, FR, M, Ad), conj(dFL), -λl, 1; maxiter = 1)
        # @assert info.converged==1
        # errL = ein"abc,cba ->"(FL, ξl)[]
        # abs(errL) > 1e-1 && throw("FL and ξl aren't orthometric. err = $(errL)")
        dAu = -ein"((adf,fgh),dgeb),ceh -> abc"(FL, Ad, M, ξl) 
        dAd = -ein"((adf,abc),dgeb),ceh -> fgh"(FL, Au, M, ξl)
        dM = -ein"(adf,abc),(fgh,ceh) -> dgeb"(FL, Au, Ad, ξl)
        return NoTangent(), conj(dAu), conj(dAd), conj(dM), NoTangent()...
    end
    return (λl, FL), back
end

function rightenv(Au, Ad, Hm, FR)
    function FRmap(FR)
        FR = ein"((ceh,abc),dgeb),fgh -> adf"(FR,Au,Hm,conj(Ad))
        FR[:,W,:] = FR[:,W,:] - Matrix{eltype(FL)}(I,D,D)*mean(diag(FR[:,W,:]))
        return FR
    end
    
    λs, FRs, info = eigsolve(FRmap, FR, 1, :LM; ishermitian = false)
    return λs[1], FRs[1]
end

function ChainRulesCore.rrule(::typeof(rightenv), Au::AbstractArray{T}, Ad::AbstractArray{T}, M::AbstractArray{T}, FL::AbstractArray{T}; kwargs...) where {T}
    λr, FR = rightenv(Au, Ad, M, FL)
    # @show λl
    function back((dλ, dFR))
        dFR -= Array(ein"abc,abc ->"(conj(FR), dFR))[] * FR
        ξr, info = linsolve(FL -> ein"((abc,adf),dgeb),fgh -> ceh"(Au, FL, M, Ad), conj(dFR), -λr, 1; maxiter = 1)
        # @assert info.converged==1
        # errL = ein"abc,cba ->"(FL, ξl)[]
        # abs(errL) > 1e-1 && throw("FL and ξl aren't orthometric. err = $(errL)")
        dAu = -ein"((ceh,fgh),dgeb),adf -> abc"(FR, Ad, M, ξr)
        dAd = -ein"((ceh,abc),dgeb),adf -> fgh"(FR, Au, M, ξr)
        dM = -ein"(ceh,abc),(fgh,adf) -> dgeb"(FR, Au, Ad, ξr)
        return NoTangent(), conj(dAu), conj(dAd), conj(dM), NoTangent()...
    end
    return (λr, FR), back
end

function loss_kit(H, Hm, LHL, D, beta)
    HL = prepare_HL(H,LHL)
    HR = prepare_HR(H,LHL)

    function loss_eig(AL)
        AL = reshape(AL,(D,2,D))
        AR = getAR(AL)
        
        FL = leftenv(AL,AL,Hm,HL)[2]
        FL = real(FL./FL[1,W,1])

        FR = rightenv(AR,AR,Hm,HR)[2]
        FR = real(FR./FR[1,1,1])
    
        h_eff_0, h_eff_1 = effective_hamiltonian_01(FL,FR,Hm,AL,AR)
        f_per_site = free_energy_persite(h_eff_0, h_eff_1, beta)
        # @show f_per_site
        return f_per_site
    end
    return loss_eig
end

loss = loss_kit(H,Hm,LHL,D,beta)

using Optim
using Zygote

AL = reshape(AL,(D*2,D))
@show loss(AL)
@show gradient(loss,AL)[1]
manifold = Optim.Stiefel()

res = optimize(loss, x->gradient(loss,x)[1], AL, LBFGS(manifold=manifold), Optim.Options(show_trace=true, iterations = 1000); inplace = false)
AL =  Optim.minimizer(res)
print(Optim.minimum(res))
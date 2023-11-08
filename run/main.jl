using ADMPS
using LinearAlgebra
using CUDA
using KrylovKit
using ITensors
using OMEinsum
using LogExpFunctions

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

D = 16
beta = 1.0


AL = Array(qr(randn(Float64,D*2,D)).Q)
LHL = Int(log2(D))
H = getH(10)
Hm = eltype(AL).(H[5])

function loss_kit(H, Hm, LHL, D, beta)
    HL = prepare_HL(H,LHL)
    HR = prepare_HR(H,LHL)
    W = size(Hm,1)
    Hrm, Hlm = copy(Hm), copy(Hm)
    Hrm[:,:,1,:] .= 0.0
    Hlm[4,:,:,:] .= 0.0

    function loss_eig(AL)

        AL = reshape(AL,(D,2,D))

        # AR = reshape(AR,(D,2,D))
        # CR, AR, λR = rightorth(reshape(AL,(D,2,D)))
        # ein"ij,jkl,lm->ikm"(inv(CR),AL,((CR))) ./ AR
        # AL, CL, λL = leftorth(reshape(AL,(D,2,D)))

        λ, ρ = norm_FR(AL,conj(AL),Matrix{eltype(AL)}(I,D,D))
        
        # FRn./tr(FRn) ~ adjoint(CR)*CR
        # Cc = cholesky(Hermitian(FRn/tr(FRn))).U

        # A more stable substitution for cholesky decomposition
        F = svd(real(ρ./tr(ρ)))
        C = Diagonal(sqrt.(F.S)) * F.Vt
        _, C = ADMPS.qrpos(C)

        AR = ein"ij,jkl,lm->ikm"(inv(adjoint(C)),AL,(adjoint(C))) 

        λL, FL = leftenv(AL, conj(AL), Hrm, HL)
        λR, FR = rightenv(AR, conj(AR), Hlm, HR)
        # λR, FR = rightenv(AL, conj(AL), Hlm, HR)
    
        FL = real(FL./FL[1,4,1])
        FR = real(FR./FR[1,1,1])
        # FR = FR./(FR[1,1,1])*FRn[1,1,1]

        h_eff_0 = reshape(ein"ijk,ajb->iakb"(FL,FR),(D^2,D^2))
        h_eff_1 = reshape(ein"ijk,abc,jubv->ivakuc"(FL,FR,Hm),(2 * D^2,2 * D^2))

        w0 = eigvals(Hermitian(h_eff_0))
        w1 = eigvals(Hermitian(h_eff_1))
        f0 = -1/beta*logsumexp(-beta * w0)
        f1 = -1/beta*logsumexp(-beta * w1)
        f_per_site = f1 - f0
        # @show f_per_site

        return f_per_site
    end

    # function loss_power(A)
        AL = reshape(AL,(D,2,D))
        AR = reshape(AR,(D,2,D))
        FLp,FRp = HL,HR
        for i = 1:1000
            FLp = ein"((adf,abc),dgeb),fgh -> ceh"(FLp,AL,Hm,conj(AL))
            FRp = ein"((ceh,abc),dgeb),fgh -> adf"(FRp,AR,Hm,conj(AR))
            
            # if i % 10 == 0
            h_eff_0 = reshape(ein"ijk,ajb->iakb"(FLp,FRp),(D^2,D^2))
            h_eff_1 = reshape(ein"ijk,abc,jubv->ivakuc"(FLp,FRp,Hm),(2 * D^2,2 * D^2))
            
            w0 = eigvals(Hermitian(h_eff_0))
            w1 = eigvals(Hermitian(h_eff_1))
            f0 = -1/beta*logsumexp(-beta * w0)
            f1 = -1/beta*logsumexp(-beta * w1)
            f_per_site = f1 - f0
            @show i,f_per_site, f0
        end
    #             # end
    #     return f_per_site
    # end
    return loss_eig
end

loss = loss_kit(H,Hm,LHL,D,beta)

using Optim
using Zygote

@show loss(AL)
@show gradient(loss,AL)[1]
manifold = Optim.Stiefel()

res = optimize(loss, x->gradient(loss,x)[1], AL, LBFGS(manifold=manifold), Optim.Options(show_trace=true, iterations = 5); inplace = false)
AL =  Optim.minimizer(res)
print(Optim.minimum(res))
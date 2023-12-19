using FileIO
using JLD2
using Optim, LineSearches
using LinearAlgebra: I, norm, tr
using TimerOutputs
using OMEinsum: get_size_dict, optimize_greedy,  MinSpaceDiff
using Zygote
using Parameters

@with_kw mutable struct Params
    atype = Array
    D::Int = 2
    χ1::Int = 5
    χ2::Int = 2
    tol::Real = 1e-10
    f_tol::Real = 1e-6
    opiter::Int = 100
    optimmethod = LBFGS(m = 20)
    verbose= true
    mapsteps = 10
    updown = true
    downfromup = false
    savefile = true
    infolder = "./data/"
    outfolder = "./data/"
end

"""
    logoverlap(Au, Ad, M)
````
    ┌── Au──┐                                 a ────┬──── c
    │   │   │         ┌──  Ad ──┐             │     b     │
    FL─ M ──FR  - 1/2 FL_n │   FR_n           ├─ d ─┼─ e ─┤
    │   │   │         └──  Ad ──┘             │     g     │   
    └── Ad──┘                                 f ────┴──── h 
````
"""
function logoverlap(mps_u, mps_d, M, Params)
    @unpack D, χ1, χ2 = Params
    Au, Mu = toMPSMPO(mps_u; D=D, χ1=χ1, χ2=χ2) 
    Ad, Md = toMPSMPO(mps_d; D=D, χ1=χ1, χ2=χ2)

    Au = reshape(ein"abc,defb->adecf"(Au, Mu), χ1*χ2,D,χ1*χ2)
    Ad = reshape(ein"abc,defb->adecf"(Ad, Md), χ1*χ2,D,χ1*χ2)

    _, FLud = Zygote.@ignore leftenv(Au, conj(Ad), M)
    _, FRud = Zygote.@ignore rightenv(Au, conj(Ad), M)
    _, FLd_n = Zygote.@ignore norm_FL(Ad, conj(Ad))
    _, FRd_n = Zygote.@ignore norm_FR(Ad, conj(Ad))

    nd = ein"(ad,acb),(dce,be) ->"(FLd_n,Ad,conj(Ad),FRd_n)[]/ein"ab,ab ->"(FLd_n,FRd_n)[]
    Ad /= sqrt(nd)
    AuM = ein"(((adf,abc),dgeb),fgh),ceh -> "(FLud,Au,M,conj(Ad),FRud)[]/ein"abc,abc -> "(FLud,FRud)[]
    -log(abs(AuM))
end

"""
````
    a ────┬──── c
    │     b     │
    ├─ d ─┼─ e ─┤
    │     f     │
    ├─ g ─┼─ h ─┤
    │     i     │
    j ────┴──── k
````
"""
function compress_fidelity(mps_u, mps_d, M, Params)
    @unpack D, χ1, χ2 = Params
    Au, Mu = toMPSMPO(mps_u; D=D, χ1=χ1, χ2=χ2) 
    Ad, Md = toMPSMPO(mps_d; D=D, χ1=χ1, χ2=χ2)

    Au = reshape(ein"abc,defb->adecf"(Au, Mu), χ1*χ2,D,χ1*χ2)
    Ad = reshape(ein"abc,defb->adecf"(Ad, Md), χ1*χ2,D,χ1*χ2)

    _, FLd_n = norm_FL(Ad, conj(Ad))
    _, FRd_n = norm_FR(Ad, conj(Ad))

    Md = permutedims(conj(M),(1,4,3,2))
    _, FL4ud = bigleftenv( Au, conj(Au), M, Md)
    _, FR4ud = bigrightenv(Au, conj(Au), M, Md)

    nd = ein"(ad,acb),(dce,be) ->"(FLd_n,Ad,conj(Ad),FRd_n)[]/ein"ab,ab ->"(FLd_n,FRd_n)[]
    Ad /= sqrt(nd)
    nu = ein"((((adgj,abc),dfeb),gihf),jik),cehk -> "(FL4ud,Au,M,Md,conj(Au),FR4ud)[]/ein"abcd,abcd ->"(FL4ud,FR4ud)[]
    Au /= sqrt(nu)

    _, FLud = leftenv(Au, conj(Ad), M)
    _, FRud = rightenv(Au, conj(Ad), M)
    AuM = ein"(((adf,abc),dgeb),fgh),ceh -> "(FLud,Au,M,conj(Ad),FRud)[]/ein"abc,abc -> "(FLud,FRud)[]
    abs2(AuM)
end


function overlap(mps_u, mps_d, Params)
    @unpack D, χ1, χ2 = Params
    Au, Mu = toMPSMPO(mps_u; D=D, χ1=χ1, χ2=χ2) 
    Ad, Md = toMPSMPO(mps_d; D=D, χ1=χ1, χ2=χ2)

    Au = reshape(ein"abc,defb->adecf"(Au, Mu), χ1*χ2,D,χ1*χ2)
    Ad = reshape(ein"abc,defb->adecf"(Ad, Md), χ1*χ2,D,χ1*χ2)

    _, FLu_n = norm_FL(Au, conj(Au))
    _, FRu_n = norm_FR(Au, conj(Au))
    _, FLd_n = norm_FL(Ad, conj(Ad))
    _, FRd_n = norm_FR(Ad, conj(Ad))

    nu = ein"(ad,acb),(dce,be) ->"(FLu_n,Au,conj(Au),FRu_n)[]/ein"ab,ab ->"(FLu_n,FRu_n)[]
    Au /= sqrt(nu)
    nd = ein"(ad,acb),(dce,be) ->"(FLd_n,Ad,conj(Ad),FRd_n)[]/ein"ab,ab ->"(FLd_n,FRd_n)[]
    Ad /= sqrt(nd)

    _, FLud_n = norm_FL(Au, conj(Ad))
    _, FRud_n = norm_FR(Au, conj(Ad))
    abs2(ein"(ad,acb),(dce,be) ->"(FLud_n,Au,conj(Ad),FRud_n)[]/ein"ab,ab ->"(FLud_n,FRud_n)[])
end

function init_mps(Params)
    @unpack D, χ1, χ2, infolder, atype, tol, verbose = Params
    in_chkp_file = infolder*"/D$(D)_χ₁$(χ1)_χ₂$(χ2)_tol$(tol).jld2"
    if isfile(in_chkp_file)
        mps =  atype(load(in_chkp_file)["mps"])
        verbose && println("load mps from $in_chkp_file")
    else
        mps = atype(rand(ComplexF64,χ1*D*χ1+χ2*D*χ2*D))
        verbose && println("random initial mps $in_chkp_file")
    end
    normalize!(mps)
    return mps
end

toMPSMPO(mps; D::Int, χ1::Int, χ2::Int) = reshape(mps[1:χ1*D*χ1], χ1,D,χ1), reshape(mps[χ1*D*χ1+1:end], χ2,D,χ2,D)

function onestep(M::AbstractArray, Params)
    mps = init_mps(Params)
    
    mps_u, mps_d = mps, mps
    to = TimerOutput()
    f(mps_d) = @timeit to "forward" logoverlap(mps_u, mps_d, M, Params)
    ff(mps_d) = logoverlap(mps_u, mps_d, M, Params)
    g(mps_d) = @timeit to "backward" Zygote.gradient(ff,mps_d)[1]
    if Params.verbose 
        message = "time  iter   loss           grad_norm\n"
        printstyled(message; bold=true, color=:blue)
        flush(stdout)
    end
    res = optimize(f, g, 
        mps_d, Params.optimmethod,inplace = false,
        Optim.Options(f_tol=Params.f_tol, iterations=Params.opiter,
        extended_trace=true,
        callback=os->writelog(os, Params)),
        )
    mps_d = Optim.minimizer(res)

    if Params.verbose 
        message = "compress error   = $(-log(compress_fidelity(mps_u, mps_d, M, Params)))\npower error = $(-log(overlap(mps_u, mps_d, Params))) \n"
        printstyled(message; bold=true, color=:green)
        flush(stdout)
    end
    return mps_d
end

"""
    writelog(os::OptimizationState, key=nothing)

return the optimise infomation of each step, including `time` `iteration` `energy` and `g_norm`, saved in `/data/model_D_chi_tol_maxiter.log`. Save the final `ipeps` in file `/data/model_D_chi_tol_maxiter.jid2`
"""
function writelog(os::OptimizationState, Params)
    @unpack outfolder, D, χ1, χ2, tol, savefile, verbose = Params
    message = "$(round(os.metadata["time"],digits=1))    $(os.iteration)    $(round(os.value,digits=8))    $(round(os.g_norm,digits=8))\n"

    if verbose
        printstyled(message; bold=true, color=:blue)
        flush(stdout)
    end

    if savefile 
        !(isdir(outfolder)) && mkpath(outfolder)
        logfile = open(outfolder*"/D$(D)_χ₁$(χ1)_χ₂$(χ2)_tol$(tol).log", "a")
        write(logfile, message)
        close(logfile)
        save(outfolder*"/D$(D)_χ₁$(χ1)_χ₂$(χ2)_tol$(tol).jld2", "mps", Array(os.metadata["x"]))
    end
    return false
end

function optimisemps(M::AbstractArray, Params)
    mps_u = nothing
    mps_d = nothing
    Md = permutedims(M,(1,4,3,2))
    infolder = Params.infolder
    outfolder = Params.outfolder
    for i in 1:Params.mapsteps
        if Params.verbose 
            message = "iter: $i   \n"
            printstyled(message; bold=true, color=:red)
            flush(stdout)
        end
        direction = "↑"
        Params.infolder = infolder*"$(direction)/"
        Params.outfolder = outfolder*"$(direction)/"
        mps_u = onestep(M, Params)
        if Params.updown
            !Params.downfromup && (direction = "↓")
            Params.infolder = infolder*"$(direction)/"
            Params.outfolder = outfolder*"$(direction)/"
            mps_d = onestep(Md, Params)
            if Params.verbose
                message = "AuAd overlap = $(overlap(mps_u, mps_d, Params)) \n"
                printstyled(message; bold=true, color=:red)
                flush(stdout)
            end
        else
            mps_d = mps_u
        end
    end
    return mps_u, mps_d
end
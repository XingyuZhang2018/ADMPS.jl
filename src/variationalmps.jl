using FileIO
using JLD2
using Optim, LineSearches
using LinearAlgebra: I, norm
using TimerOutputs
using OMEinsum: get_size_dict, optimize_greedy,  MinSpaceDiff
using Zygote

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
function logoverlap(Au, Ad, M)
    Ad /= norm(Ad)
    _, FL = leftenv(Au, conj(Ad), M)
    _, FR = rightenv(Au, conj(Ad), M)
    _, FLd_n = norm_FL(Ad, conj(Ad))
    _, FRd_n = norm_FR(Ad, conj(Ad))
    -(log(abs(ein"(((adf,abc),dgeb),fgh),ceh -> "(FL,Au,M,conj(Ad),FR)[]/ein"abc,abc -> "(FL,FR)[])) - 1/2 * (log(abs(ein"(ad,acb),(dce,be) ->"(FLd_n,Ad,conj(Ad),FRd_n)[]/ein"ab,ab ->"(FLd_n,FRd_n)[]))))
end

function init_mps(;infolder = "./data/", D::Int = 2, χ::Int = 5, tol::Real = 1e-10, verbose::Bool = true)
    in_chkp_file = infolder*"/D$(D)_chi$(χ)_tol$(tol).jld2"
    if isfile(in_chkp_file)
        mps = load(in_chkp_file)["mps"]
        verbose && println("load mps from $in_chkp_file")
    else
        mps = rand(ComplexF64,χ,D,χ)
        verbose && println("random initial mps $in_chkp_file")
    end
    mps /= norm(mps)
end

function onestep(M::AbstractArray; atype = Array, infolder = "./data/", outfolder = "./data/", χ::Int = 5, tol::Real = 1e-10, f_tol::Real = 1e-6, opiter::Int = 100, optimmethod = LBFGS(m = 20), verbose= true, savefile = true)
    D = size(M, 1)
    mps = init_mps(infolder = infolder, D = D, χ = χ, tol = tol, verbose = verbose)
    
    Au, Ad = mps, mps
    to = TimerOutput()
    f(Ad) = @timeit to "forward" logoverlap(Au, Ad, M)
    ff(Ad) = logoverlap(Au, Ad, M)
    g(Ad) = @timeit to "backward" Zygote.gradient(ff,atype(Ad))[1]
    if verbose 
        message = "time  iter   loss           grad_norm\n"
        printstyled(message; bold=true, color=:blue)
        flush(stdout)
    end
    res = optimize(f, g, 
        Ad, optimmethod,inplace = false,
        Optim.Options(f_tol=f_tol, iterations=opiter,
        extended_trace=true,
        callback=os->writelog(os, outfolder, D, χ, tol, savefile)),
        )
    Ad = Optim.minimizer(res)
end

"""
    writelog(os::OptimizationState, key=nothing)

return the optimise infomation of each step, including `time` `iteration` `energy` and `g_norm`, saved in `/data/model_D_chi_tol_maxiter.log`. Save the final `ipeps` in file `/data/model_D_chi_tol_maxiter.jid2`
"""
function writelog(os::OptimizationState, outfolder, D, χ, tol, savefile)
    message = "$(round(os.metadata["time"],digits=1))    $(os.iteration)    $(round(os.value,digits=8))    $(round(os.g_norm,digits=8))\n"

    printstyled(message; bold=true, color=:blue)
    flush(stdout)

    if savefile 
        !(isdir(outfolder)) && mkpath(outfolder)
        logfile = open(outfolder*"/D$(D)_chi$(χ)_tol$(tol).log", "a")
        write(logfile, message)
        close(logfile)
        save(outfolder*"/D$(D)_chi$(χ)_tol$(tol).jld2", "mps", os.metadata["x"])
    end
    return false
end

function optimisemps(M::AbstractArray; atype = Array, infolder = "./data/", outfolder = "./data/", χ::Int = 5, tol::Real = 1e-10, f_tol::Real = 1e-6, opiter::Int = 100, optimmethod = LBFGS(m = 20), verbose= true,  mapsteps = 10, updown = true, downfromup = false)
    Au = nothing
    Ad = nothing
    Md = permutedims(M,(1,4,3,2))
    for i in 1:mapsteps
        if verbose 
            message = "iter: $i   \n"
            printstyled(message; bold=true, color=:red)
            flush(stdout)
        end
        direction = "↑"
        Au = onestep(M; atype = atype, infolder = infolder*"$(direction)/", outfolder = outfolder*"$(direction)/", χ = χ, tol = tol, f_tol = f_tol, opiter = opiter, optimmethod = optimmethod, verbose = verbose, savefile = true)
        if updown
            !downfromup && (direction = "↓")
            Ad = onestep(Md; atype = atype, infolder = infolder*"$(direction)/", outfolder = outfolder*"$(direction)/", χ = χ, tol = tol, f_tol = f_tol, opiter = opiter, optimmethod = optimmethod, verbose = verbose, savefile = true)
        else
            Ad = Au
        end
    end
    return Au, Ad
end
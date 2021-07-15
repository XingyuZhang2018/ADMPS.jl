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
    FL─ M ──FR  - 1/2 FL   │   FR             ├─ d ─┼─ e ─┤
    │   │   │         └──  Ad ──┘             │     g     │   
    └── Ad──┘                                 f ────┴──── h 
````
"""
function logoverlap(Au, Ad, M)
    Ad /= norm(Ad)
    _, FL = leftenv(Au, Ad, M)
    _, FR = rightenv(Au, Ad, M)
    _, FLd_n = norm_FL(Ad, Ad)
    _, FRd_n = norm_FR(Ad, Ad)
    -(log(abs(ein"(((adf,abc),dgeb),fgh),ceh -> "(FL,Au,M,Ad,FR)[]/ein"abc,abc -> "(FL,FR)[])) - 1/2 * (log(ein"(ad,acb),(dce,be) ->"(FLd_n,Ad,Ad,FRd_n)[]/ein"ab,ab ->"(FLd_n,FRd_n)[])))
end

function init_mps(model::HamiltonianModel; atype = Array, direction::String, D::Int, χ::Int, tol::Real, maxiter::Int, verbose = true)
    key = (model, atype, direction, D, χ, tol, maxiter)
    folder = "./data/$(model)_$(atype)/"
    mkpath(folder)
    chkp_file = folder*"$(model)_$(atype)_$(direction)_D$(D)_chi$(χ)_tol$(tol)_maxiter$(maxiter).jld2"
    # if isfile(chkp_file)
    #     mps = load(chkp_file)["mps"]
    #     verbose && println("load mps from $chkp_file")
    # else
        mps = rand(χ,D,χ)
        verbose && println("random initial mps $chkp_file")
    # end
    mps /= norm(mps)
    return mps, key
end

function optimisemps(mps, key; f_tol = 1e-6, opiter = 100, maxiter = 20, verbose= false, optimmethod = LBFGS(m = 20)) 
    model, atype, _, _, _, _ = key
    β = model.β
    M = atype(model_tensor(model, β))
    Au, Ad = mps, mps
    to = TimerOutput()
    Ads = Array{atype{Float64,3},1}(undef, maxiter)
    for i = 1:maxiter
        verbose && println("iteration $i")
        f(Ad) = @timeit to "forward" logoverlap(Au, Ad, M)
        ff(Ad) = logoverlap(Au, Ad, M)
        g(Ad) = @timeit to "backward" Zygote.gradient(ff,atype(Ad))[1]
        res = optimize(f, g, 
            Ad, optimmethod,inplace = false,
            Optim.Options(f_tol=f_tol, iterations=opiter,
            extended_trace=true,
            callback=os->writelog(os, key)),
            )
        Au = Optim.minimizer(res)
        Ad = Optim.minimizer(res)
        Ads[i] = Ad
    end
    Ads
end

"""
    writelog(os::OptimizationState, key=nothing)

return the optimise infomation of each step, including `time` `iteration` `energy` and `g_norm`, saved in `/data/model_D_chi_tol_maxiter.log`. Save the final `ipeps` in file `/data/model_D_chi_tol_maxiter.jid2`
"""
function writelog(os::OptimizationState, key=nothing)
    message = "$(round(os.metadata["time"],digits=2))   $(os.iteration)   $(os.value)   $(os.g_norm)\n"

    # printstyled(message; bold=true, color=:red)
    # flush(stdout)

    model, atype, direction, D, χ, tol, maxiter = key
    if !(key === nothing)
        logfile = open("./data/$(model)_$(atype)/$(model)_$(atype)_$(direction)_D$(D)_chi$(χ)_tol$(tol)_maxiter$(maxiter).log", "a")
        write(logfile, message)
        close(logfile)
        save("./data/$(model)_$(atype)/$(model)_$(atype)_$(direction)_D$(D)_chi$(χ)_tol$(tol)_maxiter$(maxiter).jld2", "mps", os.metadata["x"])
    end
    return false
end
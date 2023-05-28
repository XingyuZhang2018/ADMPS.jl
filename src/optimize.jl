using HDF5

"""
    Required Input: (Au, Ad, M)
    Au: Initialized Up MPS Tensor (left-physical-right)
    Ad: Initialized Dn MPS Tensor (left-physical-right)
    M: Bulk Tensor (left-down-right-up)

    Optional Input:
        gradtol: tol of gradients in each power step.
        ptol: tol of overlap between two power steps.
        opiter: max number of iterations in each power step.
        verbosity: 0: no output, 1: output loss and gradient norm, 2: output loss, gradient norm and time.
        poweriter: max number of iterations of power steps.


    alg:
    alg = GradientDescent(;maxiter=maxiter, gradtol=gradtol, verbosity=2)
    alg = ConjugateGradient(;maxiter=maxiter, gradtol=gradtol, verbosity=2)
"""
function optimizemps(Au, Ad, M; 
    gradtol::Real = 1e-12, 
    opiter::Int = 1000,
    verbosity = 2,
    alg = LBFGS(20; maxiter=opiter, gradtol=gradtol, verbosity=verbosity),
    savefile = nothing,
    poweriter = 1000, ptol = 1E-14)

    # Fix array size
    χ,D = size(Au,2), size(M,2)
    ArrayType = _arraytype(Au)

    # M for down mapping
    Md = permutedims(M,(1,4,3,2))

    # create cached environment for up dn measurement
    env = create_cached_one(χ,D,ArrayType)

    # create onestep function from factory
    onestep_up = factory_onestep(χ,D, ArrayType; alg=alg, verbosity=verbosity,shift=0.0)
    onestep_dn = factory_onestep(χ,D, ArrayType; alg=alg, verbosity=verbosity,shift=0.0)

    # functions used after power
    logoverlap = factory_logoverlap(χ,D,ArrayType,env)

    SdlogZ = -logoverlap(Ad, Ad, M)[1]
    SulogZ = -logoverlap(Au, Au, M)[1]
    logZ = -logoverlap(Au, Ad, M)[1]
    ovlp = abs(norm_FL(reshape(Au,(χ,D,χ)), reshape(Ad,(χ,D,χ)),env["FL"])[1])
    message = "0 $(SulogZ) $(SdlogZ)  $(logZ) $(logZ-log(ovlp)) $(ovlp) 1.0 1.0 1.0 1.0\n"
    print(message)

    new_Au = Au
    new_Ad = Ad
    for i in 1:poweriter

        # Print information about power steps
        if verbosity > 0
            message = "\npower iter: $i   \n"
            printstyled(message; bold=true, color=:red)
            flush(stdout)
        end

        new_Au, stepinfo_up = onestep_up(M , Au, new_Au)
        new_Ad, stepinfo_dn = onestep_dn(Md, Ad, new_Ad)
        
        # If power method is converged, stop.
        dn_convergence =  (abs(norm_FL(reshape(new_Ad,(χ,D,χ)), reshape(Ad,(χ,D,χ)),env["FL"])[1]))
        up_convergence = (abs(norm_FL(reshape(new_Au,(χ,D,χ)), reshape(Au,(χ,D,χ)),env["FL"])[1]))
        if -log(dn_convergence) < ptol && -log(up_convergence) < ptol
            @info "Power method converged. Stop."
            return new_Au, new_Ad
        else
            Au, Ad = new_Au, new_Ad
        end

        # Print Log(Z)
        if verbosity > -2
            # Think it deeper and write a graph, you will finally realize this is right
            SdlogZ = -logoverlap(Ad, Ad, M)[1]
            SulogZ = -logoverlap(Au, Au, M)[1]
            logZ = -logoverlap(Au, Ad, M)[1]

            ovlp = abs(norm_FL(reshape(Au,(χ,D,χ)), reshape(Ad,(χ,D,χ)),env["FL"])[1])

            # message = "Ref:$(SlogZ)\n<l|M|r> =$(logZ) \nlog(Z)= $(logZ-log(ovlp)) \nAuAd overlap = $(ovlp) \n\n"
            message = "$(i) $(SulogZ) $(SdlogZ) $(logZ) $(logZ-log(ovlp)) $(ovlp) $(stepinfo_up) $(stepinfo_dn) $(up_convergence) $(dn_convergence)\n"
            print(message)
            # printstyled(message; bold=true, color=:red)
            flush(stdout)
        end

        if savefile !== nothing
            h5open(savefile,isfile(savefile) ? "r+" : "w") do file
                file["Au$(i)"] = Au
                file["Ad$(i)"] = Ad
            end
        end

    end

    return Au, Ad
end
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
    gradtol::Real = 1e-6, 
    opiter::Int = 20, 
    verbosity = 2,
    alg = LBFGS(20; maxiter=opiter, gradtol=gradtol, verbosity=2),
    poweriter = 100, ptol = 1E-11)

    # Fix array size
    χ,D = size(Au,2), size(M,2)
    ArrayType = _arraytype(Au)

    # M for down mapping
    Md = permutedims(M,(1,4,3,2))

    # create cached environment for up dn measurement
    env = create_cached_one(χ,D,ArrayType)

    # create onestep function from factory
    onestep_up = factory_onestep(χ,D, ArrayType; alg=alg, verbosity=verbosity)
    onestep_dn = factory_onestep(χ,D, ArrayType; alg=alg, verbosity=verbosity)

    # functions used after power
    logoverlap = factory_logoverlap(χ,D,ArrayType,env)


    for i in 1:poweriter
        # Print information about power steps
        if verbosity > 0
            message = "\npower iter: $i   \n"
            printstyled(message; bold=true, color=:red)
            flush(stdout)
        end

        new_Au = onestep_up(M , Au, Au)
        new_Ad = onestep_dn(Md, Ad, Ad)
        
        # If power method is converged, stop.
        dn_convergence =  (abs(norm_FL(reshape(new_Ad,(χ,D,χ)), reshape(Ad,(χ,D,χ)),env["FL"])[1]))
        up_convergnce = (abs(norm_FL(reshape(new_Au,(χ,D,χ)), reshape(Au,(χ,D,χ)),env["FL"])[1]))
        if -log(dn_convergence) < ptol && -log(up_convergnce) < ptol
            @info "Power method converged. Stop."
            return new_Au, new_Ad
        else
            Au, Ad = new_Au, new_Ad
        end

        # Print Log(Z)
        if verbosity > 0
            message = "log(Z)= $(-logoverlap(Au, Ad, Md)[1]) \n"
            printstyled(message; bold=true, color=:red)
            flush(stdout)
        end

        # Print information about up down vector overlap
        if verbosity > 1
            message = "AuAd overlap = $(abs(norm_FL(reshape(Au,(χ,D,χ)), reshape(Ad,(χ,D,χ)),env["FL"])[1])) \n"
            printstyled(message; bold=true, color=:red)
            flush(stdout)
        end
    end

    return Au, Ad
end
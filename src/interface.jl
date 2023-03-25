using FileIO
using OptimKit, LineSearches
using LinearAlgebra: I, norm, tr
using Zygote, ChainRulesCore

function obs_env(M,Au,Ad)
    _, FL = leftenv(Au, Ad, M)
    _, FR = rightenv(Au, Ad, M)
    _, FL_n = norm_FL(Au, Ad)
    _, FR_n = norm_FR(Au, Ad)
    return Dict("M"=>M, "Au"=>Au, "Ad"=>Ad, 
        "FLM"=>FL, "FRM"=>FR, "FL"=>FL_n, "FR"=>FR_n)
end

"""
    logoverlap(Au, Ad, M)
````
    ┌── Au──┐     a ────┬──── c
    │   │   │     │     b     │
    FL─ M ──FR    ├─ d ─┼─ e ─┤
    │   │   │     │     g     │   
    └── Ad──┘     f ────┴──── h 
````
return value and gradient to Ad
"""
function logoverlap(Au, Ad, M)
    χ,D = size(Au,2), size(M,2)
    Au = reshape(Au,(χ,D,χ))
    Ad = reshape(Ad,(χ,D,χ))
    
    # No need to compute the gradient of the environment tensors, due to Feynmann Hellmann theorem
    _, FLud = ChainRulesCore.@ignore_derivatives leftenv(Au, Ad, M)
    _, FRud = ChainRulesCore.@ignore_derivatives rightenv(Au, Ad, M)

    G = ein"((adf,abc),dgeb),ceh -> fgh"(FLud,conj(Au),M,FRud)/ein"abc,abc -> "(FLud,FRud)[]
    contract_value = ein"ijk,ijk->"(G,Ad)[]

    value, gradient = -log(abs(contract_value)), conj(-1/contract_value*G)
    return value, ChainRulesCore.@ignore_derivatives projectcomplement!(reshape(gradient,(χ*D,χ)),reshape(Ad,(χ*D,χ)))
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
function compress_fidelity(Au, Ad, M)
    χ,D = size(Au,2), size(M,2)
    Au = reshape(Au,(χ,D,χ))
    Ad = reshape(Ad,(χ,D,χ))

    _, FL4ud = bigleftenv( Au, Au, M)
    _, FR4ud = bigrightenv(Au, Au, M)

    nu = ein"((((adgj,abc),dfeb),gihf),jik),cehk -> "(FL4ud,conj(Au),M,M,Au,FR4ud)[]/ein"abcd,abcd ->"(FL4ud,FR4ud)[]

    _, FLud = leftenv(Au, Ad, M)
    _, FRud = rightenv(Au, Ad, M)
    AuM = ein"(((adf,abc),dgeb),fgh),ceh -> "(FLud,conj(Au),M,Ad,FRud)[]/ein"abc,abc -> "(FLud,FRud)[]
    return abs(AuM/sqrt(nu))
end

function random_mps(χ,D;atype=Array,dtype=ComplexF64)
    return atype(qr(rand(dtype,χ*D,χ)).Q)
end

# Maybe weird for you, but just see OptimKit/linesearch.254
_inner(x, ξ1, ξ2) = real(dot(precondition(x,ξ1),precondition(x,ξ2))) 

_add!(η, ξ, β) = LinearAlgebra.axpy!(β, ξ, η)
_scale!(η, β) = LinearAlgebra.rmul!(η, β)

"""
    Find a good approximation of Ad = M * Au
"""
function onestep(M, Au, Ad; verbosity= 2,
    alg = LBFGS(20; maxiter=10, gradtol=1e-10, verbosity=verbosity)
    )
    fg(Ad) = logoverlap(Au, Ad, M)
    if verbosity > 0
        message = "time  iter   loss           grad_norm\n"
        printstyled(message; bold=true, color=:blue)
        flush(stdout)
    end

    res = optimize(fg, 
        Ad, alg;
        retract = retract,
        precondition = precondition,
        transport! = transport!,
        inner= _inner,
        add! = _add!,
        scale! = _scale!
        )
    Ad, fx, gx, numfg, normgradhistory = res

    if verbosity > 0
        χ,D = size(Au,2), size(M,2)
        message = "compress fidelity   = $(compress_fidelity(Au, Ad, M))\npower convergence = $(abs(norm_FL(reshape(Au,(χ,D,χ)), reshape(Ad,(χ,D,χ)))[1])) \n"
        printstyled(message; bold=true, color=:green)
        flush(stdout)
    end
    return Ad
end

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

    χ,D = size(Au,2), size(M,2)
    # M for down mapping
    Md = permutedims(M,(1,4,3,2))
    
    for i in 1:poweriter
        # Print information about power steps
        if verbosity > 0
            message = "\npower iter: $i   \n"
            printstyled(message; bold=true, color=:red)
            flush(stdout)
        end

        new_Au = onestep(M , Au, Au; alg=alg, verbosity = verbosity)
        new_Ad = onestep(Md, Ad, Ad; alg=alg, verbosity = verbosity)
        
        # If power method is converged, stop.
        dn_convergence =  (abs(norm_FL(reshape(new_Ad,(χ,D,χ)), reshape(Ad,(χ,D,χ)))[1]))
        up_convergnce = (abs(norm_FL(reshape(new_Au,(χ,D,χ)), reshape(Au,(χ,D,χ)))[1]))
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
            message = "AuAd overlap = $(abs(norm_FL(reshape(Au,(χ,D,χ)), reshape(Ad,(χ,D,χ)))[1])) \n"
            printstyled(message; bold=true, color=:red)
            flush(stdout)
        end
    end


    return Au, Ad
end
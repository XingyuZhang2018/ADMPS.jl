# Factory to produce cache-binded functions
using CUDA

function factory_logoverlap(χ::Int,D::Int,ArrayType=Array, cached_env=create_cached_one(χ,D,ArrayType), λ=0.0)
    FML = cached_env["FML"]
    FMR = cached_env["FMR"]

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
        Au = reshape(Au,(χ,D,χ))
        Ad = reshape(Ad,(χ,D,χ))
        
        # No need to compute the gradient of the environment tensors, due to Feynmann Hellmann theorem
        wl, FLud = ChainRulesCore.@ignore_derivatives leftenv(Au, Ad, M, FML, λ)
        wr, FRud = ChainRulesCore.@ignore_derivatives rightenv(Au, Ad, M, FMR, λ)
        # @assert wl ≈ wr atol = 1E-12

        G = @CUDA.allowscalar ein"((adf,abc),dgeb),ceh -> fgh"(FLud,conj(Au),M,FRud)/ein"abc,abc -> "(FLud,FRud)[]
        contract_value = @CUDA.allowscalar ein"ijk,ijk->"(G,Ad)[]

        # T = reshape(ein"(adf,dgeb),ceh -> abcfgh"(FLud,M,FRud)/ein"abc,abc ->"(FLud,FRud)[],(χ^2*D,χ^2*D))
        # w,v = eigsolve(T, 1, :LM; ishermitian = false)
        # @show abs.(w)

        value, gradient = -log(abs(contract_value)), -conj(1/contract_value*G)
        return value, ChainRulesCore.@ignore_derivatives projectcomplement!(reshape(gradient,(χ*D,χ)),reshape(Ad,(χ*D,χ)))
    end

    return logoverlap
end

function factory_compress_fidelity(χ,D,ArrayType=Array,cached_env=create_cached_one(χ,D,ArrayType))
    FMML = cached_env["FMML"]
    FMMR = cached_env["FMMR"]
    FML = cached_env["FML"]
    FMR = cached_env["FMR"]

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
        Au = reshape(Au,(χ,D,χ))
        Ad = reshape(Ad,(χ,D,χ))
        Mu = permutedims(M,(1,4,3,2)) # MPO act on Au

        wl, FL4ud = bigleftenv( Au, Au, Mu, FMML)
        wr, FR4ud = bigrightenv(Au, Au, Mu, FMMR)
        @assert wl ≈ wr atol = 1E-12
        
        nu = @CUDA.allowscalar ein"((((adgj,abc),dbef),gihf),jik),cehk -> "(FL4ud,conj(Au),conj(Mu),Mu,Au,FR4ud)[]/ein"abcd,abcd ->"(FL4ud,FR4ud)[]

        wl, FLud = leftenv(Au, Ad, M, FML)
        wr, FRud = rightenv(Au, Ad, M, FMR)
        @assert wl ≈ wr atol = 1E-12
        
        AuM = @CUDA.allowscalar ein"(((adf,abc),dgeb),fgh),ceh -> "(FLud,conj(Au),M,Ad,FRud)[]/ein"abc,abc -> "(FLud,FRud)[]

        return abs(AuM/sqrt(nu))
    end

    return compress_fidelity
end

function factory_onestep(χ::Int, D::Int, ArrayType; cached_env::Dict =create_cached_one(χ,D,ArrayType),
    verbosity::Int = 2,
    alg = LBFGS(20; maxiter=20, gradtol=1e-10, verbosity=verbosity),
    # alg = GradientDescent(maxiter=20, gradtol=1e-10, verbosity=verbosity),
    shift = 2.0,
    logoverlap::Function = factory_logoverlap(χ,D,ArrayType, cached_env, shift),
    compress_fidelity::Function = factory_compress_fidelity(χ,D,ArrayType,cached_env)
    )
    FL = cached_env["FL"]

    """
    Find a good approximation of Td = M * Tu
    Direction of M: (left-down-right-up) M act on Td
    """
    function onestep(M, Tu, Td)

        fg(Td) = logoverlap(Tu, Td, M) # produced from factory with caches
        res = optimize(fg, 
            Td, alg;
            retract = retract,
            precondition = precondition,
            transport! = transport!,
            inner= _inner,
            add! = _add!,
            scale! = _scale!
            )
        Td, fx, gx, numfg, normgradhistory = res

        if verbosity > 1
            message = "compress fidelity   = $(compress_fidelity(Tu, Td, M))\npower convergence = $(abs(norm_FL(reshape(Tu,(χ,D,χ)), reshape(Td,(χ,D,χ)),FL)[1])) \n"            
            printstyled(message; bold=true, color=:green)
            flush(stdout)
        end
        return Td, compress_fidelity(Tu, Td, M)
    end

    return onestep
end

# solve the MPO-MPS equation: Ax=0

export solveAx0

"""
    Input A: MPS(χ,D,χ)
    Solve Ax=0 return x
"""
function solveAx0(M,A, eigenvalues, atype;
    cached_env::Dict=create_cached_one(size(A,2),size(M,1),atype),
    verbosity=2,
    alg = LBFGS(20; maxiter=10, gradtol=1e-10, verbosity=verbosity),
    ftol=1E-10)

    FL4 = cached_env["FMML"]
    FR4 = cached_env["FMMR"]
    FL3 = cached_env["FML"]
    FR3 = cached_env["FMR"]
    χ,D = size(A,2),size(A,1)÷size(A,2)

    
    function xAAx_fg(A)
        A = reshape(A,(χ,D,χ))

        _, FL4 = bigleftenv(A, A, M, FL4)
        _, FR4 = bigrightenv(A, A, M, FR4)
        nf = @CUDA.allowscalar ein"abcd,abcd->"(FL4,FR4)[]
        @show nf,2

        G = ein"(((adgj,abc),dbef),gihf),cehk -> jik"(FL4,conj(A),conj(M),M,FR4)/nf
        value1 = @CUDA.allowscalar abs(ein"jik,jik->"(G,A)[])
        gradient1 = G
        
        _, FL3 = leftenv(A,A,M,FL3)
        _, FR3 = rightenv(A,A,M,FR3)
        nf = @CUDA.allowscalar ein"abc,abc->"(FL3,FR3)[]
        @show nf,1

        G = ein"((adf,abc),dgeb),ceh -> fgh"(FL3,conj(A),M,FR3)/nf
        value2 = @CUDA.allowscalar ein"ijk,ijk->"(G,A)[]
        gradient2 = G

        # (M-lambda)^2 = M^2 - 2lambda*M + lambda^2
        @show value1, value2, eigenvalues
        value = real(value1 - 2*eigenvalues*value2 + abs2(eigenvalues))
        gradient = 2*conj(gradient1 - 2*conj(eigenvalues)*gradient2)

        return value, projectcomplement!(reshape(gradient,(χ*D,χ)),reshape(A,(χ*D,χ)))
    end

    res = optimize(xAAx_fg, 
    A, alg;
    retract = retract,
    precondition = precondition,
    transport! = transport!,
    inner= _inner,
    add! = _add!,
    scale! = _scale!
    )
    A, fx, gx, numfg, normgradhistory = res
    if fx > ftol
        print("Warn: the solution of Ax=0 is not good enough, fx = $fx\n")
    end
    return reshape(A,(χ*D,χ))  
end
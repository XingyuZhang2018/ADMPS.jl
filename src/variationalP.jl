function normality(十)
    D1,D2 = size(十)[[1,2]]
    十⁺ = permutedims(conj(十),(1,4,3,2))
    十₁ = reshape(ein"abcd,efgb -> aefdcg"(十, 十⁺), D1^2,D2^2,D1^2)
    十₂ = reshape(ein"abcd,efgb -> aefdcg"(十⁺, 十), D1^2,D2^2,D1^2)

    overlap(十₁, 十₂)
end

function bulid_丰(十, p)
    χ1,D = size(十)[[1,2]]
    χ2 = size(p, 1)
    P    = Zygote.Buffer(p, χ2,D,χ2,D)
    Pinv = Zygote.Buffer(p, χ2,D,χ2,D)
    for i in 1:χ2, j in 1:χ2
        P[i,:,j,:]    = exp(p[i,:,j,:])
        Pinv[i,:,j,:] = exp(-p[i,:,j,:])
    end

    reshape(ein"aieh,bjfi,cdgj -> abcdefgh"(copy(P), 十, copy(Pinv)), χ2^2*χ1, D, χ2^2*χ1, D)
end

function optim_P(十::AbstractArray, D::Int, χ2::Int; 
                 verbose::Bool=true,
                 f_tol::Float64=1e-6, 
                 iters::Int=100, 
                 show_every::Int=10)
    p = rand(ComplexF64, χ2,D,χ2,D)
    
    f(p) = -log(normality(bulid_丰(十, p)))
    g(p) = Zygote.gradient(f, p)[1]

    if verbose 
        message = "time  iter   loss           grad_norm\n"
        printstyled(message; bold=true, color=:blue)
        flush(stdout)
    end
    res = optimize(f, g, 
        p, LBFGS(m = 20), inplace = false,
        Optim.Options(f_tol=f_tol, iterations=iters,
        extended_trace=true,
        callback=os->writelog(os, iters, show_every, verbose)),
        )

    Optim.minimizer(res)
end

function writelog(os::OptimizationState, iters, show_every, verbose)
    message = "$(round(os.metadata["time"],digits=1))    $(os.iteration)    $(round(os.value,digits=8))    $(round(os.g_norm,digits=8))\n"

    if verbose && (iters % show_every == 0)
        printstyled(message; bold=true, color=:blue)
        flush(stdout)
    end

    return false
end

using ADMPS
using ADMPS: num_grad,Zofβ,logoverlap,Z,obs_env,magofβ,eneofβ,overlap,leftenv,rightenv
using CUDA
using KrylovKit
using LinearAlgebra: svd, norm
using LineSearches, Optim
using OMEinsum
using Plots
using Random
using Test
using Zygote

@testset "gradient with $atype{$dtype}" for atype in [Array], dtype in [Float64]
    Random.seed!(0)
    D,χ = 2,5
    β = 0.8
    model = Ising(β)
    M = atype(model_tensor(model, β))
    mps, key = init_mps(model;atype = atype, D=D, χ=χ, tol=1e-10, maxiter=10)
    Au = mps
    Ad = mps
    ff(Ad) = logoverlap(Au, Ad, M)
    # @show logoverlap(Au, mps, M),Zygote.gradient(ff,Ad)
    gradzygote = first(Zygote.gradient(mps) do x
        logoverlap(Au, x, M)
    end)
    gradnum = num_grad(mps, δ=1e-3) do x
        logoverlap(Au, mps, M)
    end
    @test isapprox(gradzygote, gradnum, atol=1e-5)
end

@testset "optimize Ad mps" for atype in [Array], dtype in [Float64]
    seed_number = 10012345
    β = 0.8
    D,χ = 2,10
    maxiter = 30
    model = Ising(β)
    M = atype(model_tensor(model, β))
    Random.seed!(seed_number)
    mps, key = init_mps(model;atype = atype, direction = "up",D=D, χ=χ, tol=1e-10, maxiter=maxiter, verbose = false)
    Aus = optimisemps(mps, key; f_tol = 1e-6, maxiter = maxiter, verbose = true)
    Random.seed!(seed_number + 1)
    mps, key = init_mps(model;atype = atype, direction = "down",D=D, χ=χ, tol=1e-10, maxiter=maxiter, verbose = false)
    Ads = optimisemps(mps, key; f_tol = 1e-6, maxiter = maxiter, verbose = true)
    env = Array{Tuple,1}(undef, maxiter)
    x = []
    yZ = []
    ymag = []
    yene = []
    yZerr = []
    ymagerr = []
    yenerr = []
    overlaps = []
    iter = 1:maxiter
    for i = iter
        env[i] = obs_env(M,Aus[i],Ads[i])
        env_up = obs_env(M,Aus[i],Aus[i])
        env_down = obs_env(M,Ads[i],Ads[i])
        x = [x; i]
        yZ = [yZ; Z(env[i])]
        ymag = [ymag; magnetisation(env[i],Ising(β))]
        yene = [yene; energy(env[i],Ising(β))]
        yZerr = [yZerr; yZ[i]-Zofβ(Ising(β),β)]
        ymagerr = [ymagerr; ymag[i]-magofβ(Ising(β),β)]
        yenerr = [yenerr; yene[i]-eneofβ(Ising(β),β)]
        overlaps = [overlaps; overlap(env[i])]
        println("$(magnetisation(env_up,Ising(β))) | $(magnetisation(env_down,Ising(β))) | $(ymag[i]) | $(overlaps[i])")
    end
    Zplot = plot()
    magnetisationplot = plot()
    energyplot = plot()
    Zerrplot = plot()
    magnetisationerrplot = plot()
    energyerrplot = plot()
    overlapplot = plot()
    plot!(Zplot, x, yZ, seriestype = :scatter, title = "Z", label = "Z", lw = 3)
    plot!(Zplot, x, [Zofβ(Ising(β),β) for _ =iter], title = "Z", label = "β = $(β) exact", lw = 3)
    plot!(magnetisationplot, x, ymag, seriestype = :scatter, title = "magnetisation", label = "magnetisation", lw = 3)
    plot!(magnetisationplot, x, [magofβ(Ising(β),β) for _ =iter], title = "magnetisation", label = "β = $(β) exact", lw = 3)
    plot!(energyplot, x, yene, seriestype = :scatter, title = "energy", label = "energy", lw = 3)
    plot!(energyplot, x, [eneofβ(Ising(β),β) for _ =iter], title = "energy", label = "β = $(β) exact", lw = 3)
    plot!(Zerrplot, x, yZerr, seriestype = :scatter, label = "Z error", ylabel = "error", lw = 3)
    plot!(magnetisationerrplot, x, ymagerr, seriestype = :scatter,  label = "magnetisation error", ylabel = "error", lw = 3)
    plot!(energyerrplot, x, yenerr, seriestype = :scatter, label = "energy error", ylabel = "error", lw = 3)
    plot!(overlapplot, x, overlaps, seriestype = :scatter, label = "overlap", ylabel = "overlap", lw = 3)
    obs = plot(Zplot, Zerrplot, magnetisationplot, magnetisationerrplot, energyplot, energyerrplot, layout = (3,2), xlabel="iterate", size = [1000, 1000])
    lo = @layout [a; b{0.2h}]
    p = plot(obs, overlapplot, layout = lo, xlabel="iterate", size = [1000, 1000])
    savefig(p,"./plot/β$(β)_Ising_Z&O-iterate.svg")
end
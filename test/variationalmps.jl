using ADMPS
using ADMPS: num_grad,Zofβ,logoverlap,Z,obs_env,magofβ,eneofβ,overlap, onestep, isingβc
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
    Random.seed!(100)
    D,χ = 2,5
    β = 0.4
    model = Ising(β)
    M = atype(model_tensor(model))
    mps = init_mps(D = D, χ = χ)
    Au = mps
    Ad = mps
    ff(Ad) = logoverlap(Au, Ad, M)
    # @show logoverlap(Au, mps, M),Zygote.gradient(ff,Ad)
    gradzygote = first(Zygote.gradient(mps) do x
        logoverlap(Au, x, M)
    end)
    gradnum = num_grad(mps, δ=1e-4) do x
        logoverlap(Au, x, M)
    end
    @test gradzygote ≈ gradnum atol=1e-5
end

@testset "onestep optimize Ad" for atype in [Array], dtype in [Float64]
    seed_number = 100
    β = 0.8
    D,χ = 2,10
    maxiter = 5
    infolder, outfolder = "./data/", "./data/"
    model = Ising(β)
    M = atype(model_tensor(model))
    Random.seed!(seed_number)

    Ad = onestep(M; infolder = infolder*"$(model)/", outfolder = outfolder*"$(model)/", D = D, χ = χ, verbose= true, savefile = false)
    @test Ad !== nothing
end

@testset "optimize mps" for atype in [Array], dtype in [Float64]
    seed_number = 100
    β = 0.8
    D,χ = 2,20
    mapsteps = 20
    infolder, outfolder = "./data/", "./data/"

    model = Ising(β)
    M = atype(model_tensor(model))
    Random.seed!(seed_number)

    Ad = optimisemps(M; infolder = infolder*"$(model)/", outfolder = outfolder*"$(model)/", D = D, χ = χ, mapsteps = mapsteps, verbose= true)

    env = obs_env(M,Ad,Ad)
    @test magnetisation(env,model) ≈ magofβ(model) atol=1e-6
    @test energy(env,model) ≈ eneofβ(model) atol=1e-6
end


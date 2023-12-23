using ADMPS
using ADMPS: normality, num_grad,Zofβ,logoverlap,Z,obs_env,magofβ,eneofβ,overlap,onestep,isingβc,init_mps
using CUDA
using KrylovKit
using LinearAlgebra: svd, norm
using LineSearches, Optim
using OMEinsum
using Random
using Test
using Zygote

@testset "gradient with $atype{$dtype}" for atype in [Array], dtype in [ComplexF64]
    Random.seed!(100)
    β = 0.4
    model = Ising(β)
    M = atype{dtype}(model_tensor(model))
    D,χ1,χ2 = 2,5,2
    params = ADMPS.Params(D=D, χ1=χ1, χ2=χ2, optimU=true)
    mps = init_mps(params)
    mps_u = mps
    mps_d = mps
    ff(mps_d) = logoverlap(mps_u, mps_d, M, params)
    gradzygote = first(Zygote.gradient(mps) do x
        logoverlap(mps_u, x, M, params)
    end)
    gradnum = num_grad(mps, δ=1e-4) do x
        logoverlap(mps_u, x, M, params)
    end
    # @test gradzygote ≈ gradnum atol=1e-5
    @test gradzygote[χ1*D*χ1+1:end] ≈ gradnum[χ1*D*χ1+1:end] atol=1e-5
    # @show norm(gradzygote-gradnum)
end

@testset "onestep optimize Ad with $atype{$dtype}" for atype in [Array], dtype in [ComplexF64]
    seed_number = 100
    β = 0.8
    model = Ising(β)
    M = atype{dtype}(model_tensor(model))
    Random.seed!(seed_number)

    Ad = onestep(M, ADMPS.Params(D=2, χ1=5, χ2=2, opiter=5, 
    infolder="./data/$model/", outfolder="./data/$model/",
    optimU=false
    ))
    @test Ad !== nothing
end

@testset "oneside optimize mps" for atype in [Array], dtype in [ComplexF64]
    seed_number = 100
    β = 0.8
    infolder, outfolder = "./data/", "./data/"

    model = Ising(β)
    M = atype{dtype}(model_tensor(model))
    Random.seed!(seed_number)

    params = ADMPS.Params(D=2, χ1=5, χ2=1, 
                          opiter = 10, 
                          f_tol = 1e-15,
                          show_every = 10,
                          mapsteps = 1,
                          optimU = true,
                          infolder=infolder*"$(model)/", 
                          outfolder=outfolder*"$(model)/", 
                          updown=false)
    mps_u, mps_d = optimisemps(M, params)


    env = obs_env(M,mps_u, mps_d, params)
    @test magnetisation(env,model) ≈ magofβ(model) atol=1e-6
    @test energy(env,model) ≈ eneofβ(model) atol=1e-6
    @show energy(env,model)+1.414213779415974 # β = isingβc
end

@testset "oneside optimize mps" for atype in [Array], dtype in [ComplexF64]
    seed_number = 100
    β = 0.8
    infolder, outfolder = "./data/", "./data/"

    model = Ising(β)
    M = atype{dtype}(model_tensor(model))
    Random.seed!(seed_number)

    params = ADMPS.Params(D=2, χ1=5, χ2=2, 
                          opiter=20, 
                          f_tol = 1e-10,
                          mapsteps = 20,
                          infolder=infolder*"$(model)/", 
                          outfolder=outfolder*"$(model)/", 
                          updown=true,
                          downfromup=true)
    mps_u, mps_d = optimisemps(M, params)


    env = obs_env(M,mps_u, mps_d, params)
    @test magnetisation(env,model) ≈ magofβ(model) atol=1e-6
    @test energy(env,model) ≈ eneofβ(model) atol=1e-6
    @show energy(env,model)+1.414213779415974 # β = isingβc
end
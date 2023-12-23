using ADMPS
using ADMPS: normality, num_grad, bulid_丰
using Test
using Zygote

@testset "normality of 十" begin
    β = 0.8
    model = Ising(β)
    十 = model_tensor(model)

    @test normality(十) ≈ 1.0 

    十 = rand(ComplexF64, 2,2,2,2)
    @test normality(十) < 1
end

@testset "gradient with $atype{$dtype}" for atype in [Array], dtype in [ComplexF64]
    Random.seed!(100)
    β = 0.4
    model = Ising(β)
    十 = atype{dtype}(model_tensor(model))

    f(p) = -log(normality(bulid_丰(十, p)))
    p = rand(ComplexF64, 1,2,1,2)
    @test Zygote.gradient(f, p)[1] ≈ num_grad(f, p) 
end

@testset "optim_P" begin
    Random.seed!(100)
    β = 0.4
    model = Ising(β)
    十 = model_tensor(model)

    P = optim_P(十, 2, 1; 
                verbose=true,
                f_tol=1e-6, 
                iters=100, 
                show_every=10)
    @test normality(bulid_丰(十, P)) ≈ 1.0
end

@testset "optim_P" begin
    Random.seed!(100)
    β = isingβc
    τ = 0.6
    model = IsingP(β, τ)
    十 = model_tensor(model)

    @test normality(十) < 1
    P = optim_P(十, 2, 1; 
                verbose=true,
                f_tol=1e-6, 
                iters=100, 
                show_every=10)
    @test normality(bulid_丰(十, P)) ≈ 1.0
end
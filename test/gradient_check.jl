using ADMPS, Random, LinearAlgebra, Zygote, OMEinsum, ChainRulesCore, Test
atype = Array
dtype = ComplexF64

using ADMPS: logoverlap, projectcomplement!
@testset "gradient with $atype{$dtype}" for atype in [Array], dtype in [ComplexF64]
    Random.seed!(101)
    D,χ = 2,5

    M = rand(2,2,2,2)
    Au = Matrix(qr(rand(dtype,χ*D,χ)).Q)
    Ad = Matrix(qr(rand(dtype,χ*D,χ)).Q)

    gradzygote = Zygote.gradient(x->logoverlap(Au, x, M)[1],Ad)[1]
    gradexact = logoverlap(Au, Ad, M)[2]

    @test projectcomplement!(gradzygote,Ad) ≈ gradexact atol=1e-12
end

"""
    A helper function to retract
"""
function poorman_retract(x,g,α)
    U,S,V = svd(x+α*g)
    return (U*V',g)
end

using ADMPS:retract
using ADMPS:precondition
@testset "Line Search with $atype{$dtype}" for atype in [Array], dtype in [ComplexF64]
    Random.seed!(101)
    D,χ = 2,5

    M = rand(2,2,2,2)
    Au = Matrix(qr(rand(dtype,χ*D,χ)).Q)
    Ad = Matrix(qr(rand(dtype,χ*D,χ)).Q)

    f,g = Ad->logoverlap(Au, Ad, M)[1], Ad->logoverlap(Au, Ad, M)[2]

    for α in [0.001,0.01]
        # Check the decrease
        print("normal   decrease $α: ", f(retract(Ad, -g(Ad), α)[1]) - f(Ad) ,"\n")

        # Line search direction is right
        @test f(retract(Ad, -g(Ad), α)[1]) < f(Ad)  
    end

    for α in [0.001,0.01,0.1]
        # Check the decrease
        print("precondition decrease $α: ", f(retract(Ad, -precondition(Ad,g(Ad)), α)[1]) - f(Ad) ,"\n")
        
        # Line search direction with precondition is right
        @test f(retract(Ad, -precondition(Ad,g(Ad)), α)[1]) < f(Ad)
    end
end
using FileIO
using OptimKit, LineSearches
using LinearAlgebra: I, norm, tr
using Zygote, ChainRulesCore,Random

"""
    Generate a random input function with χ,D and given types
"""
function random_mps(χ,D;atype=Array,dtype=ComplexF64)
    return atype(Array(qr(rand(dtype,χ*D,χ)).Q))
end


"""
    Interface left to calculate other physical quantities
"""
function obs_env(M,Au,Ad)
    _, FL = leftenv(Au, Ad, M)
    _, FR = rightenv(Au, Ad, M)
    _, FL_n = norm_FL(Au, Ad)
    _, FR_n = norm_FR(Au, Ad)
    return Dict("M"=>M, "Au"=>Au, "Ad"=>Ad, 
        "FLM"=>FL, "FRM"=>FR, "FL"=>FL_n, "FR"=>FR_n)
end
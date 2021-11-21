function obs_env(M,Au,Ad)
    _, FL = leftenv(Au, Ad, M)
    _, FR = rightenv(Au, Ad, M)
    _, FL_n = norm_FL(Au, Ad)
    _, FR_n = norm_FR(Au, Ad)
    M, Au, Ad, FL, FR, FL_n, FR_n
end

"""
    Z(env)

return the up and down partition function of the `env`.
"""
function Z(env)
    M, Au, Ad, FL, FR, FL_n, FR_n = env
    z = ein"(((adf,abc),dgeb),ceh),fgh -> "(FL,Au,M,FR,Ad)[]
    λ = ein"abc,abc -> "(FL,FR)[]
    overlap = ein"(ad,acb),(dce,be) ->"(FL_n,Au,Ad,FR_n)[]/ein"ab,ab ->"(FL_n,FR_n)[]
    return z/λ/overlap
end

"""
    magnetisation(env, model::MT, β)

return the up and down magnetisation of the `model`. Requires that `mag_tensor` are defined for `model`.
"""
function magnetisation(env, model::MT) where {MT <: HamiltonianModel}
    M, Au, Ad, FL, FR, _, _ = env
    Mag = _arraytype(M)(mag_tensor(model))
    mag = ein"(((adf,abc),dgeb),ceh),fgh -> "(FL,Au,Mag,FR,Ad)[]
    λ = ein"(((adf,abc),dgeb),ceh),fgh -> "(FL,Au,M,FR,Ad)[]
    return mag/λ
end

"""
    energy(env, model::MT, β)

return the up and down energy of the `model` as a function of the inverse
temperature `β` and the environment bonddimension `D` as calculated with
vumps. Requires that `model_tensor` are defined for `model`.
"""
function energy(env,model::MT) where {MT <: HamiltonianModel}
    M, Au, Ad, FL, FR, _, _ = env
    Ene = _arraytype(M)(energy_tensor(model))
    energy = ein"(((adf,abc),dgeb),ceh),fgh -> "(FL,Au,Ene,FR,Ad)[]
    λ = ein"(((adf,abc),dgeb),ceh),fgh -> "(FL,Au,M,FR,Ad)[]
    return energy/λ*2 # factor 2 for counting horizontal and vertical links
end

function overlap(env)
    _, Au, Ad, _, _, FL_n, FR_n = env
    _, FLu_n = norm_FL(Au, Au)
    _, FRu_n = norm_FR(Au, Au)
    _, FLd_n = norm_FL(Ad, Ad)
    _, FRd_n = norm_FR(Ad, Ad)
    Au /= sqrt(ein"(ad,acb),(dce,be) ->"(FLu_n,Au,Au,FRu_n)[]/ein"ab,ab ->"(FLu_n,FRu_n)[])
    Ad /= sqrt(ein"(ad,acb),(dce,be) ->"(FLd_n,Ad,Ad,FRd_n)[]/ein"ab,ab ->"(FLd_n,FRd_n)[])
    ein"(ad,acb),(dce,be) ->"(FL_n,Au,Ad,FR_n)[]/ein"ab,ab ->"(FL_n,FR_n)[]
end

"""
    magofβ(::Ising,β)

return the analytical result for the magnetisation at inverse temperature
`β` for the 2d classical ising model.
"""
magofβ(model::Ising) = model.β > isingβc ? (1-sinh(2*model.β)^-4)^(1/8) : 0.

"""
    magofdβ(::Ising,β)

return the analytical result for the derivative of magnetisation at inverse temperature
`β` for the 2d classical ising model.
"""
magofdβ(model::Ising) = model.β > isingβc ? (coth(2*model.β)*csch(2*model.β)^4)/(1-csch(2*model.β)^4)^(7/8) : 0.

"""
    eneofβ(::Ising,β)

return some the numerical integrations of analytical result for the energy at inverse temperature
`β` for the 2d classical ising model.
"""
function eneofβ(model::Ising)
    β = model.β
    if β == 0.0
        return 0
    elseif β == 0.2
        return -0.42822885693016843
    elseif β == 0.4
        return -1.1060792706185651
    elseif β == 0.6
        return -1.909085845408498
    elseif β == 0.8
        return -1.9848514445364174
    elseif β == 1.0
        return -1.997159425837225
    end
end

"""
    Zofβ(::Ising,β)

return some the numerical integrations of analytical result for the partition function at inverse temperature
`β` for the 2d classical ising model.
"""
function Zofβ(model::Ising)
    β = model.β
    if β == 0.0
        return 2.0
    elseif β == 0.2
        return 2.08450374046259
    elseif β == 0.4
        return 2.4093664345022363
    elseif β == 0.6
        return 3.3539286863974582
    elseif β == 0.8
        return 4.96201030069517
    elseif β == 1.0
        return 7.3916307004743125
    end
end
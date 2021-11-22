using ChainRulesCore
using LinearAlgebra
using KrylovKit


# patch since it's currently broken otherwise
function ChainRulesCore.rrule(::typeof(Base.typed_hvcat), ::Type{T}, rows::Tuple{Vararg{Int}}, xs::S...) where {T,S}
    y = Base.typed_hvcat(T, rows, xs...)
    function back(ȳ)
        return NoTangent(), NoTangent(), NoTangent(), permutedims(ȳ)...
    end
    return y, back
end

# improves performance compared to default implementation, also avoids errors
# with some complex arrays
function ChainRulesCore.rrule(::typeof(LinearAlgebra.norm), A::AbstractArray)
    n = norm(A)
    function back(Δ)
        return NoTangent(), Δ .* A ./ (n + eps(0f0)), NoTangent()
    end
    return n, back
end

"""
    ChainRulesCore.rrule(::typeof(leftenv), Au::AbstractArray{T}, Ad::AbstractArray{T}, M::AbstractArray{T}, FL::AbstractArray{T}; kwargs...) where {T}

```
           ┌──   Au  ──┐ 
           │     │     │ 
dM    = - FL ──    ──  ξl
           │     │     │ 
           └──   Ad  ──┘ 

           ┌──       ──┐   
           │     │     │   
dAu  = -  FL ──  M ──  ξl  
           │     │     │   
           └──   Ad  ──┘   

           ┌──   Au  ──┐ 
           │     │     │ 
dAd  = -  FL  ── M ──  ξl
           │     │     │ 
           └──       ──┘ 
```
"""

function ChainRulesCore.rrule(::typeof(leftenv), Au::AbstractArray{T}, Ad::AbstractArray{T}, M::AbstractArray{T}, FL::AbstractArray{T}; kwargs...) where {T}
    λl, FL = leftenv(Au, Ad, M, FL)
    # @show λl
    function back((dλ, dFL))
        ξl, info = linsolve(FR -> ein"((abc,ceh),dgeb),fgh -> adf"(Au, FR, M, Ad), dFL, -λl, 1; maxiter = 1)
        # @assert info.converged==1
        # errL = ein"abc,cba ->"(FL, ξl)[]
        # abs(errL) > 1e-1 && throw("FL and ξl aren't orthometric. err = $(errL)")
        dAu = -ein"((adf,fgh),dgeb),ceh -> abc"(FL, Ad, M, ξl) 
        dAd = -ein"((adf,abc),dgeb),ceh -> fgh"(FL, Au, M, ξl)
        dM = -ein"(adf,abc),(fgh,ceh) -> dgeb"(FL, Au, Ad, ξl)
        return NoTangent(), dAu, dAd, dM, NoTangent()...
    end
    return (λl, FL), back
end

"""
    ChainRulesCore.rrule(::typeof(rightenv), AR::AbstractArray{T}, M::AbstractArray{T}, FR::AbstractArray{T}; kwargs...) where {T}

```
           ┌──   Au  ──┐ 
           │     │     │ 
dM    = - ξr  ──   ──  FR
           │     │     │ 
           └──   Ad  ──┘ 

           ┌──       ──┐   
           │     │     │   
dAu  = -  ξr  ── M ──  FR  
           │     │     │   
           └──   Ad  ──┘   

           ┌──   Au  ──┐ 
           │     │     │ 
dAu  = -  ξr  ── M ──  FR
           │     │     │ 
           └──       ──┘
```
"""
function ChainRulesCore.rrule(::typeof(rightenv), Au::AbstractArray{T}, Ad::AbstractArray{T}, M::AbstractArray{T}, FR::AbstractArray{T}; kwargs...) where {T}
    λr, FR = rightenv(Au, Ad, M, FR)
    # @show λr
    function back((dλ, dFR))
        ξr, info = linsolve(FL -> ein"((adf,abc),dgeb),fgh -> ceh"(FL, Au, M, Ad), dFR, -λr, 1; maxiter = 1)
        # @assert info.converged==1
        # errR = ein"abc,cba ->"(ξr, FR)[]
        # abs(errR) > 1e-1 && throw("FR and ξr aren't orthometric. err = $(errR) $(info)")
        dAu = -ein"((adf,fgh),dgeb),ceh -> abc"(ξr, Ad, M, FR) 
        dAd = -ein"((adf,abc),dgeb),ceh -> fgh"(ξr, Au, M, FR)
        dM = -ein"(adf,abc),(fgh,ceh) -> dgeb"(ξr, Au, Ad, FR)
        return NoTangent(), dAu, dAd, dM, NoTangent()...
    end
    return (λr, FR), back
end

"""
    function ChainRulesCore.rrule(::typeof(norm_FL), Au::AbstractArray{T}, Ad::AbstractArray{T}, FL::AbstractArray{T}; kwargs...) where {T}

```
           ┌──       ──┐   
           │     │     │   
dAu  = -  FL ────┼──── ξl  
           │     │     │   
           └──   Ad  ──┘   

           ┌──   Au  ──┐ 
           │     │     │ 
dAd  = -  FL  ───┼───  ξl
           │     │     │ 
           └──       ──┘ 
```
"""

function ChainRulesCore.rrule(::typeof(norm_FL), Au::AbstractArray{T}, Ad::AbstractArray{T}, FL::AbstractArray{T}; kwargs...) where {T}
    λl, FL = norm_FL(Au, Ad, FL)
    function back((dλ, dFL))
        ξl, info = linsolve(FR -> ein"(be,acb), dce -> ad"(FR,Au,Ad), dFL, -λl, 1; maxiter = 1)
        # @assert info.converged==1
        # errL = ein"abc,cba ->"(FL, ξl)[]
        # abs(errL) > 1e-1 && throw("FL and ξl aren't orthometric. err = $(errL)")
        dAu = -ein"(ad,dce), be -> acb"(FL, Ad, ξl) 
        dAd = -ein"(ad,acb), be -> dce"(FL, Au, ξl)
        return NoTangent(), dAu, dAd, NoTangent()...
    end
    return (λl, FL), back
end

"""
    ChainRulesCore.rrule(::typeof(rightenv), AR::AbstractArray{T}, M::AbstractArray{T}, FR::AbstractArray{T}; kwargs...) where {T}

```
           ┌──   Au  ──┐ 
           │     │     │ 
dM    = - ξr  ──   ──  FR
           │     │     │ 
           └──   Ad  ──┘ 

           ┌──       ──┐   
           │     │     │   
dAu  = -  ξr  ── M ──  FR  
           │     │     │   
           └──   Ad  ──┘   

           ┌──   Au  ──┐ 
           │     │     │ 
dAu  = -  ξr  ── M ──  FR
           │     │     │ 
           └──       ──┘
```
"""
function ChainRulesCore.rrule(::typeof(norm_FR), Au::AbstractArray{T}, Ad::AbstractArray{T}, FR::AbstractArray{T}; kwargs...) where {T}
    λr, FR = norm_FR(Au, Ad, FR)
    # @show λr
    function back((dλ, dFR))
        ξr, info = linsolve(FL -> ein"(ad,acb), dce -> be"(FL,Au,Ad), dFR, -λr, 1; maxiter = 1)
        # @assert info.converged==1
        # errR = ein"abc,cba ->"(ξr, FR)[]
        # abs(errR) > 1e-1 && throw("FR and ξr aren't orthometric. err = $(errR) $(info)")
        dAu = -ein"(ad,dce), be -> acb"(ξr, Ad, FR) 
        dAd = -ein"(ad,acb), be -> dce"(ξr, Au, FR)
        return NoTangent(), dAu, dAd, NoTangent()...
    end
    return (λr, FR), back
end

@doc raw"
    num_grad(f, K::Real; [δ = 1e-5])

return the numerical gradient of `f` at `K` calculated with
`(f(K+δ/2) - f(K-δ/2))/δ`

# example

```jldoctest; setup = :(using TensorNetworkAD)
julia> TensorNetworkAD.num_grad(x -> x * x, 3) ≈ 6
true
```
"
num_grad(f, K::Real; δ::Real=1e-5) = (f(K + δ / 2) - f(K - δ / 2)) / δ

@doc raw"
    num_grad(f, K::AbstractArray; [δ = 1e-5])
    
return the numerical gradient of `f` for each element of `K`.

# example

```jldoctest; setup = :(using TensorNetworkAD, LinearAlgebra)
julia> TensorNetworkAD.num_grad(tr, rand(2,2)) ≈ I
true
```
"
function num_grad(f, a::AbstractArray; δ::Real=1e-5)
    b = Array(copy(a))
    df = map(CartesianIndices(b)) do i
        foo = x -> (ac = copy(b); ac[i] = x; f(_arraytype(a)(ac)))
        num_grad(foo, b[i], δ=δ)
    end
    return _arraytype(a)(df)
    # map(CartesianIndices(a)) do i
    #     foo = x -> (ac = copy(a); ac[i] = x; f(ac))
    #     num_grad(foo, a[i], δ=δ)
    # end
end
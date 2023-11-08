using ChainRulesCore
using LinearAlgebra
using KrylovKit

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
function num_grad(f, K; δ::Real=1e-5)
    if eltype(K) == ComplexF64
        (f(K + δ / 2) - f(K - δ / 2)) / δ + 
            (f(K + δ / 2 * 1.0im) - f(K - δ / 2 * 1.0im)) / δ * 1.0im
    else
        (f(K + δ / 2) - f(K - δ / 2)) / δ
    end
end


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
        dFL -= Array(ein"abc,abc ->"(conj(FL), dFL))[] * FL
        ξl, info = linsolve(FR -> ein"((abc,ceh),dgeb),fgh -> adf"(Au, FR, M, Ad), conj(dFL), -λl, 1; maxiter = 1)
        # @assert info.converged==1
        # errL = ein"abc,cba ->"(FL, ξl)[]
        # abs(errL) > 1e-1 && throw("FL and ξl aren't orthometric. err = $(errL)")
        dAu = -ein"((adf,fgh),dgeb),ceh -> abc"(FL, Ad, M, ξl) 
        dAd = -ein"((adf,abc),dgeb),ceh -> fgh"(FL, Au, M, ξl)
        dM = -ein"(adf,abc),(fgh,ceh) -> dgeb"(FL, Au, Ad, ξl)
        return NoTangent(), conj(dAu), conj(dAd), conj(dM), NoTangent()...
    end
    return (λl, FL), back
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
        dFL -= Array(ein"ab,ab ->"(conj(FL), dFL))[] * FL
        ξl, info = linsolve(FR -> ein"(be,acb), dce -> ad"(FR,Au,Ad), conj(dFL), -λl, 1; maxiter = 1)
        # @assert info.converged==1
        # errL = ein"abc,cba ->"(FL, ξl)[]
        # abs(errL) > 1e-1 && throw("FL and ξl aren't orthometric. err = $(errL)")
        dAu = -ein"(ad,dce), be -> acb"(FL, Ad, ξl) 
        dAd = -ein"(ad,acb), be -> dce"(FL, Au, ξl)
        return NoTangent(), conj(dAu), conj(dAd), NoTangent()...
    end
    return (λl, FL), back
end

function ChainRulesCore.rrule(::typeof(qrpos), A::AbstractArray{T,2}) where {T}
    Q, R = qrpos(A)
    function back((dQ, dR))
        M = Array(R * dR' - dQ' * Q)
        dA = (UpperTriangular(R + I * 1e-12) \ (dQ + Q * _arraytype(Q)(Hermitian(M, :L)))' )'
        return NoTangent(), _arraytype(Q)(dA)
    end
    return (Q, R), back
end

function ChainRulesCore.rrule(::typeof(lqpos), A::AbstractArray{T,2}) where {T}
    L, Q = lqpos(A)
    function back((dL, dQ))
        M = Array(L' * dL - dQ * Q')
        dA = LowerTriangular(L + I * 1e-12)' \ (dQ + _arraytype(Q)(Hermitian(M, :L)) * Q)
        return NoTangent(), _arraytype(Q)(dA)
    end
    return (L, Q), back
end
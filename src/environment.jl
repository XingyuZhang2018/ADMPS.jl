using LinearAlgebra
using KrylovKit
using OMEinsum

"""
tensor order graph: from left to right, top to bottom.
```
a ────┬──── c    a──────┬──────b   
│     b     │    │      │      │                     
├─ d ─┼─ e ─┤    │      c      │                  
│     g     │    │      │      │  
f ────┴──── h    d──────┴──────e    
```
"""

"""
    λ, FL = leftenv(Au, Ad, M, FL = _arraytype(Au)(rand(eltype(Au), size(Au,1), size(M,1), size(Au,1))); kwargs...)

Compute the left environment tensor for MPS `AL` and MPO `M`, by finding the left fixed point
of `Au - M - Ad` contracted Aung the physical dimension.
```
┌──  Au─       ┌──       a ────┬──── c 
│    │         │         │     b     │ 
FL ─ M ─  = λL FL─       ├─ d ─┼─ e ─┤ 
│    │         │         │     g     │ 
┕──  Ad─       ┕──       f ────┴──── h 
```
"""
function leftenv(Au, Ad, M, FL = _arraytype(Au)(rand(eltype(Au), size(Au,1), size(M,1), size(Ad,1))); kwargs...)
    λs, FLs, info = eigsolve(FL -> ein"((adf,abc),dgeb),fgh -> ceh"(FL,Au,M,Ad), FL, 1, :LM; ishermitian = false, kwargs...)
    return real(λs[1]), real(FLs[1])
end

"""
    λ, FR = rightenv(Au, Ad, M, FR = _arraytype(Au)(randn(eltype(Au), size(Au,1), size(M,3), size(Au,1))); kwargs...)

Compute the right environment tensor for MPS `AR` and MPO `M`, by finding the right fixed point
of `Au - M - Ad` contracted Aung the physical dimension.
```
 ─ Au ──┐         ──┐      a ────┬──── c 
    │   │           │      │     b     │ 
 ─  M ──FR   = λR ──FR     ├─ d ─┼─ e ─┤ 
    │   │           │      │     g     │ 
 ─ Ad ──┘         ──┘      f ────┴──── h 
```
"""
function rightenv(Au, Ad, M, FR = _arraytype(Au)(randn(eltype(Au), size(Au,1), size(M,3), size(Ad,1))); kwargs...)
    λs, FRs, info = eigsolve(FR -> ein"((abc,ceh),dgeb),fgh -> adf"(Au,FR,M,Ad), FR, 1, :LM; ishermitian = false, kwargs...)
    return real(λs[1]), real(FRs[1])
end

"""
    λ, FL = norm_FL(Au, Ad, FL; kwargs...)

Compute the left environment tensor for normalization, by finding the left fixed point
of `Au - Ad` contracted Aung the physical dimension.
```
┌── Au─      ┌──         a──────┬──────b
FL  │  =  λL FL                 │       
┕── Ad─      ┕──                c       
                                │       
                         d──────┴──────e
```
"""
function norm_FL(Au, Ad, FL = _arraytype(Au)(rand(eltype(Au), size(Au,1), size(Ad,1))); kwargs...)
    λs, FLs, info = eigsolve(FL -> ein"(ad,acb), dce -> be"(FL,Au,Ad), FL, 1, :LM; ishermitian = false, kwargs...)
    return real(λs[1]), real(FLs[1])
end

"""
    λ, FR = norm_FR(Au, Ad, FR; kwargs...)

Compute the right environment tensor for normalization, by finding the right fixed point
of `Au - Ad` contracted Aung the physical dimension.
```
 ─ AR──┐       ──┐       a──────┬──────b
   │   FR  = λR  FR             │       
 ─ AR──┘       ──┘              c       
                                │       
                         d──────┴──────e
```
"""
function norm_FR(Au, Ad, FR = _arraytype(Au)(randn(eltype(Au), size(Au,3), size(Ad,3))); kwargs...)
    λs, FRs, info = eigsolve(FR -> ein"(be,acb), dce -> ad"(FR,Au,Ad), FR, 1, :LM; ishermitian = false, kwargs...)
    return real(λs[1]), real(FRs[1])
end
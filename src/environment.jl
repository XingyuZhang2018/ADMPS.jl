using LinearAlgebra
using KrylovKit
using OMEinsum
using IterTools
"""
tensor order graph: from left to right, top to bottom.
```
a ────┬──── c    a──────┬──────b   
│     b     │    │      │      │                     
├─ d ─┼─ e ─┤    │      c      │                  
│     g     │    │      │      │  
f ────┴──── h    d──────┴──────e   

a ────┬──── c  
│     b     │
├─ d ─┼─ e ─┤
│     f     │
├─ g ─┼─ h ─┤           
│     i     │
j ────┴──── k     
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
    λs, FLs, info = eigsolve(FL -> ein"((adf,abc),dgeb),fgh -> ceh"(FL,Au,M,Ad), FL, 1, :LM; ishermitian = false, maxiter = 100)
    info == 0 && @warn "eigsolve not converged"
    if length(λs) > 1 && norm(abs(λs[1]) - abs(λs[2])) < 1e-12
        @show λs
        if real(λs[1]) > 0
            return λs[1], FLs[1]
        else
            return λs[2], FLs[2]
        end
    end
    return λs[1], FLs[1]
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
function rightenv(Au, Ad, M, FR = _arraytype(Au)(rand(eltype(Au), size(Au,1), size(M,3), size(Ad,1))); kwargs...)
    Au = permutedims(Au,(3,2,1))
    Ad = permutedims(Ad,(3,2,1))
    ML = permutedims(M,(3,2,1,4))
    return leftenv(Au, Ad, ML, FR; kwargs...)
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
    λs, FLs, info = eigsolve(FL -> ein"(ad,acb),dce -> be"(FL,Au,Ad), FL, 1, :LM; ishermitian = false, kwargs...)
    if length(λs) > 1 && norm(abs(λs[1]) - abs(λs[2])) < 1e-12
        @show λs
        if real(λs[1]) > 0
            return λs[1], FLs[1]
        else
            return λs[2], FLs[2]
        end
    end
    return λs[1], FLs[1]
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
    Au = permutedims(Au,(3,2,1))
    Ad = permutedims(Ad,(3,2,1))
    return norm_FL(Au, Ad, FR; kwargs...)
end

"""
    λ, FL4 = bigleftenv(AL, M, FL4 = rand(eltype(AL), size(AL,1), size(M,1), size(M,1), size(AL,1)); kwargs...)
Compute the left environment tensor for MPS `AL` and MPO `M`, by finding the left fixed point
of `AL - M - M - conj(AL)` contracted along the physical dimension.
```
┌── Au─       ┌──     a ────┬──── c      
│   │         │       │     b     │      
│── M ─       │──     ├─ d ─┼─ e ─┤  
FL4 │    = λL FL4     │     f     │
│── M ─       │──     ├─ g ─┼─ h ─┤           
│   │         │       │     i     │
┕── Ad─       ┕──     j ────┴──── k          
```
"""
function bigleftenv(Au, Ad, Mu, Md, FL4 = _arraytype(Au)(rand(eltype(Au), size(Au,1), size(Mu,1), size(Md,1), size(Ad,1))); kwargs...)
    λFL4s, FL4s, info = eigsolve(FL4 -> ein"(((adgj,abc),dfeb),gihf),jik -> cehk"(FL4,Au,Mu,Md,Ad), FL4, 1, :LM; ishermitian = false, kwargs...)
    # @show λFL4s
    return λFL4s[1], FL4s[1]
end

"""
    λ, FR4 = bigrightenv(AR, M, FR4 = randn(eltype(AR), size(AR,1), size(M,3), size(M,3), size(AR,1)); kwargs...)
Compute the right environment tensor for MPS `AR` and MPO `M`, by finding the right fixed point
of `AR - M - conj(AR)`` contracted along the physical dimension.
```
 ─ AR──┐         ──┐    a ────┬──── c
   │   │           │    │     b     │
 ─ M ──│         ──│    ├─ d ─┼─ e ─┤
   │  FR4   = λR  FR4   │     f     │
 ─ M ──│         ──│    ├─ g ─┼─ h ─┤
   │   │           │    │     i     │
 ─ AR──┘         ──┘    j ────┴──── k
```
"""
function bigrightenv(Au, Ad, Mu, Md, FR4 = _arraytype(Au)(rand(eltype(Au), size(Au,1), size(Mu,3), size(Md,3), size(Ad,1))); kwargs...)
    λFR4s, FR4s, info = eigsolve(FR4 -> ein"(((cehk,abc),dfeb),gihf),jik -> adgj"(FR4,Au,Mu,Md,Ad), FR4, 1, :LM; ishermitian = false, kwargs...)
    # @show λFR4s
    return λFR4s[1], FR4s[1]
end
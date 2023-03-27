# Low level functions for computing the environment tensors

using LinearAlgebra
using KrylovKit
using OMEinsum
using IterTools
using ChainRulesCore

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
    λ, FML = leftenv(Au, Ad, M, FML = _arraytype(Au)(rand(eltype(Au), size(Au,1), size(M,1), size(Au,1))); kwargs...)

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
function leftenv(Au, Ad, M, FML = _arraytype(Au)(rand(eltype(Au), size(Au,1), size(M,1), size(Ad,1))); kwargs...)
    refresh_cache!(FML)
    
    λs, FLs, info = eigsolve(FL -> ein"((adf,abc),dgeb),fgh -> ceh"(FL,conj(Au),M,Ad), FML, 1, :LM; ishermitian = false, kwargs...)
    if length(λs) > 1 && norm(abs(λs[1]) - abs(λs[2])) < 1e-12
        @show λs
        if real(λs[1]) > 0
            return λs[1], FLs[1]
        else
            return λs[2], FLs[2]
        end
    end
    return λs[1], copyto!(FML, FLs[1])
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
function rightenv(Au, Ad, M, FMR = _arraytype(Au)(randn(eltype(Au), size(Au,1), size(M,3), size(Ad,1))); kwargs...)
    Au = permutedims(Au,(3,2,1))
    Ad = permutedims(Ad,(3,2,1))
    ML = permutedims(M,(3,2,1,4))
    return leftenv(Au, Ad, ML, FMR; kwargs...)
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
    refresh_cache!(FL)

    λs, FLs, info = eigsolve(FL -> ein"(ad,acb),dce -> be"(FL,conj(Au),Ad), FL, 1, :LM; ishermitian = false, kwargs...)
    if length(λs) > 1 && norm(abs(λs[1]) - abs(λs[2])) < 1e-12
        @show λs
        if real(λs[1]) > 0
            return λs[1], FLs[1]
        else
            return λs[2], FLs[2]
        end
    end
    return λs[1], copyto!(FL, FLs[1])
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
function bigleftenv(Au, Ad, M, FMML = _arraytype(Au)(rand(eltype(Au), size(Au,1), size(M,1), size(M,1), size(Ad,1))); kwargs...)
    refresh_cache!(FMML)

    λFL4s, FL4s, info = eigsolve(FL4 -> ein"(((adgj,abc),dbef),gihf),jik -> cehk"(FL4,conj(Au),conj(M),M,Ad), FMML, 1, :LM; ishermitian = false, kwargs...)
    # @show λFL4s
    return λFL4s[1], copyto!(FMML, FL4s[1])
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
function bigrightenv(Au, Ad, M, FMMR = _arraytype(Au)(rand(eltype(Au), size(Au,1), size(M,3), size(M,1), size(Ad,1))); kwargs...)
    refresh_cache!(FMMR)

    λFR4s, FR4s, info = eigsolve(FR4 -> ein"(((cehk,abc),dbef),gihf),jik -> adgj"(FR4,conj(Au),conj(M),M,Ad), FMMR, 1, :LM; ishermitian = false, kwargs...)
    # @show λFR4s
    return λFR4s[1], copyto!(FMMR, FR4s[1])
end
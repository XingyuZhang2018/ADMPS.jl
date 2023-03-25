# Dominant Eigensolver (Krylov benchmark)

## Julia

```julia
using LinearAlgebra, BenchmarkTools, SparseArrays, KrylovKit, CUDA
A = rand(ComplexF64,4000,4000)
b = rand(ComplexF64,4000)
cuA,cub = CuArray(A),CuArray(b)
@benchmark eigsolve(  $A, $b , 1, :LM; ishermitian = false, tol = 1E-10)
@benchmark @sync eigsolve($cuA,$cub, 1, :LM; ishermitian = false, tol = 1E-10)
```
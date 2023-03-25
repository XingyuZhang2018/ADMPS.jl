using CUDA
using LinearAlgebra
CUDA.allowscalar(false)

_arraytype(x::Array{T}) where {T} = Array
_arraytype(x::CuArray{T}) where {T} = CuArray
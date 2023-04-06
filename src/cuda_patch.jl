using CUDA
import LinearAlgebra
const CublasFloat = Union{Float64,Float32,ComplexF64,ComplexF32}
const CublasReal = Union{Float64,Float32}
CUDA.allowscalar(false)

_arraytype(x::Array{T}) where {T} = Array
_arraytype(x::CuArray{T}) where {T} = CuArray

# LinearAlgebra.mul!(y::CuArray, x::CuArray, α::T) where {T <: CublasFloat} = (y .= α .* x)
# LinearAlgebra.axpy!(α::Complex{T}, x::CuArray{T}, y::CuArray{Complex{T}}) where {T <: CublasReal} = (y .+= α .*x)
# LinearAlgebra.axpy!(α::Integer, x::ComplexF64, ::Ptr{ComplexF64}, ::Int64, ::CuPtr{ComplexF64}, ::Int64)
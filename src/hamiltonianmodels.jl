using OMEinsum

abstract type HamiltonianModel end

const σx = Float64[0 1; 1 0]
const σy = ComplexF64[0 -1im; 1im 0]
const σz = Float64[1 0; 0 -1]
const id2 = Float64[1 0; 0 1]

@doc raw"
    hamiltonian(model<:HamiltonianModel)

return the hamiltonian of the `model` as a two-site tensor operator.
"
function hamiltonian end

struct Ising{T<:Real} <: HamiltonianModel 
    β::T
end

struct diaglocal{T<:Vector} <: HamiltonianModel 
    diag::T
end

"""
    diaglocal(diag::Vector)

return the 2-site Hamiltonian with single-body terms given
by the diagonal `diag`.
"""
function hamiltonian(model::diaglocal)
    diag = model.diag
    n = length(diag)
    h = ein"i -> ii"(diag)
    id = Matrix(I,n,n)
    reshape(h,n,n,1,1) .* reshape(id,1,1,n,n) .+ reshape(h,1,1,n,n) .* reshape(id,n,n,1,1)
end

@doc raw"
    TFIsing(hx::Real)

return a struct representing the transverse field ising model with magnetisation `hx`.
"
struct TFIsing{T<:Real} <: HamiltonianModel
    hx::T
end

"""
    hamiltonian(model::TFIsing)

return the transverse field ising hamiltonian for the provided `model` as a
two-site operator.
"""
function hamiltonian(model::TFIsing)
    hx = model.hx
    -2 * ein"ij,kl -> ijkl"(σz,σz) -
        hx/2 * ein"ij,kl -> ijkl"(σx, id2) -
        hx/2 * ein"ij,kl -> ijkl"(id2, σx)
end

@doc raw"
    Heisenberg(Jz::T,Jx::T,Jy::T) where {T<:Real}

return a struct representing the heisenberg model with magnetisation fields
`Jz`, `Jx` and `Jy`..
"
struct Heisenberg{T<:Real} <: HamiltonianModel
    Jz::T
    Jx::T
    Jy::T
end
Heisenberg() = Heisenberg(1.0,1.0,1.0)

"""
    hamiltonian(model::Heisenberg)

return the heisenberg hamiltonian for the `model` as a two-site operator.
"""
function hamiltonian(model::Heisenberg)
    h = model.Jz * ein"ij,kl -> ijkl"(σz, σz) -
        model.Jx * ein"ij,kl -> ijkl"(σx, σx) -
        model.Jy * ein"ij,kl -> ijkl"(σy, σy)
    h = ein"ijcd,kc,ld -> ijkl"(h,σx,σx')
    real(h ./ 2)
end

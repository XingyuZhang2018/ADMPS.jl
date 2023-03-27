using LinearAlgebra, OMEinsum
export projectisometric!, projectcomplement!, retract, transport!, precondition

projectcomplement!(g, x) = mul!(g,x,adjoint(x)*g,-1.0,true)

function projectisometric!(x)
    U, S, V = svd(x,full=false)
    return mul!(x,U,V') # x=U*V'
end

function retract(W, G, α)
    U, S, V = svd(G)
    WV = W*V
    cSVd, sSVd = Diagonal(cos.(α.*S))*V', Diagonal(sin.(α.*S)) * V'  # sin(S)*V, cos(S)*V'
    W′ = projectisometric!(WV*cSVd + U*sSVd)
    sSSVd = lmul!(Diagonal(S), sSVd)
    cSSVd = lmul!(Diagonal(S), cSVd)
    Z′ = projectcomplement!(U*cSSVd - WV*sSSVd, W′)
    return W′, Z′
end

function precondition(x,g; FL = Matrix{ComplexF64}(I,size(x,2),size(x,2)))
    χ = size(x,2)
    D = size(x,1)÷χ
    Tx = reshape(x,(χ,D,χ))
    
    FL = eigsolve(FL -> ein"(ad,acb),dce -> be"(FL,conj(Tx),Tx),
    FL, 1, :LM; 
    ishermitian = false)[2][1]
    U,S,V= svd(FL)

    g /= U*Diagonal(S)*U'+Matrix(I,χ,χ)*norm(g)
    return g
end

"""
    Not checked
"""
function transport!(Z, W, G, α, W′)
    U, S, V = svd(G)
    WVd = W*V'
    UdΘ = adjoint(U)*Z
    sSUdθ, cSUdθ = Diagonal(sin.(α.*S))*UdΘ, Diagonal(cos.(α.*S))*UdΘ 
    cSm1UdΘ = axpy!(-1, UdΘ, cSUdθ) # (cos(S)-1)*U'*Θ
    Z′ = axpy!(true, U*cSm1UdΘ - WVd*sSUdθ, Z)
    Z′ = projectcomplement!(Z′, W′)
    return Z′
end

# Tools functions for OptimKit.jl

# Maybe weird for you, but just see OptimKit/linesearch.254
_inner(x, ξ1, ξ2) = real(dot(precondition(x,ξ1),precondition(x,ξ2))) 

_add!(η, ξ, β) = LinearAlgebra.axpy!(β, ξ, η)
_scale!(η, β) = LinearAlgebra.rmul!(η, β)
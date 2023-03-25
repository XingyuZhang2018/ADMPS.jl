using LinearAlgebra, OMEinsum
export projectisometric!, projectcomplement!, retract, transport!, precondition

function projectcomplement!(g, x)
    P = adjoint(x)*g
    g = mul!(g,x,P,-1.0,true)
    return g
end 

function projectisometric!(x)
    U, S, V = svd(x,full=false)
    return mul!(x,U,V') # x=U*V'
end

function retract(W, G, α)
    U, S, V = svd(G)
    WVd = W*V'
    cSV, sSV = Diagonal(cos.(α.*S))*V, Diagonal(sin.(α.*S)) * V'  # sin(S)*V, cos(S)*V'
    W′ = projectisometric!(WVd*cSV + U*sSV)
    sSSV = lmul!(Diagonal(S), sSV)
    cSSV = lmul!(Diagonal(S), cSV)
    Z′ = projectcomplement!(U*cSSV - WVd*sSSV, W′)
    return W′, Z′
end

function precondition(x,g)
    χ = size(x,2)
    D = size(x,1)÷χ
    
    Tx = reshape(x,(χ,D,χ))
    FL = rand(ComplexF64,(χ,χ)) |> x -> (x+x')/norm(x)
    FL = eigsolve(FL -> ein"(ad,acb),dce -> be"(FL,Tx,conj(Tx)),
    FL, 1, :LM; 
    ishermitian = false)[2][1]
    U,S,V= svd(FL)

    g /= U*Diagonal(S)*U'+Matrix(I,χ,χ)*norm(g)
    return g
end

function transport!(Z, W, G, α, W′)
    U, S, V = svd(G)
    WVd = W*V'
    UdΘ = adjoint(U)*Z
    sSUdθ, cSUdθ = Diagonal(sin.(α.*S))*UdΘ, Diagonal(cos.(α.*S))*UdΘ # sin(S)*U'*Θ, cos(S)*U'*Θ
    cSm1UdΘ = axpy!(-1, UdΘ, cSUdθ) # (cos(S)-1)*U'*Θ
    Z′ = axpy!(true, U*cSm1UdΘ - WVd*sSUdθ, Z)
    Z′ = projectcomplement!(Z′, W′)
    return Z′
end
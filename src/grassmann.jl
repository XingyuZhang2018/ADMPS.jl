import Optim: project_tangent, project_tangent!, retract!, Manifold

struct Grassmann <: Manifold
end
export Grassmann

"""
    Project a vector g onto a Grassmann (ij)(k) manifold at point x
    x: shape: (i,j)(k)
    g: shape: (i,j)(k)
"""
project_tangent(M::Grassmann, g, x) = project_tangent!(M::Grassmann, copy(g), x)
function project_tangent!(::Grassmann, g, x)
    g .-= ein"deg,(abc,abg)->dec"(x, g, conj(x)) 
end

"""
    Retrct a vector g onto a Grassmann manifold at point x.
     This is a SVD based retraction and same as it in the Stiefel manifold.
"""
function retract!(::Grassmann, x)
    χ,d,_ = size(x)
    # bad retract!
    # F = svd(reshape(x,(χ*d,χ)))
    # x .= reshape(F.U * F.V, (χ,d,χ))

    # good retract!
    AL, _, _ = leftorth(reshape(x, (χ,d,χ,1,1)))
    x .= reshape(AL, (χ,d,χ))
end
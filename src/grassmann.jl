import Optim: project_tangent!, retract!, Manifold

struct Grassmann <: Manifold
end

"""
    Project a vector g onto a Grassmann (ij)(k) manifold at point x
    x: shape: (i,j)(k)
    g: shape: (i,j)(k)
"""
function Optim.project_tangent!(::Grassmann, g, x)
    g .-= ein"deg,(abc,abg)->dec"(x, g, conj(x)) 
end

"""
    Retrct a vector g onto a Grassmann manifold at point x.
     This is a SVD based retraction and same as it in the Stiefel manifold.
"""
function Optim.retract!(::Grassmann, x)
    χ,d,_ = size(x)
    F = svd(reshape(x,(χ*d,χ)))
    x .= reshape(F.U * F.V, (χ,d,χ))
end
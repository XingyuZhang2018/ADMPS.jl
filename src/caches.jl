# Create caches spaces

const CACHE_RATE = 0.0
# only cache parts of previous result to speed up. Introduce some perturbation to avoid always converge to the same. 
function refresh_cache!(v)
    v .= CACHE_RATE * v / sum(abs,v) * length(v)
    return axpy!(1-CACHE_RATE, randn(eltype(v), size(v)),v)
end

"""
    create CACHED environment; only one set: FL, FR....
"""
function create_cached_one(χ,D,ArrayType=Array)
    # Init, FL=FR, FML,FMR, FMML, FMMR
    v = ArrayType(randn(ComplexF64, χ, χ))
    Mv = ArrayType(randn(ComplexF64, χ, D, χ))
    MMv = ArrayType(randn(ComplexF64, χ, D, D, χ))

    return Dict([("FL", v), ("FR", deepcopy(v)),
    ("FML", Mv), ("FMR", deepcopy(Mv)),
    ("FMML", MMv), ("FMMR", deepcopy(MMv))
    ])
end
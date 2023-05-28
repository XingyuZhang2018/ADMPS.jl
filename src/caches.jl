# Create caches spaces

# only cache parts of previous result to speed up. Introduce some perturbation to avoid always converge to the same. 
# Experimental, with limited tests, this cache does not fail for 1.0.

if CACHE_RATE < 1.0
    @info "ADMPS CACHE: Cache rate is less than 1.0, cache will be used."
    function refresh_cache!(v)
        v .= CACHE_RATE * v / sum(abs,v) * length(v)
        return axpy!(1-CACHE_RATE, _arraytype(v)(ones(eltype(v), size(v))),v)
    end    
else
    @info "Cache rate is 1.0, no refresh of cache."

    """
        CACHE_RATE=1.0, refresh_cache!(v) = v, does not refresh cache.
    """
    refresh_cache!(v) = v
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
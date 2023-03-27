# Create caches spaces

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
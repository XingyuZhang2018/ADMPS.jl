using HDF5, LinearAlgebra
using ADMPS
using ADMPS: factory_logoverlap

# ====== initialization ======
Au = h5read("A.h5","Au")
Ad = h5read("A.h5","Ad")
M = h5read("A.h5","M")
Md = h5read("A.h5","Md")

χ,D = size(Au,2),size(M,2)

@show norm(Au-Ad)
logoverlap = factory_logoverlap(χ,D,Array)

using ADMPS:norm_FL
# ========= Try ========
logoverlap(Au,Ad,M)
logoverlap(Ad,Ad,M)
λ = norm_FL(reshape(Au,(χ,D,χ)),reshape(Ad,(χ,D,χ)))[1] |> abs
log(λ) + logoverlap(Au,Ad,M)[1]
# EXACT: 1.3813564717043518 0.3230659669
using ADMPS, Random, LinearAlgebra, OMEinsum, CUDA
atype = Array
dtype = ComplexF64

if ARGS[2] == "gpu"
    atype = CuArray
else
    atype = Array
end

Random.seed!(109)
D,χ = 4,12

Au = random_mps(χ,D;atype=atype)
Ad = random_mps(χ,D;atype=atype)

β,g = 1.2,2.0
TII = zeros(4,4,4,4)
for i in -1:2:1
    for j in -1:2:1
        for k in -1:2:1
            for l in -1:2:1
                a,b,c,d = (i+1)÷2,(j+1)÷2,(k+1)÷2,(l+1)÷2
                x,y,z,w = a*2+b+1,b*2+c+1,d*2+c+1,a*2+d+1
                TII[x,y,z,w] = exp(β/2*(i*j+i*l+k*j+k*l)-β*g*(i*k+j*l))
            end
        end
    end
end

Au, Ad = optimizemps(Au, Ad, atype(TII),verbosity=-1,poweriter=2000)
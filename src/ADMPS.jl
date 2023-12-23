module ADMPS

using Parameters

export Ising, IsingP, TFIsing, Heisenberg
export Z,magnetisation, energy
export hamiltonian, model_tensor, mag_tensor, energy_tensor
export num_grad, init_mps, optimisemps, optim_P

include("cuda_patch.jl")
include("hamiltonianmodels.jl")
include("exampletensors.jl")
include("exampleobs.jl")
include("environment.jl")
include("autodiff.jl")
include("variationalmps.jl")
include("variationalP.jl")

end

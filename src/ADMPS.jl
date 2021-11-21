module ADMPS

export Ising, TFIsing, Heisenberg
export Z,magnetisation, energy
export hamiltonian, model_tensor, mag_tensor, energy_tensor
export num_grad, optimisemps

include("cuda_patch.jl")
include("hamiltonianmodels.jl")
include("exampletensors.jl")
include("exampleobs.jl")
include("environment.jl")
include("autodiff.jl")
include("variationalmps.jl")

end

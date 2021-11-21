using ADMPS
using ADMPS: model_tensor, tensorfromclassical
using Test

@testset "exampletensor" begin
    β = rand()
    @test model_tensor(Ising(β)) ≈ tensorfromclassical([β -β; -β β])
    @test mag_tensor(Ising(β)) !== nothing
    @test energy_tensor(Ising(β)) !== nothing
end

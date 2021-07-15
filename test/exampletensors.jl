using ADMPS
using ADMPS: model_tensor, tensorfromclassical
using Test

@testset "exampletensor" begin
    β = rand()
    @test model_tensor(Ising(),β) ≈ tensorfromclassical([β -β; -β β])
end

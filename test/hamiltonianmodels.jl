using ADMPS
using ADMPS:HamiltonianModel

@testset "hamiltonianmodels" begin
    @test Ising() isa HamiltonianModel
    @test TFIsing(1.0) isa HamiltonianModel
    @test Heisenberg() isa HamiltonianModel
end
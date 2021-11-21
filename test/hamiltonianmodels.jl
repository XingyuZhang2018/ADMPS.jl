using ADMPS
using ADMPS:HamiltonianModel

@testset "hamiltonianmodels" begin
    @test Ising(1.0) isa HamiltonianModel
end
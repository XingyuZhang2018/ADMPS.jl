using ADMPS
using Test

print("Running on Test Files:", readdir("./"), "\n")
for filename in readdir("./")
    if endswith(filename, ".jl") && filename != "runtests.jl" && filename[1] != '#'
        @testset "$filename" begin
            println("Running $filename")
            include("$filename")
        end
    end
end
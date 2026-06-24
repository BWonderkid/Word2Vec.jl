using Test
using Word2Vec

vec_path = joinpath(@__DIR__, "data", "tiny.vec")
bin_path = joinpath(@__DIR__, "data", "tiny.bin")

@testset "Word2Vec loaders" begin

    vec_model = load_model(vec_path)
    bin_model = load_model(bin_path)

    v1 = get_embedding(vec_model, "computer")
    v2 = get_embedding(bin_model, "computer")

    @test v1 !== nothing
    @test v2 !== nothing

    @test length(v1) == vec_model.dim
    @test length(v2) == bin_model.dim
end
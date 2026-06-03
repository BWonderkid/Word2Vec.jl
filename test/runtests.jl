using Test
using Word2Vec

@testset "Word2Vec loaders" begin

    vec_model = load_model("models/cc.en.300.vec")
    bin_model = load_model("models/GoogleNews-vectors-negative300.bin")

    v1 = get_embedding(vec_model, "computer")
    v2 = get_embedding(bin_model, "computer")

    @test v1 !== nothing
    @test v2 !== nothing

    @test length(v1) == vec_model.dim
    @test length(v2) == bin_model.dim
end
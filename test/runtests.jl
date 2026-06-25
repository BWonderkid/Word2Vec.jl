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

    @test vocab_size(vec_model) == length(vec_model.embeddings)
    @test embedding_dim(vec_model) == vec_model.dim
    @test has_word(vec_model, "computer")
    @test !has_word(vec_model, "not-in-the-model")

    @test get_embedding(vec_model, "not-in-the-model") === nothing
    @test get_embedding(bin_model, "not-in-the-model") === nothing

    single_entry_dir = mktempdir()
    single_entry_path = joinpath(single_entry_dir, "single.vec")
    open(single_entry_path, "w") do io
        println(io, "1 3")
        println(io, "solo 1.0 2.0 3.0")
    end

    single_entry_model = load_model(single_entry_path)
    @test single_entry_model.dim == 3
    @test vocab_size(single_entry_model) == 1
    @test embedding_dim(single_entry_model) == 3
    @test has_word(single_entry_model, "solo")
    @test get_embedding(single_entry_model, "solo") == Float32[1.0, 2.0, 3.0]

    malformed_dir = mktempdir()
    malformed_path = joinpath(malformed_dir, "broken.vec")
    open(malformed_path, "w") do io
        println(io, "not-a-valid-header")
    end

    @test_throws Exception load_model(malformed_path)
end

@testset "Word2Vec savers" begin
    original = load_model(vec_path)

    tmp = mktempdir()

    saved_vec_path = joinpath(tmp, "saved.vec")
    save_model(original, saved_vec_path)
    reloaded_vec = load_model(saved_vec_path)

    @test vocab_size(reloaded_vec) == vocab_size(original)
    @test embedding_dim(reloaded_vec) == embedding_dim(original)
    for (word, vec) in original.embeddings
        @test has_word(reloaded_vec, word)
        @test get_embedding(reloaded_vec, word) ≈ vec
    end

    saved_bin_path = joinpath(tmp, "saved.bin")
    save_model(original, saved_bin_path)
    reloaded_bin = load_model(saved_bin_path)

    @test vocab_size(reloaded_bin) == vocab_size(original)
    @test embedding_dim(reloaded_bin) == embedding_dim(original)
    for (word, vec) in original.embeddings
        @test has_word(reloaded_bin, word)
        @test get_embedding(reloaded_bin, word) ≈ vec
    end

    @test_throws ArgumentError save_model(original, "model.txt")
end

@testset "Similarity functions" begin
    model = load_model(vec_path)
    words = collect(keys(model.embeddings))
    w1, w2 = words[1], words[2]

    # cosine similarity of a vector with itself is 1
    vec = get_embedding(model, w1)
    @test cosine_similarity(vec, vec) ≈ 1.0f0

    # cosine similarity is symmetric
    v1 = get_embedding(model, w1)
    v2 = get_embedding(model, w2)
    @test cosine_similarity(v1, v2) ≈ cosine_similarity(v2, v1)

    # similarity of a word with itself is 1
    @test similarity(model, w1, w1) ≈ 1.0f0

    # similarity returns nothing for unknown words
    @test similarity(model, w1, "not-in-vocab") === nothing
    @test similarity(model, "not-in-vocab", w1) === nothing

    # most_similar returns nothing for unknown word
    @test most_similar(model, "not-in-vocab") === nothing

    # most_similar returns at most n results, none of which is the query word
    results = most_similar(model, w1, 3)
    @test results !== nothing
    @test length(results) <= 3
    @test all(r[1] != w1 for r in results)

    # results are sorted by descending similarity
    scores = [r[2] for r in results]
    @test scores == sort(scores, rev=true)

    # requesting more results than vocab allows returns all other words
    all_results = most_similar(model, w1, 10_000)
    @test length(all_results) == vocab_size(model) - 1
end

using Test
using Word2Vec
using Plots

include(joinpath(@__DIR__, "..", "scripts", "AddExampleFiles.jl"))

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

    computer_view = SubString("computer", firstindex("computer"), lastindex("computer"))
    @test get_embedding(vec_model, computer_view) == v1

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

    wrong_dim_path = joinpath(malformed_dir, "wrong-dim.vec")
    open(wrong_dim_path, "w") do io
        println(io, "2 3")
        println(io, "good 1.0 2.0 3.0")
        println(io, "bad 4.0 5.0")
    end

    @test_logs (:warn, r"Skipping malformed embedding row") begin
        wrong_dim_model = load_model(wrong_dim_path)
        @test has_word(wrong_dim_model, "good")
        @test !has_word(wrong_dim_model, "bad")
    end
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

    vec = get_embedding(model, w1)
    @test cosine_similarity(vec, vec) ≈ 1.0f0

    v1 = get_embedding(model, w1)
    v2 = get_embedding(model, w2)
    @test cosine_similarity(v1, v2) ≈ cosine_similarity(v2, v1)

    @test similarity(model, w1, w1) ≈ 1.0f0

    @test similarity(model, w1, "not-in-vocab") === nothing
    @test similarity(model, "not-in-vocab", w1) === nothing

    @test most_similar(model, "not-in-vocab") === nothing

    results = most_similar(model, w1, 3)
    @test results !== nothing
    @test length(results) <= 3
    @test all(r[1] != w1 for r in results)

    scores = [r[2] for r in results]
    @test scores == sort(scores, rev=true)

    all_results = most_similar(model, w1, 10_000)
    @test length(all_results) == vocab_size(model) - 1
end

@testset "Tokenizer" begin
    @test tokenize("Hello, World!") == ["hello", "world"]
    @test tokenize("one two  three") == ["one", "two", "three"]
    @test tokenize("cats123dogs") == ["cats", "dogs"]
    @test tokenize("") == []
end

@testset "Word2Vec training" begin
    corpus = "the cat sat on the mat the cat ate the rat on the mat"

    model_sg = train_word2vec(corpus; dim=10, window=2, epochs=10, min_count=1, architecture=:skipgram)
    @test model_sg isa WordEmbeddingModel
    @test embedding_dim(model_sg) == 10
    @test has_word(model_sg, "cat")
    @test has_word(model_sg, "mat")
    @test !has_word(model_sg, "dog")
    @test length(get_embedding(model_sg, "cat")) == 10

    model_mc = train_word2vec(corpus; dim=5, min_count=3, epochs=5)
    @test !has_word(model_mc, "ate")
    @test has_word(model_mc, "the")

    seeded_1 = train_word2vec(corpus; dim=8, window=2, epochs=3, min_count=1, seed=123)
    seeded_2 = train_word2vec(corpus; dim=8, window=2, epochs=3, min_count=1, seed=123)
    @test get_embedding(seeded_1, "cat") ≈ get_embedding(seeded_2, "cat")

    tokens = tokenize(corpus)
    model_cbow = train_word2vec(tokens; dim=8, window=2, epochs=5, min_count=1, architecture=:cbow)
    @test model_cbow isa WordEmbeddingModel
    @test embedding_dim(model_cbow) == 8

    score = similarity(model_sg, "cat", "rat")
    @test score isa Float32
    @test -1.0f0 <= score <= 1.0f0

    neighbors = most_similar(model_sg, "cat", 3)
    @test neighbors !== nothing
    @test length(neighbors) <= 3

    @test_throws ArgumentError train_word2vec(tokens; architecture=:transformer)
    @test_throws ArgumentError train_word2vec(["x", "y"]; min_count=100)
end

@testset "ConEc training" begin
    corpus = "the cat sat on the mat the cat ate the rat on the mat"

    model = train_conec(corpus; dim=10, window=2, epochs=10, min_count=1)
    @test model isa ConEcModel
    @test embedding_dim(model) == 10
    @test has_word(model, "cat")
    @test !has_word(model, "dog")

    @test length(get_embedding(model, "cat")) == 10
    @test length(get_context_embedding(model, "cat")) == 10

    @test get_embedding(model, "unknown") === nothing
    @test get_context_embedding(model, "unknown") === nothing

    model_mc = train_conec(corpus; dim=5, min_count=3, epochs=5)
    @test !has_word(model_mc, "ate")
    @test has_word(model_mc, "the")

    seeded_conec_1 = train_conec(corpus; dim=8, window=2, epochs=3, min_count=1, seed=123)
    seeded_conec_2 = train_conec(corpus; dim=8, window=2, epochs=3, min_count=1, seed=123)
    @test get_embedding(seeded_conec_1, "cat") ≈ get_embedding(seeded_conec_2, "cat")

    wem = to_embedding_model(model)
    @test wem isa WordEmbeddingModel
    @test embedding_dim(wem) == 10
    @test has_word(wem, "cat")

    wem_combined = to_embedding_model(model, combined=true)
    v_word    = get_embedding(model, "cat")
    v_context = get_context_embedding(model, "cat")
    v_combined = get_embedding(wem_combined, "cat")
    @test v_combined ≈ (v_word .+ v_context) ./ 2.0f0

    score = similarity(wem, "cat", "mat")
    @test score isa Float32
    @test -1.0f0 <= score <= 1.0f0

    @test vocab_size(model) == vocab_size(wem)

    @test_throws ArgumentError train_conec(["x"]; min_count=100)
    @test_throws ArgumentError train_conec(["x"]; window=0)
end

@testset "Example file helpers" begin
    tmp = mktempdir()
    zip_in = joinpath(tmp, "sample.zip")
    extracted = joinpath(tmp, "out.txt")

    zf = ZipFile.Writer(zip_in)
    try
        file = ZipFile.addfile(zf, "sample.txt")
        write(file, "hello zip")
        close(file)
    finally
        close(zf)
    end

    zip_file_decompress(zip_in, extracted)
    @test read(extracted, String) == "hello zip"
end

@testset "Analogy evaluation" begin
    model = load_model(vec_path)
    words = collect(keys(model.embeddings))

    @test solve_analogy(model, words[1], words[2], "not-in-vocab") === nothing

    a, b, c = words[1], words[2], words[3]
    results = solve_analogy(model, a, b, c; n=2)
    @test results !== nothing
    @test length(results) <= 2
    @test all(r[1] ∉ (a, b, c) for r in results)
    @test all(-1.0f0 <= r[2] <= 1.0f0 for r in results)

    bad = [("x", "y", "z", "w")]
    result = evaluate_analogies(model, bad)
    @test result.skipped == 1
    @test result.total == 0
    @test result.accuracy == 0.0

    good = [(a, b, c, words[4])]
    result2 = evaluate_analogies(model, good)
    @test result2.total == 1
    @test result2.correct + (1 - result2.correct) == 1
end

@testset "Visualization" begin
    model = load_model(vec_path)
    words = collect(keys(model.embeddings))

    X = reduce(vcat, [get_embedding(model, w)' for w in words])
    coords_pca = reduce_pca(X; dims=2)
    @test size(coords_pca) == (length(words), 2)
    @test eltype(coords_pca) == Float32

    p = plot_embeddings(model, words[1:3]; method=:pca)
    @test p isa Plots.Plot

    mixed = [words[1], words[2], "not-in-vocab"]
    @test_logs (:warn, r"Skipping words not found in the vocabulary") begin
        p2 = plot_embeddings(model, mixed; method=:pca)
        @test p2 isa Plots.Plot
    end

    @test_logs (:warn, r"Skipping words not found in the vocabulary") begin
        @test_throws ArgumentError plot_embeddings(model, ["not-in-vocab"]; method=:pca)
    end
    @test_logs (:warn, r"Skipping words not found in the vocabulary") begin
        @test_throws ArgumentError plot_embeddings(model, [words[1], "not-in-vocab"]; method=:pca)
    end

    @test_throws ArgumentError plot_embeddings(model, words[1:2]; method=:umap)
end

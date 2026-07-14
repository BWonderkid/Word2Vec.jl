using Word2Vec

"""
    inspect_word(model, word::String; n::Int=3)

Inspect a word in a trained embedding model.

This helper prints the word, its embedding dimension, the full embedding vector,
and the `n` most similar words. If the word is not present in the vocabulary,
a message is printed and `nothing` is returned.

# Arguments
- `model`: A trained `WordEmbeddingModel`.
- `word::String`: The word to inspect.

# Keyword Arguments
- `n::Int=3`: Number of nearest neighbours to display.

# Returns
- `Vector{Float32}` if the embedding exists, otherwise `nothing`.
"""
function inspect_word(model, word::String; n::Int=3)
    emb = get_embedding(model, word)
    println("\nword = ", word)

    if emb === nothing
        println("  not in vocabulary")
        return nothing
    end

    println("  embedding_dim = ", length(emb))
    println("  embedding = ", emb)
    println("  most similar = ", most_similar(model, word, n))
    return emb
end

"""
    auto_query_words(tokens::Vector{String}; k::Int=5)::Vector{String}

Select up to `k` most frequent words from the token list.
"""
function auto_query_words(tokens::Vector{String}; k::Int=5)::Vector{String}
    counts = Dict{String, Int}()
    for t in tokens
        counts[t] = get(counts, t, 0) + 1
    end
    ranked = sort(collect(keys(counts)), by = w -> counts[w], rev = true)
    return ranked[1:min(k, length(ranked))]
end

"""
    run_toy_demo(; dataset_path, text, query_words, dim, window, epochs, architecture, model_out_path)

Train a small Word2Vec model on either a dataset file or raw text.

Priority:
1. If `dataset_path` is given, it is used.
2. Otherwise if `text` is given, it is used.
3. Otherwise the built-in default toy text is used.

If `query_words` is not provided, the function automatically picks the most
frequent words from the training text.

"""
function run_toy_demo(;
    dataset_path::Union{Nothing,String} = nothing,
    text::String = "toy car doll block ball teddy bear train puzzle toy car doll block",
    query_words::Union{Nothing,Vector{String}} = nothing,
    dim::Int = 20,
    window::Int = 2,
    epochs::Int = 10,
    model_out_path::Union{Nothing,String} = nothing,
    architecture::Symbol = :cbow
)
    if dataset_path !== nothing
        isfile(dataset_path) || throw(ArgumentError("Dataset file not found: $dataset_path"))
        text = read(dataset_path, String)
    end

    tokens = tokenize(text)
    model = train_word2vec(tokens; dim=dim, window=window, epochs=epochs, architecture=architecture)

    println("vocab_size = ", vocab_size(model))
    println("embedding_dim = ", embedding_dim(model))

    final_query_words = query_words === nothing || isempty(query_words) ? auto_query_words(tokens; k=5) : query_words
    for word in final_query_words
        inspect_word(model, word)
    end

    if model_out_path !== nothing
        save_model(model, model_out_path)
        println("model saved to: ", model_out_path)
    end

    return model
end
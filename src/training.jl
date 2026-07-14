using Random
using LinearAlgebra
using Unicode

sigmoid(x::Float32) = 1.0f0 / (1.0f0 + exp(-x))

struct Vocabulary
    word_to_idx::Dict{String, Int}
    idx_to_word::Vector{String}
    counts::Vector{Int}
end

function build_vocab(tokens::Vector{String}, min_count::Int)::Vocabulary
    freq = Dict{String, Int}()
    for t in tokens
        freq[t] = get(freq, t, 0) + 1
    end

    pairs = sort([(w, c) for (w, c) in freq if c >= min_count], by=x -> x[2], rev=true)

    idx_to_word = [p[1] for p in pairs]
    word_to_idx = Dict(w => i for (i, w) in enumerate(idx_to_word))
    counts      = [p[2] for p in pairs]

    return Vocabulary(word_to_idx, idx_to_word, counts)
end

# Pre-built O(1) sampler: each slot in the table holds a word index
# proportional to freq^0.75 — the standard Word2Vec noise distribution.
function build_unigram_table(counts::Vector{Int}, table_size::Int=Int(1e6))::Vector{Int}
    weights  = Float64.(counts) .^ 0.75
    weights ./= sum(weights)

    table    = Vector{Int}(undef, table_size)
    word_idx = 1
    cumprob  = weights[1]

    for i in 1:table_size
        table[i] = word_idx
        if i / table_size > cumprob && word_idx < length(weights)
            word_idx += 1
            cumprob  += weights[word_idx]
        end
    end

    return table
end

@views function skipgram_step!(W_in, W_out, center::Int, context::Int,
                               table::Vector{Int}, negative::Int, lr::Float32,
                               grad::AbstractVector{Float32}, rng)
    fill!(grad, 0f0)

    for k in 0:negative
        if k == 0
            target = context
            label  = 1.0f0
        else
            target = table[rand(rng, 1:length(table))]
            target == context && continue
            label = 0.0f0
        end

        err = (sigmoid(dot(W_out[target, :], W_in[center, :]) |> Float32) - label) * lr
        grad              .+= err .* W_out[target, :]
        W_out[target, :]  .-= err .* W_in[center, :]
    end

    W_in[center, :] .-= grad
end

@views function cbow_step!(W_in, W_out, ctx_indices::Vector{Int}, ctx_len::Int,
                           target::Int, table::Vector{Int}, negative::Int,
                           lr::Float32, ctx_mean::AbstractVector{Float32},
                           grad::AbstractVector{Float32}, rng)
    fill!(ctx_mean, 0f0)
    for idx in 1:ctx_len
        ci = ctx_indices[idx]
        ctx_mean .+= W_in[ci, :]
    end
    inv_ctx_len = 1f0 / ctx_len
    ctx_mean .*= inv_ctx_len

    fill!(grad, 0f0)

    for k in 0:negative
        if k == 0
            t     = target
            label = 1.0f0
        else
            t     = table[rand(rng, 1:length(table))]
            t == target && continue
            label = 0.0f0
        end

        err = (sigmoid(dot(W_out[t, :], ctx_mean) |> Float32) - label) * lr
        grad         .+= err .* W_out[t, :]
        W_out[t, :] .-= err .* ctx_mean
    end

    for idx in 1:ctx_len
        ci = ctx_indices[idx]
        @inbounds for d in eachindex(grad)
            W_in[ci, d] -= grad[d] * inv_ctx_len
        end
    end
end

"""
    tokenize(text::AbstractString)::Vector{String}

Split raw text into a sequence of normalized lowercase word tokens.

Normalization steps:
- Unicode NFKC normalization
- lowercasing
- splitting on any non-letter characters (`[^\\p{L}]+`)

This is more robust than ASCII-only tokenization and works better with
punctuation-heavy or multilingual corpora.

# Example
```julia
tokenize("The quick brown fox!")
# ["the", "quick", "brown", "fox"]
```
"""
function tokenize(text::AbstractString)::Vector{String}
    s = Unicode.normalize(text, :NFKC)
    s = lowercase(s)
    return String.(split(s, r"[^\p{L}]+"; keepempty=false))
end

"""
    train_word2vec(tokens::Vector{String}; kwargs...)::WordEmbeddingModel

Train a Word2Vec model on a tokenized corpus using negative sampling.

# Arguments
- `tokens::Vector{String}`: Flat sequence of tokens forming the corpus

# Keyword Arguments
- `dim::Int=100`: Dimensionality of the embedding vectors
- `window::Int=5`: Maximum context window radius
- `min_count::Int=1`: Minimum frequency for a word to enter the vocabulary
- `epochs::Int=5`: Number of full passes over the corpus
- `learning_rate::Float32=0.025f0`: Initial learning rate (decays linearly to near zero)
- `negative::Int=5`: Number of negative samples per positive training pair
- `architecture::Symbol=:skipgram`: `:skipgram` or `:cbow`

Calling `train_word2vec(corpus)` without keywords uses the defaults above.

# Returns
- `WordEmbeddingModel`: Trained model compatible with `get_embedding`, `similarity`, etc.

# Example
```julia
tokens = tokenize("the cat sat on the mat the cat ate the rat")
model  = train_word2vec(tokens, dim=10, epochs=20, min_count=1)
get_embedding(model, "cat")
```
"""
function train_word2vec(tokens::Vector{String};
                        dim::Int           = 100,
                        window::Int        = 5,
                        min_count::Int     = 1,
                        epochs::Int        = 5,
                        learning_rate      = 0.025f0,
                        negative::Int      = 5,
                        architecture::Symbol = :skipgram,
                        seed::Union{Nothing,Int} = nothing)::WordEmbeddingModel

    architecture in (:skipgram, :cbow) ||
        throw(ArgumentError("architecture must be :skipgram or :cbow, got :$architecture"))

    dim > 0         || throw(ArgumentError("dim must be > 0, got $dim"))
    window > 0      || throw(ArgumentError("window must be > 0, got $window"))
    min_count > 0   || throw(ArgumentError("min_count must be > 0, got $min_count"))
    epochs > 0      || throw(ArgumentError("epochs must be > 0, got $epochs"))
    negative >= 0   || throw(ArgumentError("negative must be >= 0, got $negative"))
    Float32(learning_rate) > 0f0 ||
        throw(ArgumentError("learning_rate must be > 0, got $learning_rate"))

    rng = seed === nothing ? Random.default_rng() : MersenneTwister(seed)

    vocab = build_vocab(tokens, min_count)
    V     = length(vocab.idx_to_word)
    V == 0 && throw(ArgumentError("Vocabulary is empty — lower min_count or provide more tokens."))

    lr = Float32(learning_rate)

    # W_in: small random values (standard Word2Vec init); W_out: zeros
    W_in  = (rand(rng, Float32, V, dim) .- 0.5f0) ./ Float32(dim)
    W_out = zeros(Float32, V, dim)
    grad  = zeros(Float32, dim)
    ctx_mean = zeros(Float32, dim)
    ctx_buf = Vector{Int}(undef, 2window)

    table   = build_unigram_table(vocab.counts)
    indices = [vocab.word_to_idx[t] for t in tokens if haskey(vocab.word_to_idx, t)]
    N       = length(indices)

    for epoch in 1:epochs
        for i in 1:N
            # learning rate decays linearly from lr to ~0 over all steps
            progress = Float32((epoch - 1) * N + i) / Float32(epochs * N)
            cur_lr   = lr * max(0.0001f0, 1.0f0 - progress)

            w  = rand(rng, 1:window)
            lo = max(1, i - w)
            hi = min(N, i + w)

            if architecture == :skipgram
                for j in lo:hi
                    j == i && continue
                    skipgram_step!(W_in, W_out, indices[i], indices[j], table,
                                   negative, cur_lr, grad, rng)
                end
            else
                ctx_len = 0
                for j in lo:hi
                    j == i && continue
                    ctx_len += 1
                    ctx_buf[ctx_len] = indices[j]
                end
                ctx_len == 0 && continue
                cbow_step!(W_in, W_out, ctx_buf, ctx_len, indices[i], table,
                           negative, cur_lr, ctx_mean, grad, rng)
            end
        end
    end

    embeddings = Dict(vocab.idx_to_word[i] => W_in[i, :] for i in 1:V)
    return WordEmbeddingModel(embeddings, dim)
end

"""
    train_word2vec(text::AbstractString; kwargs...)::WordEmbeddingModel

Convenience overload that tokenizes raw text before training.

# Example
```julia
model = train_word2vec("the cat sat on the mat"; dim=10, epochs=20, min_count=1)
```
"""
function train_word2vec(text::AbstractString; kwargs...)::WordEmbeddingModel
    return train_word2vec(tokenize(text); kwargs...)
end

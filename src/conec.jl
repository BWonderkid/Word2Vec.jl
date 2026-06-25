"""
    ConEcModel

A trained ConEc model holding separate word (W) and context (C) embedding matrices.

In contrast to standard Word2Vec, ConEc learns two distinct representations per word:
- `W`: how a word behaves as a **target** (the word being predicted)
- `C`: how a word behaves as **context** (when it surrounds another word)

# Fields
- `W::Matrix{Float32}`: word embedding matrix (vocab_size × dim)
- `C::Matrix{Float32}`: context matrix (vocab_size × dim)
- `word_to_idx::Dict{String,Int}`: maps each word to its row index in W and C
- `idx_to_word::Vector{String}`: maps each row index back to a word
- `dim::Int`: embedding dimensionality
"""
struct ConEcModel
    W::Matrix{Float32}
    C::Matrix{Float32}
    word_to_idx::Dict{String, Int}
    idx_to_word::Vector{String}
    dim::Int
end

"""
    vocab_size(model::ConEcModel)::Int

Return the number of words in the ConEc model vocabulary.
"""
vocab_size(model::ConEcModel) = length(model.idx_to_word)

"""
    embedding_dim(model::ConEcModel)::Int

Return the dimensionality of the ConEc model embeddings.
"""
embedding_dim(model::ConEcModel) = model.dim

"""
    has_word(model::ConEcModel, word::AbstractString)::Bool

Return `true` if `word` exists in the ConEc model vocabulary.
"""
has_word(model::ConEcModel, word::AbstractString) = haskey(model.word_to_idx, word)

"""
    get_embedding(model::ConEcModel, word::String)::Union{Vector{Float32}, Nothing}

Retrieve the word embedding vector (from W) for a given word.

Returns `nothing` if the word is not in the vocabulary.
"""
function get_embedding(model::ConEcModel, word::String)
    idx = get(model.word_to_idx, word, nothing)
    idx === nothing && return nothing
    return model.W[idx, :]
end

"""
    get_context_embedding(model::ConEcModel, word::String)::Union{Vector{Float32}, Nothing}

Retrieve the context embedding vector (from C) for a given word.

This represents how the word influences nearby words when it appears as context,
as opposed to `get_embedding` which returns how the word is represented as a target.

Returns `nothing` if the word is not in the vocabulary.
"""
function get_context_embedding(model::ConEcModel, word::String)
    idx = get(model.word_to_idx, word, nothing)
    idx === nothing && return nothing
    return model.C[idx, :]
end

"""
    to_embedding_model(model::ConEcModel; combined::Bool=false)::WordEmbeddingModel

Convert a `ConEcModel` to a standard `WordEmbeddingModel`.

# Keyword Arguments
- `combined::Bool=false`: if `true`, averages the word (W) and context (C) vectors
  for each word before returning. Combining both representations can improve
  performance on similarity tasks.

# Example
```julia
model = train_conec("the cat sat on the mat"; dim=10, epochs=20, min_count=1)
wem   = to_embedding_model(model, combined=true)
get_embedding(wem, "cat")
```
"""
function to_embedding_model(model::ConEcModel; combined::Bool=false)::WordEmbeddingModel
    embeddings = Dict{String, Vector{Float32}}()
    for (i, word) in enumerate(model.idx_to_word)
        embeddings[word] = combined ? (model.W[i, :] .+ model.C[i, :]) ./ 2.0f0 : model.W[i, :]
    end
    return WordEmbeddingModel(embeddings, model.dim)
end

@views function conec_step!(W, C, center::Int, ctx_indices::Vector{Int},
                            table::Vector{Int}, negative::Int, lr::Float32)
    dim = size(W, 2)

    # context embedding h = mean of C rows for the context words (sparse operation)
    h = zeros(Float32, dim)
    for ci in ctx_indices
        h .+= C[ci, :]
    end
    h ./= length(ctx_indices)

    # accumulate the gradient to propagate back to C after the positive/negative loop
    grad_C = zeros(Float32, dim)

    for k in 0:negative
        if k == 0
            target = center
            label  = 1.0f0
        else
            target = table[rand(1:length(table))]
            target == center && continue
            label  = 0.0f0
        end

        err = (sigmoid(dot(W[target, :], h) |> Float32) - label) * lr

        # accumulate gradient for C using the current W[target] before updating W
        grad_C       .+= err .* W[target, :]
        W[target, :] .-= err .* h
    end

    # distribute the accumulated gradient equally to each context word's row in C
    for ci in ctx_indices
        C[ci, :] .-= grad_C ./ length(ctx_indices)
    end
end

"""
    train_conec(tokens::Vector{String}; kwargs...)::ConEcModel

Train a ConEc model on a tokenized corpus using negative sampling.

ConEc (Horn 2017) extends Word2Vec by learning a separate context matrix C
alongside the standard word embedding matrix W. The context representation
for a window is the mean of the C rows of the surrounding words — a sparse
operation since only words in the window contribute.

# Arguments
- `tokens::Vector{String}`: flat sequence of tokens forming the corpus

# Keyword Arguments
- `dim::Int=100`: embedding dimensionality
- `window::Int=5`: maximum context window radius
- `min_count::Int=1`: minimum word frequency to enter the vocabulary
- `epochs::Int=5`: number of full passes over the corpus
- `learning_rate::Float32=0.025f0`: initial learning rate (decays linearly)
- `negative::Int=5`: number of negative samples per training step

# Returns
- `ConEcModel`: trained model with both W and C matrices

# Example
```julia
tokens = tokenize("the cat sat on the mat the cat ate the rat")
model  = train_conec(tokens, dim=10, epochs=20, min_count=1)
get_embedding(model, "cat")          # word vector from W
get_context_embedding(model, "cat")  # context vector from C
wem = to_embedding_model(model, combined=true)
```
"""
function train_conec(tokens::Vector{String};
                     dim::Int         = 100,
                     window::Int      = 5,
                     min_count::Int   = 1,
                     epochs::Int      = 5,
                     learning_rate    = 0.025f0,
                     negative::Int    = 5)::ConEcModel

    vocab = build_vocab(tokens, min_count)
    V     = length(vocab.idx_to_word)
    V == 0 && throw(ArgumentError("Vocabulary is empty — lower min_count or provide more tokens."))

    lr = Float32(learning_rate)

    W = (rand(Float32, V, dim) .- 0.5f0) ./ Float32(dim)
    C = zeros(Float32, V, dim)

    table   = build_unigram_table(vocab.counts)
    indices = [vocab.word_to_idx[t] for t in tokens if haskey(vocab.word_to_idx, t)]
    N       = length(indices)

    for epoch in 1:epochs
        for i in 1:N
            progress = Float32((epoch - 1) * N + i) / Float32(epochs * N)
            cur_lr   = lr * max(0.0001f0, 1.0f0 - progress)

            w  = rand(1:window)
            lo = max(1, i - w)
            hi = min(N, i + w)

            ctx = [indices[j] for j in lo:hi if j != i]
            isempty(ctx) && continue

            conec_step!(W, C, indices[i], ctx, table, negative, cur_lr)
        end
    end

    return ConEcModel(W, C, vocab.word_to_idx, vocab.idx_to_word, dim)
end

"""
    train_conec(text::AbstractString; kwargs...)::ConEcModel

Convenience overload that tokenizes raw text before training.

# Example
```julia
model = train_conec("the cat sat on the mat"; dim=10, epochs=20, min_count=1)
```
"""
function train_conec(text::AbstractString; kwargs...)::ConEcModel
    return train_conec(tokenize(text); kwargs...)
end

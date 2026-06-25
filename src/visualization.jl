using Statistics
using Plots
using TSne

"""
    reduce_pca(X::Matrix{Float32}; dims::Int=2)::Matrix{Float32}

Reduce an embedding matrix to `dims` dimensions using Principal Component Analysis (PCA).

Each row of `X` is a word embedding vector. The data is mean-centred before
decomposition. PCA is fast and deterministic, making it a good first choice for
inspecting embedding structure.

# Arguments
- `X::Matrix{Float32}`: n × d matrix of embedding vectors (n words, d dimensions)
- `dims::Int=2`: number of output dimensions

# Returns
- `Matrix{Float32}`: n × dims matrix of reduced coordinates
"""
function reduce_pca(X::Matrix{Float32}; dims::Int=2)::Matrix{Float32}
    dims      = min(dims, size(X, 1), size(X, 2))
    X_centred = X .- mean(X, dims=1)
    F         = svd(X_centred)
    return Float32.(X_centred * F.V[:, 1:dims])
end

"""
    reduce_tsne(X::Matrix{Float32}; dims::Int=2, perplexity::Real=30.0, max_iter::Int=1000)::Matrix{Float32}

Reduce an embedding matrix to `dims` dimensions using t-SNE.

t-SNE preserves local neighbourhood structure and often produces more visually
intuitive clusters than PCA, but it is non-deterministic and slower.

# Arguments
- `X::Matrix{Float32}`: n × d matrix of embedding vectors
- `dims::Int=2`: number of output dimensions (almost always 2)
- `perplexity::Real=30.0`: roughly controls the number of effective neighbours;
  typical values are between 5 and 50
- `max_iter::Int=1000`: number of optimisation iterations

# Returns
- `Matrix{Float32}`: n × dims matrix of reduced coordinates
"""
function reduce_tsne(X::Matrix{Float32};
                     dims::Int        = 2,
                     perplexity::Real = 30.0,
                     max_iter::Int    = 1000)::Matrix{Float32}
    return Float32.(tsne(X, dims, 0, max_iter, perplexity, verbose=false, progress=false))
end

"""
    plot_embeddings(model::WordEmbeddingModel, words::Vector{String};
                    method::Symbol=:pca, perplexity::Real=30.0, max_iter::Int=1000)

Plot word embeddings projected to 2D using PCA or t-SNE.

Words not found in the vocabulary are silently skipped.

# Arguments
- `model::WordEmbeddingModel`: the embedding model
- `words::Vector{String}`: words to visualise

# Keyword Arguments
- `method::Symbol=:pca`: `:pca` (fast, deterministic) or `:tsne` (slower, non-deterministic)
- `perplexity::Real=30.0`: t-SNE perplexity (ignored when `method=:pca`)
- `max_iter::Int=1000`: t-SNE iterations (ignored when `method=:pca`)

# Returns
A `Plots.Plot` object. Call `savefig(p, "plot.png")` to save it to disk.

# Example
```julia
model = load_model("embeddings.vec")
words = ["cat", "dog", "king", "queen", "man", "woman"]
p     = plot_embeddings(model, words, method=:pca)
savefig(p, "embeddings.png")
```
"""
function plot_embeddings(model::WordEmbeddingModel, words::Vector{String};
                         method::Symbol   = :pca,
                         perplexity::Real = 30.0,
                         max_iter::Int    = 1000)
    method in (:pca, :tsne) || throw(ArgumentError("method must be :pca or :tsne"))

    valid_words = filter(w -> has_word(model, w), words)
    length(valid_words) < 2 && throw(ArgumentError("At least 2 words must be in the vocabulary for a 2D plot"))

    X = Matrix{Float32}(reduce(hcat, [get_embedding(model, w) for w in valid_words])')

    coords = method == :pca ? reduce_pca(X) : reduce_tsne(X; perplexity=perplexity, max_iter=max_iter)

    p = scatter(coords[:, 1], coords[:, 2],
                legend=false,
                title="Word Embeddings — $(uppercase(string(method)))",
                xlabel="Component 1",
                ylabel="Component 2",
                markersize=4)

    for (i, word) in enumerate(valid_words)
        annotate!(p, coords[i, 1], coords[i, 2], text(" " * word, 8, :left, :bottom))
    end

    return p
end

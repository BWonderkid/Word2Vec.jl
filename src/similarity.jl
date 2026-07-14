using LinearAlgebra

"""
    cosine_similarity(a::Vector{Float32}, b::Vector{Float32})::Float32

Compute the cosine similarity between two embedding vectors.

Cosine similarity measures the angle between two vectors, returning a value
between -1 (opposite) and 1 (identical direction), regardless of magnitude.

# Arguments
- `a::Vector{Float32}`: First embedding vector
- `b::Vector{Float32}`: Second embedding vector

# Example
```julia
a = Float32[1.0, 0.0, 0.0]
b = Float32[0.0, 1.0, 0.0]
cosine_similarity(a, b)  # returns 0.0
```
"""
function cosine_similarity(a::Vector{Float32}, b::Vector{Float32})
    return dot(a, b) / (norm(a) * norm(b))
end

"""
    similarity(model::WordEmbeddingModel, word1::String, word2::String)::Union{Float32, Nothing}

Compute the cosine similarity between two words in the model.

# Arguments
- `model::WordEmbeddingModel`: The loaded embedding model
- `word1::String`: First word
- `word2::String`: Second word

# Returns
- `Float32` similarity score between -1 and 1, or `nothing` if either word is not in the vocabulary

# Example
```julia
model = load_model("embeddings.vec")
score = similarity(model, "king", "queen")
```
"""
function similarity(model::WordEmbeddingModel, word1::String, word2::String)
    v1 = get_embedding(model, word1)
    v2 = get_embedding(model, word2)
    (v1 === nothing || v2 === nothing) && return nothing
    return cosine_similarity(v1, v2)
end

"""
    most_similar(model::WordEmbeddingModel, word::String, n::Int=10)::Union{Vector{Tuple{String, Float32}}, Nothing}

Find the `n` most similar words to the given word, ranked by cosine similarity.

# Arguments
- `model::WordEmbeddingModel`: The loaded embedding model
- `word::String`: The query word
- `n::Int`: Number of results to return (default: 10)

# Returns
- `Vector{Tuple{String, Float32}}`: List of `(word, score)` pairs sorted by descending similarity,
  or `nothing` if the query word is not in the vocabulary

# Example
```julia
model = load_model("embeddings.vec")
neighbors = most_similar(model, "king", 5)
for (word, score) in neighbors
    println("\$word: \$score")
end
```
"""
function most_similar(model::WordEmbeddingModel, word::String, n::Int=10)
    query = get_embedding(model, word)
    query === nothing && return nothing

    scores = [(w, cosine_similarity(query, vec)) for (w, vec) in model.embeddings if w != word]
    k = min(n, length(scores))
    k <= 0 && return empty(scores)

    partialsort!(scores, 1:k, by=x -> x[2], rev=true)
    return scores[1:k]
end

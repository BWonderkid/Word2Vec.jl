"""
    get_embedding(model::WordEmbeddingModel, word::String)::Union{Vector{Float32}, Nothing}

Retrieve the embedding vector for a given word from a loaded model.

# Arguments
- `model::WordEmbeddingModel`: The loaded embedding model
- `word::String`: The word to look up

# Returns
- `Vector{Float32}`: The embedding vector for the word, or `nothing` if not found

# Example
```julia
model = load_model("embeddings.vec")
embedding = get_embedding(model, "hello")
if embedding !== nothing
    println("Found embedding of dimension \$(length(embedding))")
end
```
"""
function get_embedding(model::WordEmbeddingModel, word::String)
    return get(model.embeddings, word, nothing)
end
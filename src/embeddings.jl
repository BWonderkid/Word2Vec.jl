function get_embedding(model::WordEmbeddingModel, word::String)
    return get(model.embeddings, word, nothing)
end
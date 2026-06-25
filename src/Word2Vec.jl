module Word2Vec

include("loaders.jl")
include("embeddings.jl")

export load_model, save_model, get_embedding, vocab_size, embedding_dim, has_word

end

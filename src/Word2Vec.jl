module Word2Vec

include("loaders.jl")
include("embeddings.jl")
include("similarity.jl")
include("training.jl")

export WordEmbeddingModel, load_model, save_model, get_embedding, vocab_size, embedding_dim, has_word
export cosine_similarity, similarity, most_similar
export train_word2vec, tokenize

end

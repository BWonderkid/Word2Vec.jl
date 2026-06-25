module Word2Vec

include("loaders.jl")
include("embeddings.jl")
include("similarity.jl")
include("training.jl")
include("conec.jl")
include("evaluation.jl")
include("visualization.jl")

export WordEmbeddingModel, load_model, save_model, get_embedding, vocab_size, embedding_dim, has_word
export cosine_similarity, similarity, most_similar
export train_word2vec, tokenize
export ConEcModel, train_conec, get_context_embedding, to_embedding_model
export solve_analogy, evaluate_analogies
export reduce_pca, reduce_tsne, plot_embeddings

end

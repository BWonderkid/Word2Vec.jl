# Word2Vec.jl

Pure Julia implementation of Word2Vec and ConEc word embeddings — train models from scratch, load pre-trained models, query embeddings, evaluate with analogies, and visualize with PCA or t-SNE.

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://BWonderkid.github.io/Word2Vec.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://BWonderkid.github.io/Word2Vec.jl/dev/)
[![Build Status](https://github.com/BWonderkid/Word2Vec.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/BWonderkid/Word2Vec.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/BWonderkid/Word2Vec.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/BWonderkid/Word2Vec.jl)

## Features

- **Load and save** pre-trained models in `.vec` (text) and `.bin` (binary) formats
- **Train** Word2Vec models from scratch using Skip-gram or CBOW with negative sampling
- **ConEc extension** — learns separate word and context embeddings (Horn 2017)
- **Similarity** — cosine similarity between words and nearest-neighbour search
- **Evaluation** — solve word analogies (king − man + woman ≈ queen)
- **Visualization** — project embeddings to 2D with PCA or t-SNE

## Installation

Requires Julia 1.11 or later. Install directly from GitHub:

```julia
using Pkg
Pkg.add(url="https://github.com/BWonderkid/Word2Vec.jl")
```

## Running the Tests

```julia
using Pkg
Pkg.test("Word2Vec")
```

The test suite uses small built-in models (`test/data/tiny.vec` and `test/data/tiny.bin`) so no downloads are needed and everything runs in seconds.

## Usage

### Loading and saving a pre-trained model

```julia
using Word2Vec

# load from text (.vec) or binary (.bin) format
model = load_model("path/to/embeddings.vec")

vocab_size(model)       # number of words
embedding_dim(model)    # vector dimension
has_word(model, "cat")  # true / false

vec = get_embedding(model, "cat")  # Vector{Float32} or nothing

# save to a different format
save_model(model, "embeddings.bin")
```

Popular pre-trained models that work out of the box:
- [FastText English vectors](https://fasttext.cc/docs/en/english-vectors.html) (`.vec`)
- [Google News Word2Vec](https://code.google.com/archive/p/word2vec/) (`.bin`)

### Training a model from scratch

```julia
using Word2Vec

# tokenize raw text
tokens = tokenize("the cat sat on the mat the cat ate the rat")

# train with Skip-gram (default)
model = train_word2vec(tokens, dim=100, window=5, epochs=10, min_count=1)

# or pass raw text directly — tokenization is done automatically
model = train_word2vec("the cat sat on the mat"; dim=100, epochs=10)

# switch to CBOW
model = train_word2vec(tokens; dim=100, architecture=:cbow)
```

Key keyword arguments:

| Argument | Default | Description |
|---|---|---|
| `dim` | `100` | Embedding dimensionality |
| `window` | `5` | Context window radius |
| `min_count` | `1` | Minimum word frequency |
| `epochs` | `5` | Training passes over the corpus |
| `negative` | `5` | Negative samples per positive pair |
| `architecture` | `:skipgram` | `:skipgram` or `:cbow` |

### Similarity

```julia
# cosine similarity between two words
similarity(model, "cat", "dog")   # Float32 between -1 and 1

# find the 5 most similar words
most_similar(model, "cat", 5)
# => [("dog", 0.91), ("kitten", 0.88), ...]

# raw vector similarity
cosine_similarity(vec_a, vec_b)
```

### ConEc extension

ConEc (Horn 2017) learns two separate embeddings per word: one for when the word is the **target** and one for when it appears as **context**.

```julia
# train a ConEc model
model = train_conec(tokens; dim=100, window=5, epochs=10)

# word embedding (from W matrix)
get_embedding(model, "cat")

# context embedding (from C matrix)
get_context_embedding(model, "cat")

# convert to a standard WordEmbeddingModel
# combined=true averages W and C, which often improves similarity tasks
wem = to_embedding_model(model, combined=true)
similarity(wem, "cat", "dog")
```

### Analogy evaluation

```julia
# solve: man : king :: woman : ?
results = solve_analogy(model, "man", "king", "woman")
# => [("queen", 0.87)]

# batch evaluation — returns accuracy, correct count, total, skipped
analogies = [
    ("man",   "king",   "woman", "queen"),
    ("paris", "france", "rome",  "italy"),
]
result = evaluate_analogies(model, analogies)
println("Accuracy: $(round(result.accuracy * 100, digits=1))%")
```

### Visualization

```julia
words = ["cat", "dog", "king", "queen", "man", "woman", "paris", "france"]

# PCA (fast, deterministic)
p = plot_embeddings(model, words, method=:pca)
savefig(p, "embeddings_pca.png")

# t-SNE (slower, better cluster separation)
p = plot_embeddings(model, words, method=:tsne, perplexity=5.0)
savefig(p, "embeddings_tsne.png")

# get the raw 2D coordinates without plotting
X = reduce(hcat, [get_embedding(model, w) for w in words])'  |> Matrix{Float32}
coords = reduce_pca(X, dims=2)   # n×2 matrix
coords = reduce_tsne(X, dims=2)  # n×2 matrix
```

## References

- Mikolov et al. (2013) — [Distributed Representations of Words and Phrases](https://papers.nips.cc/paper/5021-distributed-representations-of-words-and-phrases-and-their-compositionality.pdf)
- Horn (2017) — [Context encoders as a simple but powerful extension of word2vec](https://arxiv.org/abs/1706.02496)

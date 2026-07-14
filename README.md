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

Requires Julia 1.11 or later. For a fresh clone of this repository, activate the project and instantiate the dependencies:

```julia
using Pkg
Pkg.activate(".")
Pkg.instantiate()
```

To install directly from GitHub:

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

## Example models download

The repository includes a helper script for downloading example models into `models/`.

```julia
include("scripts/AddExampleFiles.jl")
add_example_models()
```

You can pass `silent=true` to disable progress messages:

```julia
add_example_models(; silent=true)
```

## Example datasets download

The repository includes a helper script for downloading example datasets into `data/`
and generating reusable toy datasets.

If you are running this from a local clone, activate the project first and instantiate the dependencies before calling the helper.

```julia
include("scripts/AddExampleFiles.jl")
add_example_datasets()
```

You can pass `silent=true` to disable progress messages:

```julia
add_example_datasets(; silent=true)
```

## Generate toy datasets

To generate toy datasets from the downloaded source text:

```julia
include("scripts/PrepareToyDataset.jl")
generate_toy_datasets(joinpath("data", "text8"); out_dir=joinpath("test", "data"))
```

You can adjust the output directory, number of files, and line counts as needed: 

```julia
generate_toy_datasets(
    joinpath("data", "text8");
    out_dir=joinpath("test", "data"),
    sizes=[2000],      # tokens per dataset file
    n_per_size=3,      # how many files per size
    n_lines=20,        # number of lines per file
    seed=42,
    prefix="toy_dataset"
)
```

This generates files like:
- `test/data/toy_dataset_1.txt`
- `test/data/toy_dataset_2.txt`
- `test/data/toy_dataset_3.txt`

Pre-generated toy datasets are also included under `test/data/` for direct use.

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

If you run `add_example_models()`, the example model files mentioned above will appear in the `models/` folder.

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
| `learning_rate` | `0.025f0` | Initial learning rate, decayed linearly during training |
| `negative` | `5` | Negative samples per positive pair |
| `architecture` | `:skipgram` | `:skipgram` or `:cbow` |

Calling `train_word2vec(corpus)` without keywords uses the defaults in this table.

### Toy demo for REPL

A small REPL-friendly demo is available in `scripts/ToyDemo.jl`.

```julia
include("scripts/ToyDemo.jl")
model = run_toy_demo()
```

`run_toy_demo` input priority is:
1. `dataset_path` (if provided)
2. `text` (if `dataset_path` is not provided)
3. built-in default toy text (if neither is provided)

If `query_words` is not provided (or is empty), it is selected automatically from the most frequent tokens.

```julia
# from dataset file
model = run_toy_demo(
    dataset_path=joinpath("test", "data", "toy_dataset_1.txt"),
    dim=20,
    window=2,
    epochs=10,
    architecture=:cbow
)

# from custom raw text
model = run_toy_demo(
    text="toy car doll block ball teddy bear train puzzle",
    query_words=["toy", "car", "doll"],
    dim=20,
    window=2,
    epochs=10,
    architecture=:cbow
)

# optional save
run_toy_demo(
    dataset_path=joinpath("test", "data", "toy_dataset_1.txt"),
    model_out_path=joinpath("models", "toy_model.vec"),
    architecture=:cbow
)
```

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
- [text8 dataset](http://mattmahoney.net/dc/text8.zip)
- [GoogleNews-vectors-negative300.bin](https://huggingface.co/NathaNn1111/word2vec-google-news-negative-300-bin/resolve/main/GoogleNews-vectors-negative300.bin)
- [cc.en.300.vec.gz](https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.en.300.vec.gz)

# Training & Features

## Training a Word2Vec Model

You can train a model from scratch using Skip-gram (default) or CBOW with negative sampling. Pass either a pre-tokenized list of words or a raw string — tokenization is done automatically.

```julia
using Word2Vec

# tokenize raw text yourself
tokens = tokenize("the cat sat on the mat the cat ate the rat")

# train with Skip-gram (default)
model = train_word2vec(tokens; dim=100, window=5, epochs=10, min_count=1)

# or pass raw text directly
model = train_word2vec("the cat sat on the mat"; dim=100, epochs=10)

# switch to CBOW
model = train_word2vec(tokens; dim=100, architecture=:cbow)
```

**Key keyword arguments:**

| Argument | Default | Description |
|---|---|---|
| `dim` | `100` | Embedding dimensionality |
| `window` | `5` | Context window radius |
| `min_count` | `1` | Minimum word frequency to include |
| `epochs` | `5` | Training passes over the corpus |
| `learning_rate` | `0.025f0` | Initial learning rate, decayed linearly during training |
| `negative` | `5` | Negative samples per positive pair |
| `architecture` | `:skipgram` | `:skipgram` or `:cbow` |

Calling `train_word2vec(corpus)` without keywords uses the defaults in this table.

After training the result is a standard `WordEmbeddingModel`, so all similarity and analogy functions work on it directly.

## ConEc Extension

ConEc (Horn 2017) extends Word2Vec by keeping **two separate embedding matrices** per word: one for when the word is the target (`W`) and one for when it appears as context (`C`). This often improves similarity task performance.

```julia
# train a ConEc model
model = train_conec(tokens; dim=100, window=5, epochs=10)

# word embedding (from the W matrix)
get_embedding(model, "cat")

# context embedding (from the C matrix)
get_context_embedding(model, "cat")

# convert to a standard WordEmbeddingModel
# combined=true averages W and C, which often helps similarity tasks
wem = to_embedding_model(model; combined=true)
similarity(wem, "cat", "dog")
```

`train_conec` accepts the same keyword arguments as `train_word2vec`.

## Analogy Evaluation

Word embeddings capture semantic relationships via vector arithmetic: `king − man + woman ≈ queen`.

```julia
# solve: man : king :: woman : ?
results = solve_analogy(model, "man", "king", "woman")
# => [("queen", 0.87)]

# request more candidates
results = solve_analogy(model, "man", "king", "woman"; n=5)
```

For batch evaluation, pass a list of `(a, b, c, expected)` tuples:

```julia
analogies = [
    ("man",   "king",   "woman", "queen"),
    ("paris", "france", "rome",  "italy"),
]

result = evaluate_analogies(model, analogies)
println("Accuracy: $(round(result.accuracy * 100, digits=1))%")
println("Correct:  $(result.correct) / $(result.total)")
println("Skipped:  $(result.skipped)")   # words not in vocabulary
```

## Visualization

Project embeddings down to 2D for plotting using PCA (fast, deterministic) or t-SNE (slower, better cluster separation).

```julia
words = ["cat", "dog", "king", "queen", "man", "woman", "paris", "france"]

# PCA plot
p = plot_embeddings(model, words; method=:pca)
savefig(p, "embeddings_pca.png")

# t-SNE plot
p = plot_embeddings(model, words; method=:tsne, perplexity=5.0)
savefig(p, "embeddings_tsne.png")
```

You can also get the raw 2D coordinates without plotting:

```julia
X = Matrix{Float32}(reduce(hcat, [get_embedding(model, w) for w in words])')

coords = reduce_pca(X; dims=2)   # n×2 Float32 matrix
coords = reduce_tsne(X; dims=2)  # n×2 Float32 matrix
```

## References

- Mikolov et al. (2013) — [Distributed Representations of Words and Phrases](https://papers.nips.cc/paper/5021-distributed-representations-of-words-and-phrases-and-their-compositionality.pdf)
- Horn (2017) — [Context encoders as a simple but powerful extension of word2vec](https://arxiv.org/abs/1706.02496)

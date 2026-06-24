# Getting Started

## Installation

To use Word2Vec.jl, first make sure you have installed the package:

```julia
using Pkg
Pkg.add("Word2Vec")
```

## Loading a Model

Word2Vec.jl supports loading pre-trained word embedding models in two formats:

- `.vec` - Text format (e.g., FastText models)
- `.bin` - Binary format (e.g., Word2Vec models)

To load a model, simply use the `load_model` function:

```@repl
using Word2Vec

# Load a model from a file
model = load_model(joinpath(pkgdir(Word2Vec), "test", "data", "tiny.vec"))
```

## Retrieving Word Embeddings

Once you have loaded a model, you can retrieve the embedding vector for any word using `get_embedding`:

```@repl
using Word2Vec

# Load the tiny model
model = load_model(joinpath(pkgdir(Word2Vec), "test", "data", "tiny.vec"))

# Get the embedding for a word
embedding = get_embedding(model, "hello")

# Check if the word exists in the model
if embedding !== nothing
    println("Found embedding with dimension: $(length(embedding))")
else
    println("Word not found in model")
end
```

## Working with Embeddings

Once you have an embedding vector, you can perform various operations such as:

- **Computing similarity** between words using distance metrics
- **Finding nearest neighbors** in embedding space
- **Performing arithmetic operations** on embeddings

```julia
# Example: Get embeddings for multiple words
using Word2Vec

model = load_model(joinpath(pkgdir(Word2Vec), "test", "data", "tiny.vec"))
words = ["the", "a", "is"]
embeddings = [get_embedding(model, word) for word in words if get_embedding(model, word) !== nothing]

# You can then use these embeddings for downstream tasks
```

## Supported File Formats

### Text Format (.vec)

The text format is human-readable and commonly used for FastText models. The file structure is:

```
vocab_size embedding_dim
word1 value1 value2 ... valueN
word2 value1 value2 ... valueN
...
```

### Binary Format (.bin)

The binary format is more space-efficient. Use `load_model` or `load_bin` to load binary models:

```julia
model = load_model("path/to/embeddings.bin")
```

## Example: Complete Workflow

```@example
using Word2Vec

# Load the tiny model included in the package
model = load_model(joinpath(pkgdir(Word2Vec), "test", "data", "tiny.vec"))

# Get embedding for a word
embedding = get_embedding(model, "the")

# Check the dimensionality
if embedding !== nothing
    println("Embedding dimension: $(length(embedding))")
    println("First 5 values: $(embedding[1:5])")
end
```

For more detailed API documentation, see the [Home](index.md) page.
```
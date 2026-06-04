# Word2Vec

Pure Julia implementation for loading pretrained Word2Vec embeddings.

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://BWonderkid.github.io/Word2Vec.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://BWonderkid.github.io/Word2Vec.jl/dev/)
[![Build Status](https://github.com/BWonderkid/Word2Vec.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/BWonderkid/Word2Vec.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/BWonderkid/Word2Vec.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/BWonderkid/Word2Vec.jl)

## Features

- Load `.vec` (FastText text format)
- Load `.bin` (Google Word2Vec binary format)
- Query embeddings

## Running Tests

The repository includes small test models:

```text
test/data/tiny.vec
test/data/tiny.bin
```

Run:

```julia
using Pkg
Pkg.test()
```

These files are intentionally tiny so tests run quickly and do not require downloading large pretrained models.

## Usage

Large pretrained models are not included in the repository.

Examples:

- GoogleNews Word2Vec (`.bin`)
- FastText English vectors (`.vec`)

Place downloaded models in a local `models/` directory:

```text
models/
├── GoogleNews-vectors-negative300.bin
└── cc.en.300.vec
```

Then load and query embeddings:

```julia
using Word2Vec

model = load_model("models/cc.en.300.vec")
vec = get_embedding(model, "computer")
```
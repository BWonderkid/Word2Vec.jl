# Scripts

This project includes helper scripts under `scripts/`:

- `AddExampleFiles.jl`
- `PrepareToyDataset.jl`
- `ToyDemo.jl`

## 1) AddExampleFiles.jl

Downloads example models/datasets and handles gzip/zip helpers.

```julia
include(joinpath(pkgdir(Word2Vec), "scripts", "AddExampleFiles.jl"))

# models/
add_example_models()

# data/
add_example_datasets()
```

## 2) PrepareToyDataset.jl

Generates reusable toy datasets from a larger source dataset (e.g. `data/text8`).

```julia
include(joinpath(pkgdir(Word2Vec), "scripts", "PrepareToyDataset.jl"))

generate_toy_datasets(
    joinpath(pkgdir(Word2Vec), "data", "text8");
    out_dir=joinpath(pkgdir(Word2Vec), "test", "data"),
    sizes=[2000],
    n_per_size=3,
    n_lines=20,
    seed=42,
    prefix="toy_dataset"
)
```

## 3) ToyDemo.jl

Runs a REPL-friendly Word2Vec demo from either a dataset file or raw text.

```julia
include(joinpath(pkgdir(Word2Vec), "scripts", "ToyDemo.jl"))

# from dataset file
model = run_toy_demo(dataset_path=joinpath(pkgdir(Word2Vec), "test", "data", "toy_dataset_1.txt"))

# from raw text
model = run_toy_demo(text="toy car doll block toy train puzzle"; architecture=:cbow)

# optional save
run_toy_demo(dataset_path=joinpath(pkgdir(Word2Vec), "test", "data", "toy_dataset_1.txt"),
             model_out_path="toy_model.vec")
```
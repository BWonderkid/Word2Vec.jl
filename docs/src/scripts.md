# Scripts

This package includes helper scripts under `scripts/` for dataset/model setup and quick demo workflows:

- `AddExampleFiles.jl`
- `PrepareToyDataset.jl`
- `ToyDemo.jl`

Typical workflow:

1. Download example files (`AddExampleFiles.jl`)
2. Generate reusable toy datasets (`PrepareToyDataset.jl`)
3. Train and inspect a toy model (`ToyDemo.jl`)

---

## `AddExampleFiles.jl`

This script provides file utilities and download helpers.

### What it does

- Downloads example pre-trained models into `models/`
- Downloads example source datasets into `data/`
- Handles archive operations (`.gz`, `.zip`) via helper functions

### Main functions

- `add_example_models(; silent=false)`
- `add_example_datasets(; silent=false)`
- `gunzip_file_decompress(...)`
- `gunzip_file_compress(...)`
- `zip_file_decompress(...)`
- `zip_file_compress(...)`

### Usage

```julia
include(joinpath(pkgdir(Word2Vec), "scripts", "AddExampleFiles.jl"))

# Download example models into models/
add_example_models()

# Download example datasets into data/
add_example_datasets()

# Optional: quiet mode
add_example_models(; silent=true)
add_example_datasets(; silent=true)
```

---

## `PrepareToyDataset.jl`

This script generates reusable toy training files from a larger source dataset (for example `data/text8`).

### Why use it

Instead of manually writing small texts, you can create deterministic toy datasets with controlled size and line count.

### Main function

- `generate_toy_datasets(dataset_path; out_dir, sizes, n_per_size, n_lines, seed, prefix)`

### Parameters

- `dataset_path`: source text path
- `out_dir`: output folder (default typically `data/` or `test/data/`)
- `sizes`: token counts per generated dataset (e.g. `[1000, 2000]`)
- `n_per_size`: number of files per size
- `n_lines`: number of lines per output file (lines are sampled from different positions)
- `seed`: RNG seed for reproducibility
- `prefix`: output filename prefix (e.g. `toy_dataset`)

### Example

```julia
include(joinpath(pkgdir(Word2Vec), "scripts", "PrepareToyDataset.jl"))

paths = generate_toy_datasets(
    joinpath(pkgdir(Word2Vec), "data", "text8");
    out_dir=joinpath(pkgdir(Word2Vec), "test", "data"),
    sizes=[2000],
    n_per_size=3,
    n_lines=20,
    seed=42,
    prefix="toy_dataset"
)

println(paths)
# -> .../test/data/toy_dataset_1.txt
# -> .../test/data/toy_dataset_2.txt
# -> .../test/data/toy_dataset_3.txt
```

---

## `ToyDemo.jl`

This script runs a REPL-friendly end-to-end toy training demo.

### What it does

- Reads training input from either a dataset file or raw text
- Trains Word2Vec (CBOW or Skip-gram)
- Auto-selects query words if not provided
- Prints basic inspection output
- Optionally saves the trained model

### Main function

- `run_toy_demo(; dataset_path, text, query_words, dim, window, epochs, architecture, model_out_path)`

### Input priority

`run_toy_demo` chooses training input in this order:

1. `dataset_path` (if provided)
2. `text` (if no `dataset_path`)
3. built-in default toy text

If `query_words` is omitted or empty, top frequent tokens are selected automatically.

### Examples

```julia
include(joinpath(pkgdir(Word2Vec), "scripts", "ToyDemo.jl"))

# 1) Train from generated toy dataset file
model = run_toy_demo(
    dataset_path=joinpath(pkgdir(Word2Vec), "test", "data", "toy_dataset_1.txt"),
    architecture=:cbow
)

# 2) Train from raw text
model = run_toy_demo(
    text="toy car doll block toy train puzzle",
    architecture=:cbow
)

# 3) Save trained model
run_toy_demo(
    dataset_path=joinpath(pkgdir(Word2Vec), "test", "data", "toy_dataset_1.txt"),
    model_out_path=joinpath(pkgdir(Word2Vec), "models", "toy_model.vec"),
    architecture=:cbow
)
```

---

## Notes

- For reproducible toy dataset generation, keep `seed` fixed.
- Prefer `test/data/` for files used by tests and demos.
- Prefer `models/` for saved vectors (`.vec` / `.bin`).
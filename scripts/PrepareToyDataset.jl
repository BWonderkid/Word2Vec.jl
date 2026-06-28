using Random
using Word2Vec
include(joinpath(@__DIR__, "AddExampleModels.jl"))

"""
    load_dataset_text(path::AbstractString)::String

Load raw text from a dataset file.

Supported:
- `.txt` / `.text`
- `.gz` (decompressed via `gunzip_file_decompress`)
"""
function load_dataset_text(path::AbstractString)::String
    isfile(path) || throw(ArgumentError("Dataset file not found: $path"))
    lower = lowercase(path)

    if endswith(lower, ".gz")
        tmp = tempname() * ".txt"
        gunzip_file_decompress(String(path), tmp)
        try
            return read(tmp, String)
        finally
            isfile(tmp) && rm(tmp; force=true)
        end
    else
        return read(path, String)
    end
end

"""
    generate_toy_datasets(
        dataset_path::AbstractString;
        out_dir::AbstractString=joinpath(@__DIR__, "..", "test", "data"),
        sizes::Vector{Int}=[2000],
        n_per_size::Int=3,
        n_lines::Int=20,
        seed::Int=42,
        prefix::AbstractString="toy_dataset"
    )::Vector{String}

Generate toy dataset files by sampling random contiguous token windows from
a larger dataset text file.

Each generated file contains `n_lines` lines. Every line is sampled from a
different random start position in the source token stream.
"""
function generate_toy_datasets(
    dataset_path::AbstractString;
    out_dir::AbstractString=joinpath(@__DIR__, "..", "test", "data"),
    sizes::Vector{Int}=[2000],
    n_per_size::Int=3,
    n_lines::Int=20,
    seed::Int=42,
    prefix::AbstractString="toy_dataset"
)::Vector{String}

    n_per_size > 0 || throw(ArgumentError("n_per_size must be > 0"))
    n_lines > 0 || throw(ArgumentError("n_lines must be > 0"))
    all(>(0), sizes) || throw(ArgumentError("All sizes must be > 0"))

    mkpath(out_dir)
    rng = MersenneTwister(seed)

    text = load_dataset_text(dataset_path)
    tokens = tokenize(text)
    N = length(tokens)
    N > 0 || throw(ArgumentError("No tokens found in dataset: $dataset_path"))

    generated = String[]
    file_idx = 1

    for sz in sizes
        N >= sz || throw(ArgumentError("Requested size $sz exceeds token count $N"))

        line_len = max(1, cld(sz, n_lines))
        max_start = N - line_len + 1
        max_start >= n_lines || throw(ArgumentError("Dataset too short for $n_lines lines of length $line_len"))

        for _ in 1:n_per_size
            starts = randperm(rng, max_start)[1:n_lines]
            lines = Vector{String}(undef, n_lines)

            for (i, start_idx) in enumerate(starts)
                slice = tokens[start_idx:(start_idx + line_len - 1)]
                lines[i] = join(slice, " ")
            end

            out_path = joinpath(out_dir, "$(prefix)_$(file_idx).txt")
            open(out_path, "w") do io
                write(io, join(lines, "\n"))
            end

            push!(generated, out_path)
            file_idx += 1
        end
    end

    return generated
end
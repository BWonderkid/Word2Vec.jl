abstract type EmbeddingModel end

struct WordEmbeddingModel <: EmbeddingModel
    embeddings::Dict{String, Vector{Float32}}
    dim::Int
end

"""
    load_model(path::String)::WordEmbeddingModel

Load a word embedding model from a file.

Supports both `.vec` (text format) and `.bin` (binary format) files.

# Arguments
- `path::String`: Path to the embedding model file

# Returns
- `WordEmbeddingModel`: A struct containing word embeddings and dimension

# Throws
- `error`: If the file format is not recognized or supported

# Example
```julia
model = load_model("models/embeddings.vec")
```
"""
function load_model(path::AbstractString)
    if endswith(lowercase(path), ".vec")
        return load_vec(path)
    elseif endswith(lowercase(path), ".bin")
        return load_bin(path)
    else
        error("Unsupported format: $path")
    end
end

"""
    load_vec(path::String)::WordEmbeddingModel

Load word embeddings from a text format (.vec) file.

The file format should have a header line with vocabulary size and dimension,
followed by one word and its embedding vector per line.

# Arguments
- `path::String`: Path to the .vec format embedding file

# Returns
- `WordEmbeddingModel`: A struct containing word embeddings and dimension

# Example
```julia
model = load_vec("embeddings.vec")
```
"""
function load_vec(path::AbstractString)
    embeddings = Dict{String, Vector{Float32}}()

    open(path, "r") do io
        header = split(readline(io))
        dim = parse(Int, header[2])

        for line in eachline(io)
            parts = split(line)
            length(parts) < 2 && continue

            word = parts[1]
            vec = parse.(Float32, parts[2:end])

            embeddings[word] = vec
        end

        return WordEmbeddingModel(embeddings, dim)
    end
end

"""
    load_bin(path::String)::WordEmbeddingModel

Load word embeddings from a binary format (.bin) file.

The file format should have a text header with vocabulary size and dimension,
followed by binary-encoded word-vector pairs.

# Arguments
- `path::String`: Path to the .bin format embedding file

# Returns
- `WordEmbeddingModel`: A struct containing word embeddings and dimension

# Example
```julia
model = load_bin("embeddings.bin")
```
"""
function load_bin(path::AbstractString)
    open(path, "r") do io

        # header: vocab_size dim
        header = split(readuntil(io, '\n'))
        vocab_size = parse(Int, header[1])
        dim = parse(Int, header[2])

        embeddings = Dict{String, Vector{Float32}}()

        for _ in 1:vocab_size
            word = read_word(io)
            vec = read_vector(io, dim)
            embeddings[word] = vec
        end

        return WordEmbeddingModel(embeddings, dim)
    end
end

"""
    read_word(io)::String

Read a null-terminated word from a binary stream.

Reads bytes from the IO stream until a space character (0x20) is encountered,
then returns the accumulated bytes as a String.

# Arguments
- `io`: An IO stream positioned at a word in binary format

# Returns
- `String`: The word read from the stream

# Internal
This is an internal helper function used by `load_bin`.
"""
function read_word(io)
    bytes = UInt8[]

    while true
        b = read(io, UInt8)
        if b == 0x20  # space
            break
        end
        push!(bytes, b)
    end

    return String(bytes)
end

"""
    read_vector(io, dim::Int)::Vector{Float32}

Read a binary-encoded embedding vector from a stream.

Reads `dim` number of Float32 values from the IO stream.

# Arguments
- `io`: An IO stream positioned at a vector in binary format
- `dim::Int`: The dimensionality of the embedding vector

# Returns
- `Vector{Float32}`: The embedding vector

# Internal
This is an internal helper function used by `load_bin`.
"""
function read_vector(io, dim::Int)
    vec = Vector{Float32}(undef, dim)
    read!(io, vec)
    return vec
end
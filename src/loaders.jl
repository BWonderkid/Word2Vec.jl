struct WordEmbeddingModel
    embeddings::Dict{String, Vector{Float32}}
    dim::Int
end

"""
    vocab_size(model::WordEmbeddingModel)::Int

Return the number of words stored in the embedding model.
"""
vocab_size(model::WordEmbeddingModel) = length(model.embeddings)

"""
    embedding_dim(model::WordEmbeddingModel)::Int

Return the dimensionality of the embedding vectors stored in the model.
"""
embedding_dim(model::WordEmbeddingModel) = model.dim

"""
    has_word(model::WordEmbeddingModel, word::AbstractString)::Bool

Return `true` if `word` exists in the model vocabulary.
"""
has_word(model::WordEmbeddingModel, word::AbstractString) = haskey(model.embeddings, word)

"""
    load_model(path::String)::WordEmbeddingModel

Load a word embedding model from a file.

Supports both `.vec` (text format) and `.bin` (binary format) files.

# Arguments
- `path::String`: Path to the embedding model file

# Returns
- `WordEmbeddingModel`: A struct containing word embeddings and dimension

# Throws
- `ArgumentError`: If the file format is not recognized or supported

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
        throw(ArgumentError("Unsupported format: $path"))
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
    save_model(model::WordEmbeddingModel, path::String)

Save a word embedding model to a file.

Supports both `.vec` (text format) and `.bin` (binary format) files.

# Arguments
- `model::WordEmbeddingModel`: The model to save
- `path::String`: Destination file path

# Throws
- `ArgumentError`: If the file extension is not `.vec` or `.bin`

# Example
```julia
model = load_model("embeddings.vec")
save_model(model, "embeddings_copy.bin")
```
"""
function save_model(model::WordEmbeddingModel, path::AbstractString)
    if endswith(lowercase(path), ".vec")
        save_vec(model, path)
    elseif endswith(lowercase(path), ".bin")
        save_bin(model, path)
    else
        throw(ArgumentError("Unsupported format: $path"))
    end
end

"""
    save_vec(model::WordEmbeddingModel, path::String)

Save word embeddings to a text format (.vec) file.

# Arguments
- `model::WordEmbeddingModel`: The model to save
- `path::String`: Destination file path
"""
function save_vec(model::WordEmbeddingModel, path::AbstractString)
    open(path, "w") do io
        println(io, "$(vocab_size(model)) $(model.dim)")
        for (word, vec) in model.embeddings
            println(io, "$word $(join(vec, ' '))")
        end
    end
end

"""
    save_bin(model::WordEmbeddingModel, path::String)

Save word embeddings to a binary format (.bin) file.

# Arguments
- `model::WordEmbeddingModel`: The model to save
- `path::String`: Destination file path
"""
function save_bin(model::WordEmbeddingModel, path::AbstractString)
    open(path, "w") do io
        write(io, "$(vocab_size(model)) $(model.dim)\n")
        for (word, vec) in model.embeddings
            write(io, word)
            write(io, UInt8(0x20))
            write(io, vec)
        end
    end
end

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


function read_vector(io, dim::Int)
    vec = Vector{Float32}(undef, dim)
    read!(io, vec)
    return vec
end
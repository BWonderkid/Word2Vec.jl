abstract type EmbeddingModel end

struct WordEmbeddingModel <: EmbeddingModel
    embeddings::Dict{String, Vector{Float32}}
    dim::Int
end

function load_model(path::String)
    if endswith(lowercase(path), ".vec")
        return load_vec(path)
    elseif endswith(lowercase(path), ".bin")
        return load_bin(path)
    else
        error("Unsupported format: $path")
    end
end

function load_vec(path::String)::WordEmbeddingModel
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

function load_bin(path::String)::WordEmbeddingModel
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
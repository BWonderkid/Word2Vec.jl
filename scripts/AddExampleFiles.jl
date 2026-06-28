using CodecZlib
using ZipFile

"""
    gunzip_file_decompress(input_path::String, output_path::String)

Decompresses a `.gz` file from `input_path` and writes the decompressed
contents to `output_path`.
"""
function gunzip_file_decompress(input_path::String, output_path::String)
    open(GzipDecompressorStream, input_path, "r") do gzip_stream
        open(output_path, "w") do output_file
            write(output_file, read(gzip_stream))
        end
    end
end

"""
    gunzip_file_compress(input_path::String, output_path::String)

Compresses a file from `input_path` into a `.gz` file at `output_path`.
"""

function gunzip_file_compress(input_path::String, output_path::String)
    open(GzipCompressorStream, output_path, "w") do gzip_stream
        open(input_path, "r") do input_file
            write(gzip_stream, read(input_file))
        end
    end
end

"""
    zip_file_decompress(input_path::String, output_path::String; member_name::Union{Nothing,String}=nothing)

Extract a file from a `.zip` archive and write it to `output_path`.

If `member_name` is not provided, the first file in the archive is extracted.
"""
function zip_file_decompress(input_path::String, output_path::String; member_name::Union{Nothing,String}=nothing)
    ZipFile.Reader(input_path) do zf
        isempty(zf.files) && error("Zip archive is empty: $input_path")

        entry = if member_name === nothing
            zf.files[1]
        else
            found = findfirst(f -> f.name == member_name, zf.files)
            found === nothing && error("File '$member_name' not found in archive: $input_path")
            zf.files[found]
        end

        mkpath(dirname(output_path))
        open(output_path, "w") do output_file
            write(output_file, read(entry))
        end
    end
end

"""
    zip_file_compress(input_path::String, output_path::String)

Compress `input_path` into a `.zip` archive at `output_path`.
"""
function zip_file_compress(input_path::String, output_path::String)
    isfile(input_path) || error("Input file not found: $input_path")
    mkpath(dirname(output_path))

    ZipFile.Writer(output_path) do zf
        ZipFile.addfile(zf, basename(input_path)) do file
            open(input_path, "r") do input_file
                write(file, read(input_file))
            end
        end
    end
end

"""
    add_example_models(; silent::Bool=false)

Downloads the example model files into `models/` and decompresses the
compressed FastText vector file.

If `silent` is `false`, progress messages are printed.
"""
function add_example_models(; silent::Bool=false)
    mkpath("models")
    !silent && println("Downloading example models...")
    !silent && println("Downloading GoogleNews-vectors-negative300.bin...")
    download("https://huggingface.co/NathaNn1111/word2vec-google-news-negative-300-bin/resolve/main/GoogleNews-vectors-negative300.bin", "models/GoogleNews-vectors-negative300.bin")
    !silent && println("Downloading cc.en.300.vec.gz...")
    download("https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.en.300.vec.gz", "models/cc.en.300.vec.gz")
    !silent && println("Unzipping cc.en.300.vec.gz...")
    gunzip_file_decompress("models/cc.en.300.vec.gz", "models/cc.en.300.vec")
    !silent && println("The example models have been downloaded and unzipped successfully.")
end

function add_example_datasets(; silent::Bool=false)
    mkpath("data")
    !silent && println("Downloading example datasets...")
    !silent && println("Downloading text8.zip")
    download("http://mattmahoney.net/dc/text8.zip", "data/text8.zip")
    !silent && println("Unzipping text8.zip...")
    zip_file_decompress("data/text8.zip", "data/text8")
    !silent && println("The example datasets have been downloaded successfully.")
end


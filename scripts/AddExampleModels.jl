using CodecZlib

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


"""
    solve_analogy(model::WordEmbeddingModel, a::String, b::String, c::String; n::Int=1)

Solve the word analogy **a : b :: c : ?** using vector arithmetic.

The answer is estimated as the word whose embedding is closest to `b - a + c`.
The three input words are excluded from the results.

# Arguments
- `model::WordEmbeddingModel`: the loaded or trained embedding model
- `a`, `b`, `c`: the three analogy words (e.g., "man", "king", "woman")
- `n::Int=1`: number of candidate answers to return

# Returns
- `Vector{Tuple{String, Float32}}`: top `n` `(word, score)` pairs, or `nothing` if
  any input word is missing from the vocabulary

# Example
```julia
model   = load_model("embeddings.vec")
results = solve_analogy(model, "man", "king", "woman")
# expected top result: ("queen", score)
```
"""
function solve_analogy(model::WordEmbeddingModel, a::String, b::String, c::String;
                       n::Int=1)
    va = get_embedding(model, a)
    vb = get_embedding(model, b)
    vc = get_embedding(model, c)
    any(isnothing, (va, vb, vc)) && return nothing

    target   = vb .- va .+ vc
    excluded = Set([a, b, c])

    scores = [(w, cosine_similarity(target, vec))
              for (w, vec) in model.embeddings if w ∉ excluded]
    sort!(scores, by=x -> x[2], rev=true)
    return first(scores, n)
end

"""
    evaluate_analogies(model::WordEmbeddingModel, analogies)

Evaluate model accuracy on a set of word analogies.

Each analogy is a 4-tuple `(a, b, c, expected)` representing the task
**a : b :: c : expected** (e.g., `("man", "king", "woman", "queen")`).
An analogy is counted as correct if `expected` is the top-1 result of
`solve_analogy`.

# Arguments
- `model::WordEmbeddingModel`: the embedding model to evaluate
- `analogies`: a collection of `(a, b, c, expected)` string 4-tuples

# Returns
A `NamedTuple` with:
- `accuracy::Float64`: fraction of non-skipped analogies answered correctly
- `correct::Int`: number of correct answers
- `total::Int`: number of analogies attempted (after skipping)
- `skipped::Int`: number skipped because a word was missing from the vocabulary

# Example
```julia
analogies = [
    ("man",   "king",   "woman", "queen"),
    ("paris", "france", "rome",  "italy"),
]
result = evaluate_analogies(model, analogies)
println("Accuracy: \$(round(result.accuracy * 100, digits=1))%")
```
"""
function evaluate_analogies(model::WordEmbeddingModel, analogies)
    correct = 0
    skipped = 0

    for (a, b, c, expected) in analogies
        result = solve_analogy(model, a, b, c; n=1)
        if result === nothing
            skipped += 1
            continue
        end
        result[1][1] == expected && (correct += 1)
    end

    total = length(analogies) - skipped
    return (accuracy=total > 0 ? correct / total : 0.0,
            correct=correct,
            total=total,
            skipped=skipped)
end

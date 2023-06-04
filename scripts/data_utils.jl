struct Dataset{A, B, C, D}
    features::A
    labels::B
    indices::C
    feature_indices::D
    N::Int
end

function iterate_batch!(dataset::Dataset)
    batch_size = length(dataset.indices)
    dataset.indices .= ((dataset.indices .+ (batch_size - 1)) .% dataset.N) .+ 1
    nothing
end
function current_batch(dataset::Dataset)
    return (view(dataset.features, dataset.feature_indices..., dataset.indices), view(dataset.labels, dataset.indices)) 
end

include("mnist_utils.jl")
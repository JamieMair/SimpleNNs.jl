function create_flux_model(img_size, in_channels, device::Symbol)
    conv_to_dense_units = reduce(*, (img_size .- (5, 5) .+ 1) .- (3, 3) .+ 1)*4
    model = Flux.Chain(
        Flux.Conv((5,5), in_channels=>16, Flux.tanh_fast; bias=randn(Float32, 16)),
        Flux.Conv((3,3), 16=>4, Flux.relu; bias=randn(Float32, 4)),
        x->Flux.flatten(x),
        Flux.Dense(conv_to_dense_units, 32, Flux.tanh_fast),    
        Flux.Dense(32, 10),
    )
    if device == :gpu
        return model |> Flux.gpu
    else
        return model
    end
end
function create_simple_nn_model(img_size, in_channels, device::Symbol)
    model = SimpleNNs.chain(
        SimpleNNs.Static((img_size..., in_channels)),
        SimpleNNs.Conv((5,5), 16; activation_fn=SimpleNNs.tanh_fast),
        SimpleNNs.Conv((3,3), 4; activation_fn=SimpleNNs.relu),
        SimpleNNs.Flatten(),
        SimpleNNs.Dense(32, activation_fn=SimpleNNs.tanh_fast),
        SimpleNNs.Dense(10, activation_fn=SimpleNNs.identity)
    )
    if device == :gpu
        return model |> SimpleNNs.GPU.gpu
    else
        return model
    end
end

struct Dataset{A, B, C, D}
    features::A
    labels::B
    indices::C
    feature_indices::D
    N::Int
end

function load_data(batch_size::Int, device::Symbol)
    use_gpu = (device==:gpu)
    to_device = use_gpu ? Flux.gpu : identity
    dataset = MNIST(:train);
    images, labels = dataset[:];


    images =  reshape(images |> to_device, 28, 28, 1, :)
    labels =  (labels .+ 1) |> to_device

    feature_indices = (1:28, 1:28, 1:1)
    N = length(labels)
    indices = collect(1:batch_size) |> to_device
    return Dataset(images, labels, indices, feature_indices, N)
end

function iterate_batch!(dataset::Dataset)
    batch_size = length(dataset.indices)
    dataset.indices .= ((dataset.indices .+ (batch_size - 1)) .% dataset.N) .+ 1
    nothing
end
function current_batch(dataset::Dataset)
    return (view(dataset.features, dataset.feature_indices..., dataset.indices), view(dataset.labels, dataset.indices)) 
end

import SimpleNNs
import SimpleNNs.GPU
using Random
using MLDatasets
using ProgressBars
import CUDA
using BenchmarkTools


function create_simple_nn_model(img_size, in_channels, device::Symbol)
    model = SimpleNNs.chain(
        SimpleNNs.Static((img_size..., in_channels)),
        SimpleNNs.Conv((5,5), 16; activation_fn=SimpleNNs.relu),
        SimpleNNs.Conv((3,3), 4; activation_fn=SimpleNNs.relu),
        SimpleNNs.Flatten(),
        SimpleNNs.Dense(32, activation_fn=SimpleNNs.relu),
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
    to_device = use_gpu ? SimpleNNs.GPU.gpu : identity
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


device = :gpu;
batch_size = 128;
dataset = load_data(batch_size, device);

img_size = (28,28);
in_channels = 1;
models = [create_simple_nn_model(img_size, in_channels, device) for _ in 1:16];
params = [m.parameters for m in models];
caches = [SimpleNNs.preallocate(m, batch_size) for m in models];
input_size = (img_size..., in_channels, batch_size);
batch_features, batch_indices = current_batch(dataset);
for (c, p) in zip(caches, params)
    randn!(p)
    p.*= (1/1000)
    SimpleNNs.set_inputs!(c, reshape(batch_features, input_size));
end

function infer_single!(cache, model, num_epochs)
    for _ in 1:num_epochs
        SimpleNNs.forward!(cache, model)
    end
end

function infer!(caches, models, num_epochs, num_models)
    Threads.@sync begin
        for i in 1:num_models
            Threads.@spawn infer_single!(caches[i], models[i], num_epochs)
        end
    end
    nothing
end
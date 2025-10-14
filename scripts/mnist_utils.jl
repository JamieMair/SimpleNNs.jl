import Flux
using MLDatasets
function load_mnist_data(batch_size::Int, device::Symbol)
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
function load_mnist_test_data(batch_size::Int, device::Symbol)
    use_gpu = (device==:gpu)
    to_device = use_gpu ? Flux.gpu : identity
    dataset = MNIST(:test);
    images, labels = dataset[:];


    images =  reshape(images |> to_device, 28, 28, 1, :)
    labels =  (labels .+ 1) |> to_device

    feature_indices = (1:28, 1:28, 1:1)
    N = length(labels)
    indices = collect(1:batch_size) |> to_device
    return Dataset(images, labels, indices, feature_indices, N)
end
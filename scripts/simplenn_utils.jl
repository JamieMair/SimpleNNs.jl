import SimpleNNs

function create_simple_nn_mnist_model(img_size, in_channels, device::Symbol)
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

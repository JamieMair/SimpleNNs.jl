import Flux

function create_flux_mnist_model(img_size, in_channels, device::Symbol)
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

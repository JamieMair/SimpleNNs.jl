using SimpleNNs
import SimpleNNs: conv

img = Float64.(reshape(1:25, 5, 5, 1))
kernel = Float64.(reshape(1:9, 3, 3))
kernel_size = size(kernel)
conv_layer = Conv(1, Val(false), Val(Float64), size(img), 1, kernel_size, identity)

parameters = (kernel, )
output = zeros(eltype(img), map(x->x-2, size(img)[1:2])..., 1)

SimpleNNs.forward!(output, conv_layer, parameters, img)

using SimpleNNs

function test()
    img_size = (9, 9)
    kernel_size = (3,3)
    in_channels = 1
    out_channels = 3
    model = chain(
        Static((img_size..., in_channels)),
        Conv(kernel_size, out_channels; activation_fn=relu),
        Flatten(), # 147 outputs 
        Dense(10, activation_fn=relu),
        Dense(5, activation_fn=relu),
        Dense(1)
    )
    batch_size = 4
    input_size = (img_size..., in_channels, batch_size)
    forward_cache = preallocate(model, batch_size)
    set_inputs!(forward_cache, reshape(1:reduce(*, input_size), input_size))

    forward!(forward_cache, model)
end

@enter test()
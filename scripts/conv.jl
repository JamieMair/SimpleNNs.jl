using SimpleNNs

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

# TODO: Error in the forward pass calculation, output dims should be (W-k_1+1)by(H-k_2+1). This needs changing.
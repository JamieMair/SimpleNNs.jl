@testitem "Convolutional NN Init" begin
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
    num_parameters = reduce(*, kernel_size) * in_channels * out_channels + out_channels
    num_parameters += (mapreduce((x,y)->x-2*(y ÷ 2), *, img_size, kernel_size) * out_channels) * 10 + 10
    num_parameters += 10*5 + 5
    num_parameters += 5 + 1
    @test length(model.layers) == 6
    @test length(model.parameters) == num_parameters
end

@testitem "Convolutional NN Forward" begin
    img_size = (9, 9)
    kernel_size = (3,3)
    in_channels = 1
    out_channels = 3
    model = chain(
        Static((img_size..., in_channels)),
        Conv((3,3), 3; activation_fn=relu),
        Flatten(), # 147 outputs 
        Dense(10, activation_fn=relu),
        Dense(5, activation_fn=relu),
        Dense(1)
    )
    batch_size = 4
    input_size = (img_size..., in_channels, batch_size)
    forward_cache = preallocate(model, batch_size)
    set_inputs!(forward_cache, reshape(1:reduce(*, input_size), input_size))

    @test typeof(forward!(forward_cache, model)) <: Any
end

@testitem "Convolutional NN Forward Multi-Channel" begin
    img_size = (9, 9)
    kernel_size = (3,3)
    in_channels = 6
    out_channels = 3
    model = chain(
        Static((img_size..., in_channels)),
        Conv((3,3), 3; activation_fn=relu),
        Flatten(), # 147 outputs 
        Dense(10, activation_fn=relu),
        Dense(5, activation_fn=relu),
        Dense(1)
    )
    batch_size = 4
    input_size = (img_size..., in_channels, batch_size)
    forward_cache = preallocate(model, batch_size)
    set_inputs!(forward_cache, reshape(1:reduce(*, input_size), input_size))

    @test typeof(forward!(forward_cache, model)) <: Any
end

@testitem "Convolutional NN Forward Asymmetric Kernel" begin
    img_size = (9, 9)
    kernel_size = (5,3)
    in_channels = 2
    out_channels = 1
    model = chain(
        Static((img_size..., in_channels)),
        Conv((3,3), 3; activation_fn=relu),
        Flatten(), # 147 outputs 
        Dense(10, activation_fn=relu),
        Dense(5, activation_fn=relu),
        Dense(1)
    )
    batch_size = 4
    input_size = (img_size..., in_channels, batch_size)
    forward_cache = preallocate(model, batch_size)
    set_inputs!(forward_cache, reshape(1:reduce(*, input_size), input_size))

    @test typeof(forward!(forward_cache, model)) <: Any
end
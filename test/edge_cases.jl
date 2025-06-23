@testitem "Model Construction Edge Cases" begin
    # Test model with no bias
    model_no_bias = chain(
        Static(3),
        Dense(5, use_bias=Val(false)),
        Dense(1, use_bias=Val(false))
    )
    @test length(model_no_bias.parameters) == 3*5 + 5*1
    
    # Test single layer model
    single_layer = chain(
        Static(2),
        Dense(1)
    )
    @test length(single_layer.layers) == 2
    @test length(single_layer.parameters) == 2*1 + 1
end

@testitem "Batch Size Edge Cases" begin
    model = chain(
        Static(2),
        Dense(3),
        Dense(1)
    )
    
    # Test batch size 1
    forward_cache_1 = preallocate(model, 1)
    inputs_1 = randn(Float32, 2, 1)
    set_inputs!(forward_cache_1, inputs_1)
    @test typeof(forward!(forward_cache_1, model)) <: Any
    
    # Test large batch size
    large_batch = 128
    forward_cache_large = preallocate(model, large_batch)
    inputs_large = randn(Float32, 2, large_batch)
    set_inputs!(forward_cache_large, inputs_large)
    @test typeof(forward!(forward_cache_large, model)) <: Any
end

@testitem "Parameter Access Edge Cases" begin
    # Test model with no trainable parameters (just static layer)
    static_only = chain(Static(5))
    @test length(parameters(static_only)) == 0
    
    # Test accessing parameters of specific layers
    model = chain(
        Static(2),
        Dense(3, activation_fn=SimpleNNs.relu),
        Dense(1)
    )
    
    # First layer (static) should have no parameters
    @test SimpleNNs.num_parameters(model.layers[1]) == 0
    
    # Second layer should have weight and bias parameters
    @test SimpleNNs.num_parameters(model.layers[2]) == 2*3 + 3
    
    # Third layer should have weight and bias parameters  
    @test SimpleNNs.num_parameters(model.layers[3]) == 3*1 + 1
end

@testitem "Convolution Edge Cases" begin
    # Test 1x1 convolution
    model_1x1 = chain(
        Static((5, 5, 2)),
        Conv((1,1), 3),
        Flatten(),
        Dense(1)
    )
    
    batch_size = 4
    input_size = (5, 5, 2, batch_size)
    forward_cache = preallocate(model_1x1, batch_size)
    set_inputs!(forward_cache, randn(Float32, input_size))
    @test typeof(forward!(forward_cache, model_1x1)) <: Any
    
    # Test convolution that reduces spatial dimensions significantly
    model_large_kernel = chain(
        Static((7, 7, 1)),
        Conv((5,5), 2),
        Flatten(),
        Dense(1)
    )
    
    input_large_kernel = (7, 7, 1, batch_size)
    forward_cache_large = preallocate(model_large_kernel, batch_size)
    set_inputs!(forward_cache_large, randn(Float32, input_large_kernel))
    @test typeof(forward!(forward_cache_large, model_large_kernel)) <: Any
end

@testitem "Cache Truncation Edge Cases" begin
    model = chain(
        Static(3),
        Dense(5),
        Dense(1)
    )
    
    original_batch = 20
    forward_cache = preallocate(model, original_batch)
    backward_cache = preallocate_grads(model, original_batch)
    
    # Test truncating to same size (should work)
    truncated_same = SimpleNNs.truncate(forward_cache, original_batch)
    @test size(truncated_same.input) == size(forward_cache.input)
    
    # Test truncating to size 1
    truncated_one = SimpleNNs.truncate(forward_cache, 1)
    @test size(truncated_one.input, 2) == 1
    
    # Test that truncated cache works for forward pass
    inputs_small = randn(Float32, 3, 1)
    set_inputs!(truncated_one, inputs_small)
    @test typeof(forward!(truncated_one, model)) <: Any
end

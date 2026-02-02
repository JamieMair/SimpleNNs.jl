@testitem "Dense Network Performance" begin
    using Random
    Random.seed!(42)
    
    model = chain(
        Static(8),
        Dense(32, activation_fn=SimpleNNs.relu),
        Dense(16, activation_fn=SimpleNNs.tanh),
        Dense(1, activation_fn=identity)
    )
    
    batch_size = 64
    inputs = randn(Float32, 8, batch_size)
    targets = randn(Float32, 1, batch_size)
    loss = SimpleNNs.MSELoss(targets)
    
    # Preallocate
    forward_cache = preallocate(model, batch_size)
    backward_cache = preallocate_grads(model, batch_size)
    set_inputs!(forward_cache, inputs)
    
    # Warm up
    forward!(forward_cache, model)
    backprop!(backward_cache, forward_cache, model, loss)
    
    # Test forward pass allocations
    forward_allocs = @allocations forward!(forward_cache, model)
    @test forward_allocs == 0
    
    # Test backward pass allocations
    backward_allocs = @allocations backprop!(backward_cache, forward_cache, model, loss)
    @test backward_allocs == 0
end

@testitem "Convolutional Network Performance" begin
    using Random
    Random.seed!(42)
    
    img_size = (8, 8)
    model = chain(
        Static((img_size..., 2)),
        Conv((3,3), 4, activation_fn=SimpleNNs.relu),
        Flatten(),
        Dense(5, activation_fn=SimpleNNs.relu),
        Dense(1)
    )
    
    batch_size = 16
    input_size = (img_size..., 2, batch_size)
    inputs = randn(Float32, input_size)
    targets = randn(Float32, 1, batch_size)
    loss = SimpleNNs.MSELoss(targets)
    
    # Preallocate
    forward_cache = preallocate(model, batch_size)
    backward_cache = preallocate_grads(model, batch_size)
    set_inputs!(forward_cache, inputs)
    
    # Warm up
    forward!(forward_cache, model)
    backprop!(backward_cache, forward_cache, model, loss)
    
    # Test forward pass allocations
    forward_allocs = @allocations forward!(forward_cache, model)
    @test forward_allocs == 0
    
    # Test backward pass allocations  
    backward_allocs = @allocations backprop!(backward_cache, forward_cache, model, loss)
    @test backward_allocs == 0
end

@testitem "Cache Truncation Performance" begin
    using Random
    Random.seed!(42)
    
    model = chain(
        Static(4),
        Dense(8, activation_fn=SimpleNNs.relu),
        Dense(1)
    )
    
    full_batch_size = 32
    small_batch_size = 16
    
    inputs_full = randn(Float32, 4, full_batch_size)
    inputs_small = randn(Float32, 4, small_batch_size)
    
    # Preallocate for full batch
    forward_cache = preallocate(model, full_batch_size)
    backward_cache = preallocate_grads(model, full_batch_size)
    
    # Run the truncations to avoid picking up extra allocations in JIT
    SimpleNNs.truncate(forward_cache, small_batch_size)
    SimpleNNs.truncate(backward_cache, small_batch_size)
    # Test truncation creates minimal allocations (1 is usually unavoidable - maybe compiler could get it eventually)
    truncate_allocs = @allocations SimpleNNs.truncate(forward_cache, small_batch_size)
    @test truncate_allocs <= 1
    
    truncate_back_allocs = @allocations SimpleNNs.truncate(backward_cache, small_batch_size)
    @test truncate_back_allocs <=1 
end

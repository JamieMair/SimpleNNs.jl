@testitem "MSE Loss Forward and Backward" begin
    using Random
    Random.seed!(42)
    
    model = chain(
        Static(3),
        Dense(5, activation_fn=SimpleNNs.tanh),
        Dense(1, activation_fn=identity)
    )
    
    batch_size = 16
    inputs = randn(Float32, 3, batch_size)
    targets = randn(Float32, 1, batch_size)
    loss = SimpleNNs.MSELoss(targets)
    
    # Test forward pass
    forward_cache = preallocate(model, batch_size)
    set_inputs!(forward_cache, inputs)
    forward!(forward_cache, model)
    
    outputs = get_outputs(forward_cache)
    @test size(outputs) == (1, batch_size)
    
    # Test backward pass
    backward_cache = preallocate_grads(model, batch_size)
    total_loss = backprop!(backward_cache, forward_cache, model, loss)
    
    @test total_loss isa Real
    @test total_loss >= 0
    @test length(gradients(backward_cache)) == length(model.parameters)
end

@testitem "Logit Cross Entropy Loss" begin
    using Random
    Random.seed!(42)
    
    num_classes = 5
    model = chain(
        Static(10),
        Dense(16, activation_fn=SimpleNNs.relu),
        Dense(num_classes, activation_fn=identity)
    )
    
    batch_size = 32
    inputs = randn(Float32, 10, batch_size)
    targets = rand(1:num_classes, batch_size)
    loss = SimpleNNs.LogitCrossEntropyLoss(targets, num_classes)
    
    # Test forward pass
    forward_cache = preallocate(model, batch_size)
    set_inputs!(forward_cache, inputs)
    forward!(forward_cache, model)
    
    outputs = get_outputs(forward_cache)
    @test size(outputs) == (num_classes, batch_size)
    
    # Test backward pass
    backward_cache = preallocate_grads(model, batch_size)
    total_loss = backprop!(backward_cache, forward_cache, model, loss)
    
    @test total_loss isa Real
    @test total_loss >= 0
    @test length(gradients(backward_cache)) == length(model.parameters)
end

@testitem "Loss Truncation" begin
    batch_size = 20
    smaller_batch = 10
    
    # MSE Loss truncation
    targets_mse = randn(Float32, 2, batch_size)
    loss_mse = SimpleNNs.MSELoss(targets_mse)
    truncated_mse = SimpleNNs.truncate(loss_mse, smaller_batch)
    @test size(truncated_mse.targets) == (2, smaller_batch)
    
    # Cross Entropy Loss truncation
    targets_ce = rand(1:5, batch_size)
    loss_ce = SimpleNNs.LogitCrossEntropyLoss(targets_ce, 5)
    truncated_ce = SimpleNNs.truncate(loss_ce, smaller_batch)
    @test length(truncated_ce.targets) == smaller_batch
end

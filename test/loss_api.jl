@testitem "Loss API - has_loss, add_loss, remove_loss" begin
    using Random
    Random.seed!(42)
    
    # Create a basic model without loss
    model = chain(
        Static(10),
        Dense(32, activation_fn=SimpleNNs.relu),
        Dense(5, activation_fn=identity)
    )
    
    @test has_loss(model) == false
    
    # Test add_loss
    batch_size = 16
    targets = zeros(Int, batch_size)
    loss_layer = BatchCrossEntropyLoss(targets=targets, num_classes=5)
    
    model_with_loss = add_loss(model, loss_layer)
    @test has_loss(model_with_loss) == true
    @test length(model_with_loss.layers) == length(model.layers) + 1
    @test length(model_with_loss.parameters) == length(model.parameters)  # Loss has no params
    
    # Test remove_loss
    model_restored = remove_loss(model_with_loss)
    @test has_loss(model_restored) == false
    @test length(model_restored.layers) == length(model.layers)
    @test length(model_restored.parameters) == length(model.parameters)
    
    # Test that parameters are preserved
    model.parameters .= randn(Float32, length(model.parameters))
    model_with_loss2 = add_loss(model, loss_layer)
    @test all(model_with_loss2.parameters .≈ model.parameters)
    
    model_restored2 = remove_loss(model_with_loss2)
    @test all(model_restored2.parameters .≈ model.parameters)
end

@testitem "Loss API - get_predictions without loss" begin
    using Random
    Random.seed!(42)
    
    model = chain(
        Static(3),
        Dense(8, activation_fn=SimpleNNs.tanh),
        Dense(2, activation_fn=identity)
    )
    
    batch_size = 16
    inputs = randn(Float32, 3, batch_size)
    
    forward_cache = preallocate(model, batch_size)
    set_inputs!(forward_cache, inputs)
    forward!(forward_cache, model)
    
    # Without loss, predictions should be the same as outputs
    predictions = get_predictions(model, forward_cache)
    outputs = get_outputs(forward_cache)
    
    @test predictions === outputs
    @test size(predictions) == (2, batch_size)
end

@testitem "Loss API - get_predictions with loss" begin
    using Random
    Random.seed!(42)
    
    model = chain(
        Static(3),
        Dense(8, activation_fn=SimpleNNs.tanh),
        Dense(5, activation_fn=identity)
    )
    
    batch_size = 16
    targets = rand(1:5, batch_size)
    loss_layer = BatchCrossEntropyLoss(targets=targets, num_classes=5)
    
    model_with_loss = add_loss(model, loss_layer)
    
    inputs = randn(Float32, 3, batch_size)
    
    forward_cache = preallocate(model_with_loss, batch_size)
    set_inputs!(forward_cache, inputs)
    forward!(forward_cache, model_with_loss)
    
    # With loss, predictions should be the output before the loss layer
    predictions = get_predictions(model_with_loss, forward_cache)
    final_output = get_outputs(forward_cache)
    
    @test predictions !== final_output  # Should be different arrays
    @test size(predictions) == (5, batch_size)  # Predictions before loss
    @test size(final_output) == (1, batch_size)  # Loss output (scalar per batch)
end

@testitem "Loss API - warning when adding loss to model with loss" begin
    model = chain(
        Static(5),
        Dense(3, activation_fn=identity)
    )
    
    loss1 = BatchCrossEntropyLoss(targets=zeros(Int, 10), num_classes=3)
    model_with_loss = add_loss(model, loss1)
    
    # Adding loss to a model that already has one should warn
    loss2 = BatchCrossEntropyLoss(targets=zeros(Int, 10), num_classes=3)
    @test_logs (:warn, r"Model already has a loss layer") add_loss(model_with_loss, loss2)
end

@testitem "Loss API - warning when removing loss from model without loss" begin
    model = chain(
        Static(5),
        Dense(3, activation_fn=identity)
    )
    
    # Removing loss from a model without one should warn
    @test_logs (:warn, r"Model does not have a loss layer") remove_loss(model)
end

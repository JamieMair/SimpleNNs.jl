
@testitem "Backward pass no allocations" begin
    using Random
    
    model = chain(
        Static(4), # Layer 1
        Dense(16, activation_fn=SimpleNNs.tanh), # Layer 2
        Dense(8, activation_fn=SimpleNNs.tanh), # Layer 3
        Dense(1, activation_fn=identity, use_bias=Val(false)), # Layer 4
    );    
    batch_size = 32;
    inputs = rand(Float32, 4, batch_size)
    outputs = rand(Float32, 1, batch_size) .*2 .- 1;
    loss = SimpleNNs.MSELoss(outputs)
    randn!(model.parameters)
    model.parameters .*= 0.01
    # Preallocate Model
    forward_cache = preallocate(model, batch_size)
    backward_cache = preallocate_grads(model, batch_size)
    set_inputs!(forward_cache, inputs) # Copies inputs into the cache
    
    # Dry run for compilation
    forward!(forward_cache, model)
    function test_fn(backward_cache, forward_cache, model, loss)
        backprop!(backward_cache, forward_cache, model, loss)
        nothing
    end
    test_fn(backward_cache, forward_cache, model, loss)

    num_allocations = @allocations test_fn(backward_cache, forward_cache, model, loss)
    @test num_allocations == 0
end
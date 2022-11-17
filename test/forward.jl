@testitem "Dense Forward Pass" begin
    square(x) = x*x

    model = chain(
        Static(1), # Layer 1
        Dense(4, activation_fn=square), # Layer 2
        Dense(1), # Layer 3
    );
    (weights_2, biases_2) = parameters(model.layers[2])
    (weights_3, biases_3) = parameters(model.layers[3])
    weights_2 .= 1:4
    biases_2 .= -1
    weights_3 .= 1
    biases_3 .= 1
    
    inputs = reshape(Float32[1.2f0, -1.8f0, 0.0f0, 5.0f0], 1, :)
    expected_outputs = (reshape(weights_3, 1, 4) * square.(reshape(weights_2, 4, 1)*inputs .+ biases_2) .+ biases_3)
    # Preallocate Model
    forward_cache = preallocate(model, length(inputs))
    set_inputs!(forward_cache, inputs) # Copies inputs into the cache
    
    @test typeof(forward!(forward_cache, model)) <: Any
    
    outputs = get_outputs(forward_cache)
    
    @test isapprox(expected_outputs, outputs)
end
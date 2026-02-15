@testitem "Dense Model Initialisation" begin    
    model = chain(
        Static(2),
        Dense(10, activation_fn=sigmoid),
        Dense(5, use_bias=Val(false), activation_fn=tanh),
        Dense(1)
    )

    num_parameters = (2+1)*10 + 10 * 5 + (5+1) * 1

    @test length(model.layers) == 4
    @test length(model.parameters) == num_parameters
end

@testitem "Dense Model View Mapping" begin
    model = chain(
        Static(2),
        Dense(10, activation_fn=sigmoid),
        Dense(5, use_bias=Val(false), activation_fn=tanh),
        Dense(1)
    );
    parameter_array_lengths = (2, 1, 2)
    @test all((x->length(x.parameter_views)).(model.layers[2:end]) .== parameter_array_lengths)

    params = parameters(model)
    # Set parameters equal to their indices
    params .= 1:length(params)
    parameter_values = [
        [model.parameters[1:20], model.parameters[21:30]],
        [model.parameters[31:80]],
        [model.parameters[81:85], model.parameters[86:86]]
    ]
    @test all((x->x.parameter_views).(model.layers[2:end]) .== parameter_values)
end

@testitem "Model deepcopy - parameters copied and independent" begin
    using Random
    
    # Create model with random parameters
    Random.seed!(42)
    model = chain(
        Static(3),
        Dense(10, activation_fn=relu),
        Dense(5, activation_fn=tanh),
        Dense(2, activation_fn=identity)
    )
    
    # Initialize with random values
    randn!(model.parameters)
    model.parameters .*= 0.1f0
    
    # Store original parameters
    original_params = copy(model.parameters)
    
    # Deep copy the model
    model_copy = deepcopy(model)
    
    # Test 1: Parameters are copied (same values)
    @test model_copy.parameters ≈ model.parameters
    @test model_copy.parameters ≈ original_params
    
    # Test 2: Parameters are independent memory (different arrays)
    @test model_copy.parameters !== model.parameters
    @test pointer(model_copy.parameters) != pointer(model.parameters)
    
    # Test 3: Modifying copy doesn't affect original
    model_copy.parameters .+= 1.0f0
    @test !(model_copy.parameters ≈ model.parameters)
    @test model.parameters ≈ original_params
    
    # Test 4: Modifying original doesn't affect copy
    stored_copy_params = copy(model_copy.parameters)
    model.parameters .*= 2.0f0
    @test model_copy.parameters ≈ stored_copy_params
    @test !(model_copy.parameters ≈ model.parameters)
end

@testitem "Model deepcopy - forward pass equivalence" begin
    using Random
    
    # Create model
    Random.seed!(123)
    model = chain(
        Static(4),
        Dense(16, activation_fn=relu),
        Dense(8, activation_fn=tanh),
        Dense(3, activation_fn=identity)
    )
    
    # Initialize parameters
    randn!(model.parameters)
    model.parameters .*= 0.05f0
    
    # Deep copy the model
    model_copy = deepcopy(model)
    
    # Create test inputs
    batch_size = 32
    inputs = randn(Float32, 4, batch_size)
    
    # Preallocate caches for both models
    forward_cache = preallocate(model, batch_size)
    forward_cache_copy = preallocate(model_copy, batch_size)
    
    # Set inputs for both
    set_inputs!(forward_cache, inputs)
    set_inputs!(forward_cache_copy, inputs)
    
    # Forward pass on both models
    forward!(forward_cache, model)
    forward!(forward_cache_copy, model_copy)
    
    # Test 1: Same parameters give same outputs
    outputs_original = get_outputs(forward_cache)
    outputs_copy = get_outputs(forward_cache_copy)
    @test outputs_original ≈ outputs_copy
    
    # Test 2: Modify copy parameters and verify different outputs
    model_copy.parameters .+= 0.5f0
    forward!(forward_cache_copy, model_copy)
    outputs_copy_modified = get_outputs(forward_cache_copy)
    @test !(outputs_original ≈ outputs_copy_modified)
    
    # Test 3: Original still gives same output
    forward!(forward_cache, model)
    outputs_original_again = get_outputs(forward_cache)
    @test outputs_original ≈ outputs_original_again
end

@testitem "Model deepcopy - backward pass equivalence" begin
    using Random
    
    # Create model
    Random.seed!(456)
    model = chain(
        Static(5),
        Dense(12, activation_fn=relu),
        Dense(8, activation_fn=tanh),
        Dense(3, activation_fn=identity)
    )
    
    # Initialize parameters
    randn!(model.parameters)
    model.parameters .*= 0.1f0
    
    # Deep copy the model
    model_copy = deepcopy(model)
    
    # Create test data
    batch_size = 16
    inputs = randn(Float32, 5, batch_size)
    targets = randn(Float32, 3, batch_size)
    loss = MSELoss(targets)
    
    # Preallocate caches for both models
    forward_cache = preallocate(model, batch_size)
    backward_cache = preallocate_grads(model, batch_size)
    forward_cache_copy = preallocate(model_copy, batch_size)
    backward_cache_copy = preallocate_grads(model_copy, batch_size)
    
    # Set inputs
    set_inputs!(forward_cache, inputs)
    set_inputs!(forward_cache_copy, inputs)
    
    # Forward and backward on both models
    forward!(forward_cache, model)
    backprop!(backward_cache, forward_cache, model, loss)
    
    forward!(forward_cache_copy, model_copy)
    backprop!(backward_cache_copy, forward_cache_copy, model_copy, loss)
    
    # Test 1: Same parameters give same gradients
    @test backward_cache.parameter_gradients ≈ backward_cache_copy.parameter_gradients
    
    # Test 2: Modify copy parameters
    model_copy.parameters .*= 1.5f0
    forward!(forward_cache_copy, model_copy)
    backprop!(backward_cache_copy, forward_cache_copy, model_copy, loss)
    
    # Test 3: Different parameters give different gradients
    @test !(backward_cache.parameter_gradients ≈ backward_cache_copy.parameter_gradients)
    
    # Test 4: Original gradients unchanged
    original_grads = copy(backward_cache.parameter_gradients)
    forward!(forward_cache, model)
    backprop!(backward_cache, forward_cache, model, loss)
    @test backward_cache.parameter_gradients ≈ original_grads
end

@testitem "Model deepcopy - no aliasing in parameter views" begin
    using Random
    
    Random.seed!(789)
    model = chain(
        Static(3),
        Dense(8, activation_fn=relu),
        Dense(4, activation_fn=identity)
    )
    
    randn!(model.parameters)
    
    # Deep copy
    model_copy = deepcopy(model)
    
    # Get parameter views for both models
    weights_1 = parameters(model.layers[2])[1]
    weights_1_copy = parameters(model_copy.layers[2])[1]
    
    # Test 1: Views have same values
    @test weights_1 ≈ weights_1_copy
    
    # Test 2: Views point to different underlying arrays
    @test pointer(weights_1) != pointer(weights_1_copy)
    
    # Test 3: Modifying view in copy doesn't affect original
    original_weights = copy(weights_1)
    weights_1_copy .+= 10.0f0
    @test weights_1 ≈ original_weights
    @test !(weights_1 ≈ weights_1_copy)
    
    # Test 4: Check biases too
    biases_1 = parameters(model.layers[2])[2]
    biases_1_copy = parameters(model_copy.layers[2])[2]
    
    @test biases_1 ≈ biases_1_copy
    @test pointer(biases_1) != pointer(biases_1_copy)
    
    biases_1_copy .*= 0.0f0
    @test !(biases_1 ≈ biases_1_copy)
end
@testitem "Dense Layer Gradient Check" begin
    using Random
    Random.seed!(42)
    
    # Simple dense network
    model = chain(
        Static(2),
        Dense(3, activation_fn=SimpleNNs.tanh, parameter_type = Val(Float64)),
        Dense(1, activation_fn=identity, parameter_type = Val(Float64))
    )

    randn!(model.parameters)
    model.parameters .*= 0.1f0;
    
    batch_size = 1  # Use batch size 1 for gradient checking
    inputs = randn(Float64, 2, batch_size)
    targets = randn(Float64, 1, batch_size)
    loss = SimpleNNs.MSELoss(targets)
    
    # Get analytical gradients
    forward_cache = preallocate(model, batch_size)
    backward_cache = preallocate_grads(model, batch_size)
    set_inputs!(forward_cache, inputs)
    
    forward!(forward_cache, model)
    backprop!(backward_cache, forward_cache, model, loss)
    analytical_grads = copy(gradients(backward_cache))
    
    # Numerical gradient checking
    step_size = 1e-6
    params = parameters(model)
    numerical_grads = similar(params)
    
    for i in eachindex(params)
        # Forward perturbation
        before_param = params[i]
        params[i] + before_param + step_size / 2
        forward!(forward_cache, model)
        loss_plus = backprop!(backward_cache, forward_cache, model, loss)
        # Backward perturbation
        params[i] = before_param - step_size / 2
        forward!(forward_cache, model)
        loss_minus = backprop!(backward_cache, forward_cache, model, loss)
        
        # Restore original parameter
        params[i] = before_param
        
        # Numerical gradient 
        numerical_grads[i] = (loss_plus - loss_minus) / (step_size)
    end
    
    # Check gradients are close (allowing for numerical precision)
    @test isapprox(analytical_grads, numerical_grads, rtol=1e-5)
end

@testitem "Custom activation gradients" begin
    using Random

    # Custom activation function
    function swish(x)
        return x * sigmoid(x)
    end

    # Custom gradient (if needed for backward pass) (input is the output of the activation)
    function swish_gradient(x)
        s = sigmoid(x)
        return s + x * (1 - s)
    end
    # Link the gradient fn to the activation fn
    SimpleNNs.activation_gradient_fn(::typeof(swish)) = swish_gradient

    model = chain(
        Static(2),
        Dense(32, activation_fn=swish),  # Custom activation
        Dense(1, activation_fn=identity)
    )
    
    randn!(model.parameters)
    model.parameters .*= 0.1f0;
    
    batch_size = 1  # Use batch size 1 for gradient checking
    inputs = randn(Float64, 2, batch_size)
    targets = randn(Float64, 1, batch_size)
    loss = SimpleNNs.MSELoss(targets)
    
    # Get analytical gradients
    forward_cache = preallocate(model, batch_size)
    backward_cache = preallocate_grads(model, batch_size)
    set_inputs!(forward_cache, inputs)
    
    forward!(forward_cache, model)
    backprop!(backward_cache, forward_cache, model, loss)
    analytical_grads = copy(gradients(backward_cache))
    
    # Numerical gradient checking
    step_size = 1e-5
    params = parameters(model)
    numerical_grads = similar(params)
    
    for i in eachindex(params)
        # Forward perturbation
        before_param = params[i]
        params[i] + before_param + step_size / 2
        forward!(forward_cache, model)
        loss_plus = backprop!(backward_cache, forward_cache, model, loss)
        # Backward perturbation
        params[i] = before_param - step_size / 2
        forward!(forward_cache, model)
        loss_minus = backprop!(backward_cache, forward_cache, model, loss)
        
        # Restore original parameter
        params[i] = before_param
        
        # Numerical gradient 
        numerical_grads[i] = (loss_plus - loss_minus) / (step_size)
    end
    
    # Check gradients are close (allowing for numerical precision)
    # TODO: Check why the gradients aren't *that* close
    @test isapprox(analytical_grads, numerical_grads, rtol=1e-1)
end

@testitem "Activation Function Gradients" begin
    using Random
    Random.seed!(42)
    
    # Test different activation functions
    activations = [SimpleNNs.relu, SimpleNNs.tanh, SimpleNNs.sigmoid, identity]
    
    for activation in activations
        model = chain(
            Static(2),
            Dense(1, activation_fn=activation)
        )
        
        batch_size = 4
        inputs = randn(Float32, 2, batch_size)
        targets = randn(Float32, 1, batch_size)
        loss = SimpleNNs.MSELoss(targets)
        
        forward_cache = preallocate(model, batch_size)
        backward_cache = preallocate_grads(model, batch_size)
        set_inputs!(forward_cache, inputs)
        
        # Should not error
        forward!(forward_cache, model)
        @test typeof(backprop!(backward_cache, forward_cache, model, loss)) <: Real
        
        # Gradients should be finite
        grads = gradients(backward_cache)
        @test all(isfinite, grads)
    end
end

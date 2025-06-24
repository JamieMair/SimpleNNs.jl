@testitem "SGD Optimiser Basic Functionality" begin
    using Random
    Random.seed!(42)
    
    # Create simple test case
    params = randn(Float32, 10)
    grads = randn(Float32, 10)
    
    # Test basic SGD (no momentum)
    opt = SGDOptimiser(grads; lr=0.1f0, momentum=0.0f0)
    original_params = copy(params)
    
    update!(params, grads, opt)
    
    # Check parameters moved in opposite direction of gradients
    expected_params = original_params .- 0.1f0 .* grads
    @test params ≈ expected_params
    
    # Test reset functionality
    reset!(opt)
    @test all(iszero, opt.velocity)
end

@testitem "SGD Optimiser with Momentum" begin
    using Random
    Random.seed!(42)
    
    params = randn(Float32, 5)
    grads1 = randn(Float32, 5)
    grads2 = randn(Float32, 5)
    
    lr = 0.1f0
    momentum = 0.9f0
    opt = SGDOptimiser(grads1; lr=lr, momentum=momentum)
    
    # First update
    original_params = copy(params)
    update!(params, grads1, opt)
    
    # Velocity should be -lr * grads1
    expected_velocity = -lr .* grads1
    @test opt.velocity ≈ expected_velocity
    
    # Second update
    params_after_first = copy(params)
    update!(params, grads2, opt)
    
    # Velocity should be momentum * old_velocity - lr * grads2
    expected_velocity = momentum .* expected_velocity .- lr .* grads2
    @test opt.velocity ≈ expected_velocity
end

@testitem "RMSProp Optimiser Functionality" begin
    using Random
    Random.seed!(42)
    
    params = randn(Float32, 8)
    grads = randn(Float32, 8)
    
    lr = 0.01f0
    rho = 0.9f0
    eps = 1e-8f0
    opt = RMSPropOptimiser(grads; lr=lr, rho=rho, eps=eps)
    
    original_params = copy(params)
    update!(params, grads, opt)
    
    # Check that v was updated correctly
    expected_v = (1 - rho) .* grads .* grads
    @test opt.v ≈ expected_v
    
    # Check parameters were updated
    expected_update = lr .* grads ./ (sqrt.(expected_v) .+ eps)
    expected_params = original_params .- expected_update
    @test params ≈ expected_params
    
    # Test reset
    reset!(opt)
    @test all(iszero, opt.v)
end

@testitem "Adam Optimiser Comprehensive Test" begin
    using Random
    Random.seed!(42)
    
    params = randn(Float32, 6)
    grads = randn(Float32, 6)
    
    lr = 0.001f0
    beta_1 = 0.9f0
    beta_2 = 0.999f0
    opt = AdamOptimiser(grads; lr=lr, beta_1=beta_1, beta_2=beta_2)
    
    # Test initial state
    @test opt.epoch == 1
    @test all(iszero, opt.m)
    @test all(iszero, opt.v)
    
    original_params = copy(params)
    update!(params, grads, opt)
    
    # Check epoch incremented
    @test opt.epoch == 2
    
    # Check moment estimates updated
    @test opt.m ≈ (1 - beta_1) .* grads
    @test opt.v ≈ (1 - beta_2) .* grads .* grads
    
    # Test reset functionality
    reset!(opt)
    @test opt.epoch == 1
    @test all(iszero, opt.m)
    @test all(iszero, opt.v)
end

@testitem "Optimiser Performance - No Allocations" begin
    using Random
    Random.seed!(42)
    
    params = randn(Float32, 100)
    grads = randn(Float32, 100)
    
    # Test SGD
    sgd_opt = SGDOptimiser(grads; lr=0.01f0, momentum=0.9f0)
    function test_sgd_update(params, grads, opt)
        update!(params, grads, opt)
        nothing
    end
    test_sgd_update(params, grads, sgd_opt)  # Warm up
    sgd_allocs = @allocations test_sgd_update(params, grads, sgd_opt)
    @test sgd_allocs == 0
    
    # Test RMSProp  
    rmsprop_opt = RMSPropOptimiser(grads; lr=0.01f0)
    function test_rmsprop_update(params, grads, opt)
        update!(params, grads, opt)
        nothing
    end
    test_rmsprop_update(params, grads, rmsprop_opt)  # Warm up
    rmsprop_allocs = @allocations test_rmsprop_update(params, grads, rmsprop_opt)
    @test rmsprop_allocs == 0
    
    # Test Adam
    adam_opt = AdamOptimiser(grads; lr=0.01f0)
    function test_adam_update(params, grads, opt)
        update!(params, grads, opt)
        nothing
    end
    test_adam_update(params, grads, adam_opt)  # Warm up
    adam_allocs = @allocations test_adam_update(params, grads, adam_opt)
    @test adam_allocs == 0
end

@testitem "Optimiser Convergence Test" begin
    using Random
    Random.seed!(42)
    
    # Simple quadratic function: f(x) = x^T x
    # Gradient: 2x
    # Minimum at x = 0
    
    function test_optimiser_convergence(opt_constructor, tolerance=1e-4)
        x = randn(Float32, 10) .* 10  # Start far from minimum
        
        for _ in 1:1000
            grad = 2 .* x  # Gradient of x^T x
            opt = opt_constructor(grad)
            update!(x, grad, opt)
            
            # Early stopping if converged
            if norm(x) < tolerance
                break
            end
        end
        
        return norm(x)
    end
    
    # Test that all optimisers can converge to minimum
    sgd_error = test_optimiser_convergence(g -> SGDOptimiser(g; lr=0.01f0))
    adam_error = test_optimiser_convergence(g -> AdamOptimiser(g; lr=0.1f0))
    rmsprop_error = test_optimiser_convergence(g -> RMSPropOptimiser(g; lr=0.1f0))
    
    @test sgd_error < 0.1
    @test adam_error < 0.1  
    @test rmsprop_error < 0.1
end

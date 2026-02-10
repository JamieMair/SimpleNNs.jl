using SimpleNNs
using Test
using Statistics
using Random

@testset "Initialisers" begin
    @testset "GlorotNormal initialization" begin
        model = chain(
            Static(10),
            Dense(20, activation_fn=tanh, init=GlorotNormal()),
            Dense(5, activation_fn=identity, init=GlorotNormal())
        )
        
        # Before initialization, parameters should be zero
        @test all(parameters(model) .== 0)
        
        # Initialise the model
        initialise!(model)
        
        # After initialization, parameters should not all be zero
        @test !all(parameters(model) .== 0)
        
        # Check that the variance is approximately correct for first layer
        layer1 = model.layers[2]  # First Dense layer (after Static)
        weights1 = SimpleNNs.weights(layer1)
        expected_std = sqrt(2 / (10 + 20))
        @test abs(std(weights1) - expected_std) < 0.1
    end
    
    @testset "HeNormal initialization" begin
        model = chain(
            Static(10),
            Dense(20, activation_fn=relu, init=HeNormal()),
            Dense(5, activation_fn=relu, init=HeNormal())
        )
        
        initialise!(model)
        
        # Check that weights are initialised
        @test !all(parameters(model) .== 0)
        
        # Check that the variance is approximately correct for first layer
        layer1 = model.layers[2]
        weights1 = SimpleNNs.weights(layer1)
        expected_std = sqrt(2 / 10)
        @test abs(std(weights1) - expected_std) < 0.1
    end
    
    @testset "GlorotUniform initialization" begin
        model = chain(
            Static(10),
            Dense(20, activation_fn=tanh, init=GlorotUniform())
        )
        
        initialise!(model)
        
        # Check that weights are initialised
        @test !all(parameters(model) .== 0)
        
        # Check that weights are within expected bounds
        layer1 = model.layers[2]
        weights1 = SimpleNNs.weights(layer1)
        expected_limit = sqrt(6 / (10 + 20))
        @test all(abs.(weights1) .<= expected_limit)
    end
    
    @testset "Zeros initialization" begin
        model = chain(
            Static(10),
            Dense(20, activation_fn=identity, init=Zeros())
        )
        
        initialise!(model)
        
        # All parameters should still be zero
        @test all(parameters(model) .== 0)
    end
    
    @testset "Convolutional layer initialization" begin
        model = chain(
            Static((28, 28, 1)),
            Conv((3, 3), 16, activation_fn=relu, init=HeNormal()),
            Flatten(),
            Dense(10, activation_fn=identity, init=GlorotNormal())
        )
        
        initialise!(model)
        
        # Check that parameters are initialised
        @test !all(parameters(model) .== 0)
        
        # Check conv layer weights
        conv_layer = model.layers[2]
        conv_weights = SimpleNNs.weights(conv_layer)
        # fan_in for conv = kernel_sise * in_channels = 3*3*1 = 9
        expected_std = sqrt(2 / 9)
        @test abs(std(conv_weights) - expected_std) < 0.2
    end
    
    @testset "Bias initialization" begin
        model = chain(
            Static(10),
            Dense(20, use_bias=Val(true), init=GlorotNormal())
        )
        
        initialise!(model)
        
        # Biases should be initialised to zero
        layer1 = model.layers[2]
        biases1 = SimpleNNs.biases(layer1)
        @test all(biases1 .== 0)
    end
    
    @testset "Re-initialization" begin
        model = chain(
            Static(10),
            Dense(20, activation_fn=tanh, init=GlorotNormal())
        )
        
        # Initialise twice
        initialise!(model)
        params1 = copy(parameters(model))
        
        initialise!(model)
        params2 = copy(parameters(model))
        
        # Parameters should be different after re-initialization
        @test params1 != params2
    end
end

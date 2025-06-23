@testitem "GPU Model Transfer" begin
    # This test will only run if CUDA is loaded
    if isdefined(Main, :CUDA) && CUDA.functional()
        model_cpu = chain(
            Static(4),
            Dense(8, activation_fn=SimpleNNs.relu),
            Dense(2)
        )
        
        # Test GPU transfer
        model_gpu = SimpleNNs.gpu(model_cpu)
        @test typeof(model_gpu.parameters) <: CUDA.CuArray
        @test length(model_gpu.parameters) == length(model_cpu.parameters)
        @test size(model_gpu.layers) == size(model_cpu.layers)
    else
        @test_skip "CUDA not available, skipping GPU tests"
    end
end

@testitem "GPU Forward Pass" begin
    if isdefined(Main, :CUDA) && CUDA.functional()
        model = chain(
            Static(3),
            Dense(6, activation_fn=SimpleNNs.relu),
            Dense(1)
        )
        
        model_gpu = SimpleNNs.gpu(model)
        batch_size = 16
        inputs_cpu = randn(Float32, 3, batch_size)
        inputs_gpu = CUDA.cu(inputs_cpu)
        
        # CPU forward pass
        forward_cache_cpu = preallocate(model, batch_size)
        set_inputs!(forward_cache_cpu, inputs_cpu)
        forward!(forward_cache_cpu, model)
        outputs_cpu = get_outputs(forward_cache_cpu)
        
        # GPU forward pass
        forward_cache_gpu = preallocate(model_gpu, batch_size)
        set_inputs!(forward_cache_gpu, inputs_gpu)
        forward!(forward_cache_gpu, model_gpu)
        outputs_gpu = get_outputs(forward_cache_gpu)
        
        # Compare results (should be close, accounting for floating point differences)
        @test isapprox(Array(outputs_gpu), outputs_cpu, rtol=1e-5)
    else
        @test_skip "CUDA not available, skipping GPU tests"
    end
end

@testitem "GPU Convolutional Forward Pass" begin
    if isdefined(Main, :CUDA) && CUDA.functional()
        img_size = (6, 6)
        model = chain(
            Static((img_size..., 1)),
            Conv((3,3), 2, activation_fn=SimpleNNs.relu),
            Flatten(),
            Dense(2)
        )
        
        model_gpu = SimpleNNs.gpu(model)
        batch_size = 8
        input_size = (img_size..., 1, batch_size)
        inputs_cpu = randn(Float32, input_size)
        inputs_gpu = CUDA.cu(inputs_cpu)
        
        # GPU forward pass should work
        forward_cache_gpu = preallocate(model_gpu, batch_size)
        set_inputs!(forward_cache_gpu, inputs_gpu)
        @test typeof(forward!(forward_cache_gpu, model_gpu)) <: Any
    else
        @test_skip "CUDA not available, skipping GPU tests"
    end
end

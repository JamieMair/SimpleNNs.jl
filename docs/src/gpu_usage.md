# GPU Usage

SimpleNNs.jl provides full GPU support through CUDA.jl, allowing you to train and run inference on NVIDIA GPUs for significantly improved performance.

## Prerequisites

To use GPU functionality, you need to have the following packages installed:

```julia
using Pkg
Pkg.add(["CUDA", "cuDNN", "NNlib"])
```

These packages must be loaded **before** using SimpleNNs GPU functionality:

```julia
using CUDA
import cuDNN, NNlib # Need to load CUDA, cuDNN and NNlib to enable GPU functionality in SimpleNNs
using SimpleNNs
```

!!! note "GPU Requirements"
    You need an NVIDIA GPU with CUDA Compute Capability 3.5 or higher. Check your GPU compatibility with `CUDA.functional()`.

## Basic GPU Usage

### Moving Models to GPU

The simplest way to use GPU acceleration is with the `gpu` function:

```julia
# Create a model on CPU
model = chain(
    Static(784),  # MNIST flattened images
    Dense(128, activation_fn=relu),
    Dense(64, activation_fn=relu),
    Dense(10, activation_fn=identity)
)

# Move to GPU
gpu_model = gpu(model)
```

### GPU Data Transfer

You can move arrays to the GPU using the same `gpu` function:

```julia
# CPU data
cpu_data = randn(Float32, 784, 100)  # 100 samples

# Move to GPU
gpu_data = gpu(cpu_data)

# You can also use CUDA.cu() directly
gpu_data = CUDA.cu(cpu_data)
```

### Complete GPU Training Example

Here's a complete example showing GPU training:

```julia
using CUDA
import cuDNN, NNlib
using SimpleNNs
using Random

# Check GPU availability
if !CUDA.functional()
    error("CUDA not available!")
end

# Create model and move to GPU
model = chain(
    Static(10),
    Dense(32, activation_fn=tanh),
    Dense(16, activation_fn=relu),
    Dense(1, activation_fn=identity)
) |> gpu

# Generate sample data on GPU
batch_size = 128
inputs = CUDA.randn(Float32, 10, batch_size)
targets = CUDA.randn(Float32, 1, batch_size)

# Preallocate buffers
forward_cache = preallocate(model, batch_size)
backward_cache = preallocate_grads(model, batch_size)

# Set inputs and create loss
set_inputs!(forward_cache, inputs)
loss = MSELoss(targets)

# Initialise parameters
Random.seed!(42)
ps = parameters(model)
randn!(ps)
ps .*= 0.1f0

# Setup optimiser
optimiser = AdamOptimiser(backward_cache.parameter_gradients; lr=0.01f0)

# Training loop
for epoch in 1:1000
    forward!(forward_cache, model)
    total_loss = backprop!(backward_cache, forward_cache, model, loss)
    
    # Apply optimiser
    grads = gradients(backward_cache)
    update!(ps, grads, optimiser)
    
    if epoch % 100 == 0
        println("Epoch $epoch, Loss: $total_loss")
    end
end
```

## GPU Performance Benefits

### Convolutional Networks

GPU acceleration is particularly beneficial for convolutional networks:

```julia
using CUDA, cuDNN, NNlib
using SimpleNNs
using MLDatasets

# Load MNIST data
dataset = MNIST(:train)
images, labels = dataset[:]

# Reshape and move to GPU
images = reshape(images, 28, 28, 1, size(images, 3)) |> gpu
labels = (labels .+ 1) |> gpu

# Create CNN model on GPU
model = chain(
    Static((28, 28, 1)),
    Conv((5,5), 16, activation_fn=relu),
    MaxPool((2,2)),
    Conv((3,3), 8, activation_fn=relu),
    MaxPool((4,4)),
    Flatten(),
    Dense(10, activation_fn=identity)
) |> gpu

batch_size = 64
forward_cache = preallocate(model, batch_size)
backward_cache = preallocate_grads(model, batch_size)

# Training with GPU acceleration
for epoch in 1:100
    # Select random batch
    batch_indices = rand(1:size(images, 4), batch_size)
    batch_images = view(images, :, :, :, batch_indices)
    batch_labels = view(labels, batch_indices)
    
    set_inputs!(forward_cache, batch_images)
    loss = LogitCrossEntropyLoss(batch_labels, 10)
    
    forward!(forward_cache, model)
    total_loss = backprop!(backward_cache, forward_cache, model, loss)
    
    # Apply gradients (simplified)
    ps = parameters(model)
    grads = gradients(backward_cache)
    
    # Use built-in SGD optimiser for demonstration
    if !@isdefined(sgd_opt)
        sgd_opt = SGDOptimiser(grads; lr=0.001f0, momentum=0.9f0)
    end
    update!(ps, grads, sgd_opt)
end
```

## Performance Considerations

## GPU vs CPU Comparison

Here's a simple benchmark comparing GPU and CPU performance:

```julia
using BenchmarkTools

# Create identical models
cpu_model = chain(Static(100), Dense(200, activation_fn=relu), Dense(1))
gpu_model = gpu(cpu_model)

batch_size = 128
cpu_cache = preallocate(cpu_model, batch_size)
gpu_cache = preallocate(gpu_model, batch_size)

# CPU data
cpu_inputs = randn(Float32, 100, batch_size)
set_inputs!(cpu_cache, cpu_inputs)

# GPU data
gpu_inputs = gpu(cpu_inputs)
set_inputs!(gpu_cache, gpu_inputs)

# Benchmark
println("CPU Performance:")
@benchmark forward!($cpu_cache, $cpu_model)

println("GPU Performance:")
@benchmark CUDA.@sync forward!($gpu_cache, $gpu_model)
```


On my machine, I see the following results
```
CPU Benchmark:
BenchmarkTools.Trial: 10000 samples with 1 evaluation per sample.
 Range (min … max):   96.800 μs … 386.100 μs  ┊ GC (min … max): 0.00% … 0.00%
 Time  (median):     105.900 μs               ┊ GC (median):    0.00%
 Time  (mean ± σ):   111.648 μs ±  18.544 μs  ┊ GC (mean ± σ):  0.00% ± 0.00%

  ▁▄▆██▇▇▆▆▆▅▄▄▃▃▃▂▂▁▂▂▁▂▁▁▁▁       ▁                           ▂
  ████████████████████████████████▇████▇█▇██▇▅▇▆▆▅▆▅▃▅▅▄▅▃▅▅▅▄▃ █
  96.8 μs       Histogram: log(frequency) by time        190 μs <

 Memory estimate: 0 bytes, allocs estimate: 0.

GPU Benchmark:
BenchmarkTools.Trial: 10000 samples with 1 evaluation per sample.
 Range (min … max):  67.000 μs … 619.800 μs  ┊ GC (min … max): 0.00% … 0.00%
 Time  (median):     85.700 μs               ┊ GC (median):    0.00%
 Time  (mean ± σ):   96.932 μs ±  42.793 μs  ┊ GC (mean ± σ):  0.00% ± 0.00%

  ▁▁  █▄
  ███████▇▇▆▅▅▅▅▃▃▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▁▁▂▂▂▁▂
  67 μs           Histogram: frequency by time          275 μs <

 Memory estimate: 9.58 KiB, allocs estimate: 342.
```

You can see these small sizes actually show the GPU and CPU having similar performance. Keep this in mind when you are choosing which device to run your small-medium size neural networks on.
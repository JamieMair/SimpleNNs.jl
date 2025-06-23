```@meta
CurrentModule = SimpleNNs
```

# SimpleNNs.jl

`SimpleNNs.jl` is heavily inspired by [`SimpleChains.jl`](https://pumasai.github.io/SimpleChains.jl/stable/), which showed that there is space for micro-optimisations to be very important for small neural networks (see the [blog post](https://julialang.org/blog/2022/04/simple-chains/)). This project aims to expand upon `SimpleChains.jl` by introducing both CPU and GPU support with zero-allocation inference and training.

## Key Features

- **Zero-allocation inference**: Pre-allocated buffers eliminate memory allocations during forward and backward passes
- **GPU acceleration**: Full CUDA support for both training and inference
- **High performance**: Optimised implementations for small to medium-sized networks
- **Memory efficient**: Flat parameter vectors and pre-allocated caches minimise memory usage

## Package Goals

As the name suggests, this is **not** a fully featured neural network library, and most notably, it does not include auto-differentiation capabilities. The specific goals of this package are:

1. **Simple architectures**: Build neural networks with dense and convolutional layers
2. **Flat parameters**: All model parameters stored in a single vector for easy manipulation
3. **Pre-allocated computation**: Zero-allocation forward and backward passes using pre-allocated buffers
4. **Cross-platform**: Execution on both CPU and GPU (CUDA)
5. **High performance**: Optimised for small to medium neural networks where micro-optimisations matter

## Supported Features

### Layer Types
- **Dense layers**: Fully connected layers with customisable activation functions
- **Convolutional layers**: 2D convolutions with ReLU, tanh, and sigmoid activations
- **Pooling layers**: Max pooling with configurable pool sizes and strides
- **Utility layers**: Static input specification and flattening layers

### Activation Functions
- ReLU (`relu`)
- Hyperbolic tangent (`tanh`, `tanh_fast`)
- Logistic sigmoid (`sigmoid`)
- Identity (`identity`)

### Loss Functions  
- Mean Squared Error (`MSELoss`)
- Cross Entropy Loss (`LogitCrossEntropyLoss`)

### GPU Support
- Full CUDA acceleration through `CUDA.jl`, `cuDNN.jl`, and `NNlib.jl`
- Seamless CPU/GPU model transfer with the `gpu()` function
- Optimised GPU kernels for convolution and dense operations

## Quick Start

Here's a minimal example to get you started:

```julia
using SimpleNNs

# Create a simple neural network
model = chain(
    Static(4),                          # 4 input features
    Dense(8, activation_fn=relu),       # Hidden layer with ReLU
    Dense(1, activation_fn=identity)    # Output layer
)

# Generate some data
batch_size = 32
inputs = randn(Float32, 4, batch_size)
targets = randn(Float32, 1, batch_size)

# Pre-allocate computation buffers
forward_cache = preallocate(model, batch_size)
backward_cache = preallocate_grads(model, batch_size)

# Set inputs and run forward pass
set_inputs!(forward_cache, inputs)
forward!(forward_cache, model)
outputs = get_outputs(forward_cache)

# Define loss and run backward pass
loss = MSELoss(targets)
total_loss = backprop!(backward_cache, forward_cache, model, loss)

# Access gradients
grads = gradients(backward_cache)
```

## Performance Philosophy

SimpleNNs.jl is designed around the principle that for small to medium neural networks, careful memory management and micro-optimisations can provide significant performance benefits. Key design decisions include:

- **Pre-allocation**: All memory is allocated upfront, eliminating allocations during computation
- **Flat parameters**: Single parameter vector enables efficient gradient updates and serialisation  
- **Type stability**: Careful type design ensures fast, predictable performance
- **GPU optimisation**: Custom CUDA kernels and NNlib integration for accelerated computation

## When to Use SimpleNNs.jl

SimpleNNs.jl is ideal for:
- **Small to medium networks** where performance matters
- **Embedded applications** requiring minimal memory footprint
- **Research applications** needing fine control over memory and computation
- **GPU-accelerated inference** with minimal overhead
- **Applications requiring many small models** (e.g., ensemble methods)

Consider other frameworks like Flux.jl or Lux.jl for:
- Very large deep learning models
- Complex architectures requiring auto-differentiation
- Research requiring cutting-edge layer types
- Applications where development speed > runtime performance

## Documentation Structure

```@contents
Pages = [
    "getting_started.md",
    "mnist.md", 
    "gpu_usage.md",
    "advanced_usage.md",
    "api.md",
    "function_index.md"
]
Depth = 2
```

## Installation

Add the package using Julia's package manager:

```julia
using Pkg
Pkg.add("https://github.com/JamieMair/SimpleNNs.jl")
```

For GPU support, also install the CUDA ecosystem:

```julia
Pkg.add(["CUDA", "cuDNN", "NNlib"])
```

## Contributing

Contributions are welcome! Please see the GitHub repository for issue tracking and pull requests.

## Acknowledgments

This package is inspired by and builds upon the excellent work in:
- [`SimpleChains.jl`](https://github.com/PumasAI/SimpleChains.jl) - The original high-performance "small" neural network package
- [`NNLib.jl`](https://fluxml.ai/NNlib.jl/stable/) - Provides the CUDA implementation for some of the supported layers
- [`CUDA.jl`](https://github.com/JuliaGPU/CUDA.jl) - Allows seamless GPU support in this package
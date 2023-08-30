```@meta
CurrentModule = SimpleNNs
```

# SimpleNNs

`SimpleNNs.jl` is heavily inspired by [`SimpleChains.jl`](https://pumasai.github.io/SimpleChains.jl/stable/), which showed that there is space for micro-optimisations to be very important for small neural networks (see the [blog post](https://julialang.org/blog/2022/04/simple-chains/)). This project aims to expand upon `SimpleChains.jl` by introducing both CPU and GPU support.

As the name suggests, this is **not** a fully featured neural network library, and most notably, it does not include auto-differentiation capabilities. The goals of this package are the following:
1. To build simple neural network architectures, whose parameters are represented as a simple flat vector.
2. To be able to train and run these neural networks with pre-allocated buffers to avoid memory allocations.
3. To be executable on either the GPU or the CPU.

Currently, there is full support for dense layers with the traditional activation functions: $\text{ReLU}$, $\tanh$ (hyperbolic tangent) and $\sigma$ (logistic sigmoid). Custom loss functions will work on forward passes, but the gradient must be overloaded as detailed in another page of this documentation. Convolutional layers are also supported, with some limitations on parameters such as **stride**.

Head over to the [Getting Started](@ref) page to see how to use this package.

## Documentation Outline

```@contents
Pages = [
    "getting_started.md",
    "mnist.md",
    "api.md",
    "function_index.md"
]
Depth = 2
```
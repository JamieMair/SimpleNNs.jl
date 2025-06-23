"""
    gpu(x)

Move data or models to GPU using CUDA. This function requires CUDA.jl, cuDNN.jl, and NNlib.jl 
to be loaded before use.

# Arguments
- `x`: The object to move to GPU. Can be a `Model`, `AbstractArray`, or other supported types.

# Returns
- GPU version of the input object

# Examples
```julia
using CUDA, cuDNN, NNlib
using SimpleNNs

# Move model to GPU
model = chain(Static(10), Dense(5))
gpu_model = gpu(model)

# Move array to GPU  
cpu_array = randn(Float32, 10, 32)
gpu_array = gpu(cpu_array)
```

# Notes
- Requires NVIDIA GPU with CUDA support
- CUDA.jl, cuDNN.jl, and NNlib.jl must be loaded before calling this function
- For models, creates a new model with parameters on GPU
- For arrays, converts to CuArray
- Returns input unchanged with warning for unsupported types
"""
function gpu(x)
    @warn "Tried to put object of type $(typeof(x)) on the GPU, but unrecognised"
    x
end
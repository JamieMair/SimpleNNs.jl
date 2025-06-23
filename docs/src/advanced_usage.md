# Advanced Usage

This section covers advanced topics and optimisation techniques for SimpleNNs.jl.

## Custom Training Loops

### Function Approximation

Here's a simple example of fitting a non-linear function, using a custom training loop and optimiser (in this case a custom implementation of ADAM):

```julia
using SimpleNNs
using Random
using ProgressBars

# Define a non-linear target function
function target_function(x)
    freq = 4.0
    offset = 1.0
    return x * sin(x * freq - offset) + offset
end

# Create training data
N = 512
inputs = Float32.(reshape(collect(LinRange(-1.0, 1.0, N)), 1, N))
outputs = Float32.(reshape(target_function.(inputs), 1, :))

# Create a deeper network
model = chain(
    Static(1),
    Dense(16, activation_fn=tanh_fast),
    Dense(16, activation_fn=tanh_fast),
    Dense(16, activation_fn=tanh_fast),
    Dense(1, activation_fn=identity)
)

# Initialise parameters
Random.seed!(42)
ps = SimpleNNs.parameters(model)
randn!(ps)
ps .*= 0.1f0

# Preallocate
forward_cache = preallocate(model, N)
backward_cache = preallocate_grads(model, N)
set_inputs!(forward_cache, inputs)

# Loss function
loss = MSELoss(outputs)

# ADAM optimiser parameters
epochs = 5000
lr = 0.02f0 / N
β₁ = 0.9f0
β₂ = 0.999f0
ε = 1f-8

# ADAM state
m = similar(ps)
v = similar(ps)
fill!(m, 0.0f0)
fill!(v, 0.0f0)

losses = Float32[]

for epoch in ProgressBar(1:epochs)
    # Forward pass
    forward!(forward_cache, model)
    
    # Compute loss
    current_loss = backprop!(backward_cache, forward_cache, model, loss)
    push!(losses, current_loss)
    
    # ADAM update
    grads = gradients(backward_cache)
    m .= β₁ .* m .+ (1 - β₁) .* grads
    v .= β₂ .* v .+ (1 - β₂) .* grads .^ 2
    
    m_corrected = m ./ (1 - β₁^epoch)
    v_corrected = v ./ (1 - β₂^epoch)
    
    ps .-= lr .* m_corrected ./ (sqrt.(v_corrected) .+ ε)
end

# Final prediction
forward!(forward_cache, model)
predictions = get_outputs(forward_cache)

println("Final loss: ", losses[end])
```

### Batch Processing and Data Loading

For larger datasets, you'll need efficient batch processing:

```julia
using SimpleNNs
using MLDatasets
using Random
using ProgressBars

struct DataLoader{T,S}
    features::T
    labels::S
    batch_size::Int
    indices::Vector{Int}
    n_samples::Int
end

function DataLoader(features, labels, batch_size::Int)
    n_samples = size(features)[end]
    indices = collect(1:batch_size)
    return DataLoader(features, labels, batch_size, indices, n_samples)
end

function next_batch!(loader::DataLoader)
    # Randomly sample new indices
    shuffle!(loader.indices)
    return (
        view(loader.features, ntuple(i -> :, ndims(loader.features)-1)..., loader.indices),
        view(loader.labels, loader.indices)
    )
end

# Example with MNIST
dataset = MNIST(:train)
images, labels = dataset[:]
images = reshape(images, 28, 28, 1, size(images, 3))
labels = labels .+ 1  # 1-indexed

# Create data loader
batch_size = 128
loader = DataLoader(images, labels, batch_size)

# Create model
model = chain(
    Static((28, 28, 1)),
    Conv((5,5), 16, activation_fn=relu),
    MaxPool((2,2)),
    Conv((3,3), 8, activation_fn=relu),
    MaxPool((4,4)),
    Flatten(),
    Dense(10, activation_fn=identity)
)

# Preallocate
forward_cache = preallocate(model, batch_size)
backward_cache = preallocate_grads(model, batch_size)

# Training loop with batch processing
for epoch in ProgressBar(1:100)
    batch_features, batch_labels = next_batch!(loader)
    
    set_inputs!(forward_cache, batch_features)
    loss = LogitCrossEntropyLoss(batch_labels, 10)
    
    forward!(forward_cache, model)
    total_loss = backprop!(backward_cache, forward_cache, model, loss)
    
    # Apply gradients (simplified SGD)
    ps = SimpleNNs.parameters(model)
    grads = gradients(backward_cache)
    ps .-= 0.001f0 .* grads
end
```

## Memory Optimisation

### Cache Truncation

When your actual batch size is smaller than your preallocated cache, use truncation:

```julia
# Preallocate for maximum batch size
max_batch_size = 256
forward_cache = preallocate(model, max_batch_size)
backward_cache = preallocate_grads(model, max_batch_size)

# For smaller batches, truncate the cache
actual_batch_size = 64
truncated_forward = SimpleNNs.truncate(forward_cache, actual_batch_size)
truncated_backward = SimpleNNs.truncate(backward_cache, actual_batch_size)

# Use truncated caches
set_inputs!(truncated_forward, small_batch_data)
forward!(truncated_forward, model)
backprop!(truncated_backward, truncated_forward, model, loss)
```

## Performance Profiling

### Allocation Tracking

SimpleNNs.jl is designed for zero-allocation (or close to zero) inference. Some minor allocations may slip through, depending on the version of Julia you are currently running. An example of testing the allocations is below:

```julia
using SimpleNNs

model = chain(Static(10), Dense(32, activation_fn=relu), Dense(1))
forward_cache = preallocate(model, 64)
backward_cache = preallocate_grads(model, 64)

inputs = randn(Float32, 10, 64)
targets = randn(Float32, 1, 64)
loss = MSELoss(targets)

set_inputs!(forward_cache, inputs)

# Call once to remove allocations from JIT compilation
forward!(forward_cache, model)
# Check allocations in forward pass
forward_allocs = @allocations forward!(forward_cache, model)
println("Forward allocations: $forward_allocs")

# Again, call once to avoid measuring JIT
backprop!(backward_cache, forward_cache, model, loss)
# Check allocations in backward pass
backward_allocs = @allocations backprop!(backward_cache, forward_cache, model, loss)
println("Backward allocations: $backward_allocs")

# Should be 0 or very low for optimal performance
@assert forward_allocs <= 1
@assert backward_allocs <= 1
```

## Custom Activation Functions

While SimpleNNs.jl has built-in activation functions, you can define custom ones:

```julia
# Custom activation function
function swish(x)
    return x * sigmoid(x)
end

# Custom gradient (if needed for backward pass) (input is the output of the activation)
function swish_gradient(x)
    s = sigmoid(x)
    return s * (1 + x * (1 - s))
end
# Link the gradient fn to the activation fn
SimpleNNs.activation_gradient_fn(::typeof(swish)) = swish_gradient

model = chain(
    Static(10),
    Dense(32, activation_fn=swish),  # Custom activation
    Dense(1, activation_fn=identity)
)

# ... Model should now work with forward! and backprop!
```

!!! warning "Custom Activations"
    Custom activation functions work for forward passes, but you may need to implement custom backward pass logic for training.

## Model Serialisation

Save and load trained models:

```julia
using JLD2

# Save model parameters
@save "model_params.jld2" params=parameters(model)

# Load model parameters
@load "model_params.jld2" params
new_model = create_model()  # Recreate same architecture
parameters(new_model) .= params
```

This advanced usage guide should help you get the most out of SimpleNNs.jl for complex applications and high-performance computing scenarios.

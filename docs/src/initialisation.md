# Parameter Initialisation

Proper weight initialisation is crucial for training neural networks effectively. SimpleNNs.jl provides several standard initialisation schemes that can be specified when creating layers.

## Quick Start

When creating a model, you can specify the initialisation method for each layer:

```julia
using SimpleNNs

model = chain(
    Static(10),
    Dense(64, activation_fn=relu, init=HeNormal()),
    Dense(32, activation_fn=relu, init=HeNormal()),
    Dense(10, activation_fn=identity, init=GlorotNormal())
)

# Initialise all parameters
initialise!(model)
```

Without calling `initialise!`, all parameters start at zero. The `initialise!` function applies the specified initialisation scheme to each layer.

## Initialisation Methods

### Glorot Initialisation (Xavier)

Glorot initialisation is designed for layers with sigmoid or tanh activation functions. It aims to keep the variance of activations and gradients roughly the same across layers.

#### `GlorotNormal()`

Samples weights from a normal distribution:
```math
W \sim \mathcal{N}\left(0, \sqrt{\frac{2}{n_{in} + n_{out}}}\right)
```

where $n_{in}$ is the number of input units (fan-in) and $n_{out}$ is the number of output units (fan-out).

```julia
model = chain(
    Static(784),
    Dense(128, activation_fn=tanh, init=GlorotNormal()),
    Dense(10, activation_fn=identity, init=GlorotNormal())
)
```

#### `GlorotUniform()`

Samples weights from a uniform distribution:
```math
W \sim \mathcal{U}\left(-\sqrt{\frac{6}{n_{in} + n_{out}}}, \sqrt{\frac{6}{n_{in} + n_{out}}}\right)
```

```julia
Dense(64, activation_fn=sigmoid, init=GlorotUniform())
```

### He Initialisation (Kaiming)

He initialisation is specifically designed for ReLU activation functions and their variants. It accounts for the fact that ReLU neurons output zero for half their inputs.

#### `HeNormal()`

Samples weights from a normal distribution:
```math
W \sim \mathcal{N}\left(0, \sqrt{\frac{2}{n_{in}}}\right)
```

This is the recommended initialisation for ReLU networks:

```julia
model = chain(
    Static(28, 28, 1),
    Conv((3, 3), 32, activation_fn=relu, init=HeNormal()),
    MaxPool((2, 2)),
    Conv((3, 3), 64, activation_fn=relu, init=HeNormal()),
    Flatten(),
    Dense(10, activation_fn=identity, init=HeNormal())
)
```

#### `HeUniform()`

Samples weights from a uniform distribution:
```math
W \sim \mathcal{U}\left(-\sqrt{\frac{6}{n_{in}}}, \sqrt{\frac{6}{n_{in}}}\right)
```

```julia
Dense(128, activation_fn=relu, init=HeUniform())
```

### LeCun Initialisation

#### `LeCunNormal()`

LeCun normal initialisation is designed for SELU (Self-Normalizing) activation functions:
```math
W \sim \mathcal{N}\left(0, \sqrt{\frac{1}{n_{in}}}\right)
```

```julia
Dense(64, activation_fn=identity, init=LeCunNormal())
```

### Zero Initialisation

#### `Zeros()`

Initialises all weights to zero. **Warning**: This is generally not recommended for training neural networks as it breaks symmetry and prevents learning. However, it can be useful for specific scenarios or debugging.

```julia
Dense(10, activation_fn=identity, init=Zeros())
```

## Choosing the Right Initialiser

The choice of initialisation method depends primarily on the activation function used in your layers:

| Activation Function | Recommended Initialiser |
|---------------------|-------------------------|
| ReLU, Leaky ReLU    | `HeNormal()` or `HeUniform()` |
| Tanh, Sigmoid       | `GlorotNormal()` or `GlorotUniform()` |
| SELU                | `LeCunNormal()` |
| Identity (output)   | `GlorotNormal()` or layer-specific |

## Convolutional Layers

For convolutional layers, the fan-in is computed as:
```math
n_{in} = \text{kernel_height} \times \text{kernel_width} \times \text{in_channels}
```

For example, a `Conv((3, 3), 16)` layer with 1 input channel has `fan_in = 3 × 3 × 1 = 9`.

```julia
model = chain(
    Static((28, 28, 1)),
    Conv((3, 3), 16, activation_fn=relu, init=HeNormal()),  # fan_in = 9
    Conv((3, 3), 32, activation_fn=relu, init=HeNormal()),  # fan_in = 3*3*16 = 144
    Flatten(),
    Dense(10, activation_fn=identity)
)
```

## Bias Initialisation

Biases are always initialised to zero, regardless of the weight initialisation scheme. This is the standard practice in deep learning, as proper weight initialisation is sufficient to break symmetry.

## Manual Initialisation

If you prefer manual control over parameter initialisation, you can directly manipulate the parameter vector:

```julia
using Random

model = chain(
    Static(10),
    Dense(64, activation_fn=relu),
    Dense(10, activation_fn=identity)
)

# Get parameter vector
params = parameters(model)

# Manual initialisation
Random.seed!(42)
randn!(params)
params .*= 0.01f0  # Scale down
```

Note that this bypasses the layer-specific initialisation schemes and applies the same initialisation to all parameters, including biases.

## Complete Example

Here's a complete example showing proper initialisation for a classification network:

```julia
using SimpleNNs
using Random

# Create model with appropriate initialisers
model = chain(
    Static(784),                                              # MNIST flattened input
    Dense(256, activation_fn=relu, init=HeNormal()),         # ReLU → HeNormal
    Dense(128, activation_fn=relu, init=HeNormal()),         # ReLU → HeNormal
    Dense(64, activation_fn=tanh, init=GlorotNormal()),      # Tanh → GlorotNormal
    Dense(10, activation_fn=identity, init=GlorotNormal())   # Output layer
)

# Set random seed for reproducibility
Random.seed!(42)

# Initialise all parameters
initialise!(model)

# Verify parameters are no longer zero
params = parameters(model)
println("Parameters initialised: ", !all(params .== 0))
```

## Re-initialisation

You can re-initialise a model at any time by calling `initialise!` again. This is useful for experiments where you want to restart training with the same architecture but different initial weights:

```julia
# First training run
initialise!(model)
# ... train model ...

# Reset and start fresh
initialise!(model)
# ... train again with different initial weights ...
```

Each call to `initialise!` generates new random weights according to each layer's initialisation scheme.

## API Reference

```@docs
initialise!
GlorotNormal
GlorotUniform
HeNormal
HeUniform
LeCunNormal
Zeros
```

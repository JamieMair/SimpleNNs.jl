# Core Functions

## Model Creation
- [`chain`](@ref) - Create a neural network model
- [`parameters`](@ref) - Access model parameters

## Layer Types
- [`Static`](@ref) - Input layer specifying dimensions
- [`Dense`](@ref) - Fully connected layer
- [`Conv`](@ref) - Convolutional layer
- [`MaxPool`](@ref) - Max pooling layer
- [`Flatten`](@ref) - Flatten multi-dimensional data

## Forward Pass
- [`preallocate`](@ref) - Create forward pass cache
- [`set_inputs!`](@ref) - Set input data
- [`forward!`](@ref) - Execute forward pass
- [`get_outputs`](@ref) - Extract model outputs

## Backward Pass
- [`preallocate_grads`](@ref) - Create gradient cache
- [`backprop!`](@ref) - Execute backward pass
- [`gradients`](@ref) - Access computed gradients

## Loss Functions
- [`MSELoss`](@ref) - Mean squared error loss
- [`LogitCrossEntropyLoss`](@ref) - Cross entropy loss for classification

## Activation Functions
- [`relu`](@ref) - ReLU activation
- [`sigmoid`](@ref) - Sigmoid activation
- [`tanh_fast`](@ref) - Fast tanh activation

## GPU Support
- [`gpu`](@ref) - Move models/data to GPU

# Function Index

```@index
```
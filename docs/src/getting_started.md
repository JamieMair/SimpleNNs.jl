# Getting Started

Firstly, you can add this package directly using the URL:
```julia
import Pkg; Pkg.add("https://github.com/JamieMair/SimpleNNs.jl")
```
**Note: This package has plans to be registered and should be available with `] add SimpleNNs` in the future.**

Once the package is installed, go ahead and load the package.
```@example 1
using SimpleNNs
nothing # hide
```
To start with, let's create a simple test dataset:
```@example 1
batch_size = 256
x = collect(LinRange(0, 2*pi, batch_size)')
y = sin.(x)
nothing # hide
```
Note that we use the adjoint `'` so that the last dimension is the batch dimension.

We can now create our small neural network to fit a curve that maps from $x$ to $y$. The syntax will be familiar to users of `Flux.jl` or `SimpleChains.jl`.
```@example 1
model = chain(
    Static(1),
    Dense(10, activation_fn=tanh),
    Dense(10, activation_fn=sigmoid),
    Dense(1, activation_fn=identity),
);
nothing # hide
```
Here, we specify the expected feature size of the input, leaving out the batch dimension.

## Inference (Forward-Pass)

To run inference with this model, we first need to preallocate a buffer to store the intermediate forward pass values. This preallocation is by design, so that memory is only allocated once at the beginning of training.
```@example 1
forward_buffer = preallocate(model, batch_size);
nothing # hide
```
This buffer also contains the input to the neural network. We can set the inputs to the neural network via
```@example 1
set_inputs!(forward_buffer, x);
nothing # hide
```
The above function can be used to set the new inputs at each epoch.

We can access the flat parameter vector of the model via `parameters(model)` to initialise the weights of the network, i.e.
```@example 1
using Random
Random.seed!(1234)
params = parameters(model);
randn!(params);
params .*= 0.1;
nothing # hide
```
We can run inference with
```@example 1
forward!(forward_buffer, model);
yhat = get_outputs(forward_buffer);
nothing # hide
```

## Training (Backward-Pass)

We can specify a mean-squared error loss via
```@example 1
loss = MSELoss(y);
nothing # hide
```
and preallocate the buffer used for calculating the gradients via back-propagation:
```@example 1
gradient_buffer = preallocate_grads(model, batch_size);
nothing # hide
```

Now we have all the ingredients we need to write a simple training script, making use of `Optimisers.jl`.
```@example 1
import Optimisers

lr = 0.01
opt = Optimisers.setup(Optimisers.Adam(lr), params)
epochs = 1000
losses = zeros(Float32, epochs)
for i in 1:epochs
    forward!(forward_buffer, model)
    losses[i] = backprop!(gradient_buffer, forward_buffer, model, loss)
    grads = gradients(gradient_buffer) # extract the gradient vector
    # Apply the optimiser
    Optimisers.update!(opt, params, grads)
end

nothing # hide
```
We can plot the losses over time, for example using `Plots.jl`:
```@example 1
using Plots
using PlotThemes # hide
theme(:dark) # hide
plot(losses, xlabel="Epochs", ylabel="MSE Loss", lw=2, label=nothing)
```
Finally, we can run one final forward pass to get the predictions
```@example 1
forward!(forward_buffer, model);
yhat = get_outputs(forward_buffer);
nothing # hide
```
and then plot the predictions
```@example 1
using Plots
plt = plot(x', y', linestyle=:solid, label="Original", lw=2);
plot!(plt, x', yhat', linestyle=:dashdot, label="Prediction", lw=2);
xlabel!("x")
ylabel!("y")
plt
```

To see an example using convolution layers and GPU training, see the [MNIST](@ref) training example.
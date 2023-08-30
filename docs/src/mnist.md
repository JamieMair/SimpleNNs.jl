# MNIST

In this example, we will train a small neural network to classify the MNIST digit dataset. We will use `MLDatasets` to load our dataset.
```julia
using MLDatasets
dataset = MNIST(:train);
images, labels = dataset[:];
# reshape images array to add in a channel
images = reshape(images, size(images, 1), size(images, 2), 1, size(images, 3));
# Make our labels from 1 to 10 instead
labels .+= 1;

```
If you have an NVIDIA GPU, you can modify the code below to set `use_gpu` equal to `true`.
```julia
using SimpleNNs
import SimpleNNs.GPU: gpu
use_gpu = true;
to_device = use_gpu ? gpu : identity;

```
We can use this function to put our data onto the GPU, using the pipe operator:
```julia
images = images |> to_device;
labels = labels |> to_device;

```
Next, we can create our model:
```julia
img_size = size(images)[1:end-1]
model = chain(
    Static(img_size),
    Conv((5,5), 16; activation_fn=relu),
    MaxPool((2,2)),
    Conv((3,3), 8; activation_fn=relu),
    MaxPool((4,4)),
    Flatten(),
    Dense(10, activation_fn=identity)
) |> to_device;

```

For training, we will use stochastic gradient descent (with the ADAM optimiser), with a batch size of $32$. We need to preallocate our buffers, as below:
```julia
batch_size = 32;
forward_buffer = preallocate(model, batch_size);
gradient_buffer = preallocate_grads(model, batch_size);

```
Now, we write our training loop:
```julia
using Random
Random.seed!(1234)
params = parameters(model);
randn!(params)
params .*= 0.05
import Optimisers
lr = 0.01;
opt = Optimisers.setup(Optimisers.Adam(lr), params);
epochs = 1000;
losses = zeros(Float32, epochs);
training_indices = collect(1:length(labels));

for i in 1:epochs
    # Select a random batch
    Random.shuffle!(training_indices)
    batch_indices = view(training_indices, 1:batch_size)
    # Set the inputs of the forward buffer to this minibatch
    set_inputs!(forward_buffer, view(images, :, :, :, batch_indices));
    # Create a loss function that wraps the current minibatch labels
    loss = LogitCrossEntropyLoss(view(labels, batch_indices), 10);


    forward!(forward_buffer, model)
    losses[i] = backprop!(gradient_buffer, forward_buffer, model, loss)
    grads = gradients(gradient_buffer) # extract the gradient vector
    # Apply the optimiser
    Optimisers.update!(opt, params, grads)
end


```
We can plot the losses over time, for example using `Plots.jl`:
```julia
using Plots
using PlotThemes # hide
theme(:dark) # hide
plot(losses, xlabel="Epochs", ylabel="Cross Entropy Loss", lw=2, label=nothing)
```
Finally, we can test the accuracy of the model by loading the test set:
```julia
dataset = MNIST(:test);
test_images, test_labels = dataset[:];
# reshape images array to add in a channel
test_images = reshape(test_images, size(test_images, 1), size(test_images, 2), 1, size(test_images, 3));
# Make our labels from 1 to 10 instead
test_labels .+= 1;
test_images = test_images |> to_device;
test_labels = test_labels |> to_device;

```
Now, we create a forward buffer for the test images:
```julia
test_forward_buffer = preallocate(model, length(test_labels));
set_inputs!(test_forward_buffer, test_images);
forward!(test_forward_buffer, model);
logits = get_outputs(test_forward_buffer);
predictions = reshape([i[1] for i in Array(argmax(logits, dims=1))], :) |> to_device;
accuracy = sum(predictions .== test_labels) / length(test_labels) * 100
@show accuracy
```
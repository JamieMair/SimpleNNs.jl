using CUDA
using SimpleNNs
import SimpleNNs.GPU: gpu
using Random
import BSON: @load, @save
using MLDatasets

@load "results/model.bson" model parameters losses confusion_matrix

dataset = MNIST(:train);
images, labels = dataset[:];

num_samples = 128
train_features = images[:, :, 1:num_samples];
train_labels = labels[1:num_samples];


img_size = size(train_features)[1:end-1]
in_channels = 1
batch_size = num_samples
input_size = (img_size..., in_channels, batch_size)


forward_cache = preallocate(model, batch_size)
train_features = reshape(train_features, input_size)
set_inputs!(forward_cache, train_features)
forward!(forward_cache, model)

gradient_cache = preallocate_grads(model, batch_size)

loss = LogitCrossEntropyLoss(train_labels.+1, 10);

fill!(gradient_cache.parameter_gradients, zero(eltype(gradient_cache.parameter_gradients)))
backprop!(gradient_cache, forward_cache, model, loss)

gpu_model = chain(
    Static((img_size..., in_channels)),
    Conv((5,5), 16; activation_fn=SimpleNNs.relu),
    Conv((3,3), 4; activation_fn=SimpleNNs.relu),
    Flatten(),
    Dense(32, activation_fn=SimpleNNs.relu),
    Dense(10, activation_fn=identity)
) |> gpu
copyto!(gpu_model.parameters, parameters)
gpu_forward_cache = preallocate(gpu_model, batch_size)
train_features = reshape(train_features, input_size)
set_inputs!(gpu_forward_cache, train_features |> gpu)
gpu_outputs = forward!(gpu_forward_cache, gpu_model)

gpu_gradient_cache = preallocate_grads(gpu_model, batch_size)

gpu_loss = LogitCrossEntropyLoss((train_labels.+1)|>gpu, 10);

fill!(gpu_gradient_cache.parameter_gradients, zero(eltype(gpu_gradient_cache.parameter_gradients)))
CUDA.allowscalar(false)
backprop!(gpu_gradient_cache, gpu_forward_cache, gpu_model, gpu_loss)
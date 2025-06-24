using CUDA
import cuDNN, NNlib
using SimpleNNs
using Random
using MLDatasets
using ProgressBars
import SimpleNNs: gpu
using BenchmarkTools


use_gpu = false
to_device = use_gpu ? gpu : identity
dataset = MNIST(:train);
images, labels = dataset[:];


images =  images |> to_device
labels =  labels |> to_device

num_samples = 128
train_features = images[:, :, 1:num_samples];
train_labels = labels[1:num_samples];


img_size = size(train_features)[1:end-1]
in_channels = 1
model = chain(
    Static((img_size..., in_channels)),
    Conv((5,5), 16; activation_fn=relu),
    MaxPool((2,2)),
    Conv((3,3), 8; activation_fn=relu),
    MaxPool((4,4)),
    Flatten(),
    Dense(10, activation_fn=identity)
) |> to_device;
batch_size = num_samples
input_size = (img_size..., in_channels, batch_size)
forward_cache = preallocate(model, batch_size)
train_features = reshape(train_features, input_size)
set_inputs!(forward_cache, train_features)
parameters = model.parameters
randn!(parameters)
parameters .*= (1/1000)
@benchmark forward!($forward_cache, $model)

gradient_cache = preallocate_grads(model, batch_size)

loss = LogitCrossEntropyLoss(train_labels.+1, 10);

fill!(gradient_cache.parameter_gradients, zero(eltype(gradient_cache.parameter_gradients)))
@benchmark backprop!($gradient_cache, $forward_cache, $model, $loss)
# backprop!(gradient_cache, forward_cache, model, loss)


function cross_entropy_loss(outputs, loss::LogitCrossEntropyLoss{T, N}) where {N, T}
    e_y = exp.(outputs)
    p_y = e_y ./ sum(e_y, dims=1)
    
    l = zero(eltype(outputs))
    CUDA.@allowscalar for i in axes(e_y, length(size(e_y)))
        l -= log(p_y[loss.targets[i], i])
    end
    return l
end

params = parameters
epochs = Int(2^16)
losses = zeros(Float32, epochs+1)
begin
    # Use built-in Adam optimiser
    opt = AdamOptimiser(gradient_cache.parameter_gradients; lr=0.002f0/batch_size, beta_1=0.9f0, beta_2=0.999f0)
    
    indices = collect(1:batch_size)
    N_images = length(labels)
    images = reshape(images, img_size..., 1, :)
    for e in ProgressBar(1:epochs)    
        forward!(forward_cache, model)
        model_outputs = get_outputs(forward_cache)
        losses[e] = backprop!(gradient_cache, forward_cache, model, loss)
        
        # Use optimiser for parameter updates
        grads = gradients(gradient_cache)
        update!(params, grads, opt)
        
        # Shift to the next batch
        indices .= ((indices .+ (batch_size - 1)) .% N_images) .+ 1
        next_images = view(images, (1:d for d in img_size)..., 1:1, indices)
        next_labels = view(labels, indices)

        set_inputs!(forward_cache, next_images)
        loss.targets .= next_labels .+ 1
    end
    forward!(forward_cache, model)
    model_outputs = get_outputs(forward_cache)
    losses[epochs+1] = cross_entropy_loss(model_outputs, loss)
    nothing
end



test_dataset = MNIST(:test);
test_images, test_labels = test_dataset[:];
test_images = test_images |> to_device
test_labels = test_labels |> to_device
test_forward_cache = preallocate(model, length(test_labels));
set_inputs!(test_forward_cache, reshape(test_images, img_size..., 1, :));
forward!(test_forward_cache, model)
test_outputs = get_outputs(test_forward_cache)
_, model_preds = findmax(Array(test_outputs), dims=1)
model_preds = reshape((x->x.I[1]).(model_preds), :)

confusion_matrix = begin
    confusion_matrix = zeros(Int, 10, 10)
    cpu_test_labels = Array(test_labels)
    for i in eachindex(cpu_test_labels, model_preds)
        confusion_matrix[cpu_test_labels[i]+1, model_preds[i]] += 1
    end
    confusion_matrix
end


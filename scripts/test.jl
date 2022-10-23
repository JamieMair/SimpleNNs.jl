using SimpleNNs
using ProgressBars

model = chain(
    Static(1), # Layer 1
    Dense(16, activation_fn=tanh), # Layer 2
    Dense(16, activation_fn=tanh), # Layer 3
    Dense(16, activation_fn=tanh), # Layer 4
    Dense(1, activation_fn=identity), # Layer 5
);
params = model.parameters
using Random
rand!(model.parameters)

N = 512
inputs = rand(Float32, 1, N)
function output_fn(x) 
freq = 4.0
offset = 1.0
    return x*sin(x * freq - offset) + offset
end
outputs = Float32.(reshape(output_fn.(inputs), :))
# Preallocate Model
forward_cache = preallocate(model, N)
gradient_cache = preallocate_grads(model, N)
set_inputs!(forward_cache, inputs) # Copies inputs into the cache
forward!(forward_cache, model)
model_outputs = get_outputs(forward_cache)

mse(expected_outputs, model_outputs) = sum((expected_outputs.-model_outputs).^2) / length(expected_outputs)

loss = MSELoss(outputs)

backprop!(gradient_cache, forward_cache, model, loss)

lr = 0.005 / N
epochs = 100000
losses = Float32[]
begin
    for e in ProgressBar(1:epochs)
        forward!(forward_cache, model)
        model_outputs = get_outputs(forward_cache)
        push!(losses, mse(outputs, model_outputs))
        backprop!(gradient_cache, forward_cache, model, loss)
        params .-= lr .* gradient_cache.parameter_gradients
    end
    forward!(forward_cache, model)
    model_outputs = get_outputs(forward_cache)
    push!(losses, mse(outputs, model_outputs))
end

using Plots
begin
    flat_inputs = reshape(inputs, :)
    plt = scatter(flat_inputs, outputs, label="True")
    scatter!(plt, flat_inputs, reshape(model_outputs, :), label="Pred", legend=:topleft)

    return plt
end
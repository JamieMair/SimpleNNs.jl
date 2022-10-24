using SimpleNNs
import SimpleNNs.GPU: gpu
using ProgressBars
using CUDA

use_gpu = true
to_device = use_gpu ? gpu : identity

model = chain(
    Static(1), # Layer 1
    Dense(16, activation_fn=tanh_fast), # Layer 2
    Dense(16, activation_fn=tanh_fast), # Layer 3
    Dense(16, activation_fn=tanh_fast), # Layer 4
    Dense(1, activation_fn=identity), # Layer 5
) |> to_device;
params = model.parameters
using Random
rand!(model.parameters)

N = 2^9
inputs = reshape(collect(LinRange(-1.0, 1.0, N)), 1, N) |> to_device
function output_fn(x) 
    freq = convert(typeof(x), 4.0)
    offset = convert(typeof(x), 1.0)
    return x*sin(x * freq - offset) + offset
end
outputs = reshape(output_fn.(inputs) |> to_device, :)
# Preallocate Model
forward_cache = preallocate(model, N)
gradient_cache = preallocate_grads(model, N)
set_inputs!(forward_cache, inputs) # Copies inputs into the cache
forward!(forward_cache, model)
model_outputs = get_outputs(forward_cache)

mse(expected_outputs, model_outputs) = sum((expected_outputs.-model_outputs).^2) / length(expected_outputs)
square_fn(x) = x*x
function mse(expected_outputs::Array, model_outputs::Array)
    s = zero(eltype(expected_outputs))
    for i in eachindex(expected_outputs)
        s += square_fn(expected_outputs[i] - model_outputs[i])
    end
    return s / length(expected_outputs)
end
loss = MSELoss(outputs)

backprop!(gradient_cache, forward_cache, model, loss)


begin
    lr = 0.05 / N
    epochs = Int(10^4)
    losses = zeros(Float32, epochs+1)
    beta_1 = 0.9f0
    beta_2 = 0.999f0
    m = similar(gradient_cache.parameter_gradients)
    v = similar(gradient_cache.parameter_gradients)
    fill!(m, 0.0)
    fill!(v, 0.0)
    for e in ProgressBar(1:epochs)    
        forward!(forward_cache, model)
        model_outputs = get_outputs(forward_cache)
        losses[e] = mse(outputs, model_outputs)
        backprop!(gradient_cache, forward_cache, model, loss)
        m .= beta_1 .* m + (1-beta_1) .* gradient_cache.parameter_gradients
        v .= beta_1 .* v + (1-beta_2) .* gradient_cache.parameter_gradients .^ 2
        denom_1 = inv(1 - beta_1 ^ e)
        denom_2 = inv(1 - beta_2 ^ e)
        eps = convert(Float32, 1e-8)
        params .-= lr .* (m .* denom_1) ./ (sqrt.(v.*denom_2) .+ eps)
    end
    forward!(forward_cache, model)
    model_outputs = get_outputs(forward_cache)
    losses[epochs+1] = mse(outputs, model_outputs)
    nothing
end

using Plots
begin
    flat_inputs = reshape(inputs, :)
    plt = plot(Array(flat_inputs), Array(outputs), label="True")
    plot!(plt, Array(flat_inputs), Array(reshape(model_outputs, :)), label="Pred", legend=:topleft, alpha=0.8)

    return plt
end
using Revise
using SimpleNNs
Revise.track(SimpleNNs)
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
);
model = to_device(model);
params = model.parameters
using Random
randn!(model.parameters)

N = 2^9
inputs = reshape(collect(LinRange(-1.0, 1.0, N)), 1, N) |> to_device
function output_fn(x) 
    freq = convert(typeof(x), 4.0)
    offset = convert(typeof(x), 1.0)
    return x*sin(x * freq - offset) + offset
end
outputs = reshape(output_fn.(inputs) |> to_device, 1, :)
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

epochs = Int(5*10^3)
losses = zeros(Float32, epochs+1)
begin
    # Use built-in Adam optimiser
    opt = AdamOptimiser(gradient_cache.parameter_gradients; lr=0.02f0/N, beta_1=0.9f0, beta_2=0.999f0)
    
    for e in ProgressBar(1:epochs)    
        forward!(forward_cache, model)
        model_outputs = get_outputs(forward_cache)
        losses[e] = mse(outputs, model_outputs)
        backprop!(gradient_cache, forward_cache, model, loss)
        
        # Use optimiser for parameter updates
        grads = gradients(gradient_cache)
        update!(params, grads, opt)
    end
    forward!(forward_cache, model)
    model_outputs = get_outputs(forward_cache)
    losses[epochs+1] = mse(outputs, model_outputs)
    nothing
end

using Plots
begin
    flat_inputs = reshape(inputs, :)
    plt = plot(Array(flat_inputs), Array(reshape(outputs, :)), label="True", linestyle=:solid, lw=3)
    plot!(plt, Array(flat_inputs), Array(reshape(model_outputs, :)), label="Pred", legend=:topleft, alpha=0.8, linestyle=:dash, lw=3)

    return plt
end
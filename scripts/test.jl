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
epochs = Int(10^4)
losses = Float32[]
beta_1 = 0.9f0
beta_2 = 0.999f0
m = similar(gradient_cache.parameter_gradients)
v = similar(gradient_cache.parameter_gradients)
fill!(m, 0.0)
fill!(v, 0.0)
begin
    for e in ProgressBar(1:epochs)
        forward!(forward_cache, model)
        model_outputs = get_outputs(forward_cache)
        push!(losses, mse(outputs, model_outputs))
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
    push!(losses, mse(outputs, model_outputs))
end

using Plots
begin
    flat_inputs = reshape(inputs, :)
    plt = scatter(flat_inputs, outputs, label="True")
    scatter!(plt, flat_inputs, reshape(model_outputs, :), label="Pred", legend=:topleft, alpha=0.8)

    return plt
end
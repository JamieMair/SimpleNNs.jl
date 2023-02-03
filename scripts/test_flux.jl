using Flux

model = Flux.Chain(
    Dense(1, 16, tanh),    
    Dense(16, 16, tanh),    
    Dense(16, 16, tanh), 
    Dense(16, 1, identity)
);
function output_fn(x) 
    freq = 4.0
    offset = 1.0
    return x*sin(x * freq - offset) + offset
end
N = 512
inputs = rand(Float32, 1, N)
outputs = Float32.(reshape(output_fn.(inputs), 1, :))



optim = Flux.Adam()

parameters = Flux.params(model)

function train_model_one_epoch!(model, params, inputs, outputs, opt)
    grads = Flux.gradient(params) do 
        Flux.Losses.mse(model(inputs), outputs)
    end
    Flux.update!(opt, params, grads)
end

function inference(model, inputs)
    return model(inputs)
end
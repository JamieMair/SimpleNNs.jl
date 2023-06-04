import Flux
import SimpleNNs
import SimpleNNs.GPU
using Random
using ProgressBars
import CUDA
using BenchmarkTools

include("data_utils.jl")
include("simplenn_utils.jl")

function create_model_and_cache(dataset, batch_size, device)
    img_size = (28,28);
    in_channels = 1;
    model_simple = create_simple_nn_mnist_model(img_size, in_channels, device);
    randn!(model_simple.parameters);
    model_simple.parameters .*= (1/1000);

    input_size = (img_size..., in_channels, batch_size);
    forward_cache = SimpleNNs.preallocate(model_simple, batch_size);
    batch_features, batch_indices = current_batch(dataset);
    SimpleNNs.set_inputs!(forward_cache, reshape(batch_features, input_size));

    return model_simple, forward_cache
end

function inference!(parameters, model::SimpleNNs.Model, forward_cache)
    model.parameters .= parameters # copy new parameters
    SimpleNNs.forward!(forward_cache, model)
    nothing
end
function inference_loop!(parameters, model::SimpleNNs.Model, forward_cache, epochs)
    for _ in 1:epochs
        inference!(parameters, model, forward_cache)
    end
end
function run_in_parallel(parameters, models, forward_caches, epochs)
    num_models = length(models)
    CUDA.@sync Threads.@threads for i in 1:num_models
        model = models[i]
        forward_cache = forward_caches[i]
        inference_loop!(parameters, model, forward_cache, epochs)
    end
    nothing
end


device = :gpu;
batch_size = 128;
dataset = load_mnist_data(batch_size, device);
num_in_parallel = 8;
model_and_caches = [create_model_and_cache(dataset, batch_size, device) for _ in 1:num_in_parallel];
models = (x->first(x)).(model_and_caches);
forward_caches = (x->last(x)).(model_and_caches);
test_parameters = deepcopy(first(models).parameters);

## Benchmarks with 8 models (RTX 3090)
#
#
# julia> @benchmark run_in_parallel($test_parameters, $models, $forward_caches, $(1000))
#
# BenchmarkTools.Trial: 6 samples with 1 evaluation.
#  Range (min … max):  769.455 ms … 880.750 ms  ┊ GC (min … max): 0.00% … 0.00%
#  Time  (median):     876.571 ms               ┊ GC (median):    0.00%
#  Time  (mean ± σ):   858.977 ms ±  43.996 ms  ┊ GC (mean ± σ):  1.62% ± 3.87%
#  Memory estimate: 79.98 MiB, allocs estimate: 1528276.
# ~ 9.1 epochs/ ms
# julia> @benchmark run_in_parallel($test_parameters, $(view(models, 1:6)), $(view(forward_caches, 1:6)), $(1000))
# BenchmarkTools.Trial: 8 samples with 1 evaluation.
#  Range (min … max):  576.357 ms … 699.893 ms  ┊ GC (min … max): 0.00% … 16.94%
#  Time  (median):     647.946 ms               ┊ GC (median):    0.00%
#  Time  (mean ± σ):   652.371 ms ±  37.517 ms  ┊ GC (mean ± σ):  2.27% ±  5.99%
#  Memory estimate: 59.99 MiB, allocs estimate: 1146232.
# ~ 9.23 epochs / ms
# julia> @benchmark run_in_parallel($test_parameters, $(view(models, 1:4)), $(view(forward_caches, 1:4)), $(1000))
# BenchmarkTools.Trial: 12 samples with 1 evaluation.
#  Range (min … max):  392.978 ms … 555.650 ms  ┊ GC (min … max): 0.00% … 24.75%
#  Time  (median):     437.444 ms               ┊ GC (median):    0.00%
#  Time  (mean ± σ):   442.289 ms ±  38.376 ms  ┊ GC (mean ± σ):  2.59% ±  7.15%
#  Memory estimate: 39.99 MiB, allocs estimate: 764187.
# ~ 9.15 epochs / ms
# julia> @benchmark run_in_parallel($test_parameters, $(view(models, 1:2)), $(view(forward_caches, 1:2)), $(1000))
# BenchmarkTools.Trial: 21 samples with 1 evaluation.
#  Range (min … max):  219.499 ms … 337.074 ms  ┊ GC (min … max): 0.00% … 31.70%
#  Time  (median):     239.181 ms               ┊ GC (median):    0.00%
#  Time  (mean ± σ):   241.428 ms ±  22.655 ms  ┊ GC (mean ± σ):  2.11% ±  6.92%
#  Memory estimate: 20.00 MiB, allocs estimate: 382142.
# ~ 8.33 epochs / ms
# julia> @benchmark run_in_parallel($test_parameters, $(view(models, 1:1)), $(view(forward_caches, 1:1)), $(1000))
# BenchmarkTools.Trial: 37 samples with 1 evaluation.
#  Range (min … max):  125.823 ms … 175.007 ms  ┊ GC (min … max): 0.00% … 24.83%
#  Time  (median):     137.479 ms               ┊ GC (median):    0.00%
#  Time  (mean ± σ):   138.669 ms ±   7.076 ms  ┊ GC (mean ± σ):  1.49% ±  5.30%
#  Memory estimate: 10.01 MiB, allocs estimate: 191120.
# ~ 7.27 epochs / ms
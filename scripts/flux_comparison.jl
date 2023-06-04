import Flux
import SimpleNNs
import SimpleNNs.GPU
using Random
using ProgressBars
import CUDA
using BenchmarkTools

include("data_utils.jl")
include("flux_utils.jl")
include("simplenn_utils.jl")

device = :gpu;
batch_size = 128;
dataset = load_mnist_data(batch_size, device);

img_size = (28,28);
in_channels = 1;
model_flux = create_flux_mnist_model(img_size, in_channels, device);
model_simple = create_simple_nn_mnist_model(img_size, in_channels, device);
randn!(model_simple.parameters);
model_simple.parameters .*= (1/1000);

input_size = (img_size..., in_channels, batch_size);
forward_cache = SimpleNNs.preallocate(model_simple, batch_size);
batch_features, batch_indices = current_batch(dataset);
SimpleNNs.set_inputs!(forward_cache, reshape(batch_features, input_size));

parameters, re = Flux.destructure(model_flux);
function inference!(parameters, model::SimpleNNs.Model, forward_cache)
    model.parameters .= parameters # copy new parameters
    SimpleNNs.forward!(forward_cache, model)
    nothing
end
function inference!(parameters, re, forward_cache)
    model = re(parameters)
    outputs = model(forward_cache.input)
    nothing
end

## PERFORMANCE BENCHMARKS (RTX 3090) ##
# julia> @benchmark CUDA.@sync inference!($parameters, $model_simple, $forward_cache)
# BenchmarkTools.Trial: 10000 samples with 1 evaluation.
#  Range (min … max):  149.848 μs … 210.264 μs  ┊ GC (min … max): 0.00% … 0.00%
#  Time  (median):     152.472 μs               ┊ GC (median):    0.00%
#  Time  (mean ± σ):   152.761 μs ±   1.483 μs  ┊ GC (mean ± σ):  0.00% ± 0.00%
# 
#                  ▁▄▇█▇▆▅▅▅▄▂▃▂▂                                  
#   ▁▁▁▁▂▂▂▂▃▂▂▃▃▅▆██████████████▇▆▆▆▆▆▅▅▅▅▄▄▃▃▃▃▃▃▃▂▂▂▂▂▂▂▂▁▁▂▁▁ ▄
#   150 μs           Histogram: frequency by time          157 μs <
# 
#  Memory estimate: 10.23 KiB, allocs estimate: 191.
# 
# julia> @benchmark CUDA.@sync inference!($parameters, $re, $forward_cache)
# BenchmarkTools.Trial: 10000 samples with 1 evaluation.
#  Range (min … max):  214.573 μs … 59.864 ms  ┊ GC (min … max): 0.00% … 30.48%
#  Time  (median):     219.822 μs              ┊ GC (median):    0.00%
#  Time  (mean ± σ):   250.831 μs ±  1.310 ms  ┊ GC (mean ± σ):  3.55% ±  0.68%
# 
#             ▃▄▅█▇▅▄▂                                            
#   ▁▁▁▁▁▂▃▄▅█████████▇▆▅▄▃▃▃▃▃▃▄▃▃▃▃▃▃▃▃▃▃▃▃▃▃▃▃▂▂▂▂▂▂▂▁▁▁▁▁▁▁▁ ▃
#   215 μs          Histogram: frequency by time          234 μs <
# 
#  Memory estimate: 37.09 KiB, allocs estimate: 769.
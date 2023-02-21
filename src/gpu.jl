module GPU
using CUDA
using ..SimpleNNs
import SimpleNNs: Model, num_parameters, parameter_indices, _map_views, ParameterisedLayer, _inner_layer, AbstractParameterisedLayer
using Logging
function gpu(model::SimpleNNs.Model)
    parameter_offsets = cumsum(num_parameters.(model.layers))
    layer_indices = [parameter_indices(layer, offset-num_parameters(layer)) for (layer, offset) in Iterators.zip(model.layers, parameter_offsets)]
    gpu_parameters = CuArray(model.parameters)
    gpu_parameters_views = _map_views(layer_indices, gpu_parameters)
    function _inner_or_same(l)
        if typeof(l) <: AbstractParameterisedLayer
            return _inner_layer(l)
        else
            return l
        end
    end
    unwrapped_layers = Tuple(_inner_or_same(l) for l in model.layers)
    model_layers = Tuple((num_parameters(l) > 0 ? ParameterisedLayer(l, v) : l) for (l,v) in zip(unwrapped_layers, gpu_parameters_views))
    return Model(gpu_parameters, model_layers)
end
gpu(arr::AbstractArray) = CuArray(arr)
gpu(arr::CuArray) = arr
function gpu(x)
    @warn "Tried to put object of type $(typeof(x)) on the GPU, but unrecognised"
    x
end

export gpu
end
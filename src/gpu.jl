module GPU
using CUDA
using ..SimpleNNs
import SimpleNNs: Model, num_parameters, parameter_indices, _map_views, ParameterisedLayer, _inner_layer
function gpu(model::SimpleNNs.Model)
    parameter_offsets = cumsum(num_parameters.(model.layers))
    layer_indices = [parameter_indices(layer, offset-num_parameters(layer)) for (layer, offset) in Iterators.zip(model.layers, parameter_offsets)]
    gpu_parameters = CuArray(model.parameters)
    gpu_parameters_views = _map_views(layer_indices, gpu_parameters)
    model_layers = Tuple(ParameterisedLayer(l, v) for (l,v) in zip(_inner_layer.(model.layers), gpu_parameters_views))
    return Model(gpu_parameters, model_layers)
end
gpu(arr::AbstractArray) = CuArray(arr)

export gpu
end
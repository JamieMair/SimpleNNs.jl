# Utility
unimplemented() = error("Unimplemented function")
unimplemented(msg) = error("Unimplemented function: $msg")

flatten_size(size::Tuple) = reduce(*, size)
flatten_size(size::Number) = size

function _map_views(indices::AbstractArray{Q}, array::AbstractArray) where {Q<:UnitRange}
    return (x->view(array, x)).(indices)
end
function _map_views(indices::AbstractArray{T}, array::AbstractArray) where {Q<:UnitRange, T<:AbstractArray{Q}}
    return (x->_map_views(x, array)).(indices)
end
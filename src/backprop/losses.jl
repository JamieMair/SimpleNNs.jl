abstract type AbstractLoss end

"""
    MSELoss(targets)

Expects the targets in the form (`K` x `N`) where `K` is the output dimension (usually 1) and `N` is the batch size.
"""
struct MSELoss{T<:AbstractArray}
    targets::T
end

function truncate(l::MSELoss, batch_size)
    return MSELoss(_truncate_batch(l.targets, batch_size))
end

"""
    LogitCrossEntropyLoss(targets, num_classes::Int)

Expects the targets in a single vector containg class labels, which have to be between `1` and `num_classes` inclusive.
"""
struct LogitCrossEntropyLoss{T<:AbstractArray, N}
    targets::T
    num_classes::Val{N}
end
function truncate(l::LogitCrossEntropyLoss, batch_size)
    return LogitCrossEntropyLoss(_truncate_batch(l.targets, batch_size), l.num_classes)
end
LogitCrossEntropyLoss(targets::AbstractVector, n::Integer) = LogitCrossEntropyLoss(targets, Val(n))
square(x) = x*x
function pullback!(partials_buffer, inputs, loss::MSELoss)
    partials_buffer .= inputs .- loss.targets
    return sum(square, partials_buffer) / (2*length(loss.targets))
end
function pullback!(partials_buffer, inputs, loss::LogitCrossEntropyLoss{T, N}) where {T, N}
    @assert length(size(inputs)) == 2

    partials_buffer .= exp.(inputs)

    total_loss = zero(eltype(partials_buffer))
    @inbounds for i in axes(inputs, length(size(inputs)))
        true_class = loss.targets[i]
        z = zero(eltype(partials_buffer))
        @simd for k in 1:N
            z += partials_buffer[k, i]
        end
        for j in axes(inputs, 1)
            e_y = partials_buffer[j, i]
            e_y_over_z = ifelse(isfinite(e_y), e_y / z, one(eltype(partials_buffer)))
            total_loss -= ifelse(j==true_class, log(e_y_over_z), zero(typeof(total_loss)))
            partials_buffer[j, i] = (e_y_over_z - (j==true_class))
        end
    end

    return total_loss
end
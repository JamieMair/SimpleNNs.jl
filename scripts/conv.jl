using SimpleNNs
import SimpleNNs: conv

img = Float64.(reshape(1:25, 5, 5, 1))
kernel = Float64.(reshape(1:9, 3, 3))
kernel_size = size(kernel)
conv_layer = Conv(1, Val(false), Val(Float64), size(img), 1, kernel_size, identity)

parameters = (kernel, )
output = zeros(eltype(img), map(x->x-2, size(img)[1:2])..., 1)

SimpleNNs.forward!(output, conv_layer, parameters, img)
# Activation functions
sigmoid(x) = inv(one(typeof(x) + exp(-x)))
relu(x) = ifelse(x>=zero(typeof(x)), x, zero(typeof(x)))
# tanh_fast from NNlib
@inline function tanh_fast(x::Float32)
    x2 = abs2(x)
    n = evalpoly(x2, (1.0f0, 0.1346604f0, 0.0035974074f0, 2.2332108f-5, 1.587199f-8))
    d = evalpoly(x2, (1.0f0, 0.4679937f0, 0.026262015f0, 0.0003453992f0, 8.7767893f-7))
    ifelse(x2 < 66f0, x * (n / d), sign(x))
end
@inline function tanh_fast(x::Float64)
    exp2x = @fastmath exp(x + x)
    y = (exp2x - 1) / (exp2x + 1)
    x2 = x * x
    ypoly = x * evalpoly(x2, (1.0, -0.33333333333324583, 0.13333333325511604, -0.05396823125794372, 0.02186660872609521, -0.008697141630499953))
    ifelse(x2 > 900.0, sign(x), ifelse(x2 < 0.017, ypoly, y))
end
tanh_fast(x::Number) = Base.tanh(x)
using MacroTools
using Test
import CodeTracking

macro autograd(func, type)
    fn = eval(func)
    actual_type = eval(type)
    func_expr = Meta.parse(CodeTracking.@code_string fn(1.0))
    func_name = func_expr.args[1].args[1]
    arg_name = func_expr.args[1].args[2]
    body = func_expr.args[2]
    
    # Parse the function body to build the gradient
    grad_expr = build_gradient(body, arg_name)

    # Simplify the gradients
    simplified_grad_expr = simplify_gradient(grad_expr)
    
    # Construct the new function definition
    new_func = quote
        function autograd(::typeof($func_name), df::T, $arg_name::T) where {T <: $actual_type}
            dx = $simplified_grad_expr
            return df * dx
        end
    end
    
    return esc(new_func)
end

function build_gradient(expr, arg_name)
    if @capture(expr, -(a_))
        return :(-$(build_gradient(a, arg_name)))
    elseif @capture(expr, +(a_, b_))
        return :($(build_gradient(a, arg_name)) + $(build_gradient(b, arg_name)))
    elseif @capture(expr, -(a_, b_))
        return :($(build_gradient(a, arg_name)) - $(build_gradient(b, arg_name)))
    elseif @capture(expr, *(a_, b_))
        return :($b * $(build_gradient(a, arg_name)) + $a * $(build_gradient(b, arg_name)))
    elseif @capture(expr, /(a_, b_))
        return :(($(build_gradient(a, arg_name)) * $b - $a * $(build_gradient(b, arg_name))) / ($b^2))
    elseif @capture(expr, ^(a_, n_))
        if n isa Integer
            return :($n * $a^($n-1) * $(build_gradient(a, arg_name)))
        else
            error("Unsupported expression: $expr")
            # return :($n * $a^($n-1) * $(build_gradient(a, arg_name, type)) + $a^$n * log($a) * $(build_gradient(n, arg_name)))
        end
    elseif expr == arg_name
        return :(one(T))
    elseif expr isa Number
        return :(zero(T))
    else
        error("Unsupported expression: $expr")
    end
end
function simplify_gradient(expr)
    if @capture(expr, *(a_, b_))
        simplified_a = simplify_gradient(a)
        simplified_b = simplify_gradient(b)
        if simplified_a == :(zero(T)) || simplified_b == :(zero(T))
            return :(zero(T))
        elseif simplified_a == :(one(T))
            return simplified_b
        elseif simplified_b == :(one(T))
            return simplified_a
        else
            return :($simplified_a * $simplified_b)
        end
    elseif @capture(expr, +(a_, b_))
        simplified_a = simplify_gradient(a)
        simplified_b = simplify_gradient(b)
        if simplified_a == :(zero(T))
            return simplified_b
        elseif simplified_b == :(zero(T))
            return simplified_a
        else
            return :($simplified_a + $simplified_b)
        end
    elseif @capture(expr, -(a_, b_))
        simplified_a = simplify_gradient(a)
        simplified_b = simplify_gradient(b)
        if simplified_a == :(zero(T)) && simplified_b == :(zero(T))
            return :(zero(T))
        elseif simplified_b == :(zero(T))
            return simplified_a
        else
            return :($simplified_a - $simplified_b)
        end
    elseif @capture(expr, -(a_))
        simplified_a = simplify_gradient(a)
        if simplified_a == :(zero(T))
            return :(zero(T))
        else
            return :(-$simplified_a)
        end
    else
        return expr
    end
end

# Example usage
f(x) = 3x^2 - 2x + 1
@autograd f AbstractFloat
actual_df(x) = 6*x - 2
@test autograd(f, 1.0, 2.0) == 1 * actual_df(2)

@btime autograd(f, 1.0, 2.0)

using Enzyme

x = [2.0];
bx = [0.0];
y = [0.0];
by = [1.0];

function test_f!(x, y)
    y .= f.(x)
    nothing
end

@btime Enzyme.autodiff(Reverse, test_f!, Duplicated(x, bx), Duplicated(y, by));

@test bx[begin] == 1 * actual_df(x[begin])
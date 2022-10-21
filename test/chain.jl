@testitem "Dense Model Initialisation" begin    
    model = chain(
        Static(2),
        Dense(10, activation_fn=sigmoid),
        Dense(5, use_bias=false, activation_fn=tanh),
        Dense(1)
    )

    num_parameters = (2+1)*10 + 10 * 5 + (5+1) * 1

    @test length(model.layers) == 4
    @test length(model.parameters) == num_parameters
end
@testitem "Dense Model Initialisation" begin    
    model = chain(
        Static(2),
        Dense(10, activation_fn=sigmoid),
        Dense(5, use_bias=Val(false), activation_fn=tanh),
        Dense(1)
    )

    num_parameters = (2+1)*10 + 10 * 5 + (5+1) * 1

    @test length(model.layers) == 4
    @test length(model.parameters) == num_parameters
end

@testitem "Dense Model View Mapping" begin
    model = chain(
        Static(2),
        Dense(10, activation_fn=sigmoid),
        Dense(5, use_bias=Val(false), activation_fn=tanh),
        Dense(1)
    );
    parameter_array_lengths = (2, 1, 2)
    @test all((x->length(x.parameter_views)).(model.layers[2:end]) .== parameter_array_lengths)

    params = parameters(model)
    # Set parameters equal to their indices
    params .= 1:length(params)
    parameter_values = [
        [model.parameters[1:20], model.parameters[21:30]],
        [model.parameters[31:80]],
        [model.parameters[81:85], model.parameters[86:86]]
    ]
    @test all((x->x.parameter_views).(model.layers[2:end]) .== parameter_values)
end
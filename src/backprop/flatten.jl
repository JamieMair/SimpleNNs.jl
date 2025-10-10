function pullback!(input_partials, _, ::Flatten)
    # On construction, `input_partials` is put in the correct place
    return input_partials
end
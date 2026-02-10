using SimpleNNs
using Documenter

DocMeta.setdocmeta!(SimpleNNs, :DocTestSetup, :(using SimpleNNs); recursive=true)

makedocs(;
    modules=[SimpleNNs],
    authors="Jamie Mair <JamieMair@users.noreply.github.com> and contributors",
    repo="https://github.com/JamieMair/SimpleNNs.jl/blob/{commit}{path}#{line}",
    sitename="SimpleNNs.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://JamieMair.github.io/SimpleNNs.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
        "Getting Started" => "getting_started.md",
        "Examples" => Any[
            "MNIST" => "mnist.md"
        ],
        "Advanced Topics" => Any[
            "Parameter Initialisation" => "initialisation.md",
            "GPU Usage" => "gpu_usage.md",
            "Advanced Usage" => "advanced_usage.md"
        ],
        "API" => "api.md",
        "Index" => "function_index.md",
    ],
)

deploydocs(;
    repo="github.com/JamieMair/SimpleNNs.jl",
    devbranch="main",
)

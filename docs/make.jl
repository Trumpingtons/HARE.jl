using Documenter
using Literate
using HARE

# Convert examples/demo.jl → docs/src/tutorial.md, executing the code so
# that output is captured and rendered.  This is also what runs the demo as
# part of the CI docs job.
DEMO    = joinpath(@__DIR__, "..", "examples", "demo.jl")
OUT_DIR = joinpath(@__DIR__, "src")
Literate.markdown(DEMO, OUT_DIR; name = "tutorial", execute = true, documenter = true)

makedocs(
    modules  = [HARE],
    sitename = "HARE.jl",
    authors  = "Antonio Saragga Seabra",
    format   = Documenter.HTML(
        prettyurls       = get(ENV, "CI", nothing) == "true",
        canonical        = "https://Trumpingtons.github.io/HARE.jl",
        edit_link        = "main",
    ),
    pages = [
        "Home"          => "index.md",
        "Tutorial"      => "tutorial.md",
        "API Reference" => "api.md",
    ],
    doctest    = true,
    checkdocs  = :exports,
    warnonly   = false,
)

deploydocs(
    repo       = "github.com/Trumpingtons/HARE.jl.git",
    devbranch  = "main",
    push_preview = true,
)

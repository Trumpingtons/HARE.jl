# Contributing to HARE.jl

## Repository layout

```
HARE/
├── src/            # Package source code
├── test/           # Test suite (runtests.jl)
├── examples/       # demo.jl — tutorial source (executed by Literate)
├── docs/
│   ├── make.jl     # Documenter build script
│   ├── Project.toml
│   └── src/        # Hand-written pages + generated tutorial.md
└── .github/
    └── workflows/
        ├── CI.yml          # Tests + demo on every push / PR
        └── Documenter.yml  # Docs build + GitHub Pages deploy
```

---

## Running the tests

### Locally

```bash
# From the repo root
julia --project=. -e 'import Pkg; Pkg.test()'
```

Or interactively in a REPL:

```julia
julia --project=.
# then in the package REPL mode (]):
] test
```

### On GitHub

Tests run automatically on every push or pull request to `main` via
`.github/workflows/CI.yml`.  The matrix covers:

| Julia version | Platform |
|---|---|
| 1.10 | Ubuntu, macOS, Windows |
| 1.12 | macOS |

You can also trigger them manually from the **Actions** tab →
**CI** → **Run workflow**.

---

## Building the documentation

### First-time setup (local)

Register the local HARE source with the docs environment so that
`docs/make.jl` picks up your uncommitted changes:

```bash
julia --project=docs -e '
  import Pkg
  Pkg.develop(Pkg.PackageSpec(path="."))
  Pkg.instantiate()'
```

This only needs to be done once (or after adding new docs dependencies).

### Every subsequent build

```bash
julia --project=docs docs/make.jl
```

This runs `Literate.markdown(execute=true)` on `examples/demo.jl`,
generating `docs/src/tutorial.md`, then builds the full site into
`docs/build/`.  Open `docs/build/index.html` in a browser to preview.

> **Note:** The warning `Skipping deployment` is expected when building
> locally — Documenter only deploys from CI.

### Catching missing dependencies before pushing

Local builds can silently succeed even with an incomplete `docs/Project.toml`
because Julia makes HARE's own transitive dependencies available in the docs
environment.  CI is stricter and will fail if a package used directly in
`docs/make.jl` or `examples/demo.jl` is not listed explicitly.

To mirror CI exactly before pushing, run the full clean setup in one go:

```bash
julia --project=docs -e '
  import Pkg
  Pkg.develop(Pkg.PackageSpec(path="."))
  Pkg.instantiate()' && julia --project=docs docs/make.jl
```

**Rule of thumb:** any package that appears in a `using` statement in
`docs/make.jl` or `examples/demo.jl` must be listed in `docs/Project.toml`,
even if it is already a dependency of HARE itself.

### On GitHub

Docs are built and deployed automatically on every push to `main` via
`.github/workflows/Documenter.yml`.  The published site lives at
<https://Trumpingtons.github.io/HARE.jl>.

---

## Recommended development workflow

### Daily cycle

1. **Edit** source files in `src/`.
2. **Test interactively** in `examples/demo.jl` using VS Code's inline
   **Run** buttons — execute a block, inspect output in the REPL, iterate.
3. **Run the test suite** before committing:
   ```bash
   julia --project=. -e 'import Pkg; Pkg.test()'
   ```
4. **Preview the docs** locally when you change docstrings, the tutorial,
   or the API reference:
   ```bash
   julia --project=docs docs/make.jl
   ```
5. **Commit and push to `main`** — CI runs tests on all platforms and
   deploys the docs automatically:
   ```bash
   git add src/ test/ examples/ docs/ Project.toml   # stage changed files
   git commit -m "Short description of what changed"
   git push origin main
   ```
   Watch progress at `https://github.com/Trumpingtons/HARE.jl/actions`.

### Checking the demo independently

The demo script is the fastest way to catch tutorial breakage without
building the full docs (it mirrors what CI does in the `demo` job):

```bash
julia --project=. -e 'include("examples/demo.jl")'
```

### Revise.jl — hot-reload source changes

Add [Revise.jl](https://timholy.github.io/Revise.jl/stable/) to your
personal Julia environment so source edits are picked up without
restarting Julia:

```bash
# Once, in your global environment
julia -e 'import Pkg; Pkg.add("Revise")'
```

Then start your development sessions with:

```julia
using Revise
using HARE
```

Edit any file in `src/` and the changes are available immediately in the
same REPL session.

### Running a single test set

To run only one `@testset` block without executing the full suite, load
the test file directly and filter with the `JULIA_TEST_RUNNER` pattern,
or simply comment out the other sets temporarily.  A lightweight
alternative is
[TestEnv.jl](https://github.com/JuliaTesting/TestEnv.jl):

```julia
using TestEnv
TestEnv.activate()   # activates the test environment
include("test/runtests.jl")
```

---

## Adding a new estimator

1. Create `src/<estimator>.jl` and add a corresponding result type to
   `src/types.jl`.
2. Add `include("<estimator>.jl")` to `src/HARE.jl` and export the
   public functions.
3. Add a section to `examples/demo.jl` with a simulated-data example.
4. Add doctest examples to the docstring and `@testset` blocks to
   `test/runtests.jl`, covering the matrix interface, the formula
   interface, and the `gamma_vcov` / `rho` inference fields where
   applicable.
5. Run `julia --project=docs docs/make.jl` to confirm the docstring
   renders correctly and the tutorial executes without errors.

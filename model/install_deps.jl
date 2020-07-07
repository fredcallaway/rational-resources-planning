using Pkg
println("Installing packages...")
Pkg.add(readlines("deps.txt"))
Pkg.precompile()

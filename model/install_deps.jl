using Pkg
println("Installing packages...")
Pkg.add(split("Glob CSV JSON SplitApplyCombine Memoize Distributions Sobol DataStructures Parameters Printf StatsFuns Optim StatsBase StatsPlots"))
Pkg.precompile()

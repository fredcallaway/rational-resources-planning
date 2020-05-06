using Pkg
println("Installing packages...")
Pkg.add(split("Glob CSV JSON SplitApplyCombine Memoize Distributions DataStructures Parameters Printf StatsFuns Optim StatsBase StatsPlots"))
Pkg.precompile()

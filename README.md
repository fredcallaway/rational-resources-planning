# Code and data for a really cool paper on planning

## Setup
`julia install_deps.jl`
check out demo.jl

## Modeling procedure
1. Create a file in conf/, e.g. conf/cogsci-1.jl
2. Solve the meta MDPs. `julia solve.jl cogsci-1.jl setup` to create a bash script and an sbatch script (for SLURM). Execute one of them.
3. Fit the models, calculate likelihoods, and generate simulations in analysis.jl **THIS CODE NEEDS TO BE UPDATED**


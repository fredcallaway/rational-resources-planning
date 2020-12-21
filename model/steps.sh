exp=exp3

julia -p auto solve.jl all
# alternatively, solve on a cluster:
# julia solve.jl setup
# sbatch solve.sbatch

julia -p 8 Q_table.jl $exp
# Note for experiment 3, this must be done on a machine ~150 GB ram per process (!!)
# I use julia -p 5 Q_table.jl on an x1e EC2
# note to self: sudo mount /dev/xvdf mdps/V

julia -p auto model_comparison.jl $exp
julia -p auto simulate.jl $exp
julia analysis.jl $exp
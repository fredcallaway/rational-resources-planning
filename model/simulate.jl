using StatsBase
using Distributed
using Glob
using CSV
using DataFrames
isempty(ARGS) && push!(ARGS, "exp2")
include("conf.jl")
include("base.jl")
include("models.jl")    


# %% --------
all_trials = load_trials(EXPERIMENT) |> OrderedDict |> sort!
models = [Optimal, BestFirst, BreadthFirst, DepthFirst]
full_fits = deserialize("$base_path/full_fits")

# %% --------

function sbatch_script(n; minutes=5, memory=10000)
    """
    #!/usr/bin/env bash
    #SBATCH --job-name=simulate
    #SBATCH --output=out/%A_%a
    #SBATCH --array=1-$n
    #SBATCH --time=$minutes
    #SBATCH --mem-per-cpu=$memory
    #SBATCH --cpus-per-task=1
    #SBATCH --mail-type=end
    #SBATCH --mail-user=flc2@princeton.edu

    module load julia/1.3.1
    julia simulate.jl $conf \$SLURM_ARRAY_TASK_ID
    """
end

function make_row(t::Trial)
    (
        wid = t.wid,
        mdp=id(mutate(t.m, cost=0.)),
        state_rewards = t.bs[end],
        clicks = t.cs[1:end-1] .- 1,  # to 0-indexing
    )
end

if ARGS[2] == "setup"
    mkpath("$base_path/sims")
    write("simulate.sbatch", sbatch_script(length(all_trials)))
elseif ARGS[2] == "postprocess"
    all_sims = map(collect(keys(all_trials))) do wid
        deserialize("$base_path/sims/$wid")
    end
    flat_sims = all_sims |> flatten |> flatten;
    map(make_row, flat_sims) |> CSV.write("$results_path/simulations.csv")
else
    job = parse(Int, ARGS[2])
    wid = collect(keys(all_trials))[job]
    trials = all_trials[wid]

    sims = map(full_fits[job, :]) do fit
        model_wid = string(typeof(fit.model).name) * "-" * wid
        map(repeat(trials, 50)) do t
            simulate(fit.model, t.m; wid=model_wid)
        end
    end
    serialize("$base_path/sims/$wid", sims)
end
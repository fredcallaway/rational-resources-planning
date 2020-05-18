include("base.jl")

function sbatch_script(n)
    """
    #!/usr/bin/env bash
    #SBATCH --job-name=solve
    #SBATCH --output=out/%A_%a
    #SBATCH --array=1-$n
    #SBATCH --time=30
    #SBATCH --mem-per-cpu=3000
    #SBATCH --cpus-per-task=1
    #SBATCH --mail-type=end
    #SBATCH --mail-user=flc2@princeton.edu

    module load julia/1.3.1
    julia solve.jl $conf \$SLURM_ARRAY_TASK_ID
    """
end

function bash_script(n)
    """
    mkdir -p out/solve
    rm -f out/solve/*
    echo solve mdps
    for i in {1..$n}; do
        julia solve.jl $conf \$i &> out/solve/\$i &
    done
    wait
    echo Done
    """
end

if ARGS[2] == "setup"
    flat_trials = flatten(values(load_trials(EXPERIMENT)));
    all_mdps = [mutate(t.m, cost=c) for t in flat_trials, c in COSTS] |> unique
    mkpath("$base_path/mdps")
    mkpath("$base_path/V")

    for (i, m) in enumerate(all_mdps)
        serialize("$base_path/mdps/$i", m)
    end
    open("solve.sbatch", "w") do f
        write(f, sbatch_script(length(all_mdps)))
    end
    open("solve.sh", "w") do f
        write(f, bash_script(length(all_mdps)))
    end
    println(length(all_mdps), " mdps to solve with solve.sbatch or solve.sh")

else  # solve an MDP
    i = parse(Int, ARGS[2])
    m = deserialize("$base_path/mdps/$i")

    println("Begin solving MDP $i with cost ", round(m.cost; digits=1)); flush(stdout)

    hasher = @isdefined(HASH_FUNCTION) ? eval(HASH_FUNCTION) : default_hash
    println("hash function: ", hasher); flush(stdout)
    V = ValueFunction(m, hasher)
    @time v = V(initial_belief(m))
    println("Value of initial state is ", v)
    id = string(hash(m))
    serialize("$base_path/V/$id", V)
end

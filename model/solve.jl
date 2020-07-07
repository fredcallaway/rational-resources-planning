using Distributed
@everywhere include("base.jl")

function sbatch_script(n; minutes=30, memory=3000)
    """
    #!/usr/bin/env bash
    #SBATCH --job-name=solve
    #SBATCH --output=out/%A_%a
    #SBATCH --array=1-$n
    #SBATCH --time=$minutes
    #SBATCH --mem-per-cpu=$memory
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

function write_mdps()
    flat_trials = flatten(values(load_trials(EXPERIMENT)));
    all_mdps = [mutate(t.m, cost=c) for t in flat_trials, c in COSTS] |> unique
    rm("$base_path/mdps"; force=true, recursive=true)
    mkpath("$base_path/mdps")
    mkpath("$base_path/V")
    for (i, m) in enumerate(all_mdps)
        serialize("$base_path/mdps/$i", m)
    end
    all_mdps
end

function solve_mdps(jobs)
    pmap(jobs) do i
        m = deserialize("$base_path/mdps/$i")
        id = string(hash(m); base=62)
        if isfile("$base_path/V/$id")
            println("This MDP has already been solved.")
            exit()
        end

        V = ValueFunction(m)
        println("Begin solving MDP $i:  cost = $(m.cost),  hasher = $(V.hasher)"); flush(stdout)
        @time v = V(initial_belief(m))
        println("Value of initial state is ", v)
        serialize("$base_path/V/$id", V)
    end
end

function solve_mdps()
    N = length(write_mdps())
    println("Solving $N mdps.")
    solve_mdps(1:N)
end

if basename(PROGRAM_FILE) == basename(@__FILE__)
    if ARGS[2] == "setup"
        write_mdps()
        open("solve.sbatch", "w") do f
            kws = if startswith(EXPERIMENT, "webofcash-2")
                (minutes=30, memory=10000)
            else
                (minutes=30, memory=3000)
            end

            write(f, sbatch_script(length(all_mdps); kws...))
        end
        open("solve.sh", "w") do f
            write(f, bash_script(length(all_mdps)))
        end
        println(length(all_mdps), " mdps to solve with solve.sbatch or solve.sh")
    else  # solve an MDP (or several)
        if ARGS[2] == "all"
            solve_mdps()
        else
            solve_mdps(eval(Meta.parse(ARGS[2])))
        end

    end

end

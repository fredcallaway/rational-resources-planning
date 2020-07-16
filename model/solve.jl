using Distributed

@everywhere include("utils.jl")
@everywhere include("mdp.jl")

COSTS = [0:0.05:4; 100]

mkpath("mdps/withcost")
mkpath("mdps/V")


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

function write_mdps(ids)
    base_mdps = map(ids) do i
        deserialize("mdps/base/$i")
    end
    all_mdps = [mutate(m, cost=c) for m in base_mdps, c in COSTS]
    for m in base_mdps, c in COSTS
        mc = mutate(m, cost=c)
        serialize("mdps/withcost/$(id(mc))", mc)
    end
end

write_mdps() = write_mdps(readdir("mdps/base"))


@everywhere function solve_mdp(i::String)
    m = deserialize("mdps/withcost/$i")
    if isfile("mdps/V/$i")
        println("MDP $i has already been solved.")
        return
    end

    V = ValueFunction(m)
    println("Begin solving MDP $i:  cost = $(m.cost),  hasher = $(V.hasher)"); flush(stdout)
    @time v = V(initial_belief(m))
    println("Value of initial state is ", v)
    serialize("mdps/V/$i", V)
end

@everywhere do_job(id::String) = solve_mdp(id)
@everywhere do_job(idx::Int) = solve_mdp(readdir("mdps/withcost/")[idx])
do_job(jobs) = pmap(solve_mdp, jobs)

function solve_all()
    write_mdps()
    do_job(readdir("mdps/withcost/"))
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

            write("solve.sbatch", sbatch_script(length(all_mdps); kws...))
        end
        open("solve.sh", "w") do f
            write(f, bash_script(length(all_mdps)))
        end
        println(length(all_mdps), " mdps to solve with solve.sbatch or solve.sh")
    else  # solve an MDP (or several)
        if ARGS[2] == "all"
            write_mdps()
            do_job(readdir("mdps/withcost/"))
        else
            do_job(eval(Meta.parse(ARGS[2])))
        end

    end

end

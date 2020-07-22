using Distributed

@everywhere include("utils.jl")
@everywhere include("mdp.jl")

COSTS = [0:0.05:4; 100]

mkpath("mdps/withcost")
mkpath("mdps/V")


function sbatch_script(conf, n; minutes=30, memory=3000)
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

function write_mdps(ids)
    base_mdps = map(ids) do i
        deserialize("mdps/base/$i")
    end
    all_mdps = [mutate(m, cost=c) for m in base_mdps, c in COSTS]
    files = String[]
    for m in base_mdps, c in COSTS
        mc = mutate(m, cost=c)
        f = "mdps/withcost/$(id(mc))"
        serialize(f, mc)
        push!(files, f)
    end
    unsolved = filter(files) do f
        !isfile(replace(f, "withcost" => "V"))
    end
    unsolved = [string(split(f, "/")[end]) for f in unsolved]
    serialize("tmp/unsolved", unsolved)
    unsolved
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
@everywhere do_job(idx::Int) = solve_mdp(deserialize("tmp/unsolved")[idx])
do_job(jobs) = pmap(solve_mdp, deserialize("tmp/unsolved")[jobs])

function solve_all()
    todo = write_mdps()
    do_job(todo)
end

if basename(PROGRAM_FILE) == basename(@__FILE__)
    conf = ARGS[1]
    if ARGS[2] == "setup"
        todo = write_mdps()
        open("solve.sbatch", "w") do f
            kws = (minutes=20, memory=5000)

            write("solve.sbatch", sbatch_script(conf, length(todo); kws...))
        end
        println(length(all_mdps), " mdps to solve with solve.sbatch")
    else  # solve an MDP (or several)
        if ARGS[2] == "all"
            write_mdps()
            do_job(readdir("mdps/withcost/"))
        else
            do_job(eval(Meta.parse(ARGS[2])))
        end

    end

end

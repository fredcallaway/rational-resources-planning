using CSV
using Glob

@everywhere begin
    include("utils.jl")
    include("mdp.jl")
    include("data.jl")
    include("models.jl")
end

@everywhere N_SIM = 100000
@everywhere COSTS = [0:0.05:4; 100]

@everywhere function mean_reward_clicks(pol; N=N_SIM)
    @assert pol.m.cost == 0
    reward, clicks = N \ mapreduce(+, 1:N) do i
        roll = rollout(pol)
        [roll.reward, roll.n_steps - 1]
    end
    (reward=reward, clicks=clicks)
end

function sample_models(M, n)
    space = default_space(M)
    space[:ε] = 0.
    space[:β_click] = 100.
    lower, upper = bounds(space)
    seq = SobolSeq(lower, upper)
    skip(seq, n)
    map(1:n) do i
        x = next!(seq)
        create_model(M, x, [], space)
    end
end

function sample_models(::Type{RandomSelection}, n)
    # ignore n...
    map(RandomSelection, range(0, 1, length=201))
end

function sample_models(::Type{MetaGreedy}, n)
    # ignore n...
    map(COSTS) do cost
        MetaGreedy(cost, 100., 100., 0.)
    end
end

function pareto_front(M, m; n_candidate=10000, n_eval=N_SIM)
    candidates = pmap(sample_models(M, n_candidate)) do model
        mrc = mean_reward_clicks(Simulator(model, m), N=n_eval)
        (model=name(M), mdp=id(m), namedtuple(model)..., mrc...)
    end
    sort!(candidates, by=x->x.clicks)
    result = [candidates[1]]
    for x in candidates
        if x.reward > result[end].reward
            push!(result, x)
        end
    end
    result
end

function write_pareto(::Type{M}, m::MetaMDP; force=false, kws...) where M <: AbstractModel
    f = "mdps/pareto/$(id(m))-$(name(M)).csv"
    if !force && isfile(f)
        println("$f already exists")
    else
        println("Generating $f... ")
        pareto_front(M, m; kws...) |> CSV.write(f)
        println("Wrote ", f)
    end
end

function write_optimal_pareto(;force=false)
    @time pmap(readdir("mdps/withcost")) do i
        f = "mdps/pareto/$i-Optimal.csv"
        if !force && isfile(f)
            println("$f already exists")
        else
            println("Generating $f...")
            V = load_V_nomem(i)
            m = mutate(V.m, cost=0)
            pol = OptimalPolicy(m, V)
            res = (model="Optimal", mdp=id(m), cost=V.m.cost, mean_reward_clicks(pol)...)
            CSV.write(f, [res])  # one line csv
            println("Wrote $f")
        end
    end
end

function write_heuristic_pareto(;force=false)
    Hs = [:BestFirst, :BreadthFirst, :DepthFirst, :BestPlusDepth, :BestPlusBreadth
          # :BestFirstNoBestNext, :BreadthFirstNoBestNext, :DepthFirstNoBestNext
    ]
    models = [RandomSelection; MetaGreedy; [Heuristic{H} for H in Hs]]
    mdps = map(readdir("mdps/base")) do i
        deserialize("mdps/base/$i")
    end

    for M in models, m in mdps
        f = "mdps/pareto/$(id(m))-$(name(M)).csv"
        if !force && isfile(f)
            println("$f already exists")
        else
            println("Generating $f... ")
            pareto_front(M, m) |> CSV.write(f)
            println("Wrote ", f)
        end
    end
end


if basename(PROGRAM_FILE) == basename(@__FILE__)
    # if !isempty(ARGS)
    #     include("conf.jl")
    #     mdps = 
    # end
    mkpath("mdps/pareto")
    if !isempty(ARGS)
        if ARGS[1] == "optimal"
            write_optimal_pareto()
        elseif ARGS[1] == "heuristic"
            write_heuristic_pareto()
        else
            error("Bad arg")
        end
    else
        write_optimal_pareto()
        write_heuristic_pareto()
    end
end










using CSV
using Glob

@everywhere begin
    include("utils.jl")
    include("mdp.jl")
    include("data.jl")
    include("models.jl")
end

mkpath("mdps/pareto")

N_SIM = 100000
mdps = map(deserialize, glob("mdps/base/*"))

@everywhere function mean_reward_clicks(pol; N=N_SIM)
    @assert pol.m.cost == 0
    reward, clicks = N \ mapreduce(+, 1:N) do i
        roll = rollout(pol)
        [roll.reward, roll.n_steps - 1]
    end
    (reward=reward, clicks=clicks)
end

# %% ==================== Optimal ====================
let
    # FORCE = true
    @time pmap(readdir("mdps/withcost")) do i
        f = "mdps/pareto/$i-Optimal.csv"
        if !FORCE && isfile(f)
            println("$f already exists")
        else
            println("Generating $f...")
            V = load_V(i)
            m = mutate(V.m, cost=0)
            pol = OptimalPolicy(m, V)
            res = (model="Optimal", mdp=id(m), cost=V.m.cost, mean_reward_clicks(pol)...)
            CSV.write(f, [res])  # one line csv
            println("Wrote $f")
        end
    end
end


# %% ==================== Other ====================

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

Hs = [:BestFirst, :BestFirstNoBestNext, :BreadthFirst, :DepthFirst,
      # :BestFirstNoBestNext, :BreadthFirstNoBestNext, :DepthFirstNoBestNext
      ]
models = [RandomSelection]
# models = [RandomSelection; [Heuristic{H} for H in Hs]]

for M in models, m in mdps
    f = "mdps/pareto/$(id(m))-$(name(M)).csv"
    if isfile(f)
        println("$f already exists")
    else
        println("Creating ", f)
        pareto_front(M, m) |> CSV.write(f)
        println("Wrote ", f)
    end
end




using CSV
using Glob

@everywhere begin
    include("utils.jl")
    include("mdp.jl")
    include("data.jl")
    include("models.jl")
end

mkpath("mdps/pareto")
# %% ==================== Setup ====================

@everywhere function mean_reward_clicks(pol; N=100000)
    @assert pol.m.cost == 0
    reward, clicks = N \ mapreduce(+, 1:N) do i
        roll = rollout(pol)
        [roll.reward, roll.n_steps - 1]
    end
    (reward=reward, clicks=clicks)
end

# %% ==================== Optimal ====================
let
    FORCE = true
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

# %% ==================== Heuristic ====================

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

function pareto_front(M, m; n_candidate=10000, n_eval=100000)
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

mdps = map(deserialize, glob("mdps/base/*"))

for H in Hs, m in mdps
    f = "mdps/pareto/$(id(m))-$H.csv"
    if isfile(f)
        println("$f already exists")
    else
        pareto_front(Heuristic{H}, m) |> CSV.write(f)
        println("Wrote ", f)
    end
end

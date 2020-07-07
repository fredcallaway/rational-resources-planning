isempty(ARGS) && push!(ARGS, "web")
include("conf.jl")
using Glob
using Serialization
using CSV

@everywhere include("base.jl")
@everywhere include("models.jl")
mkpath("$results_path/pareto")

# %% ==================== Setup ====================

@everywhere function mean_reward_clicks(pol; N=100000)
    @assert pol.m.cost == 0
    reward, clicks = N \ mapreduce(+, 1:N) do i
        roll = rollout(pol)
        [roll.reward, roll.n_steps - 1]
    end
    (reward=reward, clicks=clicks)
end

flat_trials = load_trials(EXPERIMENT) |> values |> flatten
all_mdps = [mutate(t.m, cost=0.) for t in flat_trials] |> unique

map(all_mdps) do m
    (mdp=id(m), variance=variance_structure(m))
end |> CSV.write("$results_path/mdps.csv")


# %% ==================== Optimal ====================

opt_pareto = pmap(Iterators.product(all_mdps, COSTS)) do (m, cost)
    V = deserialize("$base_path/V/$(id(mutate(m, cost=cost)))")
    pol = OptimalPolicy(m, V)
    (model="Optimal", mdp=id(m), cost=V.m.cost, mean_reward_clicks(pol)...)
end
opt_pareto = opt_pareto[:];
sort!(opt_pareto, by=(x->x.cost))
opt_pareto |> CSV.write("$results_path/pareto/Optimal.csv")

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
        (model=name(model), mdp=id(m), namedtuple(model)..., mrc...)
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

Hs = [:BestFirst, :BreadthFirst, :DepthFirst, 
      :BestFirstNoBestNext, :BreadthFirstNoBestNext, :DepthFirstNoBestNext]

for H in Hs[5:6]
    M = Heuristic{H}
    println("Making pareto front for ", name(M))
    mapmany(all_mdps) do m
        pareto_front(M, m)
    end |> CSV.write("$results_path/pareto/$(name(M)).csv")
end

# %% --------

H = first(Hs)
M = Heuristic{H}
m = all_mdps[1]

models = sample_models(M, 2)
rollout(Simulator(models[1], m))
pareto_front(M, all_mdps[1])



# %% ==================== Old ====================

function possible_path_values(m::MetaMDP)
    dists = [m.rewards[i] for i in paths(m)[1]]
    map(sum, Iterators.product([[d.support; 0] for d in dists]...)) |> unique
end

function possible_thresholds(m::MetaMDP)
    ppv = possible_path_values(m)
    vals = map(Iterators.product(ppv, ppv)) do (v1, v2)
        abs(v1 - v2) - .1
    end |> unique |> sort
end


flat_trials = load_trials(EXPERIMENT) |> values |> flatten
all_mdps = [mutate(t.m, cost=0) for t in flat_trials] |> unique
classical_models = [BestFirst, BreadthFirst, DepthFirst]
jobs = mapmany(Iterators.product(classical_models, all_mdps)) do (M, m)
    θs = M == BreadthFirst ? (0.1:1:3.1) : possible_thresholds(m)
    map(θs) do θ
        (M, m, θ)
    end
end

pmap(jobs) do (M, m, θ)
    model = M(1e3, 1e3, θ, 0.)
    pol = Simulator(model, m)
    (model=string(M), mdp=id(m), threshold=θ, mean_reward_clicks(pol)...)
end |> CSV.write("$results_path/pareto/classical.csv")

# %% --------

function sample_models(M, n)
    M = BasFirst
    space = default_space(M)
    space[:ε] = 0.
    lower, upper = bounds(space)
    seq = SobolSeq(lower, upper)
    skip(seq, n)
    map(1:n) do i
        x = next!(seq)
        create_model(M, x, [], space)
    end
end

models = sample_models(BasFirst, 40);
pmap(models) do model
    pol = Simulator(model, m)
    x = mean_reward_clicks(pol)
    (x..., β_term=model.β_term, β_click=model.β_click, θ_term=model.θ_term, ε=model.ε)
end |> CSV.write("$results_path/pareto/soft_best_first.csv")

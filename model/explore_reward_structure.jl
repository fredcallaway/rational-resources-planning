@everywhere begin
    base_path = "tmp/explore_reward"
    results_path = "results/explore_reward"
    include("mdp.jl")
    include("utils.jl")
    include("data.jl")
    include("models.jl")
end

using CSV
mkpath(base_path)
mkpath(results_path)
# %% ==================== Setup ====================

@everywhere begin
    function depth(g, i)
        pths = paths(g)
        i == 1 && return 0
        for d in 1:maximum(length.(pths))
            for p in pths
                p[d] == i && return d
            end
        end
        @assert false
    end

    function make_rewards(graph, mult, depth_factor)
        base = mult * Float64[-2, -1, 1, 2]
        map(eachindex(graph)) do i
            i == 1 && return DiscreteNonParametric([0.])
            vs = round.(unique(base .*  depth_factor ^ (depth(graph, i)-1)))
            DiscreteNonParametric(vs)
        end
    end

    function make_mdp(factor, mult)
        g = tree([4,1,2])
        if factor < 1
             mult *= factor ^ -(length(paths(g)[1]) - 1)
         end
        rewards = make_rewards(g, float(mult), factor)
        MetaMDP(g, rewards, 0., -Inf, true)
    end

    function mean_reward_clicks(pol; N=100000)
        reward, clicks = N \ mapreduce(+, 1:N) do i
            roll = rollout(pol)
            [roll.reward, roll.n_steps - 1]
        end
        (reward=reward, clicks=clicks)
    end
end

factors = [1//2, 1//3, 2, 3]
mults = 1:3


# %% ==================== BasFirst ====================

function possible_path_values2(m::MetaMDP)
    dists = [m.rewards[i] for i in paths(m)[1]]
    map(sum, Iterators.product([[d.support; 0] for d in dists]...)) |> unique
end

function possible_thresholds(m::MetaMDP)
    ppv = possible_path_values2(m)
    vals = map(Iterators.product(ppv, ppv)) do (v1, v2)
        abs(v1 - v2)
    end |> unique |> sort
end

bf_jobs = mapmany(Iterators.product(factors, mults)) do (factor, mult)
    m = make_mdp(factor, mult)
    map(possible_thresholds(m)) do θ
        (factor, mult, θ)
    end
end

pmap(bf_jobs[1:40]) do (factor, mult, θ)
    m = make_mdp(factor, mult)
    model = BasFirst(1e3, 1e3, θ, 0.)
    pol = Simulator(model, m)
    (model="BasFirst", factor=factor, mult=mult, threshold=θ, mean_reward_clicks(pol)...)
end |> CSV.write("$results_path/bas_first.csv")
println("Wrote $results_path/bas_first.csv")


# %% ==================== Optimal ====================

@everywhere function solve(m::MetaMDP)
    V = ValueFunction(m, hash_412)
    @time v = V(initial_belief(m))
    return V
end

opt_jobs = collect(Iterators.product(factors, mults, costs))[:]
pmap(opt_jobs) do (factor, mult, cost)
    m = make_mdp(factor, mult)
    V = solve(mutate(m, cost=cost))
    pol = OptimalPolicy(m, V)
    (model="Optimal", factor=factor, mult=mult, cost=cost, mean_reward_clicks(pol)...)
end |> CSV.write("$results_path/optimal.csv")
println("Wrote $results_path/optimal.csv")




# %% ==================== OLD ====================





Vs = pmap(solve, mdps)

# %% --------
function test_depth_first(V)
    b = initial_belief(V.m)
    b[2] = minimum(V.m.rewards[2].support)
    Q(V, b, 3) - Q(V, b, 6) # prefer continuing
end

function test_breadth_first(V)
    b = initial_belief(V.m)
    b[2] = V.m.rewards[2].support[3]  # moderate positive
    v = V(initial_belief(V.m))
    Q(V, b, 6) - Q(V, b, 3)  # prefer trying another
end

function expected_reward(V)
    V(initial_belief(V.m))
end



# %% --------
using AxisArrays
VV = AxisArray(Vs; factor=factors, cost=costs, mult=mults)

map(VV[factor=4]) do V
    round.([expected_reward(V), test_depth_first(V)]; digits=2)
end

map(VV[factor=2]) do V
    round.([expected_reward(V), test_breadth_first(V)]; digits=2)
end

map(expected_reward, VV)[factor=3]
map(test_depth_first, VV)[factor=3]

map(expected_reward, VV)[factor=1]
map(test_breadth_first, VV)[factor=1]


# %% --------


let
    for (i, V) in enumerate(Vs)
        mult = V.m.rewards[2].support[3]
        cost = V.m.cost
        b = initial_belief(V.m)
        b[2] = minimum(V.m.rewards[2].support)
        v = V(initial_belief(V.m))
        d = Q(V, b)[4] - Q(V, b)[7]
        @printf "(%02d) %4d %4.1f %5.2f %4.2f\n" i mult cost v d
    end
end

# %% --------

jobs = Iterators.product(1:.2:3, 1:4)
mdps = map(jobs) do (cost, mult)
    g = tree([3,1,2])
    rewards = make_rewards(g, mult*9, 1/3)
    MetaMDP(g, rewards, cost, -Inf, true)
end;
Vs = pmap(solve, mdps);

# %% --------
let
    for (i, V) in enumerate(Vs)
        mult = V.m.rewards[4].support[3]
        cost = V.m.cost
        b = initial_belief(V.m)
        b[2] = V.m.rewards[2].support[3]
        v = V(initial_belief(V.m))
        d = Q(V, b)[4] - Q(V, b)[7]
        @printf "(%02d) %4d %4.1f %5.2f %4.2f\n" i mult cost v d
    end
end

# %% --------

V = Vs[21]
b = initial_belief(V.m)
b[2] = V.m.rewards[2].support[4]
b[6] = V.m.rewards[2].support[4]

Q(V, b)
# TODO check out (21)   18  2.8 17.64 -0.34

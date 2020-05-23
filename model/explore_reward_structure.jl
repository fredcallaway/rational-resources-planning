@everywhere begin 
    include("mdp.jl")
    include("utils.jl")
end

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

@everywhere function solve(m::MetaMDP)
    V = ValueFunction(m, hash_312)
    @time v = V(initial_belief(m))
    return V
end

function simulate(V::ValueFunction)
    pol = OptimalPolicy(V)
    bs = Belief[]
    cs = Int[]
    roll = rollout(pol) do b, c
        push!(bs, copy(b)); push!(cs, c)
    end
    bs, cs
end

function show_sim(V)
    bs, cs = simulate(V)
    for (i, c) in enumerate(cs)
        c != 0 && print(c, " ", bs[i+1][c], "   ")
    end
    return bs
end

# %% ==================== Generate MDPs ====================

factors = [1//2, 1//3, 2, 3]
costs = 1:.2:3
mults = 1:3
jobs = collect(Iterators.product(factors, costs, mults))

mdps = map(jobs) do (factor, cost, mult)
    g = tree([3,1,2])
    if factor < 1
         mult *= factor ^ -(length(paths(g)[1]) - 1)
     end
    rewards = make_rewards(g, float(mult), factor)
    MetaMDP(g, rewards, cost, -Inf, true)
end;

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

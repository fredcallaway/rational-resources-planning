# include("explore_reward_structure_base.jl")
# using JSON

# function sample_rewards(factor; N=100)
#     r = make_mdp(factor, 1).rewards
#     [Int.(rand.(r)) for i in 1:N]
# end
# rewards = Dict(factor => sample_rewards(factor) for factor in [1//3, 3])
# write("/Users/fred/heroku/webofcash2/rewards.json", json(rewards))

# # %% ====================  ====================
using JSON
include("mdp.jl")
mkpath("mdps/base")

base = "/Users/fred/heroku/webofcash2/static/json"

DIRECTIONS = ["up", "right", "down", "left"]
OFFSETS = [(0, -1), (1,0), (0, 1), (-1, 0)]


graph = Dict()
layout = Dict()
 # {"0": {"up": [0, "1"], "right": [0, "5"], "down": [0, "9"], "left": [0, "13"]}, 

function diridx(dir)
    1 + (dir + 4) % 4
end

function spirals(n, turns)
    nodes = Iterators.Stateful(string.(1:1000))
    pos = [0, 0]

    function relative_to_absolute(init, turns)
        dir = init
        map(turns) do t
            dir = (dir + t) % length(DIRECTIONS)
        end
    end

    function rec(pos, dirs)
        pos = pos .+ OFFSETS[diridx(dirs[1])]
        id = first(nodes)
        layout[id] = pos
        graph[id] = Dict()
        if length(dirs) > 1
            graph[id][DIRECTIONS[diridx(dirs[2])]] = [0, rec(pos, dirs[2:end])]
        end
        id
    end

    graph = Dict()
    layout = Dict()
    pos = (0, 0)
    layout["0"] = pos
    graph["0"] = Dict(
        DIRECTIONS[diridx(dir)] => [0, rec(pos, relative_to_absolute(dir, turns))]
        for dir in 0:n-1
    )
    (layout=layout, initial="0", graph=graph)
end

write("$base/structure/41111.json", json(spirals(4, [0, 1, -1, 1, 1])))

# ---------- REWARDS ---------- #

function make_rewards(g::Graph, kind::Symbol, factor, p1, p2)
    @assert kind in [:breadth, :depth, :constant]
    map(eachindex(g)) do i
        i == 1 && return DiscreteNonParametric([0.])
        if kind == :constant
            DiscreteNonParametric([-10, -5, 5, 10])
        elseif kind == :depth && isempty(g[i])
            DiscreteNonParametric([-2factor, factor], [1-p2, p2])
        elseif kind == :breadth && i in g[1]
            x = (p2 * (1 - factor) + factor) / (2factor)
            DiscreteNonParametric([-factor, 1, factor], round.([x, p2, (1 - x - p2)]; digits=6))
        else
            DiscreteNonParametric([-1, 1], [1-p1, p1])
        end
    end
end

function make_mdp(branching, kind, factor, p1, p2)
    g = tree(branching)
    rewards = make_rewards(g, kind, factor, p1, p2)
    MetaMDP(g, rewards, 0., -Inf, true)
end

function write_trials(name::String, m::MetaMDP)
    trials = map(1:300) do i
        rewards = rand.(m.rewards)
        tid = id(m) * "-" * string(hash(rewards); base=62)
        (trial_id=tid, stateRewards=rewards)
    end

    f = "mdps/base/$(id(m))"
    serialize(f, m)
    println("Wrote ", f)
    
    f = "$base/rewards/$name.json"
    write(f, json(trials))
    println("Wrote ", f)
end

write_trials("41111_increasing", make_mdp([4,1,1,1,1], :depth, 20, 1/2, 2/3))
write_trials("41111_decreasing", make_mdp([4,1,1,1,1], :breadth, 20, 1/2, 3/5))
write_trials("41111_constant", make_mdp([4,1,1,1,1], :constant, NaN, NaN, NaN))

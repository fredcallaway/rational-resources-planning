using JSON
# import DataFrames: DataFrame
using SplitApplyCombine
using Memoize

include("reward_structures.jl")

struct Trial
    m::MetaMDP  # Note: cost must be NaN
    wid::String
    bs::Vector{Belief}
    cs::Vector{Int}
    score::Float64
    rts::Vector
    path::Vector{Int}
end
Base.hash(t::Trial, h::UInt64) = hash_struct(t, h)
# Base.:(==)(x1::Trial, x2::Trial) = struct_equal(x2, x2)  # doesn't work because NaN ≠ NaN

struct Datum
    t::Trial
    b::Belief
    c::Int
    # c_last::Union{Int, Nothing}
end

# this is used to look up data in the Q_TABLE
function q_key(d::Datum)
    hash((d.b, d.c, d.t.m))
end

Base.hash(t::Datum, h::UInt64) = hash_struct(t, h)
# Base.:(==)(x1::Datum, x2::Datum) = struct_equal(x2, x2)


function make_mdp(map_name::String)
    if startswith(map_name, "fantasy")
        error("TODO")
            # min_reward = -300
    else
        branching, variance = split(map_name, "-")
        g = tree(parse.(Int, collect(branching)))
        MetaMDP(g, make_rewards(g, variance), NaN, -Inf, EXPAND_ONLY)
    end
end


# function parse_graph(t)
#     if haskey(t, "graph")
#         g = t["graph"]
#         if g isa String
#             return tree(parse.(Int, collect(g)))
#         end
#         return map(g) do children
#             Int.(children .+ 1)
#         end
#     else
#         edges = map(t["edges"]) do (x, y)
#             Int(x) + 1, Int(y) + 1
#         end
#         n_node = maximum(flatten(edges))
#         graph = [Int[] for _ in 1:n_node]
#         for (a, b) in edges
#             push!(graph[a], b)
#         end
#         graph
#     end
# end

# get_reward_structure(map) = startswith(map, "fantasy") ? "roadtrip" : split(map, "-")[end]

# @memoize Dict function make_meta_mdp(graph, rstruct, cost)
#     min_reward = rstruct == "roadtrip" ? -300 : -Inf
#     rewards = reward_distributions(rstruct, graph)
#     MetaMDP(graph, rewards, cost, min_reward, EXPAND_ONLY)
# end

function Trial(wid::String, t::Dict{String,Any})
    m = make_mdp(t["map"])
    # graph = parse_graph(t)

    bs = Belief[]
    cs = Int[]
    b = initial_belief(m)

    for (c, value) in t["reveals"]
        c += 1  # 0->1 indexing
        push!(bs, copy(b))
        push!(cs, c)

        # we ignore the case when c = 1 because the model assumes this
        # value is known to be 0 (it is irrelevant to the decision).
        # it actually shouldn't be allowed in the experiment...
        if c != 1
            if startswith(t["map"], "fantasy")  # road trip experiment uses units of cost
                value *= -1
            end
            b[c] = value
        end
    end
    push!(bs, b)
    push!(cs, TERM)
    path = Int.(t["route"] .+ 1)[2:end]
    rts = [x == nothing ? NaN : float(x) for x in t["rts"]]
    Trial(m, wid, bs, cs, get(t, "score", NaN), rts, path)
end

# this is memoized for the sake of future memoization based on object ID
@memoize function get_data(t::Trial)
    map(eachindex(t.bs)) do i
        # c_last = i == 1 ? nothing : t.cs[i-1]
        # tmp = zeros(length(t.bs)+1)
        Datum(t, t.bs[i], t.cs[i])
    end
end

get_data(trials::Vector{Trial}) = flatten(map(get_data, trials))


function Base.show(io::IO, t::Trial)
    print(io, "T")
end

# function load_params(experiment)::DataFrame
#     x = open(JSON.parse, "../data/$experiment/params.json")
#     DataFrame(map(namedtuple, x))
# end

@memoize function load_trials(experiment)::Dict{String,Vector{Trial}}
    data = open(JSON.parse, "../data/$experiment/trials.json")
    map(data) do wid, tt
        wid => [Trial(wid, t) for t in tt]
    end |> Dict
end

# # to ensure that precomputed Qs are correctly aligned
# function checksum(data::Vector{Datum})
#     map(data) do d
#         d.t.map, d.t.wid
#     end |> hash
# end



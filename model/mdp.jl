using Parameters
using Distributions
import Base
using Printf: @printf
using Memoize
using StatsFuns

include("dnp.jl")

const TERM = 0  # termination action
# const NULL_FEATURES = -1e10 * ones(4)  # features for illegal computation
const N_SAMPLE = 10000
const N_FEATURE = 5

Graph = Vector{Vector{Int}}

"Parameters defining a class of problems."
@with_kw struct MetaMDP
    graph::Graph
    rewards::Vector{Distribution}
    cost::Float64
    min_reward::Float64 = -Inf
    expand_only::Bool = true
end

struct Belief
    rewards::Vector{Float64}
    frontier::BitVector
end
Base.eachindex(b::Belief) = eachindex(b.rewards)
Base.length(b::Belief) = length(b.rewards)
Base.copy(b::Belief) = Belief(copy(b.rewards), copy(b.frontier))

function MetaMDP(g::Graph, rdist::Distribution, cost::Float64; kws...)
    rewards = repeat([rdist], length(g))
    MetaMDP(graph=g, rewards=rewards, cost=cost; kws...)
end

Base.:(==)(x1::MetaMDP, x2::MetaMDP) = struct_equal(x1, x2)
Base.hash(m::MetaMDP, h::UInt64) = hash_struct(m, h)
Base.length(m::MetaMDP) = length(m.graph)

function initial_belief(m::MetaMDP)
    rewards = [0; fill(NaN, length(m)-1)]
    frontier = falses(length(m))
    for i in m.graph[1]
        frontier[i] = true
    end
    Belief(rewards, frontier)
end
observed(b::Belief) = @. !isnan(b.rewards)
observed(b::Belief, c::Int) = !isnan(b.rewards[c])
unobserved(b::Belief) = [c for c in eachindex(b) if isnan(b.rewards[c])]

function tree(branching::Vector{Int})
    t = Vector{Int}[]
    function rec!(d)
        children = Int[]
        push!(t, children)
        idx = length(t)
        if d <= length(branching)
            for i in 1:branching[d]
                child = rec!(d+1)
                push!(children, child)
            end
        end
        return idx
    end
    rec!(1)
    t
end

tree(b::Int, d::Int) = tree(repeat([b], d))


# %% ====================  ====================

@memoize function paths(m::MetaMDP)
    g = m.graph
    frontier = [[1]]
    result = Vector{Int}[]

    function search!(path)
        loc = path[end]
        if isempty(g[loc])
            push!(result, path)
            return
        end
        for child in g[loc]
            push!(frontier, [path; child])
        end
    end
    while !isempty(frontier)
        search!(pop!(frontier))
    end
    [pth[2:end] for pth in result]
end

function easy_path_value(m::MetaMDP, b::Belief, path)
    d = 0.
    for i in path
        d += (observed(b, i) ? b.rewards[i] : mean(m.rewards[i]))
    end
    d
end

function path_value(m::MetaMDP, b::Belief, path)
    d = 0.
    if m.min_reward == -Inf
        return easy_path_value(m, b, path)
    end
    for i in path
        if observed(b, i)
            d += b.rewards[i]
        else
            d += m.rewards[i]
        end
    end
    d isa Float64 && return d
    map(d) do x
        max(m.min_reward, x)
    end |> mean
end

function path_values(m::MetaMDP, b::Belief)
    [path_value(m, b, path) for path in paths(m)]
end

function term_reward(m::MetaMDP, b::Belief)    
    maximum(path_values(m, b))
end

# %% ====================  ====================

# function has_observed_parent(graph, b, c)
#     any(enumerate(graph)) do (i, children)
#         c in children && observed(b, i)
#     end
# end

function allowed(m::MetaMDP, b::Belief, c::Int)
    c == TERM && return true
    !isnan(b.rewards[c]) && return false
    !m.expand_only || b.frontier[c]
end


function results(m::MetaMDP, b::Belief, c::Int)
    @assert allowed(m, b, c)
    res = Tuple{Float64,Belief,Float64}[]
    if c == TERM
        b1 = copy(b)
        push!(res, (1., b1, term_reward(m, b)))
        return res
    end
    # if !isnan(b[c])
    #     push!(res, (1., b, -Inf))
    #     return res
    # end
    for v in support(m.rewards[c])
        b1 = copy(b)
        b1.rewards[c] = v
        b1.frontier[c] = false
        for i in m.graph[c]
            b1.frontier[i] = true
        end
        p = pdf(m.rewards[c], v)
        push!(res, (p, b1, -m.cost))
    end
    res
end

function observe!(m::MetaMDP, b::Belief, c::Int)
    @assert allowed(m, b, c)
    b[c] = rand(m.rewards[c])
end

# %% ==================== Solution ====================

struct ValueFunction{F}
    m::MetaMDP
    hasher::F
    cache::Dict{UInt64, Float64}
end

function symmetry_breaking_hash(m::MetaMDP, b::Belief)
    lp = length(paths(m))
    hash(sum(hash(b[pth]) >> 3 for pth in paths(m)))
end
default_hash(m::MetaMDP, b::Belief) = hash(b)

ValueFunction(m::MetaMDP, h) = ValueFunction(m, h, Dict{UInt64, Float64}())
ValueFunction(m::MetaMDP) = ValueFunction(m, default_hash)

function Q(V::ValueFunction, b::Belief, c::Int)::Float64
    c == 0 && return term_reward(V.m, b)
    !allowed(V.m, b, c) && return -Inf 
    # !isnan(b[c]) && return -Inf  # already observed
    sum(p * (r + V(s1)) for (p, s1, r) in results(V.m, b, c))
end

Q(V::ValueFunction, b::Belief) = [Q(V,b,c) for c in 0:length(b)]

function (V::ValueFunction)(b::Belief)::Float64
    key = V.hasher(V.m, b)
    haskey(V.cache, key) && return V.cache[key]
    return V.cache[key] = maximum(Q(V, b))
end

function Base.show(io::IO, v::ValueFunction)
    print(io, "V")
end


# # ========== Policy ========== #
noisy(x, ε=1e-10) = x .+ ε .* rand(length(x))

abstract type Policy end

struct OptimalPolicy <: Policy
    m::MetaMDP
    V::ValueFunction
end
OptimalPolicy(V::ValueFunction) = OptimalPolicy(V.m, V)
(pol::OptimalPolicy)(b::Belief) = begin
    argmax(noisy([Q(pol.V, b, c) for c in 0:length(b)])) - 1
end

struct RandomPolicy <: Policy
    m::MetaMDP
end

(pol::RandomPolicy)(b) = rand(findall(allowed.(m, b, c)))

# struct MetaGreedy <: Policy
#     m::MetaMDP
# end
# (pol::MetaGreedy)(b::Belief) = begin
#     argmax(noisy([voi1(pol.m, b, c) for c in 0:length(b.matrix)])) - 1
# end

"Runs a Policy on a Problem."
function rollout(pol::Policy; initial=nothing, max_steps=100, callback=((b, c) -> nothing))
    m = pol.m
    b = initial != nothing ? initial : initial_belief(m)
    reward = 0
    for step in 1:max_steps
        c = (step == max_steps) ? TERM : pol(b)
        callback(b, c)
        if c == TERM
            reward += term_reward(m, b)
            return (reward=reward, n_steps=step, belief=b)
        else
            reward -= m.cost
            observe!(m, b, c)
        end
    end
end

function rollout(callback::Function, pol::Policy; initial=nothing, max_steps=100)
    rollout(pol::Policy; initial=initial, max_steps=max_steps, callback=callback)
end

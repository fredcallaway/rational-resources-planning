using StatsFuns: logistic

# ---------- Base code for all heuristic models ---------- #

struct Heuristic{H,T} <: AbstractModel{T}
    # Selection rule weights
    β_best::T
    β_depth::T
    # Stopping rule weights
    β_satisfice::T
    β_best_next::T
    β_depth_limit::T
    θ_term::T
    # Stopping rule thresholds
    # θ_satisfice::T
    # θ_best_next::T
    # θ_depth_limit::T
    # Lapse rate
    ε::T
end

name(::Type{Heuristic{H}}) where H = string(H)
name(::Type{Heuristic{H,T}}) where {H,T} = string(H)


Base.show(io::IO, model::Heuristic{H,T}) where {H, T} = print(io, "Heuristic{:$H}(...)")
function Base.display(model::Heuristic{H,T}) where {H, T}
    println("--------- Heuristic{:$H} ---------")
    space = default_space(Heuristic{H})
    for k in fieldnames(Heuristic)
        print("  ", lpad(k, 14), " = ")
        if length(space[k]) == 1
            println("(", space[k], ")")
        else
            println(round(getfield(model, k); sigdigits=3))
        end
            
    end
end

function action_dist!(p::Vector{T}, model::Heuristic{H,T}, φ::NamedTuple) where {H, T}
    p .= 0.
    if length(φ.frontier) == 0
        p[1] = 1.
        return p
    end

    ε = model.ε
    p_rand = ε / (1+length(φ.frontier))
    p_term = termination_probability(model, φ)
    p[1] = p_rand + (1-ε) * p_term

    # Note: we assume that p[i] is zero for all non-frontier nodes
    p_select = selection_probability(model, φ)
    for i in eachindex(p_select)
        c = φ.frontier[i] + 1
        p[c] = p_rand + (1-ε) * (1-p_term) * p_select[i]
    end
    p
end

function action_dist(model::Heuristic{H,T}, m::MetaMDP, b::Belief) where {H, T}
    φ = features(Heuristic{H,T}, m, b)
    p = zeros(T, length(b) + 1)
    action_dist!(p, model, φ)
end

function features(::Type{Heuristic{H,T}}, m::MetaMDP, b::Belief) where {H, T}
    frontier = findall(1:length(b)) do i
        allowed(m, b, i)
    end
    (
        frontier = frontier,
        frontier_values = node_values(m, b)[frontier],
        frontier_depths = node_depths(m)[frontier],
        term_reward = term_reward(m, b),
        best_vs_next = best_vs_next(m, b),
        min_depth = min_depth(m, b),
        tmp = zeros(T, length(frontier)),  # pre-allocate for use in selection_probability
    )
end

function logp(L::Likelihood, model::Heuristic{H,T}) where {H, T}
    φ = memo_map(L) do d
        features(Heuristic{H,T}, d.t.m, d.b)
    end

    tmp = zeros(T, n_action(L))
    total = zero(T)
    for i in eachindex(L.data)
        a = L.data[i].c + 1
        p = action_dist!(tmp, model, φ[i])
        if !(sum(p) ≈ 1)
            println("\n\n")
            display(model)
            println("\n\n")
        end
        @assert sum(p) ≈ 1
        total += log(p[a])
    end
    total
end

# ---------- Define parameter ranges and special cases ---------- #

default_space(::Type{Heuristic{:Full}}) = Space(
    :β_best => (1e-6, 3),
    :β_depth => (-3, 3),

    :β_satisfice => (1e-6, 3),
    :β_best_next => (1e-6, 3),
    :β_depth_limit => (1e-6, 3),

    :θ_term => (-30, 30),
    # :θ_satisfice => (0, MAX_THETA),
    # :θ_best_next => (0, MAX_THETA),
    # :θ_depth_limit => (0, 4),

    :ε => (1e-3, 1)
)

function _modify(base; kws...)
    space = default_space(Heuristic{base})
    for (k,v) in kws
        space[k] = v
    end
    space
end

default_space(::Type{Heuristic{:BestFirst}}) = _modify(:Full, β_depth=0)
default_space(::Type{Heuristic{:DepthFirst}}) = _modify(:Full, β_best=0, β_depth=(0, 3))
default_space(::Type{Heuristic{:BreadthFirst}}) = _modify(:Full, β_best=0, β_depth=(-3, 0))

default_space(::Type{Heuristic{:BestFirstNoSatisfice}}) = _modify(:BestFirst, β_satisfice=0, θ_satisfice=0)
default_space(::Type{Heuristic{:BestFirstNoBestNext}}) = _modify(:BestFirst, β_best_next=0, θ_best_next=0)
default_space(::Type{Heuristic{:BestFirstNoDepthLimit}}) = _modify(:BestFirst, β_depth_limit=0, θ_depth_limit=0)

default_space(::Type{Heuristic{:BreadthFirstNoSatisfice}}) = _modify(:BreadthFirst, β_satisfice=0, θ_satisfice=0)
default_space(::Type{Heuristic{:BreadthFirstNoBestNext}}) = _modify(:BreadthFirst, β_best_next=0, θ_best_next=0)
default_space(::Type{Heuristic{:BreadthFirstNoDepthLimit}}) = _modify(:BreadthFirst, β_depth_limit=0, θ_depth_limit=0)

default_space(::Type{Heuristic{:DepthFirstNoSatisfice}}) = _modify(:DepthFirst, β_satisfice=0, θ_satisfice=0)
default_space(::Type{Heuristic{:DepthFirstNoBestNext}}) = _modify(:DepthFirst, β_best_next=0, θ_best_next=0)
default_space(::Type{Heuristic{:DepthFirstNoDepthLimit}}) = _modify(:DepthFirst, β_depth_limit=0, θ_depth_limit=0)

default_space(::Type{Heuristic{:FullNoSatisfice}}) = _modify(:Full, β_satisfice=0, θ_satisfice=0)
default_space(::Type{Heuristic{:FullNoBestNext}}) = _modify(:Full, β_best_next=0, θ_best_next=0)
default_space(::Type{Heuristic{:FullNoDepthLimit}}) = _modify(:Full, β_depth_limit=0, θ_depth_limit=0)

default_space(::Type{Heuristic{:Foobar}}) = _modify(:Full, θ_depth_limit=0, θ_satisfice=0)

# ---------- Selection rule ---------- #

function selection_probability(model, φ)
    p = φ.tmp  # use pre-allocated array for memory efficiency
    @. p = model.β_best * φ.frontier_values + model.β_depth * φ.frontier_depths
    softmax!(p)
end

"The maximum expected value of any path going through this node."
function node_values(m::MetaMDP, b::Belief)
    nv = fill(-Inf, length(m))
    for p in paths(m)
        v = path_value(m, b, p)
        for i in p
            nv[i] = max(nv[i], v)
        end
    end
    nv
end

"Distance of each node from the start state."
@memoize function node_depths(m::MetaMDP)
    g = m.graph
    result = zeros(Int, length(g))
    function rec(i, d)
        result[i] = d
        for j in g[i]
            rec(j, d+1)
        end
    end
    rec(1, 0)
    result
end

# ---------- Stopping rule---------- #

function termination_probability(model, φ)
    v = model.θ_term + 
        model.β_satisfice * φ.term_reward +
        model.β_best_next * φ.best_vs_next +
        model.β_depth_limit * 10φ.min_depth  # put min_depth on the roughly the same scale as the others
    # v = model.β_satisfice * (φ.term_reward - model.θ_satisfice) +
    #     model.β_best_next * (φ.best_vs_next - model.θ_best_next) +
    #     model.β_depth_limit * (φ.min_depth - model.θ_depth_limit)
    logistic(v)
end

function min_depth(m::MetaMDP, b::Belief)
    nd = node_depths(m)
    # minimum(nd[c] for c in 1:length(b) if allowed(m, b, c))  # do this but handle fully revealed case
    mapreduce(min, 1:length(b); init=Inf) do c
        allowed(m, b, c) ? nd[c] : Inf
    end
end

"How much better is the best path from its competitors?"
function best_vs_next(m, b)
    pvals = path_values(m, b)
    undetermined = [isnan(b[path[end]]) for path in paths(m)]
    # find best path, breaking ties in favor of undetermined
    best = argmax(collect(zip(pvals, undetermined)))
    
    competing_value = if undetermined[best]
        # best path is undetermined -> competing value is the second best path (undetermined or not)
        partialsort(pvals, 2; rev=true)
    else
        # best path is determined -> competing value is the best undetermined path
        vals = pvals[undetermined]
        if isempty(vals)
            0.  # Doesn't matter, you have to terminate.
        else
            maximum(vals)
        end
    end
    
    pvals[best] - competing_value
end


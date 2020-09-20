using StatsFuns: logistic

# ---------- Base code for all heuristic models ---------- #

struct Heuristic{H,T} <: AbstractModel{T}
    # Selection rule weights
    β_best::T
    β_depth::T
    β_expand::T
    # Stopping rule weights
    β_satisfice::T
    β_best_next::T
    β_depth_limit::T
    # Stopping rule threshold
    θ_term::T
    # Lapse rate
    ε::T
end

name(::Type{Heuristic{H}}) where H = string(H)
name(::Type{Heuristic{H,T}}) where {H,T} = string(H)
name(::Heuristic{H,T}) where {H,T} = string(H)


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

function features(::Type{Heuristic{H,T}}, m::MetaMDP, b::Belief) where {H,T}
    frontier = get_frontier(m, b)
    expansion = map(frontier) do c
        has_observed_parent(m.graph, b, c)
    end
    (
        frontier = frontier,
        expansion=expansion,
        frontier_values = node_values(m, b)[frontier],
        frontier_depths = node_depths(m)[frontier],
        term_reward = term_reward(m, b),
        best_vs_next = best_vs_next(m, b),
        min_depth = min_depth(m, b),
        tmp = zeros(T, length(frontier)),  # pre-allocate for use in selection_probability
    )
end

function action_dist!(p::Vector{T}, model::Heuristic{H,T}, φ::NamedTuple) where {H, T}
    term_select_action_dist!(p, model, φ)
end

# ---------- Define parameter ranges and special cases ---------- #

default_space(::Type{Heuristic{:Full}}) = Space(
    :β_best => (1e-6, 3),
    :β_depth => (-3, 3),
    :β_expand => 0.,

    :β_satisfice => (1e-6, 3),
    :β_best_next => (1e-6, 3),
    :β_depth_limit => (1e-6, 3),

    :θ_term => (-30, 30),
    :ε => (1e-3, 1)
)

default_space(::Type{Heuristic{:BestFirst}}) = 
    change_space(Heuristic{:Full}, β_depth=0)
default_space(::Type{Heuristic{:DepthFirst}}) = 
    change_space(Heuristic{:Full}, β_best=0, β_depth=(0, 3))
default_space(::Type{Heuristic{:BreadthFirst}}) = 
    change_space(Heuristic{:Full}, β_best=0, β_depth=(-3, 0))

default_space(::Type{Heuristic{:BestFirstExpand}}) = 
    change_space(Heuristic{:Full}, β_depth=0, β_expand=(1e-6, 50))
default_space(::Type{Heuristic{:DepthFirstExpand}}) = 
    change_space(Heuristic{:Full}, β_best=0, β_depth=(0, 3), β_expand=(1e-6, 50))
default_space(::Type{Heuristic{:BreadthFirstExpand}}) = 
    change_space(Heuristic{:Full}, β_best=0, β_depth=(-3, 0), β_expand=(1e-6, 50))

default_space(::Type{Heuristic{:BestPlusDepth}}) = 
    change_space(Heuristic{:Full}, β_depth=(0, 3))
default_space(::Type{Heuristic{:BestPlusBreadth}}) = 
    change_space(Heuristic{:Full}, β_depth=(-3, 0))

default_space(::Type{Heuristic{:BestFirstNoSatisfice}}) = 
    change_space(Heuristic{:BestFirst}, β_satisfice=0, θ_satisfice=0)
default_space(::Type{Heuristic{:BestFirstNoBestNext}}) = 
    change_space(Heuristic{:BestFirst}, β_best_next=0, θ_best_next=0)
default_space(::Type{Heuristic{:BestFirstNoDepthLimit}}) = 
    change_space(Heuristic{:BestFirst}, β_depth_limit=0, θ_depth_limit=0)

default_space(::Type{Heuristic{:BreadthFirstNoSatisfice}}) = 
    change_space(Heuristic{:BreadthFirst}, β_satisfice=0, θ_satisfice=0)
default_space(::Type{Heuristic{:BreadthFirstNoBestNext}}) = 
    change_space(Heuristic{:BreadthFirst}, β_best_next=0, θ_best_next=0)
default_space(::Type{Heuristic{:BreadthFirstNoDepthLimit}}) = 
    change_space(Heuristic{:BreadthFirst}, β_depth_limit=0, θ_depth_limit=0)

default_space(::Type{Heuristic{:DepthFirstNoSatisfice}}) = 
    change_space(Heuristic{:DepthFirst}, β_satisfice=0, θ_satisfice=0)
default_space(::Type{Heuristic{:DepthFirstNoBestNext}}) = 
    change_space(Heuristic{:DepthFirst}, β_best_next=0, θ_best_next=0)
default_space(::Type{Heuristic{:DepthFirstNoDepthLimit}}) = 
    change_space(Heuristic{:DepthFirst}, β_depth_limit=0, θ_depth_limit=0)

default_space(::Type{Heuristic{:FullNoSatisfice}}) = 
    change_space(Heuristic{:Full}, β_satisfice=0, θ_satisfice=0)
default_space(::Type{Heuristic{:FullNoBestNext}}) = 
    change_space(Heuristic{:Full}, β_best_next=0, θ_best_next=0)
default_space(::Type{Heuristic{:FullNoDepthLimit}}) = 
    change_space(Heuristic{:Full}, β_depth_limit=0, θ_depth_limit=0)

default_space(::Type{Heuristic{:BestFirstRandomStopping}}) = Space(
    :β_best => (1e-6, 3),
    :β_depth => 0.,

    :β_satisfice => 0.,
    :β_best_next => 0.,
    :β_depth_limit => 0.,

    :θ_term => (-30, 30),
    :ε => (1e-3, 1)
)

default_space(::Type{Heuristic{:BestFirstSatisficing}}) = 
    change_space(Heuristic{:BestFirstRandomStopping}, β_satisfice = (1e-6, 3)
)
default_space(::Type{Heuristic{:BestFirstBestNext}}) = 
    change_space(Heuristic{:BestFirstRandomStopping}, β_best_next = (1e-6, 3)
)
default_space(::Type{Heuristic{:BestFirstDepth}}) = 
    change_space(Heuristic{:BestFirstRandomStopping}, β_depth_limit = (1e-6, 3)
)

# ---------- Selection rule ---------- #

function selection_probability(model::Heuristic, φ::NamedTuple)
    p = φ.tmp  # use pre-allocated array for memory efficiency
    @. p = model.β_best * φ.frontier_values + model.β_depth * φ.frontier_depths + model.β_expand * φ.expansion
    softmax!(p)
end


# ---------- Stopping rule---------- #

function termination_probability(model::Heuristic, φ::NamedTuple)
    v = model.θ_term + 
        model.β_satisfice * φ.term_reward +
        model.β_best_next * φ.best_vs_next +
        model.β_depth_limit * 10φ.min_depth  # put min_depth on the roughly the same scale as the others
    # v = model.β_satisfice * (φ.term_reward - model.θ_satisfice) +
    #     model.β_best_next * (φ.best_vs_next - model.θ_best_next) +
    #     model.β_depth_limit * (φ.min_depth - model.θ_depth_limit)
    logistic(v)
end

"Minimum depth of a frontier node"
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
    undetermined = map(paths(m)) do path
        any(isnan(b[i]) for i in path)
    end
    # find best path, breaking ties in favor of undetermined
    best = argmax(collect(zip(pvals, undetermined)))
    
    competing_value = if undetermined[best]
        # best path is undetermined -> competing value is the second best path (undetermined or not)
        partialsort(pvals, 2; rev=true)
    else
        # best path is determined -> competing value is the best undetermined path
        vals = pvals[undetermined]
        if isempty(vals)
            return NaN  # Doesn't matter, you have to terminate.
        else
            maximum(vals)
        end
    end
    pvals[best] - competing_value
end


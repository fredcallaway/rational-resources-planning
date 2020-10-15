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
    # Pruning
    β_prune::T
    θ_prune::T
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
        tmp2 = zeros(T, length(frontier)),  # pre-allocate for use in selection_probability
        tmp3 = zeros(T, length(frontier)),  # pre-allocate for use in selection_probability
    )
end

function action_dist!(p::Vector{T}, model::Heuristic{H,T}, φ::NamedTuple) where {H, T}
    term_select_action_dist!(p, model, φ)
end

# ---------- Selection rule ---------- #

function pruning(model::Heuristic{H,T}, φ::NamedTuple)::Vector{T} where {H, T}
    @. logistic(model.β_prune * (model.θ_prune - φ.frontier_values))
end

function select_pref(model::Heuristic{H,T}, φ::NamedTuple)::Vector{T} where {H, T}
    h = φ.tmp  # use pre-allocated array for memory efficiency
    @. h = model.β_best * φ.frontier_values + model.β_depth * φ.frontier_depths + model.β_expand * φ.expansion
    h
end

@memoize function cartesian_bitvectors(N::Int)::Vector{BitVector}
    (map(collect, Iterators.product(repeat([BitVector([0, 1])], N)...)))[:]
end

function selection_probability(model::Heuristic{H, T}, φ::NamedTuple)::Vector{T} where {H, T}
    h = select_pref(model, φ)
    if model.θ_prune == -Inf
        return softmax!(h)
    else
        # this part is the bottleneck, so we add type annotations and use explicit loops
        total::Vector{T} = fill!(φ.tmp2, 0.)
        p::Vector{T} = φ.tmp3

        p_prune = pruning(model, φ)
        for prune in cartesian_bitvectors(length(p_prune))
            all(prune) && continue
            pp = 1.
            for i in eachindex(prune)
                if prune[i]
                    p[i] = -1e10
                    pp *= p_prune[i]
                else
                    p[i] = h[i]
                    pp *= (1. - p_prune[i])
                end
            end
            total .+= pp .* softmax!(p)
        end
        total ./= (eps() + (1. - prod(p_prune)))
        # @assert all(isfinite(p) for p in total)
        return total
    end

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
    p_term = logistic(v)
    if model.θ_prune > -Inf
        p_term += (1-p_term) * prod(pruning(model, φ))
    end
    p_term
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

# ---------- Define parameter ranges for individual models ---------- #

default_space(::Type{Heuristic{:Random}}) = Space(
    :β_best => 0.,
    :β_depth => 0.,
    :β_expand => 0,
    :β_satisfice => 0.,
    :β_best_next => 0.,
    :β_depth_limit => 0.,
    :θ_term => (-5, 5),
    :β_prune => 0,
    :θ_prune => -Inf,
    :ε => 0,
)

PARAMS = Dict(
    "Best" => (β_best=(0, 3),),
    "Depth" => (β_depth=(0, 3),),
    "Breadth" => (β_depth=(-3, 0),),
    "Satisfice" => (β_satisfice=(0, 3), θ_term=(-90, 90)),
    "BestNext" => (β_best_next=(0, 3), θ_term=(-90, 90)),
    "DepthLimit" => (β_depth_limit=(0, 3), θ_term=(-90, 90)),
    "Prune" => (β_prune=(0, 3), θ_prune=(-30, 30)),
    "Expand" => (β_expand=(0, 50),),
)


function default_space(::Type{Heuristic{M}}) where M
    x = string(M)
    components = Set()
    spec = split(x, "_")
    push!(components, popfirst!(spec))

    for ex in spec
        if ex == "Full"
            push!(components, "Satisfice", "BestNext", "DepthLimit", "Prune")
        elseif startswith(ex, "No")
            delete!(components, ex[3:end])
        else
            push!(components, ex)
        end
    end
    space = merge((PARAMS[k] for k in components)...)
    change_space(Heuristic{:Random}; ε=(1e-3, 1), space...)
end

default_space(::Type{Heuristic{:Exhaustive}}) = 
    change_space(Heuristic{:Random}, θ_term=-1e10)

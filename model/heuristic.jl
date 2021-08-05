using StatsFuns: logistic

# ---------- Base code for all heuristic models ---------- #
abstract type AbstractHeuristic{H,T} <: AbstractModel{T} end  # this was added for FancyHeuristic

struct Heuristic{H,T} <: AbstractHeuristic{H,T}
    # Selection rule weights
    β_best::T
    β_depth::T
    β_expand::T
    # Stopping rule weights
    β_satisfice::T
    β_best_next::T
    # Stopping rule threshold
    θ_term::T
    # Depth limits
    β_depthlim::T
    θ_depthlim::T
    # Pruning
    β_prune::T
    θ_prune::T
    # Lapse rate
    ε::T
end

name(::Type{<:Heuristic{H}}) where H = string(H)
name(::Type{<:Heuristic{H,T}}) where {H,T} = string(H)
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

function features(::Type{X}, m::MetaMDP, b::Belief) where X <: AbstractHeuristic{H,T} where {H, T}
    frontier = get_frontier(m, b)
    expansion = map(frontier) do c
        has_observed_parent(m.graph, b, c)
    end
    (
        frontier = frontier,
        expansion = expansion,
        frontier_values = node_values(m, b)[frontier],
        frontier_depths = node_depths(m)[frontier],
        satisfice = satisfice(X, m, b),
        best_next = best_next(X, m, b),
        tmp = zeros(T, length(frontier)),  # pre-allocate for use in selection_probability
        tmp2 = zeros(T, length(frontier)),  # pre-allocate for use in selection_probability
        tmp3 = zeros(T, length(frontier)),  # pre-allocate for use in selection_probability
    )
end

function action_dist!(p::Vector{T}, model::AbstractHeuristic{H,T}, φ::NamedTuple) where {H, T}
    term_select_action_dist!(p, model, φ)
end

# ---------- Selection rule ---------- #

function pruning(model::AbstractHeuristic{H,T}, φ::NamedTuple)::Vector{T} where {H, T}
    @. begin  # weird syntax prevents unnecessary memory allocation
        1 - 
        # probability of NOT pruning
        logistic(model.β_prune * (φ.frontier_values - model.θ_prune)) * 
        logistic(model.β_depthlim * (model.θ_depthlim - φ.frontier_depths))
    end
end

@inline pruning_active(model) = (model.β_prune != 1e5) || (model.β_depthlim != 1e5)

function select_pref(model::AbstractHeuristic{H,T}, φ::NamedTuple)::Vector{T} where {H, T}
    h = φ.tmp  # use pre-allocated array for memory efficiency
    @. h = model.β_best * φ.frontier_values + model.β_depth * φ.frontier_depths + model.β_expand * φ.expansion
    h
end

@memoize function cartesian_bitvectors(N::Int)::Vector{BitVector}
    (map(collect, Iterators.product(repeat([BitVector([0, 1])], N)...)))[:]
end

function selection_probability(model::AbstractHeuristic{H, T}, φ::NamedTuple)::Vector{T} where {H, T}
    h = select_pref(model, φ)
    if !pruning_active(model)
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

function termination_probability(model::Heuristic{H,T}, φ::NamedTuple)::T where {H,T}
    v = model.θ_term + 
        model.β_satisfice * φ.satisfice +
        model.β_best_next * φ.best_next
    p_term = logistic(v)
    if pruning_active(model)
        p_term += (1-p_term) * prod(pruning(model, φ))
    end
    p_term
end

"How much better is the best path from its competitors?"
function best_vs_next_value(m, b)
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

function satisfice(H::Type{<:Heuristic}, m::MetaMDP, b::Belief)
    has_component(H, "Satisfice") ? term_reward(m, b) : 0.
end

function best_next(H::Type{<:Heuristic}, m::MetaMDP, b::Belief)
    has_component(H, "BestNext") ? best_vs_next_value(m, b) : 0.
end

# ---------- Define parameter ranges for individual models ---------- #

default_space(::Type{Heuristic{:Random}}) = Space(
    :β_best => 0.,
    :β_depth => 0.,
    :β_expand => 0,
    :β_satisfice => 0.,
    :β_best_next => 0.,
    :θ_term => (-10, 10),
    :β_depthlim => 1e5,  # flag for inactive
    :θ_depthlim => 1e10,  # Inf breaks gradient
    :β_prune => 1e5,
    :θ_prune => -1e10,
    :ε => 0,
)

default_space(::Type{Heuristic{:Exhaustive}}) = 
    change_space(Heuristic{:Random}, θ_term=-1e10)

function default_space(::Type{Heuristic{H}}) where H
    ranges = Dict(
        "Best" => (β_best=(0, 3),),
        "Depth" => (β_depth=(0, 3),),
        "Breadth" => (β_depth=(-3, 0),),
        "Satisfice" => (β_satisfice=(0, 3), θ_term=(-90, 90)),
        "BestNext" => (β_best_next=(0, 3), θ_term=(-90, 90)),
        "DepthLimit" => (β_depthlim=(0, 30), θ_depthlim=(0, 5)),
        "Prune" => (β_prune=(0, 3), θ_prune=(-30, 30)),
        "Expand" => (β_expand=(0, 50),),
    )
    x = string(H)
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
    space = merge((ranges[k] for k in components)...)
    change_space(Heuristic{:Random}; ε=(1e-3, 1), space...)
end

function all_heuristic_models(base = ["Best", "Depth", "Breadth"])
    whistles = ["Satisfice", "BestNext"]
    if EXPAND_ONLY
        push!(whistles, "DepthLimit", "Prune")
    else
        push!(whistles, "Expand")
    end
    
    map(Iterators.product(base, powerset(whistles))) do (b, ws)
        spec = Symbol(join([b, ws...], "_"))
        Heuristic{spec}
    end[:]
end

function has_component(::Type{<:AbstractHeuristic{H}}, ex) where H
    sh = string(H)
    occursin(ex, sh) || occursin("Full", sh)
end
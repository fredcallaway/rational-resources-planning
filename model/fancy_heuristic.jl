"Heuristic models with stopping rules based on probability rather than values"

using StatsFuns: logistic

# ---------- Base code for fancy heuristic models ---------- #

struct FancyHeuristic{H,T} <: AbstractHeuristic{H,T}
    # Selection rule weights
    β_best::T
    β_depth::T
    β_expand::T
    # Stopping rule weights
    β_satisfice::T
    β_best_next::T
    # Stopping rule thresholds
    θ_satisfice::T
    α_term::T
    # Depth limits
    β_depthlim::T
    θ_depthlim::T
    # Pruning
    β_prune::T
    θ_prune::T
    # Lapse rate
    ε::T
end

name(::Type{<:FancyHeuristic{H}}) where H = "Fancy_" * string(H)
name(::Type{<:FancyHeuristic{H,T}}) where {H,T} = "Fancy_" * string(H)
name(::FancyHeuristic{H,T}) where {H,T} = "Fancy_" * string(H)

Base.show(io::IO, model::FancyHeuristic{H,T}) where {H, T} = print(io, "FancyHeuristic{:$H}(...)")

function Base.display(model::FancyHeuristic{H,T}) where {H, T}
    println("--------- FancyHeuristic{:$H} ---------")
    space = default_space(FancyHeuristic{H})
    for k in fieldnames(FancyHeuristic)
        print("  ", lpad(k, 14), " = ")
        if length(space[k]) == 1
            println("(", space[k], ")")
        else
            println(round(getfield(model, k); sigdigits=3))
        end
            
    end
end

function termination_probability(model::FancyHeuristic{H,T}, φ::NamedTuple)::T where {H,T}
    bpvd = φ.satisfice  # output of best_path_value_dist

    if bpvd isa Int
        p_worse = 1
    else
        # linear interpolation between multiplies of 5 to make the objective smooth
        lo = 5 * fld(model.θ_satisfice, 5); hi = 5 * cld(model.θ_satisfice, 5)
        hi_weight = (model.θ_satisfice - lo) / 5.  
        p_worse_lo = cdf(bpvd, lo-1e-3); p_worse_hi = cdf(bpvd, hi-1e-3)
        p_worse = p_worse_lo * (1 - hi_weight) + p_worse_hi * hi_weight
    end

    v = model.α_term + 
        model.β_satisfice * (1 - p_worse)
        model.β_best_next * φ.best_next
    p_term = logistic(v)
    if pruning_active(model)
        p_term += (1-p_term) * prod(pruning(model, φ))
    end
    p_term
end

function satisfice(H::Type{<:FancyHeuristic}, m::MetaMDP, b::Belief)
    has_component(H, "Satisfice") ? best_path_value_dist(m, b) : 0
end

function best_next(H::Type{<:FancyHeuristic}, m::MetaMDP, b::Belief)
    has_component(H, "BestNext") ? prob_best_maximal(m, b) : 0
end


"How likely is the best path actually the best"
function prob_best_maximal(m, b)
    pvals = path_values(m, b)
    n_known = map(paths(m)) do path
        !any(isnan(b[i]) for i in path)
    end
    # find best path, breaking ties in favor of less uncertain
    # (this is conservative because it gives more opportunity for other paths to be better)
    best = argmax(collect(zip(pvals, n_known)))
    pth = paths(m)[best]

    rewards = [-10, -5, 5, 10]
    @assert EXPERIMENT == "exp1"

    marginalizing = filter(i->!observed(b, i), pth)
    b1 = copy(b)
    mapreduce(+, Iterators.product(fill(rewards, length(marginalizing))...)) do z
        b1[marginalizing] .= z
        own_value = path_value(m, b1, pth)
        competing_value = best_path_value_dist(m, b1)
        cdf(competing_value, own_value)
    end * (.25 ^ length(marginalizing))
end

function belief_tree(m, b)
    function rec(i)
        (observed(b, i) ? b[i] : m.rewards[i], 
         Tuple(rec(child) for child in m.graph[i]))
    end
    rec(1)
end

function tree_value_dist(btree)
    self, children = btree
    isempty(children) && return self # base case
    self + maximum(map(tree_value_dist, children))
end

function best_path_value_dist(m, b)
    v = tree_value_dist(belief_tree(m, b))
    DNP([0.], [1.]) + v  # ensure it's a DNP
end


# ---------- Define parameter ranges for individual models ---------- #


function default_space(::Type{FancyHeuristic{H}}) where H
    ranges = Dict(
        "Best" => (β_best=(0, 3),),
        "Depth" => (β_depth=(0, 3),),
        "Breadth" => (β_depth=(-3, 0),),
        "Satisfice" => (β_satisfice=(0, 30), θ_satisfice=(-30,30)),
        "BestNext" => (β_best_next=(0, 30),),
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
    space = Space(
        :β_best => 0.,
        :β_depth => 0.,
        :β_expand => 0,
        :β_satisfice => 0.,
        :β_best_next => 0.,
        :θ_satisfice => 0.,
        :α_term => (-30, 30),
        :β_depthlim => 1e5,  # flag for inactive
        :θ_depthlim => 1e10,  # Inf breaks gradient
        :β_prune => 1e5,
        :θ_prune => -1e10,
        :ε => (1e-3, 1),
    )
    for (k, v) in pairs(merge((ranges[k] for k in components)...))
        @assert k in keys(space)
        space[k] = v
    end
    space
end


function all_fancy_heuristic_models()
    base = ["Best", "Depth", "Breadth"]
    whistles = ["Satisfice", "BestNext"]
    if EXPAND_ONLY
        push!(whistles, "DepthLimit", "Prune")
    else
        push!(whistles, "Expand")
    end
    
    models = map(Iterators.product(base, powerset(whistles))) do (b, ws)
        "Satisfice" in ws || "BestNext" in ws || return missing
        spec = Symbol(join([b, ws...], "_"))
        FancyHeuristic{spec}
    end[:] |> skipmissing |> collect
end
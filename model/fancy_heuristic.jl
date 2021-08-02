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
    p_better = φ.satisfice isa Int ? 0 : prob_better(φ.satisfice, model.θ_satisfice)
    v = model.α_term + 
        model.β_satisfice * p_better +
        model.β_best_next * φ.best_next

    p_term = logistic(v)
    if pruning_active(model)
        p_term += (1-p_term) * prod(pruning(model, φ))
    end
    p_term
end

# ---------- Fancy satisficing ---------- #

function satisfice(H::Type{<:FancyHeuristic}, m::MetaMDP, b::Belief)
    has_component(H, "Satisfice") ? best_paths_value_dists(m, b) : 0
end

"How likely is the best path to have value ≥ θ (maximize over ties)"
function prob_better(best_dists, θ)
    # linear interpolation between multiplies of 5 to make the objective smooth
    lo = 5 * fld(θ, 5); hi = 5 * cld(θ, 5)
    hi_weight = (θ - lo) / 5.; lo_weight = 1 - hi_weight
    maximum(best_dists) do bpvd
        p_worse_lo = cdf(bpvd, lo-1e-3); p_worse_hi = cdf(bpvd, hi-1e-3)
        p_worse = p_worse_lo * lo_weight + p_worse_hi * hi_weight
        1 - p_worse
    end
end

"Distributions of value of all paths with maximal expected value"
function best_paths_value_dists(m, b)
    rewards = [-10, -5, 5, 10]
    @assert EXPERIMENT == "exp1"
    pvals = path_values(m, b)
    max_pval = maximum(pvals)

    map(paths(m)[pvals .== max_pval]) do pth
        path_value_dist(m, b, pth)
    end |> unique
end

# ---------- Fancy best vs next ---------- #

function best_next(H::Type{<:FancyHeuristic}, m::MetaMDP, b::Belief)
    has_component(H, "BestNext") ? prob_best_maximal(m, b) : 0
end

"How likely is the best path actually the best (maximize over ties)"
function prob_best_maximal(m, b)
    rewards = [-10, -5, 5, 10]
    @assert EXPERIMENT == "exp1"
    pvals = path_values(m, b)
    max_pval = maximum(pvals)
    # if multiple best paths, take the maximum probability of any of them
    maximum(zip(paths(m), pvals)) do (pth, val)
        val != max_pval && return -Inf
        unobs = filter(i->!observed(b, i), pth)
        b1 = copy(b)
        possible_unobs_vals = Iterators.product(fill(rewards, length(unobs))...)
        sum(possible_unobs_vals) do z
            b1[unobs] .= z
            own_value = sum(b1[pth])  # same as path_value(m, b1, pth) because pth is fully observed
            cdf(max_value_dist(m, b1), own_value)
        end * (.25 ^ length(unobs))
    end
end

"Distribution of value of the best path if you knew all the rewards"
function max_value_dist(m, b)
    v = tree_value_dist(belief_tree(m, b))
    DNP([0.], [1.]) + v  # ensure it's a DNP
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

# ---------- Define parameter ranges for individual models ---------- #


function default_space(::Type{FancyHeuristic{H}}) where H
    ranges = Dict(
        "Best" => (β_best=(0, 3),),
        "Depth" => (β_depth=(0, 3),),
        "Breadth" => (β_depth=(-3, 0),),
        "Satisfice" => (β_satisfice=(0, 100), θ_satisfice=(-100,100)),
        "BestNext" => (β_best_next=(0, 100),),
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
        :α_term => (-1000, 1000),
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
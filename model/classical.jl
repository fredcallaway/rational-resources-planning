using StatsFuns: logistic

# ---------- Base code for all models ---------- #

abstract type ClassicalSearchModel{T} <: AbstractModel{T} end

default_space(::Type{M}) where M <: ClassicalSearchModel = Space(
    :β_term => (0, 3),
    :β_click => (0, 3),
    :θ_term => (0, 30),
    :ε => (0, 1)
)

function action_dist!(p::Vector{T}, model::M, phi::NamedTuple) where M <: ClassicalSearchModel{T} where T
    p .= 0.
    if length(phi.click_options) == 0
        p[1] = 1.
        return p
    end

    ε = model.ε
    p_rand = ε / (1+length(phi.click_options))

    p_term = logistic(model.β_term * (phi.best_lead - model.θ_term))
    p[1] = p_rand + (1-ε) * p_term

    p_clicks = phi.tmp  # use pre-allocated array
    p_clicks .= model.β_click .* phi.click_values
    softmax!(p_clicks)
    for i in eachindex(p_clicks)
        c = phi.click_options[i] + 1
        p[c] = p_rand + (1-ε) * (1-p_term) * p_clicks[i]
    end
    p
end

function action_dist(model::M, m::MetaMDP, b::Belief) where M <: ClassicalSearchModel{T} where T
    phi = features(M, m, b)
    p = zeros(T, length(b) + 1)
    action_dist!(p, model, phi)
end

function features(::Type{M}, m::MetaMDP, b::Belief) where M <: ClassicalSearchModel{T} where T
    np = node_preference(M, m, b)  # this function is defined for each model (best/depth/breadth)
    possible = allowed(m, b)[2:end]
    click_values = np[possible]
    (
        best_lead = best_lead(m, b),
        click_options = findall(possible),
        click_values = click_values,
        tmp = zeros(T, length(click_values)),
    )
end

"How much better is the best path from its competitors?"
function best_lead(m, b)
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

function logp(L::Likelihood, model::M)::T where M <: ClassicalSearchModel{T} where T
    phi = memo_map(L) do d
        features(M, d)
    end

    tmp = zeros(T, n_action(L))
    total = zero(T)
    for i in eachindex(L.data)
        a = L.data[i].c + 1
        p = action_dist!(tmp, model, phi[i])
        @assert sum(p) ≈ 1
        total += log(p[a])
    end
    total
end

# ---------- Specific search orders ---------- #


struct BestFirst{T} <: ClassicalSearchModel{T}
    β_term::T
    β_click::T
    θ_term::T
    ε::T
end

"What is the maximal expected path value for each node?"
function node_preference(::Type{BestFirst{T}}, m::MetaMDP, b::Belief) where T
    nv = fill(-Inf, length(m))
    for p in paths(m)
        v = path_value(m, b, p)
        for i in p
            nv[i] = max(nv[i], v)
        end
    end
    nv
end


struct DepthFirst{T} <: ClassicalSearchModel{T}
    β_term::T
    β_click::T
    θ_term::T
    ε::T
end

function node_depths(g::Graph)
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

function node_preference(::Type{DepthFirst{T}}, m::MetaMDP, b::Belief) where T
    node_depths(m.graph) .* 10  # multiply by 10 so that β_click = 3 gives hard maximization
end


struct BreadthFirst{T} <: ClassicalSearchModel{T}
    β_term::T
    β_click::T
    θ_term::T
    ε::T
end

function node_preference(::Type{BreadthFirst{T}}, m::MetaMDP, b::Belief) where T
    node_depths(m.graph) .* -10
end


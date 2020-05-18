using StatsFuns: logistic

struct NewBestFirst{T} <: AbstractModel{T}
    β_term::T
    β_click::T
    θ_term::T
    ε::T
end

default_space(::Type{NewBestFirst}) = Space(
    :β_term => (0, 3),
    :β_click => (0, 3),
    :θ_term => (0, 30),
    :ε => (0, 1)
)


function action_dist!(p::Vector{T}, model::NewBestFirst{T}, phi::NamedTuple) where T
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

function action_dist(model::M, m::MetaMDP, b::Belief) where M <: NewBestFirst{T} where T <: Real
    phi = features(M, m, b)
    p = zeros(T, length(b) + 1)
    action_dist!(p, model, phi)
end

function features(::Type{NewBestFirst{T}}, m::MetaMDP, b::Belief) where T
    nv = node_values(m, b)
    possible = allowed(m, b)[2:end]
    click_values = nv[possible]
    (
        best_lead = best_lead(m, b),
        click_options = findall(possible),
        click_values = click_values,
        tmp = zeros(T, length(click_values)),
    )
end

"What is the maximal expected path value for each node?"
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

"How much better is the best path from its competitors?"
function best_lead(m, b)
    pvals = path_values(m, b)
    undetermined = [isnan(b[path[end]]) for path in paths(m)]
    # find best path, breaking ties in favor of undetermined
    best = argmax(collect(zip(pvals, undetermined)))
    # either the best or the competitor must be undetermind
    competitors = undetermined[best] ? pvals : pvals[undetermined]
    competing_value = partialsort(competitors, 2, rev=true)
    pvals[best] - competing_value
end


function logp(L::Likelihood, model::M)::T where M <: NewBestFirst{T} where T <: Real
    phi = memo_map(L) do d
        features(M, d)
    end

    tmp = zeros(T, n_action(L))
    total = zero(T)
    for i in eachindex(L.data)
        a = L.data[i].c + 1
        p = action_dist!(tmp, model, phi[i])[a]
        total += log(p)
    end
    total
end

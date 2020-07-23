abstract type AbstractModel{T} end

function create_model(::Type{M}, x::Vector{T}, z, space::Space) where M where T
    xs = Iterators.Stateful(x)
    zs = Iterators.Stateful(z)
    args = map(fieldnames(M)) do fn
        spec = space[fn]
        if spec isa Tuple
            first(xs)
        elseif spec isa Vector
            first(zs)
        else
            T(spec)
        end
    end
    M{T}(args...)
end

# function preference(model::M, m::MetaMDP, b::Belief, c::Int) where M <: AbstractModel
#     error("preference")
#     phi = features(M, m, b)
#     preference(model, phi, c)
# end

# function preferences(model::M, m::MetaMDP, b::Belief) where M <: AbstractModel
#     error("preference")
#     phi = features(M, m, b)
#     map(0:length(b)) do c
#         preference(model, phi, c)
#     end
# end

# preference(model::AbstractModel, d::Datum) = preference(model, d.t.m, d.b, d.c)


"Computes features that don't depend on model parameters."
function features(::Type{M}, m::MetaMDP, b::Belief) where M <: AbstractModel
    error("Not implemented")
end

"Computes action probabilities."
function action_dist!(p::Vector{T}, model::AbstractModel{T}, φ) where T
    error("Not implemented")
end

"Simple one-step softmax model."
function softmax_action_dist!(p::Vector{T}, model::AbstractModel{T}, φ::NamedTuple) where T
    p .= NOT_ALLOWED
    for c in φ.options
        p[c+1] = preference(model, φ, c)
    end
    softmax!(p)
    
    ε = model.ε
    p .*= (1 - ε)
    p_rand = ε / length(φ.options)
    for i in φ.options
        p[i+1] += p_rand
    end
    p
end

"Two-step terminate-then-select model."
function term_select_action_dist!(p::Vector{T}, model::AbstractModel{T}, φ) where T
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

# This is used for simulation and can be overridden if necessary
function action_dist(model::M, m::MetaMDP, b::Belief) where M <: AbstractModel
    φ = features(M, m, b)
    p = zeros(T, length(b) + 1)
    action_dist!(p, model, φ)
end

# Convenience
features(::Type{M}, d::Datum) where M <: AbstractModel = features(M, d.t.m, d.b)
action_dist(model::AbstractModel, d::Datum) = action_dist(model, d.t.m, d.b)


function get_frontier(m, b::Belief)
    findall(1:length(b)) do i
        allowed(m, b, i)
    end
end
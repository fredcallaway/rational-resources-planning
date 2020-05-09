struct BestFirst{T} <: AbstractModel{T}
    β_node_value::T
    β_prune::T
    β_satisfice::T
    threshold_prune::T
    threshold_satisfice::T
    ε::T
end

default_space(::Type{BestFirst}) = Space(
    :β_node_value => (0, 1000),
    :β_prune => (0, 1000),
    :β_satisfice => (0, 1000),
    :threshold_prune => (-30, 30),
    :threshold_satisfice => (-30, 30),
    :ε => (0, 1)
)

function preference(model::BestFirst{T}, phi::NamedTuple, c::Int)::T where T
    c == TERM && return model.β_satisfice * (phi.term_reward - model.threshold_satisfice)
    model.β_prune * min(0, phi.node_values[c] - model.threshold_prune) + 
    model.β_node_value * phi.node_values[c]
end

function features(::Type{BestFirst{T}}, m::MetaMDP, b::Belief) where T
    (node_values=node_values(m, b), term_reward=term_reward(m, b))
end

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

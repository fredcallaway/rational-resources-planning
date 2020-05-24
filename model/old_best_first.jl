# This specifies the parameters. Ignore (but replicate) the mysterious typing syntax.
struct OldBestFirst{T} <: AbstractModel{T}
    β_node_value::T
    β_prune::T
    β_satisfice::T
    threshold_prune::T
    threshold_satisfice::T
    ε::T
end

# This specifies lower and upper bounds on each parameter.
# I don't think the order matter(, but you should have
# it match the struct definition to be safe.
default_space(::Type{OldBestFirst}) = Space(
    :β_node_value => (0, 1000),
    :β_prune => (0, 1000),
    :β_satisfice => (0, 1000),
    :threshold_prune => (-30, 30),
    :threshold_satisfice => (-30, 30),
    :ε => (0, 1)
)

# Compute the score for clicking node `c` given the features in `phi`.
# `phi` is computed by `features` (the next function).
function preference(model::OldBestFirst{T}, phi::NamedTuple, c::Int)::T where T
    c == TERM && return model.β_satisfice * (phi.term_reward - model.threshold_satisfice)
    model.β_prune * min(0, phi.node_values[c] - model.threshold_prune) + 
    model.β_node_value * phi.node_values[c]
end

# Computes features for a given belief state and a given metalevel MDP.
# Note that the metalevel MDP has NaN cost!
function features(::Type{OldBestFirst{T}}, m::MetaMDP, b::Belief) where T
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

struct MetaGreedy{T} <: AbstractModel{T}
    cost::T  # note: equivalent to θ_term
    β_select::T
    β_term::T
    ε::T
end

function voi1(m, b, c)
    mapreduce(+, results(m, b, c)) do (p, b, r)
        p * term_reward(m, b)
    end - term_reward(m, b)
end


default_space(::Type{MetaGreedy}) = Space(
    :cost => COSTS,
    :β_select => (1e-6, 50),
    :β_term => (1e-6, 50),
    :ε => (1e-3, 1)
)

function features(::Type{MetaGreedy{T}}, m::MetaMDP, b::Belief) where T
    frontier = get_frontier(m, b)
    voi = map(frontier) do c
        voi1(m, b, c)
    end
    (
        frontier=frontier,
        q_select=voi,
        q_term=[0.; voi],
        tmp_select=zeros(T, length(frontier)),
        tmp_term=zeros(T, length(frontier)+1),
    )
end
function action_dist!(p::Vector{T}, model::MetaGreedy{T}, φ::NamedTuple) where T
    term_select_action_dist!(p, model, φ)
end

function selection_probability(model::MetaGreedy, φ::NamedTuple)
    q = φ.tmp_select
    q .= model.β_select .* φ.q_select
    softmax!(q)
end

function termination_probability(model::MetaGreedy, φ::NamedTuple)
    q = φ.tmp_term
    q .= φ.q_term
    q[1] = model.cost  # equivalent to substracting from all the computations
    q .*= model.β_term
    softmax!(q, 1)
end

# For simulation

# function action_dist(model::MetaGreedy, m::MetaMDP, b::Belief)
#     p = zeros(length(b) + 1)
#     φ = _optplus_features(Float64, get_qs(m, b, model.cost), get_frontier(m, b))
#    action_dist!(p, model, φ)
# end
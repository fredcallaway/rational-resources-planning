
# %% ==================== MetaGreedy ====================

mutable struct MetaGreedy <: Preference
    cost::Float64
end

parameters(pref::MetaGreedy) = [
    :cost => Set(COSTS),
]

function apply(pref::MetaGreedy, t::Trial, b::Belief)
    m = MetaMDP(t, pref.cost)
    voc1(m, b)
end

function voc1(m::MetaMDP, b::Belief, c::Int)
    c == TERM && return 0.
    !allowed(m, b, c) && return -Inf
    q = mapreduce(+, results(m, b, c)) do (p, b1, r)
        p * (term_reward(m, b1) + r)
    end
    q - term_reward(m, b)
end

voc1(m, b) = [voc1(m, b, c) for c in 0:length(b)]


# %% ==================== Optimal ====================

mutable struct Optimal <: Preference
    cost::Float64
end

parameters(pref::Optimal) = [
    :cost => Set(COSTS),
]

@memoize function get_V_tbl()
    mdp_ids = readdir("$base_path/mdps/")
    V_tbl = asyncmap(mdp_ids) do i
        V = deserialize("$base_path/mdps/$i/V")
        (identify(V.m), V.m.cost) => V
    end |> Dict
end

function get_V(t::Trial, cost)
    tbl = get_V_tbl()
    tbl[identify(t), cost]
end

function apply(pref::Optimal, t::Trial, b::Belief)
    V = get_V(t, pref.cost)
    Q(V, b)
end


# %% ==================== Best First Search ====================

struct BestFirst <: Preference
end

function apply(pref::BestFirst, t::Trial, b::Belief)
    m = MetaMDP(t, NaN)
    [0; node_values(m, b)]
end


# %% ==================== Satisficing ====================

mutable struct Satisficing <: Preference
    threshold::Real
end

parameters(pref::Satisficing) = [
    :threshold => (-30., 30.)
]

function apply(pref::Satisficing, t::Trial, b::Belief)
    m = MetaMDP(t, NaN)
    [term_reward(m, b) - pref.threshold; zeros(length(b))]
end


# %% ==================== Pruning ====================

mutable struct Pruning <: Preference
    threshold::Real
end

parameters(pref::Pruning) = [
    :threshold => (-30., 30.)
]

function apply(pref::Pruning, t::Trial, b::Belief)
    m = MetaMDP(t, NaN)
    nv = node_values(m, b)
    map(0:length(b)) do c
        c == TERM && return 0.
        min(0, nv[c] - pref.threshold)
        # term_reward(m, b) - pref.threshold
    end
end


# %% ==================== Expansion ====================

struct Expansion <: Preference end


function apply(pref::Expansion, t::Trial, b::Belief)
    m = MetaMDP(t, NaN)
    map(0:length(b)) do c
        c == TERM && return 0.
        !allowed(m, b, c) && return -Inf
        float(has_observed_parent(t.graph, b, c))
    end
end


# # %% ==================== FollowLast ====================

# struct FollowLast <: Preference end

# function preferences(model::FollowLast, t::Trial, b::Belief)
#     m = MetaMDP(t, NaN)
#     map(0:length(b)) do c
#         c == TERM && return 0.
#         !allowed(m, b, c) && return -Inf
#         d.c_last == nothing && return 0.
#         float(c in d.t.graph[d.c_last])
#     end
# end

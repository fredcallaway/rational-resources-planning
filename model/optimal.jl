NO_β_TERM = -1  # flag

struct Optimal{T} <: AbstractModel{T}
    cost::Float64
    β::T
    # β_term::T
    ε::T
end


default_space(::Type{Optimal}) = Space(
    :cost => COSTS,
    :β => (1e-6, 50),
    # :β_expansion => # TODO
    :ε => (1e-3, 1)
)

# standard method, causes memory overflow with parallel processing
# because every worker needs to load every value function
function features(::Type{Optimal{T}}, m::MetaMDP, b::Belief) where T
    error("This method shouldn't run")
    map(COSTS) do cost
        cost => get_q(mutate(m, cost=cost), b)
    end |> Dict
end

function action_dist!(p::Vector{T}, model::Optimal{T}, φ::NamedTuple) where T
    softmax_action_dist!(p, model, φ)
end

function preference(model::Optimal{T}, phi::NamedTuple, c::Int)::T where T
    model.β * phi.q[model.cost][c+1]
end

# for likelihood: use precomputed look up table for just the
# beliefs found in the experiment

if @isdefined(base_path) && isfile("$base_path/Q_table")
    const Q_TABLE = deserialize("$base_path/Q_table")
else
    myid() == 1 && @warn "Q_table not found. Can't fit Optimal model"
end
features(::Type{Optimal{T}}, d::Datum) where T = (
    options = [0; get_frontier(d.t.m, d.b)],
    q = Q_TABLE[hash(d)],
)

# for simulation: just get the q value you need.
# parallelization code must divide work such that each
# worker only loads a few MDPs
function preferences(model::Optimal{T}, m::MetaMDP, b::Belief) where T
    model.β .* get_q(mutate(m, cost=model.cost), b)
end


# @memoize function load_V(mid::String)
#     println("Loading value function: $mid"); flush(stdout)
#     deserialize("$base_path/V/$mid")
# end

function get_q(m::MetaMDP, b::Belief)
    mid = string(hash(m); base=62)
    V = load_V(mid)
    Q(V, b)
end

# Use the Q table if you can
function action_dist(model::Optimal{T}, d::Datum) where T
    m = d.t.m
    possible = allowed(m, d.b)
    q = features(Optimal{Float64}, d)[model.cost]
    h = model.β * q
    p_rand = model.ε .* rand_prob(m, d.b) .* possible
    p_rand .+ (1-model.ε) .* softmax(h)
end



# ---------- Parameter ranges ---------- #

# default_space(::Type{Optimal{:Full}}) = Space(
#     :cost => collect(COSTS),
#     :β => (1e-6, 50),
#     :β_term => (1e-6, 50),
#     # :β_expansion => # TODO
#     :ε => (1e-3, 1)
# )

# function _modify(;kws...)
#     space = default_space(Optimal{:Full})
#     for (k,v) in kws
#         space[k] = v
#     end
#     space
# end

# default_space(::Type{Optimal{:Classic}}) = _modify(β_term = NO_β_TERM)
# default_space(::Type{Optimal{:Term}}) = _modify()


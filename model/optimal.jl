NO_β_TERM = -1  # flag

struct NewOptimal{H,T} <: AbstractModel{T}
    cost::Float64
    β::T
    β_term::T
    ε::T
end

name(::Type{Heuristic{H}}) where H = "NewOptimal$H"
name(::Type{Heuristic{H,T}}) where {H,T} = "NewOptimal$H"

function preference(model::NewOptimal{T}, phi, c)::T where T
    model.β * phi[model.cost][c+1]
end

# standard method, causes memory overflow with parallel processing
# because every worker needs to load every value function
function features(::Type{NewOptimal{T}}, m::MetaMDP, b::Belief) where T
    error("This method shouldn't run")
    map(COSTS) do cost
        cost => get_q(mutate(m, cost=cost), b)
    end |> Dict
end


# for likelihood: use precomputed look up table for just the
# beliefs found in the experiment

if @isdefined(base_path) && isfile("$base_path/Q_table")
    const Q_TABLE = deserialize("$base_path/Q_table")
else
    myid() == 1 && @warn "Q_table not found. Can't fit NewOptimal model"
end
features(::Type{NewOptimal{T}}, d::Datum) where T = Q_TABLE[hash(d)]

# for simulation: just get the q value you need.
# parallelization code must divide work such that each
# worker only loads a few MDPs
function preferences(model::NewOptimal{T}, m::MetaMDP, b::Belief) where T
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
function action_dist(model::NewOptimal{T}, d::Datum) where T
    m = d.t.m
    possible = allowed(m, d.b)
    q = features(NewOptimal{Float64}, d)[model.cost]
    h = model.β * q
    p_rand = model.ε .* rand_prob(m, d.b) .* possible
    p_rand .+ (1-model.ε) .* softmax(h)
end

# ---------- Parameter ranges ---------- #

default_space(::Type{NewOptimal{:Full}}) = Space(
    :cost => collect(COSTS),
    :β => (1e-6, 50),
    :β_term => (1e-6, 50),
    # :β_expansion => # TODO
    :ε => (1e-3, 1)
)

function _modify(;kws...)
    space = default_space(NewOptimal{:Full})
    for (k,v) in kws
        space[k] = v
    end
    space
end

default_space(::Type{NewOptimal{:Classic}}) = _modify(β_term = NO_β_TERM)
default_space(::Type{NewOptimal{:Term}}) = _modify()


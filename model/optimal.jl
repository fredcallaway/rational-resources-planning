struct Optimal{T} <: AbstractModel{T}
    cost::Float64
    β::T
    ε::T
end

default_space(::Type{Optimal}) = Space(
    :cost => collect(COSTS),
    :β => (0, 1000),
    :ε => (0, 1)
)


function preference(model::Optimal{T}, phi, c)::T where T
    model.β * phi[model.cost][c+1]
end

# standard method, causes memory overflow with parallel processing
# because every worker needs to load every value function
function features(::Type{Optimal{T}}, m::MetaMDP, b::Belief) where T
    error("This method shouldn't run")
    map(COSTS) do cost
        cost => get_q(mutate(m, cost=cost), b)
    end |> Dict
end


# for likelihood: use precomputed look up table for just the
# beliefs found in the experiment
if isfile("$base_path/Q_table")
    const Q_TABLE = deserialize("$base_path/Q_table")
else
    @warn "No file $base_path/Q_table. Can't fit Optimal model"
end
features(::Type{Optimal{T}}, d::Datum) where T = Q_TABLE[q_key(d)]


# for simulation: just get the q value you need.
# parallelization code must divide work such that each
# worker only loads a few MDPs
function preferences(model::Optimal{T}, m::MetaMDP, b::Belief) where T
    model.β .* get_q(mutate(m, cost=model.cost), b)
end


@memoize function load_V(mid::String)
    println("Loading value function: $mid")
    deserialize("$base_path/V/$mid")
end

function get_q(m::MetaMDP, b::Belief)
    V = load_V(string(hash(m)))
    Q(V, b)
end


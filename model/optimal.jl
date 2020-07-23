
if @isdefined(base_path) && isfile("$base_path/Q_table")
    const Q_TABLE = deserialize("$base_path/Q_table")
else
    myid() == 1 && @warn "Q_table not found. Can't fit Optimal model"
end

struct Optimal{T} <: AbstractModel{T}
    cost::Float64
    β::T
    ε::T
end

default_space(::Type{M}) where M <: Optimal = Space(
    :cost => COSTS,
    :β => (1e-6, 50),
    # :β_expansion => # TODO
    :ε => (1e-3, 1)
)


features(::Type{M}, d::Datum) where M <: Optimal = (
    options = [0; get_frontier(d.t.m, d.b)],
    q = Q_TABLE[hash(d)],
)

function action_dist!(p::Vector{T}, model::Optimal{T}, φ::NamedTuple) where T
    softmax_action_dist!(p, model, φ)
end

function preference(model::Optimal{T}, phi::NamedTuple, c::Int)::T where T
    model.β * phi.q[model.cost][c+1]
end

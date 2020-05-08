using DataStructures: OrderedDict
using Optim
using Sobol

const NOT_ALLOWED = -1e20

abstract type AbstractModel{T} end
Space = OrderedDict{Symbol,Tuple{Float64, Float64}}
features(::Type{T}, d::Datum) where T <: AbstractModel = features(T, d.t.m, d.b)


# %% ==================== BestFirst ====================

struct BestFirst{T} <: AbstractModel{T}
    β_node_value::T
    β_prune::T
    β_satisfice::T
    threshold_prune::T
    threshold_satisfice::T
    ε::T
end

default_space(::Type{BestFirst}) = Space(
    :β_prune => (0, 1000),
    :β_satisfice => (0, 1000),
    :β_node_value => (0, 1000),
    :threshold_prune => (-30, 30),
    :threshold_satisfice => (-30, 30),
    :ε => (0, 1)
)

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

function features(::Type{BestFirst{T}}, m::MetaMDP, b::Belief) where T
    (node_values=node_values(m, b), term_reward=term_reward(m, b))
end

function BestFirst(x::Vector{T}, space::Space=default_space(BestFirst))::BestFirst{T} where T
    x = Iterators.Stateful(x)
    args = map(fieldnames(BestFirst)) do fn
        fn in keys(space) ? first(x) : zero(T)
    end
    @assert isempty(x)
    BestFirst{T}(args...)
end

function preference(model::BestFirst{T}, phi, c)::T where T
    c == TERM && return model.β_satisfice * (phi.term_reward - model.threshold_satisfice)
    model.β_prune * min(0, phi.node_values[c] - model.threshold_prune) + 
    model.β_node_value * phi.node_values[c]
end


# %% ==================== Optimal ====================

ws = Iterators.Stateful(Iterators.cycle(workers()))
@memoize Dict function worker(m::MetaMDP)
     first(ws)
end

@memoize load_V(mid::String) = deserialize("$base_path/V/$mid")

function get_qs(m::MetaMDP, t::Trial)
    @fetchfrom worker(m) begin
        V = load_V(string(hash(m)))
        map(get_data(t)) do d
            Q(V, d.b)
        end
    end
end




# @memoize function get_Vs(cost)
#     n = length(readdir("$base_path/mdps/"))
#     X = reshape(1:n, :, length(COSTS))
#     idx = X[findfirst(x->x==1, COSTS)]
#     map(deserialize("$base_path/mdps/$i/V")) 
# end


# function compute_Q(data)
#     n = length(readdir("$base_path/mdps/"))

#     mdp_ids = readdir("$base_path/mdps/")
#     i = 1
#     V = deserialize("$base_path/mdps/$i/V")
#     V.m.cost

    
#     V_tbl = asyncmap(mdp_ids) do i
#         (identify(V.m), V.m.cost) => V
#     end |> Dict
# end


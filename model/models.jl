using DataStructures: OrderedDict
using Optim
using Sobol

struct Param{T}
    β_q::T
    β_prune::T
    β_node_value::T
    β_satisfice::T
    threshold_prune::T
    threshold_satisfice::T
    ε::T
end

function Param(space, x::Vector{T})::Param{T} where T
    x = Iterators.Stateful(x)
    args = map(fieldnames(Param)) do fn
        fn in keys(space) ? first(x) : zero(T)
    end
    Param(args...)
end


function preference(node_values, term_reward, c, prm::Param{T})::T where T
    c == TERM && return prm.β_satisfice * (term_reward - prm.threshold_satisfice)
    prm.β_prune * min(0, node_values[c] - prm.threshold_prune) + 
    prm.β_node_value * node_values[c]
end

function foo_pref(nv, tr, c, prm)
    threshold, w1, w2 = prm
    c == TERM && return w2 * (tr - threshold)
    w1 * nv[c]
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


# %% ==================== Likelihood ====================


struct Likelihood
    data::Vector{Datum}
end
Likelihood(trials::Vector{Trial}) = Likelihood(get_data(trials))
n_action(L) = length(L.data[1].b) + 1
n_datum(L) = length(L.data)

@memoize function apply(f::Function, L::Likelihood)
    map(L.data) do d
        f(d.t.m, d.b)
    end
end

# works on julia 1.4 only??
# function initial_pref(L::Likelihood, t::Type{T})::Vector{Vector{T}} where T
#     map(data) do d
#         .!allowed(d.t.m, d.b) * -1e20
#     end
# end

function initial_pref(L::Likelihood, ::Type{T})::Vector{Vector{T}} where T
    _initial_pref(L)
end

@memoize function _initial_pref(L::Likelihood)
    map(data) do d
        .!allowed(d.t.m, d.b) * -1e20
    end
end

@memoize chosen_actions(L::Likelihood) = map(d->d.c+1, L.data)

rand_prob(m::MetaMDP, b::Belief) = 1. / sum(allowed(m, b))

function mysoftmax(x)
    ex = exp.(x .- maximum(x))
    ex ./= sum(ex)
    ex
end

function softmax(tmp, h, c)
    @. tmp = exp(h - $maximum(h))
    tmp[c] / sum(tmp)
end

function logp(L::Likelihood, prm::Param{T})::T where T <: Real
    nv = apply(node_values, L)
    tr = apply(term_reward, L)
    H = initial_pref(L, T)
    p_rand = apply(rand_prob, L)
    chosen = chosen_actions(L)

    all_actions = 1:n_action(L)
    tmp = zeros(T, n_action(L))

    total = zero(T)
    for i in eachindex(L.data)
        h = H[i]
        for a in all_actions
            if h[a] != -1e20
                # h[c] = foo_pref(nv[i], tr[i], c, x)
                h[a] = preference(nv[i], tr[i], a-1, prm)
            end
        end
        p = prm.ε * p_rand[i] + (1-prm.ε) * softmax(tmp, h, chosen[i])
        total += log(p)
    end
    total
end

Space = OrderedDict{Symbol,Tuple{Float64, Float64}}

function Distributions.fit(space::Space, trials; x0=nothing, n_restart=20)
    lower, upper = invert(collect(values(space)))

    algo = Fminbox(LBFGS())
    options = Optim.Options()
    L = Likelihood(trials)

    if x0 != nothing
        x0s = [x0]
    else
        seq = SobolSeq(lower, upper)
        skip(seq, n_restart)
        x0s = [next!(seq) for i in 1:n_restart]
    end

    xs, ys = map(x0s) do x0
        opt = optimize(lower, upper, x0, algo, options, autodiff=:forward) do x
            prm = Param(space, x)
            -logp(L, prm)
        end
        @debug "Optimization" opt.time_run opt.iterations opt.f_calls

        opt.minimizer, opt.minimum
    end
    xs[argmin(ys)]  # note this breaks ties arbitrarily
end

# %% ====================  ====================

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


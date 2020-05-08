
# %% ==================== Likelihood ====================


struct Likelihood
    data::Vector{Datum}
end
n_action(L) = length(L.data[1].b) + 1
n_datum(L) = length(L.data)

@memoize function apply(f::Function, L::Likelihood)
    map(L.data) do d
        f(d.t.m, d.b)
    end
end

@memoize function initial_pref(L::Likelihood, t::Type{T})::Vector{Vector{T}} where T
    map(data) do d
        .!allowed(d.t.m, d.b) * -1e20
    end
end

@memoize chosen_actions(L::Likelihood) = map(d->d.c+1, L.data)

rand_prob(m::MetaMDP, b::Belief) = 1. / sum(allowed(m, b))

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

function mysoftmax(x)
    ex = exp.(x .- maximum(x))
    ex ./= sum(ex)
    ex
end

function softmax(tmp, h, c)
    @. tmp = exp(h - $maximum(h))
    tmp[c] / sum(tmp)
end

function foo_pref(nv, tr, c, prm)
    threshold, w1, w2 = prm
    c == TERM && return w2 * (tr - threshold)
    w1 * nv[c]
end

function logp(L::Likelihood, z::Vector, x::Vector{T})::T where T <: Real
    nv = apply(node_values, L)
    tr = apply(term_reward, L)
    H = initial_pref(L, T)
    p_rand = apply(rand_prob, L)
    chosen = chosen_actions(L)
    all_cs = 1:n_action(L) .- 1

    tmp = zeros(T, n_action(L))

    ε = x[end]
    prm = x[1:end-1]
    # ε = x[1]
    # prm = x[2:end]

    total = zero(T)
    for i in eachindex(L.data)
        h = H[i]
        for c in all_cs
            if h[c] != -1e20
                h[c] = foo_pref(nv[i], tr[i], c, prm)
            end
        end
        p = ε * p_rand[i] + (1-ε) * softmax(tmp, h, chosen[i])
        total += log(p)
    end
    total
end



L = Likelihood(data)
logp(L, [], [0.013, 5004.459, 4995.539, 0.296])
@btime logp(L, [], [0.013, 5004.459, 4995.539, 0.296])

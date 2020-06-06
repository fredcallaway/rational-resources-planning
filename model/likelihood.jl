struct Likelihood
    data::Vector{Datum}
end

Likelihood(trials::Vector{Trial}) = Likelihood(get_data(trials))
n_action(L) = length(L.data[1].b) + 1
n_datum(L) = length(L.data)

@memoize memo_map(f::Function, L::Likelihood) = map(f, L.data)

# works on julia 1.4 only??
# function initial_pref(L::Likelihood, t::Type{T})::Vector{Vector{T}} where T
#     map(data) do d
#         .!allowed(d.t.m, d.b) * NOT_ALLOWED
#     end
# end

function initial_pref(L::Likelihood, ::Type{T})::Vector{Vector{T}} where T
    _initial_pref(L)
end

@memoize function _initial_pref(L::Likelihood)
    map(L.data) do d
        .!allowed(d.t.m, d.b) * NOT_ALLOWED
    end
end

@memoize chosen_actions(L::Likelihood) = map(d->d.c+1, L.data)

rand_prob(m::MetaMDP, b::Belief) = 1. / sum(allowed(m, b))
rand_prob(d::Datum) = rand_prob(d.t.m, d.b)

function softmax(tmp, h, c)
    @. tmp = exp(h - $maximum(h))
    tmp[c] / sum(tmp)
end

function logp(model::AbstractModel, trials::Vector{Trial})
    L = Likelihood(get_data(trials))
    logp(L, model)
end

function logp(L::Likelihood, model::M)::T where M <: AbstractModel{T} where T <: Real
    H = initial_pref(L, T)
    p_rand = memo_map(rand_prob, L)
    chosen = chosen_actions(L)
    phi = memo_map(L) do d
        features(M, d)
    end

    all_actions = 1:n_action(L)
    tmp = zeros(T, n_action(L))

    total = zero(T)
    for i in eachindex(L.data)
        h = H[i]
        for a in all_actions
            if h[a] != NOT_ALLOWED
                # h[c] = foo_pref(nv[i], tr[i], c, x)
                h[a] = preference(model, phi[i], a-1)
            end
        end
        p = model.ε * p_rand[i] + (1-model.ε) * softmax(tmp, h, chosen[i])
        total += log(p)
    end
    total
end
# %% --------


@everywhere function Distributions.fit(::Type{M}, trials::Vector{Trial}; space=default_space(M), 
        x0=nothing, n_restart=20, progress=false) where M <: AbstractModel
    lower, upper = bounds(space)

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

    function opt_helper(x0, z; kws...)
        optimize(lower, upper, x0, algo, options; kws...) do x
            model = create_model(M, x, z, space)
            -logp(L, model)
        end
    end

    models, losses = map(combinations(space)) do z
        map(x0s) do x0
            opt = try
                opt_helper(x0, z; autodiff=:forward)
            catch err
                # fall back on numerical gradient when autograd fails
                (err isa ArgumentError && err.msg == "Value and slope at step length = 0 must be finite.") || rethrow(err)
                @warn("Automatic differentation failed while fitting $M to $(trials[1].wid)")
                opt_helper(x0, z)
            end
            @debug "Optimization" opt.time_run opt.iterations opt.f_calls
            progress && print(".")
            create_model(M, opt.minimizer, z, space), opt.minimum
        end
    end |> flatten |> invert
    i = argmin(losses)
    progress && println("")
    models[i], losses[i]
end





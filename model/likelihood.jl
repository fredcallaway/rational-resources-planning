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

# function logp(L::Likelihood, model::M)::T where M <: AbstractModel{T} where T <: Real
#     H = initial_pref(L, T)
#     p_rand = memo_map(rand_prob, L)
#     chosen = chosen_actions(L)
#     phi = memo_map(L) do d
#         features(M, d)
#     end

#     all_actions = 1:n_action(L)
#     tmp = zeros(T, n_action(L))

#     total = zero(T)
#     for i in eachindex(L.data)
#         h = H[i]
#         for a in all_actions
#             if h[a] != NOT_ALLOWED
#                 # h[c] = foo_pref(nv[i], tr[i], c, x)
#                 h[a] = preference(model, phi[i], a-1)
#             end
#         end
#         p = model.ε * p_rand[i] + (1-model.ε) * softmax(tmp, h, chosen[i])
#         total += log(p)
#     end
#     total
# end


function logp(L::Likelihood, model::M)::T where M <: AbstractModel{T} where T <: Real
    φ = memo_map(L) do d
        features(M, d)
    end

    tmp = zeros(T, n_action(L))
    total = zero(T)
    for i in eachindex(L.data)
        a = L.data[i].c + 1
        p = action_dist!(tmp, model, φ[i])
        if !(sum(p) ≈ 1)
            @error "bad probability vector" p sum(p)
            println("\n\n")
            display(model)
            println("\n\n")
        end
        @assert sum(p) ≈ 1
        total += log(p[a])
    end
    total
end


function print_tracked(x)
    if x[1] isa Float64
        xx = x
    else
        xx = getfield.(x, :value)
    end
    println(round.(xx; sigdigits=6))
end

@memoize function get_sobol(lower, upper, n)
    seq = SobolSeq(lower, upper)
    skip(seq, n)
    x0s = [next!(seq) for i in 1:n]
end

function bfgs_random_restarts(loss, lower, upper, n_restart; max_err=20)
    algorithms = [
        Fminbox(LBFGS()),
        Fminbox(LBFGS(linesearch=Optim.LineSearches.BackTracking())),
    ] |> Iterators.cycle |> Iterators.Stateful
    algo = first(algorithms)
    n_err = 0

    opts = map(get_sobol(lower, upper, n_restart)) do x0
        try
            optimize(loss, lower, upper, x0, algo, autodiff=:forward)
        catch err
            err isa InterruptException && rethrow(err)
            @warn "First BFGS attempt failed" err linesearch=typeof(algo.method.linesearch!).name
            # try the other line search method
            algo = first(algorithms)  # this cycles
            try
                optimize(loss, lower, upper, x0, algo, autodiff=:forward)
            catch err
                err isa InterruptException && rethrow(err)
                @error "Second BFGS attempt failed" err linesearch=typeof(algo.method.linesearch!).name
                n_err += 1
                if n_err >= max_err
                    error("Too many optimization errors")
                end
                return missing
            end
        end
    end |> skipmissing |> collect
    isempty(opts) ? missing : partialsort(opts, 1; by=o->o.minimum)
end

function Distributions.fit(::Type{M}, trials::Vector{Trial}; method=:bfgs, n_restart=20) where M <: AbstractModel
    space = default_space(M)
    lower, upper = bounds(space)
    space_size = upper .- lower
    @assert all(space_size .> 0)

    L = Likelihood(trials)

    n_call = 0
    function make_loss(z)
        x -> begin
            n_call += 1
            model = create_model(M, x, z, space)
            # L1 = sum(abs.(x) ./ space_size)
            -logp(L, model) #+ 10 * L1
        end
    end

    results = map(combinations(space)) do z
        loss = make_loss(z)

        opt = begin
            if method == :samin
                x0 = lower .+ rand(length(lower)) .* space_size
                optimize(loss, lower, upper, x0, SAMIN(verbosity=0), Optim.Options(iterations=10^6))
            elseif method == :bfgs
                bfgs_random_restarts(loss, lower, upper, n_restart)
            end
        end
        ismissing(opt) && return missing
        model = create_model(M, opt.minimizer, z, space)
        model, -logp(L, model)
    end |> skipmissing |> collect 
    if isempty(results)
        @error("Could not fit $M to $(trials[1].wid)")
        error("Fitting error")
    end
    @info "fitting complete" n_call
    models, losses = invert(results)
    i = argmin(losses)
    models[i], -logp(L, models[i])
end





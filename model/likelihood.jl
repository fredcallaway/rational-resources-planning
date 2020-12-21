struct Likelihood
    data::Vector{Datum}
end

Likelihood(trials::Vector{Trial}) = Likelihood(get_data(trials))
n_action(L) = length(L.data[1].b) + 1
n_datum(L) = length(L.data)

@memoize memo_map(f::Function, L::Likelihood) = map(f, L.data)

function logp(model::AbstractModel, trials::Vector{Trial})
    L = Likelihood(get_data(trials))
    logp(L, model)
end

function logp(L::Likelihood, model::M)::T where M <: AbstractModel{T} where T <: Real
    φ = memo_map(L) do d
        features(M, d)
    end

    tmp = zeros(T, n_action(L))
    total = zero(T)
    for i in eachindex(L.data)
        a = L.data[i].c + 1
        p = action_dist!(tmp, model, φ[i])
        # if !(sum(p) ≈ 1)
        #     @error "bad probability vector" p sum(p)
        # end
        # @assert sum(p) ≈ 1
        @assert isfinite(p[a])
        total += log(p[a])
    end
    total
end

@memoize function get_sobol(lower, upper, n)
    seq = SobolSeq(lower, upper)
    skip(seq, n)
    x0s = [Sobol.next!(seq) for i in 1:n]
end

function bfgs_random_restarts(loss, lower, upper, n_restart; max_err=n_restart/2, id="null")
    algorithms = [
        Fminbox(LBFGS()),
        Fminbox(LBFGS(linesearch=Optim.LineSearches.BackTracking())),
    ] |> Iterators.cycle |> Iterators.Stateful
    algo = first(algorithms)
    n_err = 0
    n_time = 0

    function do_opt(algo, x0)
        res = optimize(loss, lower, upper, x0, algo, Optim.Options(time_limit=600); autodiff=:forward)
        if !(res.f_converged || res.g_converged) && res.time_run > res.time_limit
            @warn "$id: Timed out" res.iterations res.f_calls x0=repr(round.(x0; digits=3))
            n_time += 1
            if n_time >= max_err
                error("$id: Too many timeouts")
            end
        end
        res
    end

    opts = map(get_sobol(lower, upper, n_restart)) do x0
        while !isfinite(loss(x0))
            # don't start with infinite loss!
            x0 = lower .+ rand(length(lower)) .* (upper .- lower)
        end
        try
            do_opt(algo, x0)
        catch err
            err isa InterruptException && rethrow(err)
            # @warn "First BFGS attempt failed" err linesearch=typeof(algo.method.linesearch!).name
            # try the other line search method
            algo = first(algorithms)  # this cycles
            try
                do_opt(algo, x0)
            catch err
                err isa InterruptException && rethrow(err)
                @warn "$id: Second BFGS attempt failed" err linesearch=typeof(algo.method.linesearch!).name x0=repr(round.(x0; digits=3))
                n_err += 1
                if n_err >= max_err
                    @error "$id: Too many optimization errors"
                    rethrow(err)
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

    # Initialize in the center of the space? This doesn't seem to help
    # q = space_size ./ 4; lower += q; upper -= q

    L = Likelihood(trials)
    
    if isempty(lower)  # no free parameters
        model = create_model(M, lower, (), space)
        return model, -logp(L, model)
    end

    function make_loss(z)
        x -> begin
            model = create_model(M, x, z, space)
            # L1 = sum(abs.(x) ./ space_size)
            -logp(L, model) #+ 10 * L1
        end
    end

    results, elapsed = @timed map(combinations(space)) do z
        loss = make_loss(z)

        opt = begin
            if method == :samin
                x0 = lower .+ rand(length(lower)) .* space_size
                optimize(loss, lower, upper, x0, SAMIN(verbosity=0), Optim.Options(iterations=10^6))
            elseif method == :bfgs
                t = trials[1]
                bfgs_random_restarts(loss, lower, upper, n_restart; id="$M-$(t.wid)-$(t.i)")
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
    # @info "Fitting complete" n_call elapsed M
    models, losses = invert(results)
    i = argmin(losses)
    models[i], -logp(L, models[i])
end

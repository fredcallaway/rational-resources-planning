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

function bfgs_random_restarts(loss, lower, upper, n_restart; 
                              time_limit=120, max_err=10, max_timeout=50, max_finite=10, id="null")
    #algorithms = [
    #    Fminbox(LBFGS()),
    #    Fminbox(LBFGS(linesearch=Optim.LineSearches.BackTracking())),
    #] |> Iterators.cycle |> Iterators.Stateful
    n_err = 0
    n_time = 0
    n_finite = 0

    function do_opt(x0)
        if !isfinite(loss(x0))  # hopeless!
            n_finite += 1
            @debug "nonfinite loss" n_finite
            return missing
        end
        try
            # >30s indicates that the optimizer is stuck, which means it's not likely to find a good minimum anyway
            res = optimize(loss, lower, upper, x0, Fminbox(LBFGS()), Optim.Options(;time_limit); autodiff=:forward)
            if !(res.f_converged || res.g_converged) && res.time_run > res.time_limit
                n_time += 1
                @debug "timeout" n_time
                return missing
            else
                return res
            end
        catch err
            err isa InterruptException && rethrow(err)
            n_err += 1
            @debug "error" n_err
            return missing
        end
    end

    x0s = SobolSeq(lower, upper)
    results = Any[]
    while length(results) < n_restart
        res = do_opt(next!(x0s))
        if !ismissing(res)
            push!(results, res)
        else
            if n_err > max_err
                @error "$id: Too many errors while optimizing"
                rethrow(err)
            elseif n_time > max_timeout
                @error "$id: Too many timeouts while optimizing"
                error("Optimization timeouts")
            elseif n_finite > max_finite
                @error "$id: Too many timeouts while optimizing"
                error("Optimization timeouts")
            end
        end
    end
    if n_err > max_err/2 || n_time > max_timeout/2 || n_finite > max_finite/2
        @warn "$id: Difficulty optimizing" n_err n_time n_finite
    end
    losses = getfield.(results, :minimum)
    very_good = minimum(losses) * 1.01
    n_good = sum(losses .< very_good)
    if n_good < 5
        @warn "$id: Only $n_good random restarts produced a very good minimum"
    end
    partialsort(results, 1; by=o->o.minimum)  # best result
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
                id = "$(name(M))-$(t.wid)-$(t.i)"
                bfgs_random_restarts(loss, lower, upper, n_restart; id)
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

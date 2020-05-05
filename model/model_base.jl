using Optim

# %% ==================== Preference model ====================

abstract type PreferenceModel end

Base.length(m::PreferenceModel) = 1


function preferences(model::PreferenceModel, d::Datum)
    preferences(model, d.t, d.b)
end

function preferences(model::PreferenceModel, t::Trial)
    map(get_data(t)) do datum
        preferences(model, datum)
    end
end

function get_params(model::T) where T <: PreferenceModel
    map(fieldnames(typeof(model))) do field
        getfield(model, field)
    end |> collect
end

n_param(model::PreferenceModel) = length(bounds(model))

function set_params!(model::PreferenceModel, x)
    for (field, val) in zip(fieldnames(typeof(model)), x)
        setfield!(model, field, val)
    end
end

function disprefer_impossible!(prefs, t, b)
    m = MetaMDP(t, NaN)
    for c in 1:length(m)
        if !allowed(m, b, c)
            prefs[c+1] = -1e20
        end
    end
end

# %% ==================== Error model ====================

struct ErrorModel{T}
    model::T
    β
    ε
end

function mysoftmax(x)
    ex = exp.(x .- maximum(x))
    ex ./= sum(ex)
    ex
end

function p_rand(prefs)
    n_option = sum(prefs .!= -1e20)
    1 / n_option
end

function logp(prefs, c::Int, β, ε)
    p_soft = mysoftmax(β .* prefs)[c+1]
    if !isfinite(p_soft)
        error("Bad prefs: $prefs")
    end
    p = ε * p_rand(prefs) + (1-ε) * p_soft
    log(p)
end

function logp(em::ErrorModel, d::Datum)
    prefs = preferences(em.model, d)
    disprefer_impossible!(prefs, d.t, d.b)
    logp(prefs, d.c, em.β, em.ε)
end

function predictive(em::ErrorModel, t::Trial, b::Belief)
    prefs = preferences(em.model, t, b)
    disprefer_impossible!(prefs, t, b)
    possible = prefs .!= -1e20
    p_ran = possible / sum(possible)
    p_soft = mysoftmax(em.β .* prefs)
    em.ε * p_ran + (1-em.ε) * p_soft
end

logp(em::ErrorModel, data::Vector{Datum}) = mapreduce(d->logp(em, d), +, data)
logp(em::ErrorModel, trial::Trial) = logp(em, get_data(trial))
logp(em::ErrorModel, trials::Vector{Trial}) = logp(em, get_data(trials))


function fit_model(model, trials)
    lower = [1e-3, 1e-3]; upper = [10., 1.]; x0 = [0.01, 0.1]
    for (low, high) in bounds(model)
        push!(lower, low); push!(upper, high)
        push!(x0, rand(Uniform(low, high)))
    end

    opt = optimize(lower, upper, x0, Fminbox(LBFGS()), autodiff=:forward) do x
        # model = model_class(x[3:end]...)
        set_params!(model, x[3:end])
        error_model = ErrorModel(model, x[1:2]...)
        -logp(error_model, trials)
    end
    x = opt.minimizer
    set_params!(model, x[3:end])
    error_model = ErrorModel(model, x[1:2]...)
end

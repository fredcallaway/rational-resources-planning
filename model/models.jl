using Optim

# %% ==================== Preference ====================

"""
A Preference defines how desirable each computation is in a given belief state.
"""
abstract type Preference end

parameters(pref::Preference) = []

function apply(pref::Preference, d::Datum)
    apply(perf, d.t, d.b)
end

function apply(pref::Preference, t::Trial)
    map(get_data(t)) do datum
        apply(pref, datum)
    end
end

function parameters(pref, kind::Symbol)
    type = Dict(:discrete => Set, :continuous => Tuple)[kind]
    filter(parameters(pref)) do (field, spec)
        spec isa type
    end
end

function set_params!(pref::Preference, kind::Symbol, x)
    x = Iterators.Stateful(x)
    for (field, spec) in parameters(pref, kind)
        setfield!(pref, field, first(x))
    end
    @assert isempty(x)
end

function get_params(pref::Preference, kind::Symbol)
    map(keys(parameters(pref, kind))) do field
        getfield(pref, field)
    end
end

include("preferences.jl")

# %% ==================== Choice model ====================

"""
A Model defines a likelihood over computations based on a softmax
over a weighted sum of preferences and a lapse rate.


p(c|b) ∝ ε ∑[ wᵢ prefᵢ(b, c)] + (1-ε) allowed(b, c)
"""
mutable struct Model
    preferences::Vector{Preference}
    weights::Vector{Real}
    ε::Real
end

function preference(model::Model, t::Trial, b::Belief)
    mapreduce(+, model.preferences, model.weights) do pref, w
        w * apply(pref, t, b)
    end
end

function mysoftmax(x)
    ex = exp.(x .- maximum(x))
    ex ./= sum(ex)
    ex
end


function disprefer_impossible!(prefs, t, b)
    m = MetaMDP(t, NaN)
    for c in 1:length(m)
        if !allowed(m, b, c)
            prefs[c+1] = -1e20
        end
    end
end

function likelihood(model::Model, t::Trial, b::Belief)
    h = preference(model, t, b)
    disprefer_impossible!(h, t, b)
    possible = h .!= -1e20
    model.ε * (possible / sum(possible)) + (1-model.ε) * mysoftmax(h)
end

logp(model::Model, d::Datum) = log(likelihood(model, d.t, d.b)[d.c+1])

# function p_rand(h)
#     n_option = sum(h .!= -1e20)
#     1 / n_option
# end
# function logp(model::Model, d::Datum)
#     h = preference(model, d)
#     disprefer_impossible!(h, d.t, d.b)
#     p_soft = mysoftmax(h)[c+1]
#     p = ε * p_rand(h) + (1-ε) * p_soft
#     log(p)
# end

logp(model::Model, data::Vector{Datum}) = mapreduce(d->logp(model, d), +, data)
logp(model::Model, trial::Trial) = logp(model, get_data(trial))
logp(model::Model, trials::Vector{Trial}) = logp(model, get_data(trials))


function parameters(model::Model, kind::Symbol)
    mapreduce(vcat, model.preferences) do pref
        parameters(pref, kind::Symbol)
    end
end

function n_continuous(model::Model)
    1 + mapreduce(+, model.preferences) do pref
        1 + length(a(pref, :continuous))
    end
end

function set_params!(model::Model, kind::Symbol, x)
    # @info "set_params!" kind x
    x = Iterators.Stateful(x)
    for pref in model.preferences
        n = length(parameters(pref, kind))
        set_params!(pref, kind, Iterators.take(x, n))
    end
    if kind == :continuous
        for i in eachindex(model.weights)
            model.weights[i] = first(x)
        end
        model.ε = first(x)
    end
    @assert isempty(x)
end

function get_params(model::Model, kind::Symbol)
    x = Float64[]
    for pref in model.preferences
        push!(x, get_params(pref, kind)...)
    end
    if kind == :continuous
        push!(x, model.weights...)
        push!(x, model.ε)
    end
end

function discrete_options(model::Model)
    specs = map(x->x[2], parameters(model, :discrete))
    Iterators.product(specs...)
end

function continuous_space(model::Model)
    bounds = Tuple{Float64, Float64}[]
    for (field, (low, high)) in parameters(model, :continuous)
        push!(bounds, (low, high))
    end
    for i in eachindex(model.weights)
        push!(bounds, (0., 1e4))
    end
    push!(bounds, (1e-4, 1.))
    bounds
end

function Distributions.fit(base_model::Model, trials)
    lower, upper = invert(continuous_space(base_model))

    models, losses = map(discrete_options(base_model)) do x_disc
        model = deepcopy(base_model)
        set_params!(model, :discrete, x_disc)
        
        r = rand(length(upper))
        x0 = @. lower + r * (upper - lower)

        algo = Fminbox(LBFGS())
        options = Optim.Options()
        # options = Optim.Options(f_tol=1e-3, successive_f_tol=10)
        opt = optimize(lower, upper, x0, algo, options, autodiff=:forward) do x
            set_params!(model, :continuous, x)
            # penalty = sum(model.weights) * 1e-3
            -logp(model, trials)# + penalty
        end

        set_params!(model, :continuous, opt.minimizer)
        model, opt.minimum
    end |> invert
    models[argmin(losses)]  # note this breaks ties arbitrarily
end

#convenience
function Model(prefs::Preference...)
    Model(collect(prefs), fill(NaN, length(prefs)), NaN)
end


# # %% ====================  ====================
# function fit_model(model, trials)
#     lower = [1e-3, 1e-3]; upper = [10., 1.]; x0 = [0.01, 0.1]

#     for (low, high) in bounds(model)
#         push!(lower, low); push!(upper, high)
#         push!(x0, rand(Uniform(low, high)))
#     end

#     opt = optimize(lower, upper, x0, Fminbox(LBFGS()), autodiff=:forward) do x
#         # model = model_class(x[3:end]...)
#         set_params!(model, x[3:end])
#         error_model = ErrorModel(model, x[1:2]...)
#         -logp(error_model, trials)
#     end
#     x = opt.minimizer
#     set_params!(model, x[3:end])
#     error_model = ErrorModel(model, x[1:2]...)
# end

using Optim
using Sobol
# %% ==================== Preference ====================

"A Preference defines how desirable each computation is in a given belief state."
abstract type Preference end

apply(pref::Preference, m::MetaMDP, b::Belief) = error("Not implemented")
apply(pref::Preference, d::Datum) = apply(pref, d.t.m, d.b)

"Returns a list of (name => spec) pairs.

if spec is a Set, it gives the possible values of a discrete variable
if spec is a Tuple, it gives bounds for a continuous variable
"
parameters(pref::Preference) = []

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

"Initialize a Model with the given preference types and all parameters NaN"
function Model(pref_types::Type...)
    prefs = map(pref_types) do T
        T(fill(NaN, length(fieldnames(T)))...)
    end |> collect
    Model(prefs, fill(NaN, length(prefs)), NaN)
end

"Total preference of the model is weighted sum of its preferences"
function preference(model::Model, m::MetaMDP, b::Belief)
    mapreduce(+, model.preferences, model.weights) do pref, w
        w * apply(pref, m, b)
    end
end
function preference(model::Model, d::Datum)
    mapreduce(+, model.preferences, model.weights) do pref, w
        w * apply(pref, d)
    end
end

function parameters(model::Model, kind::Symbol)
    mapreduce(vcat, model.preferences) do pref
        parameters(pref, kind::Symbol)
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


# %% ==================== Likelihood ====================

function mysoftmax(x)
    ex = exp.(x .- maximum(x))
    ex ./= sum(ex)
    ex
end

function disprefer_impossible!(prefs::Vector, m::MetaMDP, b::Belief)
    for c in 1:length(m)
        if !allowed(m, b, c)
            prefs[c+1] = -1e20
        end
    end
end

function likelihood(h::Vector, ε::Real, m::MetaMDP, b::Belief)
    disprefer_impossible!(h, m, b)
    possible = h .!= -1e20
    ε * (possible / sum(possible)) + (1-ε) * mysoftmax(h)
end
function likelihood(model::Model, m::MetaMDP, b::Belief)
    h = preference(model, m, b)
    likelihood(h, model.ε, m, b)
end
function likelihood(model::Model, d::Datum)
    h = preference(model, d)
    likelihood(h, model.ε, d.t.m, d.b)
end

likelihood(model::Model, t::Trial, b::Belief) = likelihood(model, t.m, b)
logp(model::Model, d::Datum) = log(likelihood(model, d)[d.c+1])
logp(model::Model, data::Vector{Datum}) = mapreduce(d->logp(model, d), +, data)
logp(model::Model, trial::Trial) = logp(model, get_data(trial))
logp(model::Model, trials::Vector{Trial}) = logp(model, get_data(trials))


# %% ==================== Fitting ====================

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

function Distributions.fit(base_model::Model, trials; x0=nothing, n_restart=20)
    lower, upper = invert(continuous_space(model))

    algo = Fminbox(LBFGS())
    options = Optim.Options()
    data = get_data(trials)

    models, losses = map(discrete_options(base_model)) do x_disc
        model = deepcopy(base_model)
        set_params!(model, :discrete, x_disc)

        if x0 != nothing
            x0s = [x0]
        else
            seq = SobolSeq(lower, upper)
            skip(seq, n_restart)
            x0s = [next!(seq) for i in 1:n_restart]
        end

        map(x0s) do x0
            opt = optimize(lower, upper, x0, algo, options, autodiff=:forward) do x
                set_params!(model, :continuous, x)
                -logp(model, data)
            end
            @debug "Optimization" opt.time_run opt.iterations opt.f_calls

            set_params!(model, :continuous, opt.minimizer)
            model, opt.minimum
        end
    end |> flatten |> invert
    models[argmin(losses)]  # note this breaks ties arbitrarily
end



# %% ==================== Simulating ====================

struct Simulator <: Policy
    model::Model
    m::MetaMDP
end

(sim::Simulator)(b::Belief) = rand(Categorical(likelihood(sim.model, sim.m, b))) - 1

function simulate(sim::Simulator)
    bs = Belief[]
    cs = Int[]
    rollout(sim) do b, c
        push!(bs, deepcopy(b)); push!(cs, c)
    end
    bs, cs
    wid = join([typeof(p) for p in model.preferences], "-")
    Trial(sim.m, wid, bs, cs, [], [])
end

simulate(model::Model, m::MetaMDP) = simulate(Simulator(model, m))


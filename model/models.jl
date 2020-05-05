include("model_base.jl")

# %% ==================== Satisficing ====================

mutable struct Satisficing <: PreferenceModel
    threshold::Real
end

bounds(model::Satisficing) = [
    (-30., 30.)
]

function preferences(model::Satisficing, t::Trial, b::Belief)
    m = MetaMDP(t, NaN)
    [term_reward(m, b) - model.threshold; zeros(length(b))]
end

# %% ==================== Pruning ====================

mutable struct Pruning <: PreferenceModel
    threshold::Real
end

bounds(model::Pruning) = [
    (-30., 30.)
]

function preferences(model::Pruning, t::Trial, b::Belief)
    m = MetaMDP(t, NaN)
    nv = node_values(m, b)
    map(0:length(b)) do c
        c == TERM && return 0.
        min(0, nv[c] - model.threshold)
        # term_reward(m, b) - model.threshold
    end
end

# %% ==================== Best First Search ====================

struct BestFirst <: PreferenceModel
end

bounds(model::BestFirst) = []

function preferences(model::BestFirst, t::Trial, b::Belief)
    m = MetaMDP(t, NaN)
    [0; node_values(m, b)]
end


# %% ==================== Combinations of preferences ====================

struct MultiPref2 <: PreferenceModel
    prefs::Vector{PreferenceModel}
    weights::Vector{Real}
end


function bounds(mp::MultiPref2)
    pbounds = mapreduce(vcat, mp.prefs) do model
        bounds(model)
    end
    wbounds = [(1e-4, 1e4) for i in eachindex(mp.weights)]
    vcat(pbounds, wbounds)
end

function preferences(mp::MultiPref2, t::Trial, b::Belief)
    mapreduce(+, mp.prefs, mp.weights) do model, w
        w * preferences(model, t, b)
    end
end

function set_params!(mp::MultiPref2, x)
    x = Iterators.Stateful(x)
    for model in mp.prefs
        set_params!(model, Iterators.take(x, n_param(model)))
    end
    for i in eachindex(mp.weights)
        mp.weights[i] = first(x)
    end
    @assert isempty(x)
end

function fit_model(model::MultiPref2, trials)
    lower = [1e-3]; upper = [1.]; x0 = [0.1]
    β = 1.  # redundant with the weights
    for (low, high) in bounds(model)
        push!(lower, low); push!(upper, high)
        push!(x0, rand(Uniform(low, high)))
    end

    opt = optimize(lower, upper, x0, Fminbox(LBFGS()), autodiff=:forward) do x
        set_params!(model, x[2:end])
        error_model = ErrorModel(model, β, x[1])
        penalty = sum(model.weights) * 1e-3
            -logp(error_model, trials) + penalty
    end
    x = opt.minimizer
    set_params!(model, x[2:end])
    ErrorModel(model, β, x[1])
end



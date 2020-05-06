# %% ==================== Load code ====================
isempty(ARGS) && push!(ARGS, "web")  # command line arg specifies to use conf/web.jl
include("conf.jl")
include("utils.jl")
include("mdp.jl")
include("data.jl")
include("models.jl")

# %% ==================== Load Data ====================

# dictionary of participant id to vector of Trials
all_trials = load_trials(EXPERIMENT);

# %% ==================== Fit a model ====================
# define model "class" as a model with all its parameters NaN
base_model = Model(BestFirst, Satisficing)
@assert isnan(logp(base_model, trials))

# fit the model to data from one participant
trials = first(values(all_trials))  # trials = all_trials["w1ebd161"]
model = fit(base_model, trials)

# total log likelihood
logp(model, trials)

# one Trial
logp(model, trials[1])

# one Datum
datum = get_data(trials)[1]
logp(model, datum)

# likelihood for ALL possible actions, first is TERM
likelihood(model, datum)

# likelihood of computation c. WARNING: note the +1 to account for TERM value in position 1
log(likelihood(model, datum)[d.c+1]) ≈ logp(model, datum)


# %% ==================== Exammple: KL divergence between models ====================

function kl_divergence(P, Q)
    mapreduce(+, P, Q) do p, q
        p == 0 && return 0.  # becuase 0/0 is NaN
        p * log(p/q)
    end
end

model2 = fit(Model(BestFirst, Satisficing, Pruning), trials)

# Hopefully the more flexible model does at least as well.
# This might not be true because the MLE optimization gets stuck in 
# local minima. (On the list of things to fix.)
@assert logp(model2, trials) >= logp(model, trials)

# Find datum with maximal KL divergence and see how the predictions differ.
klds = map(get_data(trials)) do d
    kl_divergence(likelihood(model, d), likelihood(model2, d))
end

max_kl, i = findmax(klds)
@show max_kl
distinguishing_datum = get_data(trials)[i]
likelihood(model, distinguishing_datum) .- likelihood(model2, distinguishing_datum)

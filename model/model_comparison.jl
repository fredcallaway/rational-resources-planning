using StatsBase
using Distributed
isempty(ARGS) && push!(ARGS, "web")
include("conf.jl")
@everywhere begin
    using Glob
    using Serialization
    using CSV
    include("base.jl")
    include("models.jl")
end

@everywhere results_path = "$results/$EXPERIMENT"
mkpath(results_path)
# mkpath("$results_path/cv_likelihood")
FOLDS = 5

# %% ==================== Load pilot data ====================

all_trials = load_trials(EXPERIMENT) |> OrderedDict |> sort!
flat_trials = flatten(values(all_trials));
println(length(flat_trials), " trials")
all_data = all_trials |> values |> flatten |> get_data;

# %% ==================== Fit models to full dataset ====================
@everywhere include("best_first_alt.jl")
models = [Optimal, BestFirst, NewBestFirst]
jobs = Iterators.product(values(all_trials), models);
@time full_fits = pmap(jobs) do (trials, M)
    model, nll = fit(M, trials)
    (model=model, nll=nll)
end;

# %% --------

let
    nll = getfield.(full_fits, :nll)
    total = sum(nll; dims=1)
    best_model = [p.I[2] for p in argmin(nll; dims=2)]
    n_fit = counts(best_model)
    println("Model             Likelihood   Best Fit")
    for i in eachindex(models)
        @printf "%-15s         %4d         %d\n" models[i] total[i] n_fit[i]
    end
end


# %% --------
using DataFrames

fmods = getfield.(full_fits[:, 1], :model)
fns = fieldnames(typeof(fmods[1]))
model = fmods[1]
rows = map(fmods) do model
    [getfield(model, k) for k in fns]
end
DataFrame(combinedims(rows)', collect(fns))


# %% ==================== Simulations ====================


ks = map(jobs) do (trials, M)
    trials[1].wid, M
end
fit_dict = Dict(zip(ks, getfield.(full_fits, :model)))

@everywhere fit_dict = $fit_dict
@everywhere function get_model(M::Type, t::Trial)
    fit_dict[t.wid, M]
end

sims = pmap(jobs) do (trials, M)
    wid = string(M) * "-" * trials[1].wid 
    map(repeat(trials, 50)) do t
        model = get_model(M, t)
        simulate(model, t.m; wid=wid)
    end
end

serialize("$base_path/sims", sims)


# %% --------

nll = getfield.(full_fits, :nll)
total = sum(nll; dims=1)
best_model = [p.I[2] for p in argmin(nll; dims=2)]
n_fit = counts(best_model)

let
    println("Model        Likelihood   Best Fit")
    for i in eachindex(models)
        @printf "%-10s         %4d         %d\n" models[i] total[i] n_fit[i]
    end
end


# %% ==================== Cross validation ====================

function kfold_splits(n, k)
    @assert (n / k) % 1 == 0  # can split evenly
    map(1:k) do i
        test = i:k:n
        (train=setdiff(1:n, test), test=test)
    end
end


function cross_validate(model_class; k=2, fit_kws...)
    map(kfold_splits(length(trials), k)) do (train, test)
        fit = fit_model(model_class, trials[train]; fit_kws...)
        logp(fit.model, get_data(trials[test]), fit.α, fit.ε)
    end
end

models = [Optimal, BestFirst]
n_trial = length(all_trials |> values |> first)
folds = kfold_splits(n_trial, FOLDS)
jobs = Iterators.product(values(all_trials), models, folds)
@time cv_fits = pmap(jobs) do (trials, M, fold)
    model, train_nll = fit(M, trials[fold.train])
    (model=model, train_nll=train_nll, test_nll=-logp(model, trials[fold.test]))
end;

# %% --------

# Sum over the folds
test_nll = sum(getfield.(cv_fits, :test_nll); dims=3) |> dropdims(3)
train_nll = sum(getfield.(cv_fits, :train_nll); dims=3) |> dropdims(3)
train_nll ./= (FOLDS - 1)  # each trial is counted this many times
@assert sum(flatten([f.train for f in folds]) .== 1) == (FOLDS - 1)

# Sum over participants
total_train = sum(train_nll; dims=1)
total_test = sum(test_nll; dims=1)

best_model = [p.I[2] for p in argmin(test_nll; dims=2)]
n_fit = counts(best_model)


# %% --------


let
    println("Model       Train NLL   Test NLL    Best Fit")
    for i in eachindex(models)
        @printf "%-10s  %4d  %10d  %8d\n" models[i] total_train[i] total_test[i] n_fit[i]
    end
end

# %% --------





total = sum(cv_fits; dims=1)
best_fit = counts([p.I[2] for p in argmin(cv_fits; dims=2)][:], 1:length(models))

# %% ====================  ====================




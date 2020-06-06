using StatsBase
using Distributed
isempty(ARGS) && push!(ARGS, "exp2")
include("conf.jl")
@everywhere include("base.jl")
    
using Glob
using CSV
using DataFrames

@everywhere results_path = "$results/$EXPERIMENT"
mkpath(results_path)
FOLDS = 5

# %% ==================== Load data ====================

all_trials = load_trials(EXPERIMENT) |> OrderedDict |> sort!
flat_trials = flatten(values(all_trials));
println(length(flat_trials), " trials")
all_data = all_trials |> values |> flatten |> get_data;

# %% ==================== Write Q table and load model code ====================

if !isfile("$base_path/Q_table")
    include("Q_table.jl")
    println("Creating Q_table")
    @time serialize("$base_path/Q_table", make_Q_table(all_data))
    println("Wrote $base_path/Q_table")
end

include("models.jl")  # necessary for some reason
@everywhere include("models.jl")

# %% ==================== Fit models to full dataset ====================

# fit(Optimal, all_trials |> values |> first)
# @fetchfrom 2 fit(BestFirst, all_trials |> values |> first)

models = [Optimal, BestFirst, BreadthFirst, DepthFirst]
full_jobs = Iterators.product(values(all_trials), models);
@time full_fits = pmap(full_jobs) do (trials, M)
    try
        model, nll = fit(M, trials)
        (model=model, nll=nll)
    catch e
        println("Error fitting $M to $(trials[1].wid):  $e")
        rethrow(e)
        # (model=model, nll=NaN)
    end
end;

function mle_table(M)
    i = findfirst(models .== M)
    map(zip(keys(all_trials), full_fits[:, i])) do (wid, (model, nll))
        (wid=wid, model=M, nll=nll, namedtuple(model)...)
    end |> DataFrame
end

mkpath("$results_path/mle")
for M in models
    mle_table(M) |> CSV.write("$results_path/mle/$M.csv")
end

let
    nll = getfield.(full_fits, :nll)
    total = sum(nll; dims=1)
    best_model = [p.I[2] for p in argmin(nll; dims=2)]
    n_fit = counts(best_model, 1:length(models))
    println("Model             Likelihood   Best Fit")
    for i in eachindex(models)
        @printf "%-15s         %4d         %d\n" models[i] total[i] n_fit[i]
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

n_trial = length(all_trials |> values |> first)
folds = kfold_splits(n_trial, FOLDS)
cv_jobs = Iterators.product(values(all_trials), models, folds);
@time cv_fits = pmap(cv_jobs) do (trials, M, fold)
    model, train_nll = fit(M, trials[fold.train])
    (model=model, train_nll=train_nll, test_nll=-logp(model, trials[fold.test]))
end;

function cv_table(M)
    mi = findfirst(models .== M)
    mapmany(enumerate(keys(all_trials))) do (wi, wid)
        map(1:FOLDS) do fi
            x = cv_fits[wi, mi, fi]
            (wid=wid, model=M, fold=fi, train_nll=x.train_nll, test_nll=x.test_nll, namedtuple(x.model)...)
        end
    end |> DataFrame
end

mkpath("$results_path/mle")
for M in models
    cv_table(M) |> CSV.write("$results_path/mle/$M-cv.csv")
end

let
    # Sum over the folds
    test_nll = sum(getfield.(cv_fits, :test_nll); dims=3) |> dropdims(3);
    train_nll = sum(getfield.(cv_fits, :train_nll); dims=3) |> dropdims(3);
    train_nll ./= (FOLDS - 1);  # each trial is counted this many times

    # Sum over participants
    total_train = sum(train_nll; dims=1)
    total_test = sum(test_nll; dims=1)

    best_model = [p.I[2] for p in argmin(test_nll; dims=2)];
    n_fit = counts(best_model, 1:length(models))

    let
        println("Model            Train NLL   Test NLL    Best Fit")
        for i in eachindex(models)
            @printf "%-15s  %4d  %10d  %8d\n" models[i] total_train[i] total_test[i] n_fit[i]
        end
    end
end


# %% ==================== Save model predictions ====================
fit_lookup = let
    ks = map(cv_jobs) do (trials, M, fold)
        trials[1].wid, M, fold.test[1]
    end
    @assert length(ks) == length(cv_fits)
    Dict(zip(ks, getfield.(cv_fits, :model)))
end

function get_model(M::Type, t::Trial, trial_index)
    fold = (trial_index - 1) % FOLDS + 1  # the pains of 1-indexing
    fit_lookup[t.wid, M, fold]
end

function get_preds(M::Type, t::Trial, trial_index)
    model = get_model(M, t, trial_index)
    map(get_data(t)) do d
        action_dist(model, d)
    end
end

function get_params(M::Type, t::Trial, trial_index)
    model = get_model(M, t, trial_index)
    Dict(fn => getfield(model, fn) for fn in fieldnames(typeof(model)))
end

function demo_trial(t, trial_index)
    (
        stateRewards = t.bs[end],
        demo = (
            clicks = t.cs[1:end-1] .- 1,
            path = t.path .- 1,
            predictions = Dict(string(M) => get_preds(M, t, trial_index) for M in models),
            parameters = Dict(string(M) => get_params(M, t, trial_index) for M in models)
        )
    )
end

function sort_by_score(xs)
    sort(xs, by=x->-x.score)
end

map(collect(all_trials)) do (wid, trials)
    (
        wid = wid,
        trials = demo_trial.(trials, eachindex(trials)),
        score = mean(t.score for t in trials),
        clicks = mean(length(t.cs)-1 for t in trials),
    )
end |> sort_by_score |> JSON.json |> write("$results_path/demo_viz.json")

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

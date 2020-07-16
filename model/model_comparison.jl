using Distributed
using StatsBase
using Glob
using CSV
using DataFrames

isempty(ARGS) && push!(ARGS, "exp2")
include("conf.jl")

@everywhere include("base.jl")

mkpath(results_path)
FOLDS = 5

# %% ==================== LOAD DATA ====================

all_trials = load_trials(EXPERIMENT) |> OrderedDict |> sort!
flat_trials = flatten(values(all_trials));
println(length(flat_trials), " trials")
all_data = all_trials |> values |> flatten |> get_data;

# %% ==================== SOLVE MDPS AND PRECOMPUTE Q LOOKUP TABLE ====================

include("solve.jl")
@time solve_all();

include("Q_table.jl")
@time serialize("$base_path/Q_table", make_Q_table(all_data));

# %% ==================== LOAD MODEL CODE ====================

@everywhere include("models.jl")

MODELS = [
    Optimal,
    # Heuristic{:Full},
    Heuristic{:BestFirst},
    Heuristic{:DepthFirst},
    Heuristic{:BreadthFirst},
    # Heuristic{:BestFirstNoSatisfice},
    # Heuristic{:BestFirstNoBestNext},
    # Heuristic{:BestFirstNoDepthLimit},
    # Heuristic{:FullNoSatisfice},
    # Heuristic{:FullNoBestNext},
    # Heuristic{:FullNoDepthLimit},
]

# %% ==================== FIT MODELS TO FULL DATASET ====================

@fetchfrom 2 fit(BestFirst, all_trials |> values |> first)
fit(Heuristic{:BreadthFirst}, all_trials |> values |> first)

full_jobs = Iterators.product(values(all_trials), MODELS);
@time full_fits = pmap(full_jobs) do (trials, M)
    model, nll = fit(M, trials)
    (model=model, nll=nll)
end;
serialize("$base_path/full_fits", full_fits)

function mle_table(M)
    i = findfirst(MODELS .== M)
    map(zip(keys(all_trials), full_fits[:, i])) do (wid, (model, nll))
        (wid=wid, model=name(M), nll=nll, namedtuple(model)...)
    end |> DataFrame
end

mkpath("$results_path/mle")
for M in MODELS
    mle_table(M) |> CSV.write("$results_path/mle/$(name(M)).csv")
end

let
    nll = getfield.(full_fits, :nll)
    total = sum(nll; dims=1)
    best_model = [p.I[2] for p in argmin(nll; dims=2)]
    n_fit = counts(best_model, 1:length(MODELS))
    println("Model                  Likelihood   Best Fit")
    for i in eachindex(MODELS)
        @printf "%-22s       %4d         %d\n" name(MODELS[i]) total[i] n_fit[i]
    end
end
# %% ==================== CROSS VALIDATION ====================

function kfold_splits(n, k)
    @assert (n / k) % 1 == 0  # can split evenly
    map(1:k) do i
        test = i:k:n
        (train=setdiff(1:n, test), test=test)
    end
end

n_trial = length(all_trials |> values |> first)
folds = kfold_splits(n_trial, FOLDS)
cv_jobs = Iterators.product(values(all_trials), MODELS, folds);
@time cv_fits = pmap(cv_jobs) do (trials, M, fold)
    try
        model, train_nll = fit(M, trials[fold.train])
        (model=model, train_nll=train_nll, test_nll=-logp(model, trials[fold.test]))
    catch e
        println("Error fitting $M to $(trials[1].wid):  $e")
        rethrow(e)
        # (model=model, nll=NaN)
    end
end;
serialize("$base_path/cv_fits", cv_fits)

function cv_table(M)
    mi = findfirst(MODELS .== M)
    mapmany(enumerate(keys(all_trials))) do (wi, wid)
        map(1:FOLDS) do fi
            x = cv_fits[wi, mi, fi]
            (wid=wid, model=name(M), fold=fi, train_nll=x.train_nll, test_nll=x.test_nll, namedtuple(x.model)...)
        end
    end |> DataFrame
end

mkpath("$results_path/mle")
for M in MODELS
    cv_table(M) |> CSV.write("$results_path/mle/$(name(M))-cv.csv")
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
    n_fit = counts(best_model, 1:length(MODELS))

    let
        println("Model                   Train NLL   Test NLL    Best Fit")
        for i in eachindex(MODELS)
            @printf "%-22s  %4d  %10d  %8d\n" name(MODELS[i]) total_train[i] total_test[i] n_fit[i]
        end
    end
end


# %% ==================== SAVE MODEL PREDICTIONS ====================
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

# %% --------


# %% --------
function demo_trial(t, trial_index)
    (
        stateRewards = t.bs[end],
        demo = (
            clicks = t.cs[1:end-1] .- 1,
            path = t.path .- 1,
            predictions = Dict(name(M) => get_preds(M, t, trial_index) for M in MODELS),
            parameters = Dict(name(M) => get_params(M, t, trial_index) for M in MODELS)
        )
    )
end

function sorter(xs)
    sort(xs, by=x->(x.variance, -x.score))
end

mkpath("$results_path/viz")
map(collect(all_trials)) do (wid, trials)
    (
        wid = wid,
        variance = variance_structure(trials[1].m),
        score = mean(t.score for t in trials),
        clicks = mean(length(t.cs)-1 for t in trials),
    )
end |> sorter |> JSON.json |> write("$results_path/viz/table.json")

foreach(collect(all_trials)) do (wid, trials)
    demo_trial.(trials, eachindex(trials)) |> JSON.json |> write("$results_path/viz/$wid.json")
end

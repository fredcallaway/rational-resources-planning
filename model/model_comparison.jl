using Distributed
using StatsBase
using Glob
using CSV
using DataFrames

isempty(ARGS) && push!(ARGS, "exp1")
include("conf.jl")

@everywhere include("base.jl")
mkpath(results_path)
FOLDS = 5
CV_METHOD = :random  # :stratified
OPT_METHOD = :bfgs  # :samin


# %% ==================== LOAD DATA ====================

all_trials = load_trials(EXPERIMENT) |> OrderedDict |> sort!
flat_trials = flatten(values(all_trials));

println(length(flat_trials), " trials")
all_data = all_trials |> values |> flatten |> get_data;

@assert length(unique(hash.(flat_trials))) == length(flat_trials)
@assert length(unique(hash.(all_data))) == length(all_data)

# %% ==================== WRITE MDPS ====================
# this only applies if we changed the MDP when loading it (e.g. adding/removing expand_only)
mdps = unique(getfield.(flat_trials, :m))

for m in mdps
    f = "mdps/base/$(id(m))"
    serialize(f, m)
    println("Wrote ", f)
end

# %% ==================== SOLVE MDPS AND PRECOMPUTE Q LOOKUP TABLE ====================

include("solve.jl")
todo = write_mdps()
if !isempty(todo)
    println("Solving $(length(todo)) mdps....")
    @time do_jobs(todo)
end

include("Q_table.jl")
@time serialize("$base_path/Q_table", make_Q_table(all_data));

# %% ==================== LOAD MODEL CODE ====================

@everywhere include("models.jl")

MODELS = [
    RandomSelection,
    Optimal,
    OptimalPlus,
    Heuristic{:BestFirst},
    Heuristic{:BestFirstNoBestNext},
    # Heuristic{:DepthFirst},
    # Heuristic{:BreadthFirst},

    # Heuristic{:BestFirstNoSatisfice},
    # Heuristic{:BestFirstNoBestNext},
    # Heuristic{:BestFirstNoDepthLimit},
    # Heuristic{:FullNoSatisfice},
    # Heuristic{:FullNoBestNext},
    # Heuristic{:FullNoDepthLimit},
]
serialize("$base_path/models", MODELS)

# %% ==================== FIT MODELS TO FULL DATASET ====================

if false
    @everywhere include("likelihood.jl")
    @sync begin
        @spawnat 2 @time fit(Heuristic{:BreadthFirst}, all_trials |> values |> first)
        @spawnat 3 @time fit(Optimal, all_trials |> values |> first)
        @spawnat 4 @time fit(OptimalPlus, all_trials |> values |> first)
        @spawnat 5 fit(RandomSelection, all_trials |> values |> first)
    end
end

# %% --------
full_fits = let
    full_jobs = Iterators.product(values(all_trials), MODELS);
    @time full_fits = pmap(full_jobs) do (trials, M)
        model, nll = fit(M, trials; method=OPT_METHOD)
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
    
    nll = getfield.(full_fits, :nll)
    total = sum(nll; dims=1)
    best_model = [p.I[2] for p in argmin(nll; dims=2)]
    n_fit = counts(best_model, 1:length(MODELS))
    println("Model                  Likelihood   Best Fit")
    for i in eachindex(MODELS)
        @printf "%-22s       %4d         %d\n" name(MODELS[i]) total[i] n_fit[i]
    end

    full_fits
end;



    
# %% ==================== CROSS VALIDATION ====================

using Random: randperm

function kfold_splits(n, k)
    @assert (n / k) % 1 == 0  # can split evenly
    x = Dict(
        :random => randperm(n),
        :stratified => 1:n
    )[CV_METHOD]

    map(1:k) do i
        test = x[i:k:n]
        (train=setdiff(1:n, test), test=test)
    end
end

n_trial = length(all_trials |> values |> first)
folds = kfold_splits(n_trial, FOLDS)
cv_jobs = Iterators.product(values(all_trials), MODELS, folds);

cv_fits = let
    @time cv_fits = pmap(cv_jobs) do (trials, M, fold)
        try
            model, train_nll = fit(M, trials[fold.train]; method=OPT_METHOD)
            (model=model, train_nll=train_nll, test_nll=-logp(model, trials[fold.test]))
        catch e
            println("Error fitting $M to $(trials[1].wid):  $e")
            rethrow(e)
            # (model=model, nll=NaN)
        end
    end
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

    # Sum over the folds
    test_nll = sum(getfield.(cv_fits, :test_nll); dims=3) |> dropdims(3);
    train_nll = sum(getfield.(cv_fits, :train_nll); dims=3) |> dropdims(3);
    train_nll ./= (FOLDS - 1);  # each trial is counted this many times

    # Sum over participants
    total_train = sum(train_nll; dims=1)
    total_test = sum(test_nll; dims=1)

    best_model = [p.I[2] for p in argmin(test_nll; dims=2)];
    n_fit = counts(best_model, 1:length(MODELS))

    println("Model                   Train NLL   Test NLL    Best Fit")
    for i in eachindex(MODELS)
        @printf "%-22s  %4d  %10d  %8d\n" name(MODELS[i]) total_train[i] total_test[i] n_fit[i]
    end
    cv_fits
end;



# %% ==================== SIMULATION ====================

include("simulate.jl")
@everywhere all_trials = $all_trials
@everywhere full_fits = $full_fits
pmap(run_simulations, 1:length(all_trials))
sims = process_simulations();
run(`du -h $results_path/simulations.csv`);

# %% --------

group(sims, x->x.model)




# %% ==================== SAVE MODEL PREDICTIONS ====================

@memoize function get_fold(i::Int)
    # folds are identified by their first test trial index
    first(f for f in folds if i in f.test).test[1]
end
get_fold(t::Trial) = get_fold(t.i)

fit_lookup = let
    ks = map(cv_jobs) do (trials, M, fold)
        trials[1].wid, M, fold.test[1]
    end
    @assert length(ks) == length(cv_fits)
    Dict(zip(ks, getfield.(cv_fits, :model)))
end

function get_model(M::Type, t::Trial)
    fit_lookup[t.wid, M, get_fold(t)]
end

function get_preds(M::Type, t::Trial)
    model = get_model(M, t)
    map(get_data(t)) do d
        action_dist(model, d)
    end
end

function get_params(M::Type, t::Trial)
    model = get_model(M, t)
    Dict(fn => getfield(model, fn) for fn in fieldnames(typeof(model)))
end

function get_logp(M::Type, d::Datum)
    model = get_model(M, t)
    log(action_dist(model, d.t.m, d.b)[d.c+1])
end

# %% --------


click_features(d) = (
    wid=d.t.wid,
    i=d.t.i,
    b=d.b,
    c=d.c,
    p_rand=1/sum(allowed(d.t.m, d.b)),
    predictions = Dict(name(M) => action_dist(get_model(M, d.t), d) for M in MODELS),
    n_revealed=sum(observed(d.b)) - 1,
    term_reward=term_reward(d.t.m, d.b)
)

click_features.(all_data) |> JSON.json |> write("$results_path/click_features.json")


# %% --------
term_reward(t::Trial) = term_reward(t.m, t.bs[end])

trial_features(t::Trial) = (
    wid=t.wid,
    i=t.i,
    term_reward=term_reward(t),
)

trial_features.(flat_trials) |> JSON.json |> write("$results_path/trial_features.json")

# %% --------
function demo_trial(t)
    (
        stateRewards = t.bs[end],
        demo = (
            clicks = t.cs[1:end-1] .- 1,
            path = t.path .- 1,
            predictions = Dict(name(M) => get_preds(M, t) for M in MODELS),
            parameters = Dict(name(M) => get_params(M, t) for M in MODELS)
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
    demo_trial.(trials) |> JSON.json |> write("$results_path/viz/$wid.json")
end

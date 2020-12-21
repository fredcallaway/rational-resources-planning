using Distributed
using StatsBase
using Glob
using CSV
using DataFrames
using ProgressMeter

println("Running model comparison for ", ARGS[1])
include("conf.jl")

@everywhere include("base.jl")

using Random
Random.seed!(RANDOM_SEED)

mkpath("$base_path/fits/full")
mkpath("$base_path/fits/cv")
mkpath("$base_path/fits/group")

# %% ==================== LOAD DATA ====================
all_trials = load_trials(EXPERIMENT) |> OrderedDict |> sort!
flat_trials = flatten(values(all_trials));
println(length(flat_trials), " trials")
all_data = all_trials |> values |> flatten |> get_data;

@assert length(unique(hash.(flat_trials))) == length(flat_trials)
@assert length(unique(hash.(all_data))) == length(all_data)

# %% ==================== LOAD MODEL CODE ====================

@everywhere include("models.jl")

MODELS = eval(QUOTE_MODELS)

# %% ==================== TEST CROSS VALIDATION ====================

function run_cross_validation(RANDOM_SEED)

    function kfold_splits(n, k)
        @assert (n / k) % 1 == 0  # can split evenly
        x = Dict(
            :random => randperm(MersenneTwister(RANDOM_SEED), n),  # seed again just to be sure!
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

    cv_fits = @showprogress pmap(cv_jobs) do (trials, M, fold)
        wid = trials[1].wid; mname = name(M); fold_i = fold.test[1]
        # file = "$base_path/fits/cv/$mname-$wid-$fold_i"
        # if isfile(file)
        #     # println("$file already exists, skipping.")
        #     result = deserialize(file)
        #     @assert result.fold == fold
        #     return result
        # end
        try
            model, train_nll = fit(M, trials[fold.train]; method=OPT_METHOD)
            result = (model=model, train_nll=train_nll, test_nll=-logp(model, trials[fold.test]), fold=fold)
            # serialize(file, result)
            return result
        catch e
            println("Error fitting $mname to $wid on fold $fold_i:  $e")
            return missing
            # (model=model, nll=NaN)
        end
    end
    # serialize("$base_path/cv_fits", cv_fits)

    # function cv_table(M)
    #     mi = findfirst(MODELS .== M)
    #     mapmany(enumerate(keys(all_trials))) do (wi, wid)
    #         map(1:FOLDS) do fi
    #             x = cv_fits[wi, mi, fi]
    #             (wid=wid, model=name(M), fold=fi, train_nll=x.train_nll, test_nll=x.test_nll, namedtuple(x.model)...)
    #         end
    #     end |> DataFrame
    # end

    # mkpath("$results_path/mle")
    # for M in MODELS
    #     cv_table(M) |> CSV.write("$results_path/mle/$(name(M))-cv.csv")
    # end

    # Sum over the folds
    test_nll = sum(getfield.(cv_fits, :test_nll); dims=3) |> dropdims(3);
    train_nll = sum(getfield.(cv_fits, :train_nll); dims=3) |> dropdims(3);
    train_nll ./= (FOLDS - 1);  # each trial is counted this many times

    # Sum over participants
    total_train = sum(train_nll; dims=1)
    total_test = sum(test_nll; dims=1)

    best_model = [p.I[2] for p in argmin(test_nll; dims=2)];
    n_fit = counts(best_model, 1:length(MODELS))

    # println("Model                   Train NLL   Test NLL    Best Fit")
    # for i in eachindex(MODELS)
    #     @printf "%-22s  %4d  %10d  %8d\n" name(MODELS[i]) total_train[i] total_test[i] n_fit[i]
    # end
    cv_fits
end;

mkpath("$base_path/test_cv")
cv_results = map(1:5) do seed
    println("\n\n\n" * "="^30 * "  " * string(seed) * "  " * "="^30 * "\n")
    cv_fits = run_cross_validation(seed)
    serialize("$base_path/test_cv/$seed", cv_fits)
end

# %% --------
cv_results = map(1:5) do seed
    deserialize("$base_path/test_cv/$seed")
end

cv_fits = cv_results[1];

all_test, all_nfit = map(cv_results) do cv_fits
    # Sum over the folds
    test_nll = sum(getfield.(cv_fits, :test_nll); dims=3) |> dropdims(3);
    train_nll = sum(getfield.(cv_fits, :train_nll); dims=3) |> dropdims(3);
    train_nll ./= (FOLDS - 1);  # each trial is counted this many times

    # Sum over participants
    total_train = sum(train_nll; dims=1)
    total_test = sum(test_nll; dims=1)

    best_model = [p.I[2] for p in argmin(test_nll; dims=2)];
    n_fit = counts(best_model, 1:length(MODELS))

    total_test[:], n_fit
end  |> invert |> map(combinedims)


# (models = MODELS, test_likelihood=)
# JSON.json |> write("$results_path/predictions.json")

all_nfit
rank = sortperm(sum(all_test; dims=2)[:])
X = (all_test .- all_test[1:1, :])
X[rank, :]

all_nfit[rank, :]
MODELS[rank]
# %% --------

all_nfit

is_best = map(MODELS) do m
    startswith(name(m), "Best")
end

sum(all_nfit .* is_best; dims=1)



# all_test .- all_test[1:1, :]



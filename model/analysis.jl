using Distributed
isempty(ARGS) && push!(ARGS, "cogsci-1")
include("conf.jl")
@everywhere begin
    using Glob
    using Serialization
    using CSV
    include("base.jl")
    include("models.jl")
    # include("simulation.jl")
end

@everywhere results_path = "$results/$EXPERIMENT"
mkpath(results_path)
mkpath("$base_path/fits")
mkpath("$base_path/sims")

# %% ==================== Load data ====================

all_trials = load_trials(EXPERIMENT);
flat_trials = flatten(values(all_trials));
println(length(flat_trials), " trials")
all_data = all_trials |> values |> flatten |> get_data;

# %% ====================  ====================

function define_model(prefs...)
    Model(collect(prefs), fill(NaN, length(prefs)), NaN)
end
models = [
    define_model(Optimal(NaN), Expansion()),
    define_model(BestFirst(), Expansion(), Pruning(NaN), Satisficing(NaN)),
    # define_model(Optimal(NaN), Expansion(), Pruning(NaN), Satisficing(NaN)),
    # define_model(MetaGreedy(NaN), Expansion(), Pruning(NaN), Satisficing(NaN)),
]

fits = map(models) do model
    # println(model)
    @time fits = pmap(pairs(load_trials(EXPERIMENT))) do wid, trials
        wid => fit(model, trials)
    end |> OrderedDict
    mapreduce(+, values(fits), values(all_trials)) do model, trials
        logp(model, trials)
    end |> println
    fits
end


# %% ====================  ====================
base = MultiPref2([BestFirst(), Pruning(NaN), Satisficing(NaN)], ones(3))
fits = pmap(pairs(load_trials(EXPERIMENT))) do wid, trials
    wid => fit_model(base, trials)
end |> OrderedDict

# %% ====================  ====================

mapreduce(+, values(fits), values(all_trials)) do em, trials
    logp(em, trials)
end

# %% ==================== Fit models ====================

function write_fit(model, biased, path; parallel=true)
    fits, t = @timed fit_model(model; biased=biased, parallel=parallel)
    serialize(path, fits)
    println("Wrote $path ($(round(t)) seconds)")
    return fits
end

function get_fits(models; overwrite=false)
    bias_opts = FIT_BIAS ? [false, true] : [false]
    combos = Iterators.product(models, bias_opts)
    asyncmap(combos; ntasks=3) do (model, biased)
        k = biased ? :biased : :default
        path = "$base_path/fits/$model-$k"
        if isfile(path) && !overwrite
            fits = deserialize(path)
        else
            fits = write_fit(model, biased, path; parallel=(model != Optimal))
        end
        (model, k) => fits
    end |> Dict
end


models = [Optimal, MetaGreedy, BestFirst, Random]
fits = get_fits(models)

# get_fits([Random]; overwrite=true)


# %% ==================== Likelihood ====================

function total_logp(fits::OrderedDict)
    mapreduce(+, values(fits)) do pf
        pf.logp
    end
end

let
    println("---- Total likelihood ----")
    bias_opts = FIT_BIAS ? [:default, :biased] : [:default]
    for k in bias_opts
        println("  ",k)
        for m in models
            lp = mapreduce(+, values(fits[(m, k)])) do pf
                pf.logp
            end
            println("    ", m, "  ", round(Int, lp))
        end
    end
end


# %% ==================== Features ====================

@everywhere function descrybe(d::Datum; skip_logp=true)
    (
        map="themap",
        wid=d.t.wid,
        logp=skip_logp ? NaN : logp(fits[Optimal, :default][d.t.wid], d),
        n_revealed=sum(observed(d.b)) - 1,
        is_term=d.c == TERM,
        term_reward=term_reward(d.t.m, d.b)
    )
end


using CSV
mkpath("$results_path/features")
map(descrybe, all_data) |> CSV.write("$results_path/features/Human.csv")


# %% ==================== Simulations ====================
sims = map(flat_trials) do t
    repeatedly(50) do
        simulate(bf_fits[t.wid], t.m)
    end
end |> flatten |> CSV.write("$results_path/features/$(dd[1].wid).csv")


# %% ====================  ====================
# mkpath("$base_path/sims")
# pmap(write_sim, readdir("$base_path/fits"));

@everywhere function write_features(model_id)
    path = "$results_path/features/$model_id.csv"
    # sims = deserialize("$base_path/sims/$model_id");
    sims = write_sim(model_id)
    map(get_data(flatten(sims))) do d
        descrybe(d; skip_logp=true)
    end |> CSV.write(path)
    println("Wrote $path")
end

map(write_features, readdir("$base_path/fits"));


# %% ==================== Map info ====================

map(first(values(all_trials))) do t
    m = MetaMDP(t, NaN)
    (
        map=t.map,
        shortest_path=minimum(map(length, paths(m))),
        n_node=length(initial_belief(m)) - 1,
    )
end |> unique |> CSV.write("$results_path/maps.csv")

# %% ==================== Expansion rate ====================

map(all_data) do d
    d.c == TERM && return missing
    observed(d.b, d.c) && error("Nope")
    has_observed_parent(d.t.graph, d.b, d.c)
end |> skipmissing |> collect |> mean

# %% ====================  ====================
skips = filter(all_data) do d
    d.c == TERM && return false
    !has_observed_parent(d.t.graph, d.b, d.c)
end

skips[1].b
skips[1].c








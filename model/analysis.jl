using Distributed
isempty(ARGS) && push!(ARGS, "cogsci-1")
@everywhere begin
    using Glob
    using Serialization
    using Optim
    using CSV
    include("base.jl")
    include("models.jl")
    include("simulation.jl")
    results_path = "results/$EXPERIMENT"
end


# %% ==================== Load data ====================

all_trials = load_trials(EXPERIMENT);
flat_trials = flatten(values(all_trials));
println(length(flat_trials), " trials")
all_data = all_trials |> values |> flatten |> get_data;

# %% ==================== Fit models ====================

mkpath("$base_path/fits")

function write_fit(model, biased, path)
    fits, t = @timed fit_model(model; biased=biased)
    serialize(path, fits)
    println("Wrote $path ($(round(t)) seconds)")
    return fits
end

function get_fits(models)
    combos = Iterators.product(models, [true, false])
    asyncmap(combos; ntasks=3) do (model, biased)
        k = biased ? :biased : :default
        path = "$base_path/fits/$model-$k"
        if isfile(path)
            fits = deserialize(path)
        else
            fits = write_fit(model, biased, path)
        end
        (model, k) => fits
    end |> Dict
end



models = [Optimal, MetaGreedy, BestFirst, Random]
fits = get_fits(models)

for mid in readdir("$base_path/fits")
    @spawn write_sim(mid)
end


# %% ==================== Likelihood ====================

function total_logp(fits::OrderedDict)
    mapreduce(+, values(fits)) do pf
        pf.logp
    end
end

let
    println("---- Total likelihood ----")
    for k in [:biased, :default]
        println("  ",k)
        for m in models
            lp = mapreduce(+, values(fits[(m, k)])) do pf
                pf.logp
            end
            println("    ", m, "  ", round(Int, lp))
        end
    end
end

using TypedTables
using CSV

like_tbl = map(all_data) do d
    map(collect(fits)) do ((mc, k), pfits)
        (
            model=string(mc),
            biased=(k==:biased),
            logp=logp(pfits[d.t.wid], d),
            map=d.t.map,
            wid=d.t.wid,
            n_revealed=sum(observed(d.b)) - 1,
            c=d.c,
        )
    end
end |> flatten |> Table;

like_tbl |> CSV.write("$results_path/likelihoods.csv")


# %% ==================== Features ====================

@everywhere function descrybe(d::Datum; skip_logp=false)
    m = MetaMDP(d.t, NaN)
    (
        map=d.t.map,
        wid=d.t.wid,
        logp=skip_logp ? NaN : logp(fits[Optimal, :biased][d.t.wid], d),
        n_revealed=sum(observed(d.b)) - 1,
        is_term=d.c == TERM,
        term_reward=term_reward(m, d.b)
    )
end

using CSV
mkpath("$results_path/features")
map(descrybe, all_data) |> CSV.write("$results_path/features/Human.csv")


# %% ==================== Simulations ====================

# mkpath("$base_path/sims")
# pmap(write_sim, readdir("$base_path/fits"));

@everywhere function write_features(model_id)
    path = "$results_path/features/$model_id.csv"
    # sims = deserialize("$base_path/sims/$model_id");
    sims = write_sim(model_id)
    map(get_data(flatten(sims))) do d
        descrybe(d; skip_logp=true)
    end |> CSV.write(path)
    println("Wrote $results_path/features/$model_id.csv")
end

pmap(write_features, readdir("$base_path/fits"));


# %% ==================== Map info ====================

map(first(values(all_trials))) do t
    m = MetaMDP(t, NaN)
    (
        map=t.map,
        shortest_path=minimum(map(length, paths(m))),
        n_node=length(initial_belief(m)) - 1,
    )
end |> unique |> CSV.write("$results_path/maps.csv")



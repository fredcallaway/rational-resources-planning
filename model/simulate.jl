using StatsBase
using Distributed
using Glob
using CSV
using DataFrames
include("conf.jl")

@everywhere include("base.jl")
@everywhere include("models.jl")
@everywhere FORCE = false
mkpath("$base_path/sims/")


@everywhere function purify(model::OptimalPlus)
    OptimalPlus{:Pure,Float64}(model.cost, 1e5, 1e5, 0., 0.)
end

@everywhere function run_simulation(model, wid, mdps; n_repeat=10)
    model_wid = name(model) * "-" * wid
    file = "$base_path/sims/$model_wid"
    if isfile(file) && !FORCE
        println(file, " already exists. Skipping.")
    end
    GC.gc()  # minimize risk of memory overload
    sims = map(repeat(mdps, n_repeat)) do m
        simulate(model, m; wid=model_wid)
    end
    serialize("$base_path/sims/$model_wid", sims)
    println("Wrote $base_path/sims/$model_wid")

    if name(model) == "OptimalPlus"
        run_simulation(purify(model), wid, mdps)
    # elseif name(model) == "OptimalPlusPure"
    #     pure_id = "OptimalPlusPure-$(model.cost)"
    #     cp("$base_path/sims/$model_wid", "$base_path/sims/$pure_id")
    end
end


if basename(PROGRAM_FILE) == basename(@__FILE__)   
    all_trials = load_trials(EXPERIMENT) |> OrderedDict |> sort!
    full_fits = deserialize("$base_path/full_fits")

    jobs = mapmany(enumerate(pairs(all_trials))) do (i, (wid, trials))
        mdps = [t.m for t in trials]
        map(full_fits[i, :]) do fit
            fit.model, wid, mdps
        end
    end
    # pmap(COSTS) do cost
    #     model = Optimal(cost, 1e5, 0.)
    #     run_simulate(model, "cost$cost")
    # end
    if length(ARGS) >= 2 && ARGS[2] == "optimal"
        filter!(jobs) do (model, )
            model isa OptimalPlus
        end
    elseif length(ARGS) >= 2 && ARGS[2] == "nonoptimal"
        filter!(jobs) do (model, )
            !(model isa OptimalPlus)
        end
    end

    pmap(x->run_simulation(x...), jobs)

end
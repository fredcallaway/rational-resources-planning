model_sims = map([name.(MODELS); "OptimalPlusPure"]) do mname
    mname => map(collect(keys(all_trials))) do wid
        deserialize("$base_path/sims/$mname-$wid")
    end
end |> Dict;

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


# f = first(glob("$base_path/sims/Optimal-cost*"))
# group_fits = deserialize("$base_path/group_fits")
# group_fits["Best_Satisfice_BestNext_DepthLimit_Prune"]
# group_fits[1]
mname, sims = model_sims |> first
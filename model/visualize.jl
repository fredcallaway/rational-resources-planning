# %% ==================== GENERATE VISUALIZATION JSON ====================

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

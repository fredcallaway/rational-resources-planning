isempty(ARGS) && push!(ARGS, "exp4")
include("conf.jl")
include("base.jl")
include("models.jl")

all_trials = load_trials(EXPERIMENT) |> OrderedDict |> sort!
flat_trials = flatten(values(all_trials));
all_data = all_trials |> values |> flatten |> get_data;

MODELS = eval(QUOTE_MODELS)

mkpath("$results_path/stats")
function write_tex(name, x)
    f = "$results_path/stats/$name.tex"
    println(x, " > ", f)
    write(f, string(x, "\\unskip"))
end
write_pct(name, x; digits=1) = write_tex(name, string(round(100 * x; digits=digits), "\\%"))

# %% --------

model_sims = map([name.(MODELS); "OptimalPlusPure"]) do mname
    mname => map(collect(keys(all_trials))) do wid
        deserialize("$base_path/sims/$mname-$wid")
    end
end |> Dict;

# %% ==================== trial features ====================

path_loss(t::Trial) = term_reward(t.m, t.bs[end]) - path_value(t.m, t.bs[end], t.path)
term_reward(t::Trial) = term_reward(t.m, t.bs[end])
first_revealed(t) = t.cs[1] == 0 ? NaN : t.bs[end][t.cs[1]]

leaves(m::MetaMDP) = Set([i for (i, children) in enumerate(m.graph) if isempty(children)])
get_leaves = Dict(id(m) => leaves(m) for m in unique(getfield.(flat_trials, :m)))
is_backward(t::Trial) = t.cs[1] in get_leaves[id(t.m)]

function second_same(t)
    t.m.expand_only || return NaN
    length(t.cs) < 3 && return NaN
    c1, c2 = t.cs
    c2 in t.m.graph[c1] && return 1.
    @assert c2 in t.m.graph[1]
    return 0.
end

trial_features(t::Trial) = (
    wid=t.wid,
    i=t.i,
    term_reward=term_reward(t),
    first_revealed = first_revealed(t),
    second_same=second_same(t),
    backward=is_backward(t),
    n_click=length(t.cs)-1,
    path_loss=path_loss(t),
)

trial_features.(flat_trials) |> JSON.json |> writev("$results_path/trial_features.json");

for (nam, sims) in pairs(model_sims)
    f = "$results_path/$nam-trial_features.json"
    sims |> flatten .|> trial_features |> JSON.json |> writev(f)
end

# %% ==================== click features ====================

_bf = Heuristic{:BestFirst,Float64}(10., 0.,0., 0., 0., 0., -1e10, 0.)
function is_bestfirst(d::Datum)
    (d.c != TERM) && action_dist(_bf, d)[d.c+1] > 1e-2
end

function click_features(d)
    m = d.t.m; b = d.b;
    pv = path_values(m, b)
    mpv = [max_path_value(m, b, p) for p in paths(m)]
    best = argmax(pv)
    max_path = maximum(mpv)
    mpv[best] = -Inf
    max_competing = maximum(mpv)
    (
        wid=d.t.wid,
        i=d.t.i,
        expanding = has_observed_parent(d.t.m.graph, d.b, d.c),
        # depth = d.c == TERM ? -1 : depth(m.graph, d.c),
        is_term = d.c == TERM,
        is_best=is_bestfirst(d),
        n_revealed=sum(observed(d.b)) - 1,
        term_reward=pv[best],
        max_path=max_path,
        max_competing=max_competing,
        best_next=best_vs_next(m, b),
    )
end
click_features.(all_data) |> JSON.json |> writev("$results_path/click_features.json");

for (nam, sims) in pairs(model_sims)
    f = "$results_path/$nam-click_features.json"
    sims |> flatten |> get_data .|> click_features |> JSON.json |> writev(f)
end

# # %% ==================== depth curve ====================
cummax(xs) = accumulate(max, xs)

function cummaxdepth(t)
    map(t.cs[1:end-1]) do c
        depth(t.m, c)
    end |> cummax
end

function depth_curve(trials)
    rows = mapmany(trials) do t
        cs = t.cs[1:end-1]
        dpth = map(cs) do c
            depth(t.m, c)
        end
        click = eachindex(cs)
        map(zip(click, dpth, cummax(dpth))) do x
            (t.wid, x...)
        end
    end
    Dict(["wid", "click", "depth", "cumdepth"] .=> invert(rows))
end

depth_curve(flat_trials) |> JSON.json |> writev("$results_path/depth_curve.json");

for (nam, sims) in pairs(model_sims)
    # M = MODELS[5]; sims = model_sims[M];
    sims |> flatten |> depth_curve |> JSON.json |> writev("$results_path/$nam-depth_curve.json")
end

# # %% ==================== optimal visualization ====================

# function viz_sim(t::Trial)
#     (
#         stateRewards = t.bs[end],
#         demo = (
#             clicks = t.cs[1:end-1] .- 1,
#             path = t.path .- 1,
#             # parameters = Dict(name(M) => get_params(M, t) for M in MODELS)
#             # predictions = Dict(name(M) => get_preds(M, t) for M in MODELS),
#         )
#     )
# end

# map(zip(COSTS, opt_sims)) do (cost, sim)
#     name = "Optimal-$cost"
#     viz_sim.(sim[1:50]) |> json |> write("$results_path/viz/$name.json")
#     (cost = cost, name=name)
# end |> json |> write("$results_path/viz/optimal-table.json")

# # %% ==================== best first ====================

# _bf = Heuristic{:BestFirst,Float64}(10., 0., 0., 0., 0., -1e10, 0.)
# is_bestfirst(d::Datum) = action_dist(_bf, d)[d.c+1] > 1e-2

# function best_first_rate(trials)
#     map(get_data(trials)) do d
#         d.c == TERM && return missing
#         is_bestfirst(d)
#     end |> skipmissing |> (x -> isempty(x) ? NaN : mean(x))
# end

# opt_bfr = map(best_first_rate, opt_sims)
# (
#     optimal = Dict(zip(COSTS, opt_bfr)),
#     human = valmap(best_first_rate, all_trials)
# ) |> json |> write("$results_path/bestfirst.json")

# is_bestfirst(all_data)



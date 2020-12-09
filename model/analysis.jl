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

# %% --------
map(MODELS) do m
    name(m) => length(bounds(default_space(m))[1])
end |> Dict |> JSON.json |> writev("$results_path/param_counts.json")

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

_bf = create_model(Heuristic{:Best}, [1e5, -1e5, 0], (), default_space(Heuristic{:Best}))
@everywhere function is_bestfirst(d::Datum)
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

# %% ==================== expansion ====================
if !flat_trials[1].m.expand_only
    mle_cost = let
        M =  OptimalPlus{:Expand,Float64}
        fits = first.(deserialize("$base_path/full_fits"))
        opt_fits = filter(x->x isa M, fits)
        Dict(keys(all_trials) .=> getfield.(opt_fits, :cost))
    end

    mle_qs(d::Datum) = Q_TABLE[hash(d)][mle_cost[d.t.wid]]

    function expansion_value(d::Datum)
        qs = mle_qs(d)[2:end]
        cs = eachindex(d.b)
        expanding = map(cs) do c
            has_observed_parent(d.t.m.graph, d.b, c)
        end
        (
            q_expand = maximum(qs[expanding]),
            q_jump = maximum(qs[.!expanding]),
            q_human = qs[d.c],
            expand = expanding[d.c],
            wid = d.t.wid
        )
    end

    all_data |> filter(d->d.c != TERM) .|> expansion_value |> JSON.json |> writev("$results_path/expansion.json")

    for (nam, sims) in pairs(model_sims)
        f = "$results_path/$nam-expansion.json"
        sims |> flatten |> get_data  |> filter(d->d.c != TERM) .|> expansion_value |> JSON.json |> writev(f)
    end
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
using Glob
@everywhere function best_first_rate(trials)
    map(get_data(trials)) do d
        d.c == TERM && return missing
        is_bestfirst(d)
    end |> skipmissing |> (x -> isempty(x) ? NaN : mean(x))
end

@everywhere function compute_bfr(f)
    cost, mid = match(r"cost(.*)-(.*)", f).captures
    cost = parse(Float64, cost)
    trials = deserialize(f);
    (
        mdp = mid,
        cost = cost,
        n_click = mean(length(t.cs) for t in trials) - 1,
        best_first = best_first_rate(trials),
    )
end

using ProgressMeter

res = map(glob("$base_path/sims/Optimal-cost*")) do f
    cost, mid = match(r"cost(.*)-(.*)", f).captures
    cost = parse(Float64, cost)
    trials = deserialize(f);
    (
        mdp = mid,
        cost = cost,
        n_click = mean(length(t.cs) for t in trials) - 1,
        best_first = best_first_rate(trials),
    )
end
res |> json |> write("$results_path/bestfirst.json")




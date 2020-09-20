using Distributed
isempty(ARGS) && push!(ARGS, "exp4")
include("conf.jl")
@everywhere include("base.jl")
@everywhere include("models.jl")

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
)

trial_features.(flat_trials) |> JSON.json |> writev("$results_path/trial_features.json")

# %% --------
for (nam, sims) in pairs(model_sims)
    nam != "OptimalPlusPure" && continue
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

# %% --------
thin(sims) = [s[1:100] for s in sims]
for (nam, sims) in pairs(model_sims)
    # M = MODELS[5]; sims = model_sims[M];
    f = "$results_path/$nam-click_features.json"
    sims |> thin |> flatten |> get_data .|> click_features |> JSON.json |> writev(f)
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
# %% --------
for (nam, sims) in pairs(model_sims)
    # M = MODELS[5]; sims = model_sims[M];
    sims |> thin |> flatten |> depth_curve |> JSON.json |> writev("$results_path/$nam-depth_curve.json")
end


# %% ==================== group simulations and features ====================

function run_simulations(trials::Vector{Trial}, model::AbstractModel; n_repeat=10)
    map(repeat(trials, n_repeat)) do t
        model_wid = name(model) * "-" * t.wid
        simulate(model, t.m; wid=model_wid)
    end
end

group_fits = deserialize("$base_path/group_fits")

@time group_sims = map(group_fits) do (model, _)
    name(model) => run_simulations(flat_trials, model)
end |> Dict

for (nam, sims) in pairs(group_sims)
    # M = MODELS[5]; sims = model_sims[M];
    f = "$results_path/group-$nam-click_features.json"
    sims |> get_data .|> click_features |> JSON.json |> write(f)
    println("Wrote $f")
end

# %% ==================== path choice ====================

path_loss(t::Trial) = term_reward(t.m, t.bs[end]) - path_value(t.m, t.bs[end], t.path)
pl = path_loss.(flat_trials)
write_pct("path_loss", mean(pl .== 0))

# %% --------

function viz_sim(t::Trial)
    (
        stateRewards = t.bs[end],
        demo = (
            clicks = t.cs[1:end-1] .- 1,
            path = t.path .- 1,
            # parameters = Dict(name(M) => get_params(M, t) for M in MODELS)
            # predictions = Dict(name(M) => get_preds(M, t) for M in MODELS),
        )
    )
end

map(zip(COSTS, opt_sims)) do (cost, sim)
    name = "Optimal-$cost"
    viz_sim.(sim[1:50]) |> json |> write("$results_path/viz/$name.json")
    (cost = cost, name=name)
end |> json |> write("$results_path/viz/optimal-table.json")


# %% ==================== best first ====================

_bf = Heuristic{:BestFirst,Float64}(10., 0., 0., 0., 0., -1e10, 0.)
is_bestfirst(d::Datum) = action_dist(_bf, d)[d.c+1] > 1e-2

function best_first_rate(trials)
    trials |>
    get_data |> 
    @filter(_.c != TERM) |> 
    @map(is_bestfirst) |> 
    (x -> isempty(x) ? NaN : mean(x))
end

opt_bfr = map(best_first_rate, opt_sims)
(
    optimal = Dict(zip(COSTS, opt_bfr)),
    human = valmap(best_first_rate, all_trials)
) |> json |> write("$results_path/bestfirst.json")

is_bestfirst(all_data)




# %% ==================== adaptive satisficing ====================
include("binning.jl")
etrs = -30.:5:30
bins = Dict(zip(etrs, 1:100))

function termination_matrices(trials)
    X = zeros(length(bins), 17)
    N = zeros(length(bins), 17)
    for t in trials
        for (n_click, b, c) in zip(1:100, t.bs, t.cs)
            bin = bins[term_reward(t.m, b)]
            X[bin, n_click] += (c == TERM)
            N[bin, n_click] += 1
        end
    end
    X, N
end

@time tmats = map(collect(model_sims)) do (model, sim)
    name(model) => termination_matrices(flatten(sim))
end

# TODO this doesn't exclude properly!!
h = "Human" => termination_matrices(flatten(values(all_trials)))
write("$results_path/termination.json",
      json(Dict(tmats..., h, "etrs"=>collect(etrs))))

# %% ==================== relative stopping rule ====================
include("features.jl")
vals = -60:5:60
bins = Dict(zip(vals, 1:100))

function termination_matrices(trials)
    X = zeros(length(bins), length(bins))
    N = zeros(length(bins), length(bins))
    for d in get_data(trials)
        all(observed(d.b)) && continue  # ignore forced terminations
        i = bins[term_reward(d.t.m, d.b)]
        j = bins[best_vs_next(d.t.m, d.b)]
        X[i, j] += (d.c == TERM)
        N[i, j] += 1
    end
    X, N
end

@time tmats = map(collect(model_sims)) do (model, sim)
    name(model) => termination_matrices(flatten(sim))
end


# TODO this doesn't exclude properly!
# Same problem with the simulations!
h = "Human" => termination_matrices(flatten(values(all_trials)))

write("$results_path/termination.json",
      json(Dict(tmats..., h, "etrs"=>collect(etrs))))

# function evmv(d::Datum)
#     m = d.t.m; b = d.b;
#     pv = path_values(m, b)
#     mpv = [max_path_value(m, b, p) for p in paths(m)]
#     best = argmax(pv)
#     mpv[best] = -Inf
#     pv[best], maximum(mpv)
# end
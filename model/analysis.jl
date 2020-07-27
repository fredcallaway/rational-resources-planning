using Query

isempty(ARGS) && push!(ARGS, "exp1")
include("conf.jl")

all_trials = load_trials(EXPERIMENT) |> OrderedDict |> sort!
flat_trials = flatten(values(all_trials));
all_data = all_trials |> values |> flatten |> get_data;


# @time all_sims = pmap(1:length(all_trials)) do i
    # run_simulations(i; n_repeat=50)
# end |> invert;

all_sims = map(collect(keys(all_trials))) do wid
    deserialize("$base_path/sims/$wid")
end |> invert;

model_sims = Dict(zip(MODELS, all_sims))


mkpath("../stats/$EXPERIMENT")
function write_tex(name, x)
    f = "../stats/$EXPERIMENT/$name.tex"
    println(x, " > ", f)
    write(f, string(x, "\\unskip"))
end
write_pct(name, x; digits=1) = write_tex(name, string(round(100 * x; digits=digits), "\\%"))

# %% ==================== pure optimal simulations ====================

mdps = unique(getfield.(flat_trials, :m))
N_SIM = 10000
@everywhere GC.gc()
@time opt_sims = pmap(Iterators.product(mdps, COSTS)) do (m, cost)
    V = load_V_nomem(id(mutate(m, cost=cost)))
    map(1:N_SIM) do i
        simulate(OptimalPolicy(V), "PureOptimal")
    end
end

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

# %% ==================== generate features ====================

_bf = Heuristic{:BestFirst,Float64}(10., 0., 0., 0., 0., -1e10, 0.)
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
        is_term = d.c == TERM,
        is_best=is_bestfirst(d),
        n_revealed=sum(observed(d.b)) - 1,
        term_reward=pv[best],
        max_path=max_path,
        max_competing=max_competing,
    )
end
click_features.(all_data) |> JSON.json |> write("$results_path/click_features.json")



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

function evmv(d::Datum)
    m = d.t.m; b = d.b;
    pv = path_values(m, b)
    mpv = [max_path_value(m, b, p) for p in paths(m)]
    best = argmax(pv)
    mpv[best] = -Inf
    pv[best], maximum(mpv)
end

function evmv_matrices(trials)
    X = zeros(length(bins), length(bins))
    N = zeros(length(bins), length(bins))
    for d in get_data(trials)
        b, n = evmv(d)
        i = bins[b]; j = bins[n]
        X[i, j] += (d.c == TERM)
        N[i, j] += 1
    end
    X, N
end
@time tmats = map(collect(model_sims)) do (model, sim)
    name(model) => evmv_matrices(flatten(sim))
end

# TODO this doesn't exclude properly!!
h = "Human" => evmv_matrices(flatten(values(all_trials)))

write("$results_path/evmv.json",
      json(Dict(tmats..., h, "etrs"=>collect(etrs))))

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


# %% ==================== path choice ====================

path_loss(t::Trial) = term_reward(t.m, t.bs[end]) - path_value(t.m, t.bs[end], t.path)
pl = path_loss.(flat_trials)
write_pct("path_loss", mean(pl .== 0))

# %% --------
term_reward(t::Trial) = term_reward(t.m, t.bs[end])

trial_features(t::Trial) = (
    wid=t.wid,
    i=t.i,
    term_reward=term_reward(t),
)

trial_features.(flat_trials) |> JSON.json |> write("$results_path/trial_features.json")




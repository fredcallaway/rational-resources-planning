using Query

isempty(ARGS) && push!(ARGS, "exp1")
include("conf.jl")

all_trials = load_trials(EXPERIMENT) |> OrderedDict |> sort!
flat_trials = flatten(values(all_trials));
all_data = all_trials |> values |> flatten |> get_data;

@time all_sims = map(collect(keys(all_trials))) do wid
    deserialize("$base_path/sims/$wid")
end |> invert;

MODELS = deserialize("$base_path/models")


# file = f"{self.path}/{name}.tex"
# with open(file, "w+") as f:
#     f.write(str(tex) + r"\unskip")
# print(f'wrote "{tex}" to "{file}"')


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

tmats = map(all_sims) do msim
    name = split(msim[1][1].wid, "-")[1]
    name => termination_matrices(flatten(msim))
end

# TODO this doesn't exclude properly!!
h = "Human" => termination_matrices(flatten(values(all_trials)))
write("$results_path/termination.json",
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



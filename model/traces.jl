# mixin for analysis.jl

# %% ==================== Get optimal log probabilities ====================

fit_prm = map(eachrow(opt_mle)) do row
    row.wid => (α=row.α, cost=row.cost, ε=row.ε)
end |> Dict

qd = map(all_qs) do qs
    qs.cost => qs.qs
end |> Dict;

trial_qs = map(all_trials) do wid, trials
    qs = qd[fit_prm[wid].cost][wid]
    map(trials, qs) do t, qq
        t => qq
    end
end |> flatten |> Dict


function opt_logps(t::Trial; relative=false)
    qs = trial_qs[t]
    fit = fit_prm[t.wid]
    map(qs, t.cs) do q, c
        logp(q, c, fit.α, fit.ε) - (relative ? log(p_rand(q)) : 0.)
    end
end

wid = all_trials |> keys |> first
new = mapreduce(t->sum(opt_logps(t)), +, all_trials[wid])
prev = filter(x->x.wid == wid, opt_mle).logp[1]
@assert new ≈ prev

# %% ====================  ====================

Expanding(fit::NamedTuple) = Expanding(fit.expand_bonus, fit.last_bonus)

function finite_median(xs; default=0)
    finite = [x for x in xs[2:end] if isfinite(x)]
    isempty(finite) ? default : median(finite)
end


trace = map(all_data) do d
    opt_fit = fits[Optimal, :biased][d.t.wid]
    bf_fit = fits[BestFirst, :biased][d.t.wid]
    m = make_mdp(d.t.graph, opt_fit.model.cost)
    q = collect(preferences(opt_fit.model, d))
    q .+= preferences(Expanding(opt_fit), d)

    bf = collect(preferences(bf_fit.model, d))
    bf .+= preferences(Expanding(bf_fit), d)

    # baseline = finite_median(q[2:end])
    baseline = term_reward(m, d.b)
    q .-= baseline

    (
        cost = opt_fit.model.cost,
        alpha = opt_fit.α,
        epsilon = opt_fit.ε,
        is_opt = q[d.c+1] ≈ maximum(q),
        is_bf = bf[d.c+1] ≈ maximum(bf),
        opt_logp = logp(opt_fit, d),
        bf_logp = logp(bf_fit, d),
        bf_choice = bf .== maximum(bf),
        map = d.t.map,
        b = d.b,
        c = d.c,
        # score = logp(q, d.c, fit.α, fit.ε) - log(p_rand(q)),
        q = q,
        q_loss=maximum(q) - q[d.c+1],
    )
end;



encode_b(b::Belief) = join(map(b) do x
    isnan(x) ? "X" : -Int(x)
end, "-")

encode_qs(q) = join(map(q) do x
    isfinite(x) ? round(x; digits=3) : "null"
end, ",")


js_trace = map(trace) do x
    (x..., b=encode_b(x.b), q=encode_qs(x.q))
end;
# filter!(js_trace) do x
#     5 < x.cost < 10
# end

open("real_results/trace.js", "w+") do f
    write(f, "var ACTION_TRACE = ")
    write(f, JSON.json(js_trace));
end

# %% ====================  ====================
filter(js_trace) do x
    x.map =="fantasy_map_1560976900330.png"
end |> length

# %% ====================  ====================


# %% ====================  ====================

t = trace |> @filter (_.c != 0) |> collect

t = trace[1]

trace |> @filter(_.score <0) |> @map(_.c == 0) |> mean
trace |> @filter(_.c ==0) |> @map(_.score < 0) |> mean

let
    println("Q loss for observing")
    print("Total: ")
    trace |> @filter(_.c != 0) |> @map(_.q_loss) |> sum |> println
    print("Mean: ")
    trace |> @filter(_.c != 0) |> @map(_.q_loss) |> mean |> println

    println("\nQ loss for terminating")
    print("Total: ")
    trace |> @filter(_.c == 0) |> @map(_.q_loss) |> sum |> println
    print("Mean: ")
    trace |> @filter(_.c == 0) |> @map(_.q_loss) |> mean |> println
end

# %% ==================== Write json ====================

# %% ====================  ====================
greedy_logp



using Distributed


using StatsBase
# include("base.jl")
# include("models.jl")

struct Simulator <: Policy
    model::Model
    α::Real
    ε::Real
    expand_bonus::Real
    last_bonus::Real
    t::Trial
    m::MetaMDP
end

function Simulator(fit::BiasedFit, t::Trial)
    Simulator(fit.model, fit.α, fit.ε, fit.expand_bonus, fit.last_bonus, t, MetaMDP(t, NaN))
end

function Simulator(fit::Fit, t::Trial)
    Simulator(fit.model, fit.α, fit.ε, 0., 0., t, MetaMDP(t, NaN))
end

function sample_softmax(v)
    p = softmax(α * preferences(fit.model, t, b))
    sample(eachindex(p), Weights(p))
end

function probs(pol::Simulator, b::Belief)
    prefs = preferences(pol.model, pol.t, b)
    p_soft = softmax(pol.α .* prefs)
    each_p_rand = pol.ε * p_rand(prefs)
    @. each_p_rand * (prefs > -Inf) + (1-pol.ε) * p_soft
end

(pol::Simulator)(b::Belief) = rand(Categorical(probs(pol, b))) - 1

@memoize function get_V_tbl()
    mdp_ids = readdir("$base_path/mdps/")
    V_tbl = asyncmap(mdp_ids) do i
        V = deserialize("$base_path/mdps/$i/V")
        (V.m.graph, V.m.cost) => V
    end |> Dict
end

function get_V(graph, cost)
    tbl = get_V_tbl()
    tbl[graph, cost]
end

function preferences(model::Optimal, t::Trial, b::Belief)
    V = get_V(t.graph, model.cost)
    Q(V, b)
end

function simulate(fits, t::Trial)
    pol = Simulator(fits[t.wid], t)
    bs = Belief[]
    cs = Int[]
    rollout(pol) do b, c
        push!(bs, deepcopy(b)); push!(cs, c)
    end
    mutate(t, bs=bs, cs=cs)
end

function write_sim(model_id)
    flat_trials = load_trials(EXPERIMENT) |> values |> flatten
    fits = deserialize("$base_path/fits/$model_id")
    sims, t = @timed map(flat_trials) do t
        repeatedly(50) do
            simulate(fits, t)
        end
    end
    path = "$base_path/sims/" * model_id
    serialize(path, sims)
    println("Wrote $path ($(round(t)) seconds)")
    return sims
end

# if basename(PROGRAM_FILE) == basename(@__FILE__)
#     mkpath("$base_path/sims")
#     pmap(write_sim, readdir("$base_path/fits"))
# end








using Distributed


using StatsBase
# include("base.jl")
# include("models.jl")

struct Simulator <: Policy
    model::Model
    β::Real
    ε::Real
    expand_bonus::Real
    last_bonus::Real
    t::Trial
    m::MetaMDP
end

function Simulator(fit::BiasedFit, t::Trial)
    Simulator(fit.model, fit.β, fit.ε, fit.expand_bonus, fit.last_bonus, t, MetaMDP(t, NaN))
end

function Simulator(fit::Fit, t::Trial)
    Simulator(fit.model, fit.β, fit.ε, 0., 0., t, MetaMDP(t, NaN))
end

function sample_softmax(v)
    p = softmax(β * preferences(fit.model, t, b))
    sample(eachindex(p), Weights(p))
end

function probs(pol::Simulator, b::Belief)
    prefs = preferences(pol.model, pol.t, b)
    p_soft = softmax(pol.β .* prefs)
    each_p_rand = pol.ε * p_rand(prefs)
    @. each_p_rand * (prefs > -Inf) + (1-pol.ε) * p_soft
end

(pol::Simulator)(b::Belief) = rand(Categorical(probs(pol, b))) - 1


function simulate(fits, t::Trial)
    pol = Simulator(fits[t.wid], t)
    bs = Belief[]
    cs = Int[]
    rollout(pol) do b, c
        push!(bs, deepcopy(b)); push!(cs, c)
    end
    mutate(t, bs=bs, cs=cs)
end

function simulate(sim::Simulator)
    simulate(sim, sim.t)
end

function simulate(pol::Policy)
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
    mkpath("$base_path/sims/")
    serialize(path, sims)
    println("Wrote $path ($(round(t)) seconds)")
    return sims
end

# if basename(PROGRAM_FILE) == basename(@__FILE__)
#     mkpath("$base_path/sims")
#     pmap(write_sim, readdir("$base_path/fits"))
# end








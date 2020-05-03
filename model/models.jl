# %% ==================== Base code for all models ====================

abstract type Model end

Base.length(m::Model) = 1

function preferences(model::Model, t::Trial)
    map(get_data(t)) do datum
        preferences(model, datum)
    end
end

function preferences(model::Model, d::Datum)
    preferences(model, d.t, d.b)
end

function mysoftmax(x)
    ex = exp.(x .- maximum(x))
    ex ./= sum(ex)
    ex
end

function p_rand(q::Vector{Float64})
    n_option = sum(q .> -Inf)
    1 / n_option
end

function logp(prefs::Vector{Float64}, c::Int, α, ε)
    p_soft = mysoftmax(α .* prefs)[c+1]
    if !isfinite(p_soft)
        error("Bad prefs: $prefs")
    end
    p = ε * p_rand(prefs) + (1-ε) * p_soft
    log(p)
end

function c_probs(model::Model, d::Datum, α, ε)
    prefs = preferences(model, d)
    p_soft = mysoftmax(α .* prefs)
    each_p_rand = ε * p_rand(prefs)
    @. each_p_rand * (prefs > -Inf) + (1-ε) * p_soft
end

function logp(model::Model, d::Datum, α::Float64, ε::Float64)
    prefs = preferences(model, d)
    logp(prefs, d.c, α, ε)
end

function logp(model::Model, data::Vector{Datum}, α::Float64, ε::Float64)
    mapreduce(+, data) do d
        logp(model, d, α, ε)
    end
end

function map_logp(model::Model, data::Vector{Datum}, α::Float64, ε::Float64)
    map(data) do d
        logp(model, d, α, ε)
    end
end

function fit_error_model(model::Model, data::Vector{Datum}; x0 = [0.002, 0.1], biased=false)
    biased && return fit_biased_error_model(model, data)
    lower = [1e-3, 1e-3]; upper = [10., 1.]
    all_prefs = [preferences(model, d) for d in data]
    cs = [d.c for d in data]
    opt = optimize(lower, upper, x0, Fminbox(LBFGS())) do (α, ε)
        - mapreduce(+, all_prefs, cs) do prefs, c
            logp(prefs, c, α, ε)
        end
    end
    (α=opt.minimizer[1], ε=opt.minimizer[2], logp=-opt.minimum)
end


function fit_biased_error_model(model::Model, data::Vector{Datum}; x0 = [0.002, 0.1, 0, 0])
    lower = [1e-3, 1e-3, -1e3, -1e3]; upper = [10., 1., 1e3, 1e3]
    all_prefs = [preferences(model, d) for d in data]
    expand_prefs = [preferences(Expanding(1., 0.), d) for d in data]
    last_prefs = [preferences(Expanding(0., 1.), d) for d in data]
    cs = [d.c for d in data]
    opt = optimize(lower, upper, x0, Fminbox(LBFGS())) do (α, ε, expand_bonus, last_bonus)
        - mapreduce(+, eachindex(all_prefs), cs) do i, c
            prefs = all_prefs[i] + expand_bonus * expand_prefs[i] + last_bonus * last_prefs[i]
            logp(prefs, c, α, ε)
        end
    end
    x = opt.minimizer
    (α=x[1], ε=x[2], expand_bonus=x[3], last_bonus=x[4], logp=-opt.minimum)
end

function fit_model(model_class::Type, trials; biased=false)
    data = get_data(trials)
    err_fits = map(instantiations(model_class)) do model
        fit = fit_error_model(model, data; biased=biased)
        (model=model, fit...)
    end
    best = argmax([ef.logp for ef in err_fits])
    return err_fits[best]
end

function fit_model(model_class; biased=false, parallel=true)
    mymap = parallel ? pmap : map
    mymap(pairs(load_trials(EXPERIMENT))) do wid, trials
        wid => fit_model(model_class, trials; biased=biased)
    end |> OrderedDict
end

Fit = NamedTuple{(:model, :α, :ε, :logp)}
BiasedFit = NamedTuple{(:model, :α, :ε, :expand_bonus, :last_bonus, :logp)}

function logp(fit::Fit, d::Datum)
    prefs = preferences(fit.model, d)
    logp(prefs, d.c, fit.α, fit.ε)
end

function logp(fit::BiasedFit, d::Datum)
    bias = Expanding(fit.expand_bonus, fit.last_bonus)
    prefs = preferences(fit.model, d) .+ preferences(bias, d)
    logp(prefs, d.c, fit.α, fit.ε)
end

# %% ==================== Meta Greedy ====================

struct MetaGreedy <: Model
    cost::Float64
end

instantiations(::Type{MetaGreedy}) = map(MetaGreedy, COSTS)

function preferences(model::MetaGreedy, t::Trial, b::Belief)
    m = MetaMDP(t, model.cost)
    voc1(m, b)
end

function voc1(m::MetaMDP, b::Belief, c::Int)
    c == TERM && return 0.
    !allowed(m, b, c) && return -Inf
    q = mapreduce(+, results(m, b, c)) do (p, b1, r)
        p * (term_reward(m, b1) + r)
    end
    q - term_reward(m, b)
end

voc1(m, b) = [voc1(m, b, c) for c in 0:length(b)]


# %% ==================== Best First ====================

struct BestFirst <: Model
    satisfice_threshold::Float64
    prune_threshold::Float64
end

function instantiations(::Type{BestFirst})
    vals = PRUNE_SAT_THRESHOLDS
    thresholds = filter(collect(Iterators.product(vals, vals))) do (sat, prn)
        sat > prn
    end
    map(thresholds) do (sat, prn)
        BestFirst(sat, prn)
    end
end

function best_first_value(m, b, sat_thresh, prune_thresh)
    # FIXME: how do we handle observing the start or end cities?
    nv = fill(-1e10, length(m))
    for p in paths(m)
        v = path_value(m, b, p)
        for i in p
            nv[i] = max(nv[i], v)
        end
    end
    term_r = maximum(nv)
    for i in eachindex(nv)
        if !allowed(m, b, i)
            nv[i] = -Inf
        elseif nv[i] < prune_thresh
            nv[i] = -1e20
        end
    end
    term_value = (term_r >= sat_thresh ? 1e15 : -1e15)
    [term_value; nv]
end

function preferences(model::BestFirst, t::Trial, b::Belief)
    m = MetaMDP(t, NaN)
    best_first_value(m, b, model.satisfice_threshold, model.prune_threshold)
end


# %% ==================== Expansion Bias ====================

function has_observed_parent(graph, b, c)
    any(enumerate(graph)) do (i, children)
        c in children && observed(b, i)
    end
end

function expansion_prefs(d::Datum)
    map(0:length(d.b)) do c
        c == TERM && return false
        observed(d.b, c) && return false
        has_observed_parent(d.t.graph, d.b, c)
    end
end

function last_prefs(d::Datum)
    map(0:length(d.b)) do c
        c == TERM && return false
        d.c_last == nothing && return false
        observed(d.b, c) && return false
        c in d.t.graph[d.c_last]
    end
end

struct Expanding <: Model
    expand_bonus::Float64
    last_bonus::Float64
end

function preferences(model::Expanding, d::Datum)
    model.expand_bonus * expansion_prefs(d) +
    model.last_bonus * last_prefs(d)
end

# %% ====================  ====================

struct Random <: Model
end

function preferences(model::Random, t::Trial, b::Belief)
    x = zeros(length(b)+1)
    for i in eachindex(b)
        if !isnan(b[i])
            x[i+1] = -Inf
        end
    end
    x
end

instantiations(::Type{Random}) = [Random()]

# struct Biased <: Model
#     model::Model
# end

# function preferences(biased::Biased, g::Graph, b::Belief)
#     preferences(biased.model, g, b)
# end
# %% ==================== Optimal ====================

struct Optimal <: Model
    cost::Float64
end

instantiations(::Type{Optimal}) = map(Optimal, COSTS)

@memoize function get_V_tbl()
    mdp_ids = readdir("$base_path/mdps/")
    V_tbl = asyncmap(mdp_ids) do i
        V = deserialize("$base_path/mdps/$i/V")
        (identify(V.m), V.m.cost) => V
    end |> Dict
end

function get_V(t::Trial, cost)
    tbl = get_V_tbl()
    tbl[identify(t), cost]
end

function preferences(model::Optimal, t::Trial, b::Belief)
    V = get_V(t, model.cost)
    Q(V, b)
end

# @memoize function load_qs()
#     data = load_trials(EXPERIMENT) |> values |> flatten |> get_data;
#     check = checksum(data)
#     all_qs = map(eachindex(COSTS)) do i
#         qq = deserialize("$base_path/qs/$i")
#         @assert qq.checksum == check
#         @assert qq.cost == COSTS[i]
#         qdict = map(data, qq.qs) do d, q
#             hash(d) => q
#         end |> Dict
#         qq.cost => qdict
#     end |> Dict
# end


# function preferences(model::Optimal, d::Datum)
#     load_qs()[model.cost][hash(d)]
# end

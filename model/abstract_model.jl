abstract type AbstractModel{T} end

function create_model(::Type{M}, x::Vector{T}, z, space::Space) where M where T
    xs = Iterators.Stateful(x)
    zs = Iterators.Stateful(z)
    args = map(fieldnames(M)) do fn
        spec = space[fn]
        if spec isa Tuple
            first(xs)
        elseif spec isa Vector
            first(zs)
        else
            T(spec)
        end
    end
    M{T}(args...)
end

features(::Type{T}, d::Datum) where T <: AbstractModel = features(T, d.t.m, d.b)

function preference(model::M, m::MetaMDP, b::Belief, c::Int) where M <: AbstractModel
    phi = features(M, m, b)
    preference(model, phi, c)
end

function preferences(model::M, m::MetaMDP, b::Belief) where M <: AbstractModel
    phi = features(M, m, b)
    map(0:length(b)) do c
        preference(model, phi, c)
    end
end

preference(model::AbstractModel, d::Datum) = preference(model, d.t.m, d.b, d.c)

function action_dist(model::AbstractModel, m::MetaMDP, b::Belief)
    possible = allowed(m, b)
    h = preferences(model, m, b) .+ .!possible .* -Inf
    p_rand = model.ε .* rand_prob(m, b) .* possible
    # model.ε * p_rand + (1-model.ε) * pref
    p_rand .+ (1-model.ε) .* softmax(h)
end
action_dist(model::AbstractModel, d::Datum) = action_dist(model, d.t.m, d.b)
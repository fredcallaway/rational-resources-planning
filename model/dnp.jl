using Distributions
using DataStructures: DefaultDict
using Memoize

DNP = DiscreteNonParametric

DiscreteNonParametric(x) = DNP(x, ones(length(x)) ./ length(x))

function (Base.:+)(d1::DNP, d2::DNP)::DNP
    res = DefaultDict{Float64,Float64}(0.)
    for (p1, v1) in zip(d1.p, d1.support)
        for (p2, v2) in zip(d2.p, d2.support)
            res[v1 + v2] += p1 * p2
        end
    end
    DNP(collect(keys(res)), collect(values(res)))
end

function (Base.:+)(d::DNP, x)::DNP
    DNP(d.support .+ x, d.p, )
end
(Base.:+)(x, d::DNP) = d + x

function Base.map(f, d::DNP)::DNP
    res = DefaultDict{Float64,Float64}(0.)
    for (p, v) in zip(d.p, d.support)
        res[f(v)] += p
    end
    DNP(collect(keys(res)), collect(values(res)))
end

@memoize function sum_many(d::DNP, n::Int)::DNP
    n == 1 ? d : sum_many(d, n-1) + d
end

Base.hash(d::DNP, h::UInt64) = hash_struct(d, h)

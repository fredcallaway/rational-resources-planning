using SplitApplyCombine
flatten = SplitApplyCombine.flatten  # block DataFrames.flatten
using DataStructures: OrderedDict
using Printf
using Serialization

dictkeys(d::Dict) = (collect(keys(d))...,)
dictvalues(d::Dict) = (collect(values(d))...,)

namedtuple(d::Dict{String,T}) where {T} =
    NamedTuple{Symbol.(dictkeys(d))}(dictvalues(d))

namedtuple(d::Dict{Symbol,T}) where {T} =
    NamedTuple{dictkeys(d)}(dictvalues(d))

Base.map(f::Function) = xs -> map(f, xs)
Base.map(f::Type) = xs -> map(f, xs)
Base.map(f, d::Dict) = [f(k, v) for (k, v) in d]

Base.dropdims(idx::Int...) = X -> dropdims(X, dims=idx)
Base.reshape(idx::Union{Int,Colon}...) = x -> reshape(x, idx...)


valmap(f, d::Dict) = Dict(k => f(v) for (k, v) in d)
valmap(f, d::OrderedDict) = OrderedDict(k => f(v) for (k, v) in d)
# valmap(f, d::T) where T <: AbstractDict = T(k => f(v) for (k, v) in d)

valmap(f) = d->valmap(f, d)
keymap(f, d::Dict) = Dict(f(k) => v for (k, v) in d)
juxt(fs...) = x -> Tuple(f(x) for f in fs)
repeatedly(f, n) = [f() for i in 1:n]

nanreduce(f, x) = f(filter(!isnan, x))
nanmean(x) = nanreduce(mean, x)
nanstd(x) = nanreduce(std, x)

function Base.write(fn)
    obj -> open(fn, "w") do f
        write(f, string(obj))
    end
end

# type2dict(x::T) where T = Dict(fn=>getfield(x, fn) for fn in fieldnames(T))

function mutate(x::T; kws...) where T
    return T([get(kws, fn, getfield(x, fn)) for fn in fieldnames(T)]...)
end

getfields(x) = (getfield(x, f) for f in fieldnames(typeof(x)))

# %% ====================  ====================
# ensures consistent hashes across runs
function hash_struct(s, h::UInt64=UInt64(0))
    reduce(getfields(s); init=h) do acc, x
        hash(x, acc)
    end
end

function struct_equal(s1::T, s2::T) where T
    all(getfield(s1, f) == getfield(s2, f)
        for f in fieldnames(T))
end

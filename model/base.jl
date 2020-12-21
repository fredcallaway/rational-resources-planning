if VERSION >= v"1.5"
    using StableHashes
    Base.hash(x, h::UInt64) = shash(x, h)
end

include("utils.jl")
include("conf.jl")
include("mdp.jl")
include("data.jl")

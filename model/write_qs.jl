@everywhere include("base.jl")

mkpath("$base_path/qs/")
n = length(readdir("$base_path/mdps/"))
X = reshape(1:n, :, length(COSTS))

@everywhere data = load_trials(EXPERIMENT) |> values |> flatten |> get_data;

all_qs = pmap(enumerate(eachcol(X))) do (job, idx)
    vs = map(idx) do i
        mdp = deserialize("$base_path/mdps/$i")
        mid = string(hash(mdp))
        deserialize("$base_path/V/$mid")
    end
    cost = vs[1].m.cost
    @assert all(v.m.cost == cost for v in vs)
    value_functions = Dict(identify(v.m) => v for v in vs)

    map(data) do d
        V = value_functions[identify(d.t.m)]
        @assert haskey(V.cache, V.hasher(V.m, d.b))
        Q(V, d.b)
    end
end |> invert

@assert length(all_qs) == length(data)
@assert length(all_qs[1]) == length(COSTS)
Q_tbl = map(data, all_qs) do d, dqs
    hash(d) => Dict(zip(COSTS, dqs))
end |> Dict


serialize("$base_path/Q_table", Q_tbl)
println("Wrote $base_path/Q_table")


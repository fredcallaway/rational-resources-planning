@everywhere include("base.jl")

n = length(readdir("$base_path/mdps/"))
X = reshape(1:n, :, length(COSTS))

pmap(enumerate(eachcol(X))) do (job, idx)
    @time vs = map(idx) do i
        deserialize("$base_path/mdps/$i/V")
    end
    cost = vs[1].m.cost
    @assert all(v.m.cost == cost for v in vs)
    value_functions = Dict(v.m.graph => v for v in vs)

    data = load_trials(EXPERIMENT) |> values |> flatten |> get_data;
    qs = map(data) do d
        V = value_functions[d.t.graph]
        Q(V, d.b)
    end

    mkpath("$base_path/qs/")
    serialize("$base_path/qs/$job", (cost=cost, qs=qs, checksum=checksum(data)))
    println("Wrote $base_path/qs/$job")
end
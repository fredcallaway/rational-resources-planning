function make_Q_table(data)
    n = length(readdir("$base_path/mdps/"))
    X = reshape(1:n, :, length(COSTS))

    all_qs = pmap(enumerate(eachcol(X))) do (job, idx)
        vs = map(idx) do i
            mdp = deserialize("$base_path/mdps/$i")
            mid = string(hash(mdp); base=62)
            deserialize("$base_path/V/$mid")
        end
        cost = vs[1].m.cost
        @assert all(v.m.cost == cost for v in vs)
        value_functions = Dict(v.m => v for v in vs)

        map(data) do d
            V = value_functions[mutate(d.t.m, cost=cost)]
            @assert haskey(V.cache, V.hasher(V.m, d.b))
            Q(V, d.b)
        end
    end |> invert

    @assert length(all_qs) == length(data)
    @assert length(all_qs[1]) == length(COSTS)
    map(data, all_qs) do d, dqs
        q_key(d) => Dict(zip(COSTS, dqs))
    end |> Dict
end


if basename(PROGRAM_FILE) == basename(@__FILE__)
    @everywhere include("base.jl")
    data = load_trials(EXPERIMENT) |> values |> flatten |> get_data;
    serialize("$base_path/Q_table", make_Q_table(data))
    println("Wrote $base_path/Q_table")
end





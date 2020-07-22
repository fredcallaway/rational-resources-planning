
function make_Q_table(data)
    println("Creating Q_table")
    all_mdps = map(deserialize, glob("mdps/withcost/*"))
    base_mdps = unique([d.t.m for d in data])
    filter!(all_mdps) do m
        mutate(m, cost=0.) in base_mdps
    end
    sort!(all_mdps, by=m->m.cost)
    M = reshape(all_mdps, :, length(COSTS))
    grouped_ids = eachcol(id.(M))
    all_qs = pmap(grouped_ids) do ids
        vs = map(load_V_nomem, ids)
        cost = vs[1].m.cost
        @assert all(v.m.cost == cost for v in vs)
        value_functions = Dict(v.m => v for v in vs)
        println("Processing cost: $cost")

        map(data) do d
            V = value_functions[mutate(d.t.m, cost=cost)]
            @assert haskey(V.cache, V.hasher(V.m, d.b))
            Q(V, d.b)
        end
    end |> invert

    @assert length(all_qs) == length(data)
    @assert length(all_qs[1]) == length(COSTS)
    map(data, all_qs) do d, dqs
        hash(d) => Dict(zip(COSTS, dqs))
    end |> Dict
end


if basename(PROGRAM_FILE) == basename(@__FILE__)
    @everywhere include("base.jl")
    data = load_trials(EXPERIMENT) |> values |> flatten |> get_data;
    serialize("$base_path/Q_table", make_Q_table(data))
    println("Wrote $base_path/Q_table")
end





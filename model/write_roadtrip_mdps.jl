function parse_edges(t)
    edges = map(t["edges"]) do (x, y)
        Int(x) + 1, Int(y) + 1
    end
    n_node = maximum(flatten(edges))
    graph = [Int[] for _ in 1:n_node]
    for (a, b) in edges
        push!(graph[a], b)
    end
    graph
end

data = open(JSON.parse, "../data/roadtrip-2.0/trials.json")

lookup = map(first(values(data))) do t
    m = MetaMDP(parse_edges(t), DNP([25,35,50,100]), 0., -300., false)
    serialize("mdps/base/$(id(m))", m)
    map_id = t["map"][13:end-4]
    map_id => id(m)
    # serialize("mdps/$mid", m)
end |> Dict
serialize("mdps/roadtrip_lookup", lookup)

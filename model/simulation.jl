struct Simulator <: Policy
    model::AbstractModel{Float64}
    m::MetaMDP
end

(sim::Simulator)(b::Belief) = rand(Categorical(action_dist(sim.model, sim.m, b))) - 1

function best_path(m, b)
    i = map(paths(m)) do path
        path_value(m, b, path)
    end |> argmax
    paths(m)[i]
end

function simulate(sim::Simulator, wid::String)
    bs = Belief[]
    cs = Int[]
    rollout(sim) do b, c
        push!(bs, deepcopy(b)); push!(cs, c)
    end
    Trial(sim.m, wid, bs, cs, NaN, [], best_path(sim.m, bs[end]))
end

simulate(model::AbstractModel, m::MetaMDP; wid=string(typeof(model).name)) = simulate(Simulator(model, m), wid)

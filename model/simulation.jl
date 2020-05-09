struct Simulator <: Policy
    model::AbstractModel{Float64}
    m::MetaMDP
end

(sim::Simulator)(b::Belief) = rand(Categorical(action_dist(sim.model, sim.m, b))) - 1

function simulate(sim::Simulator, wid::String)
    bs = Belief[]
    cs = Int[]
    rollout(sim) do b, c
        push!(bs, deepcopy(b)); push!(cs, c)
    end
    Trial(sim.m, wid, bs, cs, [], [])
end

simulate(model::AbstractModel, m::MetaMDP; wid=string(typeof(model).name)) = simulate(Simulator(model, m), wid)

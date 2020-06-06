include("explore_reward_structure_base.jl")
using JSON

function sample_rewards(factor; N=100)
    r = make_mdp(factor, 1).rewards
    [Int.(rand.(r)) for i in 1:N]
end
rewards = Dict(factor => sample_rewards(factor) for factor in [1//3, 3])
write("/Users/fred/heroku/webofcash2/rewards.json", json(rewards))

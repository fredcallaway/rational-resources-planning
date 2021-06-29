function belief_tree(m, b)
    function rec(i)
        (observed(b, i) ? b[i] : m.rewards[i], 
         Tuple(rec(child) for child in m.graph[i]))
    end
    rec(1)
end

function tree_value_dist(btree)
    self, children = btree
    isempty(children) && return self # base case
    self + maximum(map(tree_value_dist, children))
end

function best_path_value_dist(m, b)
    tree_value_dist(belief_tree(m, b))
end

function prob_path_maximal(m, b)
    rewards = [-10, -5, 5, 10]
    @assert EXPERIMENT == "exp1"
    map(paths(m)) do pth
        marginalizing = filter(i->!observed(b, i), pth)
        b1 = copy(b)
        mapreduce(+, Iterators.product(fill(rewards, length(marginalizing))...)) do z
            b1[marginalizing] .= z
            competing_value = best_path_value_dist(m, b1)
            own_value = path_value(m, b1, pth)
            cdf(competing_value, own_value)
        end * (.25 ^ length(marginalizing))
    end
end

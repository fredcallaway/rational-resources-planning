EXPERIMENT = "exp2"
COSTS = [0:0.05:4; 100]
MAX_THETA = 60
EXPAND_ONLY = true
FIT_BIAS = false

QUOTE_MODELS = quote 
    [
        RandomSelection,
        OptimalPlus,
        MetaGreedy,
        Heuristic{:BestFirst},
        Heuristic{:DepthFirst},
        Heuristic{:BreadthFirst},
        Heuristic{:BestPlusDepth},
        Heuristic{:BestPlusBreadth},
        Heuristic{:Full},
    ] 
end


EXPERIMENT = "exp3"
COSTS = [0:0.05:4; 100]
# COSTS = [0:0.05:4; 100]
MAX_THETA = 60
EXPAND_ONLY = false
FIT_BIAS = true

MDP_IDS = [
    "6oCv6ld5V89",
    "IXcheAhOWH7",
    "95YVmaymRm9",
]

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
    ] 
end


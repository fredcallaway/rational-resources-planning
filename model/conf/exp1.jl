EXPERIMENT = "exp1"
COSTS = [0:0.05:4; 100]
MAX_THETA = 60
EXPAND_ONLY = true
FIT_BIAS = false

QUOTE_MODELS = quote   # this quote thing allows us to refer to types that aren't defined yet
    [
        OptimalPlus{:Default},
        MetaGreedy{:Default},
        Heuristic{:Random},
        Heuristic{:BestFirst},
        Heuristic{:BestFirstNoBestNext},
    ]
end
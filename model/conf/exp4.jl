EXPERIMENT = "exp4"
EXPAND_ONLY = false
FOLDS = 7

QUOTE_MODELS = quote 
    [
        OptimalPlus{:Default},
        MetaGreedy{:Default},
        Heuristic{:Random},
        Heuristic{:BestFirst},
        Heuristic{:DepthFirst},
        Heuristic{:BreadthFirst},
        OptimalPlus{:Expand},
        MetaGreedy{:Expand},
        Heuristic{:RandomExpand},
        Heuristic{:BestFirstExpand},
        Heuristic{:DepthFirstExpand},
        Heuristic{:BreadthFirstExpand},
    ] 
end

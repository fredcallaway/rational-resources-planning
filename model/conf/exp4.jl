EXPERIMENT = "exp4"
EXPAND_ONLY = false
FOLDS = 7

QUOTE_MODELS = quote 
    [
        RandomSelection,
        OptimalPlus{:Default},
        MetaGreedy{:Default},
        Heuristic{:BestFirst},
        Heuristic{:DepthFirst},
        Heuristic{:BreadthFirst},
        OptimalPlus{:Expand},
        MetaGreedy{:Expand},
        Heuristic{:BestFirstExpand},
        Heuristic{:DepthFirstExpand},
        Heuristic{:BreadthFirstExpand},
    ] 
end

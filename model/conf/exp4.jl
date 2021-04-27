EXPERIMENT = "exp4"
EXPAND_ONLY = false
FOLDS = 7

QUOTE_MODELS = quote 
    [
        OptimalPlus{:Default},
        MetaGreedy{:Default},
        Heuristic{:Random},

        OptimalPlus{:Expand},
        MetaGreedy{:Expand},
        Heuristic{:Expand},
    ]
end

QUOTE_PARETO_MODELS = quote
    [
        RandomSelection,
        MetaGreedy,

        Heuristic{:Best_Satisfice_BestNext},
        Heuristic{:Depth_Satisfice_BestNext},
        Heuristic{:Breadth_Satisfice_BestNext},
    ]
end
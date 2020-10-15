EXPERIMENT = "exp4"
EXPAND_ONLY = false
FOLDS = 7

QUOTE_MODELS = quote 
    [
        OptimalPlus{:Default},
        MetaGreedy{:Default},
        Heuristic{:Random},
        Heuristic{:Best_Satisfice_BestNext},
        Heuristic{:Breadth_Satisfice_BestNext},
        Heuristic{:Depth_Satisfice_BestNext},

        OptimalPlus{:Expand},
        MetaGreedy{:Expand},
        Heuristic{:Expand},

        Heuristic{:Best_Satisfice_BestNext_Expand},
        Heuristic{:Breadth_Satisfice_BestNext_Expand},
        Heuristic{:Depth_Satisfice_BestNext_Expand},
    ] 
end

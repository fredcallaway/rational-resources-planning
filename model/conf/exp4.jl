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

        all_heuristic_models()...
    ]
end

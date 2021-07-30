EXPERIMENT = "exp1"
EXPAND_ONLY = true


#SKIP_GROUP = true
#SKIP_FULL = true

# this quote thing allows us to refer to types that aren't defined yet
QUOTE_MODELS = quote 
    [
        OptimalPlus{:Default},
        MetaGreedy{:Default},
        Heuristic{:Random},
        all_heuristic_models()...,
        all_fancy_heuristic_models()...
    ] 
end

QUOTE_PARETO_MODELS = quote
    [
        RandomSelection,
        MetaGreedy,
        Heuristic{:Best_Full},
    ]
end
EXPERIMENT = "exp1"
EXPAND_ONLY = true

# this quote thing allows us to refer to types that aren't defined yet
QUOTE_MODELS = quote 
    [
        #OptimalPlus{:Default},
        MetaGreedy{:Default},
        Heuristic{:Random},
        Heuristic{:Best_BestNext_Satisfice},


        #all_heuristic_models()...,
        #all_fancy_heuristic_models()...
    ] 
end

QUOTE_PARETO_MODELS = quote
    [
        RandomSelection,
        MetaGreedy,
        Heuristic{:Best_Full},
    ]
end
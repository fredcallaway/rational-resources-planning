EXPERIMENT = "exp1"
EXPAND_ONLY = true
MAX_VALUE = 30.
MAX_DEPTH = 3.

#SKIP_GROUP = true
#SKIP_FULL = true
#SKIP_CV = true

# this quote thing allows us to refer to types that aren't defined yet
QUOTE_MODELS = quote 
    [
        all_fancy_heuristic_models(["Best"])...,
    ] 
end

QUOTE_PARETO_MODELS = quote
    [
        RandomSelection,
        MetaGreedy,
        Heuristic{:Best_Full},
    ]
end
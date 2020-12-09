EXPERIMENT = "exp2"
EXPAND_ONLY = true
FIT_BIAS = false

QUOTE_MODELS = quote 
    [
        OptimalPlus{:Default},
        MetaGreedy{:Default},
        Heuristic{:Random},

        all_heuristic_models()...
    ] 
end